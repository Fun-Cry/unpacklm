import torch
import numpy as np

class ValueStateProcessor:
    def __init__(self, storage, model_id, head_dim, streamer,
                 target_samples_per_head=2000, 
                 total_sentences_to_process=1000, samples_per_collect=1):
        self.storage = storage
        self.model_id = model_id
        self.head_dim = head_dim
        self.streamer = streamer
        self.samples_per_collect = samples_per_collect
        
        collections_needed = target_samples_per_head / samples_per_collect
        self.selection_prob = min(max(collections_needed / total_sentences_to_process, 0.0), 1.0)
        
        # Streaming state (set per batch)
        self._selection_mask = None
        self._should_process = False
        self._final_var = None
        self._seq_lens = None
        self._q_padding_mask = None
        self._layer_comp_stds = {}

    # ==========================================
    #  Streaming Mode
    # ==========================================

    def on_batch_start(self, batch_size, num_heads, num_layers, sentences, seq_lens):
        """Initialize per-batch sampling state."""
        rand_matrix = torch.rand((batch_size, num_heads))
        self._selection_mask = rand_matrix < self.selection_prob
        self._should_process = self._selection_mask.any().item()
        self._seq_lens = seq_lens
        self._layer_comp_stds = {}
        
        if self._should_process:
            final_logits = self.streamer.get_unnormalized_logits()
            self._final_var = final_logits.var(dim=-1, unbiased=False).cpu().detach().numpy()
    
    def on_component(self, layer_idx, name, comp_std, centered_logits,
                     sentence_ids, sentences, seq_lens):
        """Accumulate comp_stds (tiny: (B, H, Q) per component)."""
        if not self._should_process:
            return
        if layer_idx not in self._layer_comp_stds:
            self._layer_comp_stds[layer_idx] = []
        self._layer_comp_stds[layer_idx].append(comp_std)
    
    def on_layer_complete(self, layer_idx, final_attention, value_states,
                          residual_values_np, weighted_values_np, total_std,
                          sentence_ids, sentences, seq_lens):
        """Process this layer's value state sampling."""
        if not self._should_process:
            return
        
        device = value_states.device
        comp_stds_list = self._layer_comp_stds.pop(layer_idx, [])
        if not comp_stds_list:
            return
        
        comp_stds = np.stack(comp_stds_list, axis=-1)  # (B, H, Q, num_components)
        
        final_attention_t = final_attention.to(device)
        selection_mask = self._selection_mask.to(device)
        
        # Build padding mask
        Q = final_attention.shape[2]
        if seq_lens is not None:
            seq_lens_t = torch.tensor(seq_lens, device=device)
            q_mask = torch.arange(Q, device=device).unsqueeze(0) < seq_lens_t.unsqueeze(1)
            q_padding_mask = q_mask[:, None, :, None].float()
            final_attention_t = final_attention_t * q_padding_mask
        
        w_mask = self._generate_mask(final_attention_t, selection_mask, method='weighted')
        w_stds, w_vals, w_meta, w_var = self._retrieve(comp_stds, value_states, w_mask, self._final_var)
        self._save_flat(layer_idx, 'weighted', w_stds, w_vals, w_meta, w_var, None)

        u_mask = self._generate_mask(final_attention_t, selection_mask, method='uniform')
        u_stds, u_vals, u_meta, u_var = self._retrieve(comp_stds, value_states, u_mask, self._final_var)
        self._save_flat(layer_idx, 'uniform', u_stds, u_vals, u_meta, u_var, None)
    
    def on_batch_complete(self):
        self._selection_mask = None
        self._final_var = None
        self._layer_comp_stds = {}

    # ==========================================
    #  Internal Methods
    # ==========================================

    def _generate_mask(self, attention_weights, pre_selection_mask, method='weighted'):
        b, h, s_q, s_k = attention_weights.shape
        flat_weights = attention_weights.view(b * h, -1)
        flat_selection_mask = pre_selection_mask.view(-1)
        
        if method == 'weighted':
            probs = flat_weights
        else:
            probs = (flat_weights > 0).float()
            row_sums = probs.sum(dim=-1, keepdim=True)
            probs = probs / (row_sums + 1e-9)

        prob_sums = probs.sum(dim=-1)
        valid_rows_mask = (prob_sums > 0) & flat_selection_mask
        
        valid_probs = probs[valid_rows_mask]
        sampled_indices = torch.zeros((b * h, self.samples_per_collect), dtype=torch.long, device=attention_weights.device)
        
        if valid_probs.shape[0] > 0:
            if s_k < self.samples_per_collect:
                valid_samples = torch.multinomial(valid_probs, self.samples_per_collect, replacement=True)
            else:
                valid_samples = torch.multinomial(valid_probs, self.samples_per_collect, replacement=False)
            sampled_indices[valid_rows_mask] = valid_samples

        mask_flat = torch.zeros_like(flat_weights, dtype=torch.bool)
        row_indices = torch.arange(b * h, device=attention_weights.device).unsqueeze(1).expand(-1, self.samples_per_collect)
        valid_row_indices = row_indices[valid_rows_mask].flatten()
        valid_col_indices = sampled_indices[valid_rows_mask].flatten()
        mask_flat.index_put_((valid_row_indices, valid_col_indices), torch.tensor(True, device=attention_weights.device))
        return mask_flat.view(b, h, s_q, s_k)

    def _retrieve(self, comp_stds, value_states, indices, final_var):
        nonzero = torch.nonzero(indices, as_tuple=True)
        batch_idx, head_idx, query_idx, key_idx = nonzero
        
        selected_stds = comp_stds[batch_idx.cpu().numpy(), head_idx.cpu().numpy(), query_idx.cpu().numpy()]
        selected_vals = value_states[batch_idx, head_idx, key_idx]
        metadata = torch.stack([batch_idx, head_idx, query_idx, key_idx], dim=1)
        selected_var = final_var[batch_idx.cpu().numpy(), query_idx.cpu().numpy()]
        
        return selected_stds, selected_vals, metadata, selected_var

    def _save_flat(self, layer_idx, method, stds, vals, idxs, final_var, component_names):
        if vals.shape[0] == 0: 
            return
        
        layer_col = torch.full((idxs.shape[0], 1), layer_idx, device=idxs.device, dtype=idxs.dtype)
        idxs_extended = torch.cat([idxs, layer_col], dim=1)

        base_path = f"model_{self.model_id}/value_states/{method}"
        
        self.storage.log_tensor(f"{base_path}/vectors", vals.cpu().numpy())
        self.storage.log_tensor(f"{base_path}/indices", idxs_extended.cpu().numpy())

        stds_list = list(stds) 
        self.storage.log_variable_length_tensor(f"{base_path}/component_stds", stds_list)
        
        var_array = final_var.astype(np.float32).reshape(-1, 1)
        self.storage.log_tensor(f"{base_path}/final_variance", var_array)

    def close(self):
        pass


class HeadPreferenceProcessor:
    def __init__(self, storage, model_id):
        self.storage = storage
        self.model_id = model_id
        
        self.num_layers = None
        self.num_heads = None
        self.component_names = None
        self.comp_to_idx = None
        self.num_components = None
        self.initialized = False
        
        # Streaming state
        self._batch_tensor = None      # (B, Q, num_layers, num_heads, num_components)
        self._batch_sentences = None
        self._seq_lens = None
        self._max_seq_len = None

    # ==========================================
    #  Streaming Mode
    # ==========================================

    def on_batch_start(self, batch_size, num_heads, num_layers, sentences, seq_lens):
        """Initialize per-batch accumulator."""
        self.num_heads = num_heads
        self.num_layers = num_layers
        self._batch_sentences = sentences
        self._seq_lens = seq_lens

    def on_component(self, layer_idx, name, comp_std, centered_logits,
                     sentence_ids, sentences, seq_lens):
        """Store per-token std for each (layer, component, head) combination.
        
        Skips norm_bias and attn_bias — these are decomposition artifacts,
        not residual stream components.
        """
        if name in ('norm_bias', 'attn_bias'):
            return
        # Lazy init: build component name mapping from first layer's components
        if self.comp_to_idx is None:
            self.comp_to_idx = {}
        
        if name not in self.comp_to_idx:
            self.comp_to_idx[name] = len(self.comp_to_idx)
        
        comp_idx = self.comp_to_idx[name]
        
        B, H, Q = comp_std.shape
        
        if self._max_seq_len is None:
            self._max_seq_len = Q
        else:
            self._max_seq_len = max(self._max_seq_len, Q)
        
        # Lazy init batch tensor: (B, Q, num_layers, num_heads, num_components)
        if self._batch_tensor is None:
            max_comps = max(len(self.comp_to_idx), comp_idx + 1)
            self._batch_tensor = np.zeros(
                (B, self._max_seq_len, self.num_layers, self.num_heads, max_comps),
                dtype=np.float32
            )
        
        # Grow Q dimension if needed
        if Q > self._batch_tensor.shape[1]:
            new_tensor = np.zeros(
                (B, Q, self.num_layers, self.num_heads, self._batch_tensor.shape[4]),
                dtype=np.float32
            )
            new_tensor[:, :self._batch_tensor.shape[1], :, :, :] = self._batch_tensor
            self._batch_tensor = new_tensor
        
        # Grow component dimension if needed
        if comp_idx >= self._batch_tensor.shape[4]:
            new_tensor = np.zeros(
                (B, self._batch_tensor.shape[1], self.num_layers, self.num_heads, comp_idx + 10),
                dtype=np.float32
            )
            new_tensor[:, :, :, :, :self._batch_tensor.shape[4]] = self._batch_tensor
            self._batch_tensor = new_tensor
        
        # Store per-token std: comp_std is (B, H, Q), transpose to (B, Q, H)
        std_transposed = comp_std.transpose(0, 2, 1)
        
        # Zero out padding positions
        if seq_lens is not None:
            mask = np.arange(Q)[None, :] < np.array(seq_lens)[:, None]
            std_transposed = np.where(mask[:, :, None], std_transposed, 0.0)
        
        self._batch_tensor[:, :Q, layer_idx, :, comp_idx] = std_transposed
    
    def on_layer_complete(self, layer_idx, final_attention, value_states,
                          residual_values_np, weighted_values_np, total_std,
                          sentence_ids, sentences, seq_lens):
        pass

    def on_batch_complete(self):
        """Flush per-token computational graphs to storage."""
        if self._batch_tensor is None:
            return
        
        # Finalize component names
        self.component_names = [''] * len(self.comp_to_idx)
        for name, idx in self.comp_to_idx.items():
            self.component_names[idx] = name
        self.num_components = len(self.component_names)
        
        batch_tensor = self._batch_tensor[:, :, :, :, :self.num_components]
        
        B, Q = batch_tensor.shape[0], batch_tensor.shape[1]
        sentence_ids = self.storage.register_sentences(self._batch_sentences)
        
        if self._seq_lens is not None:
            valid_mask = np.arange(Q)[None, :] < np.array(self._seq_lens)[:, None]
        else:
            valid_mask = np.ones((B, Q), dtype=bool)
        
        batch_indices, token_indices = np.where(valid_mask)
        
        valid_tensor = batch_tensor[batch_indices, token_indices]
        
        valid_sentence_ids = np.array(
            [sentence_ids[b] for b in batch_indices], dtype=np.int64
        ).reshape(-1, 1)
        
        valid_token_indices = token_indices.astype(np.int32).reshape(-1, 1)
        
        if self._seq_lens is not None:
            valid_seq_lens = np.array(
                [self._seq_lens[b] for b in batch_indices], dtype=np.int32
            ).reshape(-1, 1)
        else:
            valid_seq_lens = np.full((len(batch_indices), 1), -1, dtype=np.int32)
        
        base_path = f"model_{self.model_id}/head_preferences"
        self.storage.log_tensor(
            base_path, valid_tensor,
            metadata={"component_names_json": str(self.component_names)}
        )
        self.storage.log_tensor(f"{base_path}_sentence_ids", valid_sentence_ids)
        self.storage.log_tensor(f"{base_path}_token_indices", valid_token_indices)
        self.storage.log_tensor(f"{base_path}_seq_lens", valid_seq_lens)
        
        self._batch_tensor = None
        self._max_seq_len = None

    def close(self):
        pass


class MLPIntermediateProcessor:
    """Track per-component element-wise L2 norms through combined LN + W_up.

    For each target MLP layer, every residual stream component is marginal-
    normalized (center, divide by stream variance treated as constant, scale
    by LN weight) then projected through dense_h_to_4h.  The L2 norm of the
    resulting intermediate vector is stored per (token, target_layer, component).

    This mirrors HeadPreferenceProcessor but for the MLP pathway — no heads
    dimension, L2 norms instead of attention stds.

    Stored tensors (all flattened to valid tokens, excluding padding):
      - mlp_intermediate/norms:          (N_tokens, L, C) — per-component L2 norms
      - mlp_intermediate/sentence_ids:   (N_tokens, 1)
      - mlp_intermediate/token_indices:  (N_tokens, 1)
      - mlp_intermediate/seq_lens:       (N_tokens, 1)
    """

    def __init__(self, storage, model_id):
        self.storage = storage
        self.model_id = model_id

        self.num_layers = None
        self.comp_to_idx = None

        # Per-batch state
        self._batch_tensor = None      # (B, Q, L, C)
        self._batch_sentences = None
        self._seq_lens = None
        self._max_seq_len = None

    # ==========================================
    #  Streaming callbacks
    # ==========================================

    def on_batch_start(self, batch_size, num_heads, num_layers, sentences, seq_lens):
        self.num_layers = num_layers
        self._batch_sentences = sentences
        self._seq_lens = seq_lens
        self._batch_tensor = None
        self._max_seq_len = None
        self.comp_to_idx = None

    def on_component(self, layer_idx, name, comp_std, centered_logits,
                     sentence_ids, sentences, seq_lens):
        pass  # Attention-side callback, not used here

    def on_mlp_norms(self, layer_idx, names, norms_np,
                     sentence_ids, sentences, seq_lens, is_last_group):
        """Accumulate per-component L2 norms for one source group × target layer.

        Args:
            layer_idx:      target MLP layer
            names:          list[str] component names (may include norm_bias, mlp_up_bias)
            norms_np:       (B, S, C_group) array of L2 norms
            is_last_group:  True when all source groups for this target have arrived
        """
        mask = [i for i, n in enumerate(names) if n not in ('mlp_norm_bias', 'mlp_up_bias')]
        names, norms_np = [names[i] for i in mask], norms_np[:, :, mask]

        if self.comp_to_idx is None:
            self.comp_to_idx = {}

        for name in names:
            if name not in self.comp_to_idx:
                self.comp_to_idx[name] = len(self.comp_to_idx)

        B, S, C_group = norms_np.shape

        if self._max_seq_len is None:
            self._max_seq_len = S
        else:
            self._max_seq_len = max(self._max_seq_len, S)

        # Lazy init: (B, Q, L, C)
        if self._batch_tensor is None:
            max_comps = len(self.comp_to_idx)
            self._batch_tensor = np.zeros(
                (B, self._max_seq_len, self.num_layers, max_comps),
                dtype=np.float32,
            )

        # Grow Q dimension if needed
        if S > self._batch_tensor.shape[1]:
            new = np.zeros(
                (B, S, self.num_layers, self._batch_tensor.shape[3]),
                dtype=np.float32,
            )
            new[:, :self._batch_tensor.shape[1], :, :] = self._batch_tensor
            self._batch_tensor = new

        # Grow component dimension if needed
        max_idx = max(self.comp_to_idx[n] for n in names)
        if max_idx >= self._batch_tensor.shape[3]:
            new = np.zeros(
                (B, self._batch_tensor.shape[1], self.num_layers, max_idx + 10),
                dtype=np.float32,
            )
            new[:, :, :, :self._batch_tensor.shape[3]] = self._batch_tensor
            self._batch_tensor = new

        # Store: norms_np is (B, S, C_group)
        for i, name in enumerate(names):
            cidx = self.comp_to_idx[name]
            self._batch_tensor[:, :S, layer_idx, cidx] = norms_np[:, :, i]

    def on_layer_complete(self, layer_idx, final_attention, value_states,
                          residual_values_np, weighted_values_np, total_std,
                          sentence_ids, sentences, seq_lens):
        pass

    def on_batch_complete(self):
        """Flatten (B, Q) to valid tokens and store to HDF5."""
        if self._batch_tensor is None:
            return

        component_names = [''] * len(self.comp_to_idx)
        for name, idx in self.comp_to_idx.items():
            component_names[idx] = name
        num_components = len(component_names)

        batch_tensor = self._batch_tensor[:, :, :, :num_components]

        B, Q = batch_tensor.shape[0], batch_tensor.shape[1]
        sentence_ids = self.storage.register_sentences(self._batch_sentences)

        if self._seq_lens is not None:
            valid_mask = np.arange(Q)[None, :] < np.array(self._seq_lens)[:, None]
        else:
            valid_mask = np.ones((B, Q), dtype=bool)

        batch_indices, token_indices = np.where(valid_mask)
        valid_tensor = batch_tensor[batch_indices, token_indices]

        valid_sentence_ids = np.array(
            [sentence_ids[b] for b in batch_indices], dtype=np.int64
        ).reshape(-1, 1)
        valid_token_indices = token_indices.astype(np.int32).reshape(-1, 1)

        if self._seq_lens is not None:
            valid_seq_lens = np.array(
                [self._seq_lens[b] for b in batch_indices], dtype=np.int32
            ).reshape(-1, 1)
        else:
            valid_seq_lens = np.full((len(batch_indices), 1), -1, dtype=np.int32)

        base_path = f"model_{self.model_id}/mlp_intermediate"
        self.storage.log_tensor(
            f"{base_path}/norms", valid_tensor,
            metadata={
                "component_names_json": str(component_names),
                "description": (
                    "Per-component element-wise L2 norms after combined "
                    "post_attention_layernorm + dense_h_to_4h linear map. "
                    "LN variance treated as constant (marginal)."
                ),
            },
        )
        self.storage.log_tensor(f"{base_path}/sentence_ids", valid_sentence_ids)
        self.storage.log_tensor(f"{base_path}/token_indices", valid_token_indices)
        self.storage.log_tensor(f"{base_path}/seq_lens", valid_seq_lens)

        self._batch_tensor = None
        self._max_seq_len = None

    def close(self):
        pass
