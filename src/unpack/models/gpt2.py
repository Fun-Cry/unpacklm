import torch
import torch.nn as nn
from typing import Dict, Any
import torch.nn.functional as F

from unpack.models.base import ModelAdapter


class GPT2Adapter(ModelAdapter):
    """
    Hook manager for GPT-2 models (HuggingFace GPT2LMHeadModel).

    Architectural differences from GPT-NeoX / Pythia:
      - Sequential residual (MLP sees attention output, not parallel)
      - Conv1D projections (weight shape is transposed vs Linear)
      - Learned absolute position embeddings (no RoPE)
      - Tied embeddings (lm_head.weight == wte.weight)
      - No biases in layernorm (GPT-2 does have LN bias, unlike Gemma)

    Hook points:
        embedding              - token + position embedding (after dropout)
        attn_{L}_input         - hidden states entering attention
        attn_{L}_pre_dense     - merged heads before c_proj
        attn_{L}               - attention output after c_proj
        mlp_{L}                - MLP sublayer output
        resid_{L}              - residual stream after layer L
    """

    def __init__(self, capture_mlp_intermediates=False):
        super().__init__()
        self.model: nn.Module = None
        self.mlp_outputs = []
        self.embedding_outputs = []
        self.position_embedding_outputs = []
        self.pre_dense_inputs = []
        self.attention_input_cache: Dict[int, Dict[str, Any]] = {}
        self.pre_final_norm = None
        self.device: torch.device = None

        self.capture_mlp_intermediates = capture_mlp_intermediates
        self.mlp_intermediate_outputs = []

    def register_hooks(self, model):
        self.model = model
        self.device = model.device

        # Token embedding and position embedding as separate components
        self._wire_hook(model.transformer.wte, "embedding", "output",
                        "embedding_outputs")
        self._wire_hook(model.transformer.wpe, "pos_embedding", "output",
                        "position_embedding_outputs")

        for i, block in enumerate(model.transformer.h):
            # Attention input (captures hidden_states entering attention)
            self._wire_attn_input_hook(block.attn, i)

            # Pre-dense: input to c_proj (merged heads)
            self._wire_hook(block.attn.c_proj, f"attn_{i}_pre_dense",
                            "input", "pre_dense_inputs")

            # Attention output
            self._wire_hook(block.attn, f"attn_{i}", "tuple_output")

            # Layer input (for interventions)
            self._wire_hook(block, f"layer_{i}_input", "input")

            # MLP output
            self._wire_hook(block.mlp, f"mlp_{i}", "output",
                            "mlp_outputs")

            # Residual after full block (Block returns a tuple)
            self._wire_hook(block, f"resid_{i}", "tuple_output")

            # Layernorm inputs (for selective ablation)
            self._wire_hook(block.ln_1, f"attn_ln_{i}_input", "input")
            self._wire_hook(block.ln_2, f"mlp_ln_{i}_input", "input")

            if self.capture_mlp_intermediates:
                def _mlp_hook(mod, inp, out, _i=i):
                    self.mlp_intermediate_outputs.append(
                        out.detach().clone().cpu())
                self.handles.append(
                    block.mlp.c_fc.register_forward_hook(_mlp_hook))

        # Pre-final-norm capture
        def pre_final_norm_hook(mod, inp):
            self.pre_final_norm = inp[0].detach().clone()
        self.handles.append(
            model.transformer.ln_f.register_forward_pre_hook(
                pre_final_norm_hook))

    def _wire_attn_input_hook(self, attention_module, layer_idx):
        name = f"attn_{layer_idx}_input"

        def hook(mod, args, kwargs, _n=name, _i=layer_idx):
            hidden_states = args[0]
            self.attention_input_cache[_i] = {
                "hidden_states": hidden_states,
                "attention_mask": kwargs.get("attention_mask"),
            }
            result, changed = self._run_hook(_n, hidden_states)
            if changed:
                self.attention_input_cache[_i]["hidden_states"] = result
                return (result,) + args[1:], kwargs
        self.handles.append(
            attention_module.register_forward_pre_hook(hook,
                                                      with_kwargs=True))

    def clear(self):
        super().clear()
        self.mlp_outputs = []
        self.embedding_outputs = []
        self.position_embedding_outputs = []
        self.pre_dense_inputs = []
        self.attention_input_cache = {}
        self.pre_final_norm = None
        self.mlp_intermediate_outputs = []

    # ==========================================
    #  Streaming Source Groups
    # ==========================================

    def iter_source_groups(self):
        # Token embedding + position embedding as separate components
        if self.embedding_outputs and self.position_embedding_outputs:
            tok_emb = self.embedding_outputs[0].cpu()    # (B, S, D)
            pos_emb = self.position_embedding_outputs[0].cpu()  # (1, S, D) or (B, S, D)
            # Expand pos_emb to match batch if needed
            if pos_emb.shape[0] != tok_emb.shape[0]:
                pos_emb = pos_emb.expand_as(tok_emb)
            group = torch.stack([tok_emb, pos_emb], dim=2)  # (B, S, 2, D)
            self.embedding_outputs = []
            self.position_embedding_outputs = []
            yield group, ['embedding', 'pos_embedding'], -1

        num_layers = len(self.mlp_outputs)
        num_heads = self.model.config.n_head
        head_size = self.model.config.n_embd // num_heads

        for i in range(num_layers):
            block = self.model.transformer.h[i]
            merged = self.pre_dense_inputs[i]  # (B, S, d_model)

            # Conv1D c_proj: weight (d_model, d_model)
            # Per-head output: h_j @ W_O[j*hs:(j+1)*hs, :]
            W_O = block.attn.c_proj.weight  # (d_model, d_model) Conv1D
            b_O = block.attn.c_proj.bias    # (d_model,)

            unmerged = merged.view(*merged.shape[:-1], num_heads, head_size)

            components = []
            names = []
            for j in range(num_heads):
                h_j = unmerged[..., j, :]  # (B, S, head_size)
                W_O_j = W_O[j * head_size:(j + 1) * head_size, :]
                contrib = (h_j @ W_O_j).cpu()  # (B, S, d_model)
                components.append(contrib)
                names.append(f'attn_{i}_head_{j}')

            components.append(b_O.expand_as(self.mlp_outputs[i]).cpu())
            names.append(f'attn_{i}_bias')

            components.append(self.mlp_outputs[i].cpu())
            names.append(f'mlp_{i}')

            group_tensor = torch.stack(components, dim=2)
            del components

            self.pre_dense_inputs[i] = None
            self.mlp_outputs[i] = None

            yield group_tensor, names, i

    def free_attention_cache(self, layer_idx):
        if layer_idx in self.attention_input_cache:
            del self.attention_input_cache[layer_idx]

    # ==========================================
    #  Cross-Attention & Projection
    # ==========================================

    def get_cross_attention_scores(self, layer_idx, new_input_states,
                                   include_bias=False, side="key"):
        """Cross-attention scoring on either the key side or the query side.

        Computes attention logits where ONE side (key or query) is replaced
        with a per-component projection of `new_input_states`, while the
        OTHER side is held fixed at its original forward-pass value.

        Args:
            layer_idx:         attention layer to score against
            new_input_states:  (B, S*C, d_model) — component residuals at
                               the position to be decomposed; flattened
                               along (S, C) where C = num components
            include_bias:      if True, append the bias of the projected
                               side as an extra component (the "norm_bias"
                               / "attn_bias" placeholder slot)
            side:              "key"   → decompose K[s], hold Q[q] fixed
                                         (existing behavior; surfaces
                                          components writing to source
                                          residual)
                               "query" → decompose Q[q], hold K[s] fixed
                                         (new; surfaces components writing
                                          to query residual, e.g. S-inhibition
                                          heads writing at END that affect
                                          downstream queries at END)

        Returns:
            (attn_weights, attention_mask, v_states)
            attn_weights:   (B, H, Q, K * C_eff)   — per-component logits
                            with the decomposed side spread along the K-axis
                            of the tensor (regardless of which side it
                            actually decomposed; downstream callers reshape)
            attention_mask: causal mask
            v_states:       (B, H, K, D)  — original value states
        """
        if side not in ("key", "query"):
            raise ValueError(f"side must be 'key' or 'query', got {side!r}")

        new_input_states = new_input_states.to(self.device)
        if layer_idx not in self.attention_input_cache:
            raise KeyError(f"No cached inputs for layer {layer_idx}.")

        inputs = self.attention_input_cache[layer_idx]
        attn = self.model.transformer.h[layer_idx].attn
        original_hidden = inputs["hidden_states"]
        attention_mask = inputs["attention_mask"]

        num_heads = self.model.config.n_head
        head_size = self.model.config.n_embd // num_heads
        b, seq_len, d = original_hidden.shape

        # GPT-2 c_attn is Conv1D: x @ W + b, W shape (d_model, 3*d_model)
        W = attn.c_attn.weight     # (d_model, 3*d_model)
        bias = attn.c_attn.bias    # (3*d_model,)

        # Original Q, K, V (with bias) — needed to hold the non-decomposed
        # side fixed and to return v_states for downstream value attribution.
        qkv_orig = original_hidden @ W + bias
        q_orig, k_orig, v_orig = qkv_orig.split(d, dim=-1)
        v_states = v_orig.view(b, seq_len, num_heads, head_size).transpose(1, 2)

        # Decomposed-side projection of `new_input_states`. No bias by
        # default, since per-component decomposition treats the bias as
        # a separate "component" appended via include_bias.
        chunk_size = new_input_states.shape[1] // seq_len
        if side == "key":
            W_side = W[:, d:2 * d]
            b_side = bias[d:2 * d]
        else:  # side == "query"
            W_side = W[:, :d]
            b_side = bias[:d]
        qkv_side = new_input_states @ W_side                # (B, S*C, d_model)
        qkv_side = qkv_side.view(b, seq_len, chunk_size, d)

        if include_bias:
            expanded_bias = b_side[None, None, None, :].expand(
                b, seq_len, 1, -1).to(self.device)
            qkv_side = torch.cat([qkv_side, expanded_bias], dim=2)
        qkv_side = qkv_side.reshape(b, -1, d)
        side_states = qkv_side.view(b, -1, num_heads, head_size).transpose(1, 2)

        # Combine with the held-fixed side. The output convention is
        # (B, H, Q, K * C_eff) for K-side (last axis groups k_position then
        # component) and (B, H, Q * C_eff, K) for Q-side (intermediate axis
        # groups q_position then component). The caller reshapes to a 5-tensor
        # of shape (B, H, Q, K, C_eff) which is the same regardless of side.
        scaling = 1.0 / (head_size ** 0.5)
        if side == "key":
            # Q held fixed at original; K decomposed.
            # side_states: (B, H, S*C_eff, D)  along the source axis
            # query:       (B, H, S, D)        original
            # output:      (B, H, S, S*C_eff)
            query = q_orig.view(b, seq_len, num_heads, head_size).transpose(1, 2)
            attn_weights = torch.matmul(query, side_states.transpose(2, 3)) * scaling
        else:
            # K held fixed at original; Q decomposed.
            # side_states: (B, H, S*C_eff, D)  along the query axis
            # key:         (B, H, S, D)        original
            # output:      (B, H, S*C_eff, S)
            key = k_orig.view(b, seq_len, num_heads, head_size).transpose(1, 2)
            attn_weights = torch.matmul(side_states, key.transpose(2, 3)) * scaling

        return attn_weights, attention_mask, v_states

    def project_values(self, layer_idx, values_states):
        batch_size, num_head, seq_len, dim = values_states.shape
        # Conv1D c_proj: weight (d_model, d_model)
        # Convert to Linear convention (transpose) for einsum
        W_O = self.model.transformer.h[layer_idx].attn.c_proj.weight
        weight_per_head = W_O.T.view(W_O.shape[1], num_head, dim)
        per_head_transformed = torch.einsum(
            "bshd,ohd->bsho",
            values_states.transpose(1, 2), weight_per_head)
        return per_head_transformed.transpose(1, 2)

    def get_value_weight(self, layer_idx):
        """Return W_V for this layer, shaped (num_heads, d_model, head_dim).

        c_attn is Conv1D with weight (d_model, 3*d_model). The value
        slice is columns [2*d, 3*d). Convention: x @ W gives QKV.
        Per-head W_V[h] is (d_model, head_dim) — projecting residual
        to head h's value space.
        """
        attn = self.model.transformer.h[layer_idx].attn
        d = self.model.config.n_embd
        num_heads = self.model.config.n_head
        head_dim = d // num_heads

        W_V_full = attn.c_attn.weight[:, 2*d:3*d]  # (d_model, d_model)
        # Reshape into per-head blocks along the output dim.
        # GPT-2 layout: V[s] = LN(residual(s)) @ W_V_full
        # Then V[s] is sliced into heads as V[s].view(seq, num_heads, head_dim)
        # So per-head W_V is W_V_full[:, h*head_dim:(h+1)*head_dim].
        W_V_per_head = W_V_full.view(d, num_heads, head_dim)  # (D, H, head_dim)
        return W_V_per_head.permute(1, 0, 2).contiguous()       # (H, D, head_dim)

    def get_value_bias(self, layer_idx):
        """Return b_V (the c_attn value-bias) for this layer, shaped
        (num_heads, head_dim). This is the bias term added to V[s] in
        the forward pass: V[s] = LN(residual(s)) · W_V + b_V.

        For GPT-2, b_V = c_attn.bias[2*d : 3*d], reshaped to per-head.
        """
        attn = self.model.transformer.h[layer_idx].attn
        d = self.model.config.n_embd
        num_heads = self.model.config.n_head
        head_dim = d // num_heads

        b_V_full = attn.c_attn.bias[2*d:3*d]  # (d_model,)
        return b_V_full.view(num_heads, head_dim).contiguous()

    def get_value_at_position(self, layer_idx, hidden_states):
        """Compute the actual V[h, s, :] vectors (head_dim) from a hidden
        state. Used to score per-component value contributions against
        the head's actual value vector. Mirrors what the forward pass
        computes internally before splitting into heads.

        Args:
            hidden_states: (B, S, D)

        Returns:
            V: (B, num_heads, S, head_dim)
        """
        attn = self.model.transformer.h[layer_idx].attn
        d = self.model.config.n_embd
        num_heads = self.model.config.n_head
        head_dim = d // num_heads

        # Apply attention LN first (matches forward pass).
        ln = self.model.transformer.h[layer_idx].ln_1
        hs = ln(hidden_states.to(ln.weight.device))
        # c_attn projects QKV; take the V slice.
        V_full = hs @ attn.c_attn.weight[:, 2*d:3*d] + attn.c_attn.bias[2*d:3*d]
        # (B, S, D) -> (B, S, H, head_dim) -> (B, H, S, head_dim)
        b, s, _ = V_full.shape
        V = V_full.view(b, s, num_heads, head_dim).transpose(1, 2)
        return V

    # ==========================================
    #  Layer Parameters
    # ==========================================

    def get_norm_params(self, layer_idx):
        """Attention-side layernorm (ln_1)."""
        ln = self.model.transformer.h[layer_idx].ln_1
        return ln.weight, ln.bias, ln.eps

    def get_mlp_norm_params(self, layer_idx):
        """MLP-side layernorm (ln_2).

        GPT-2 uses sequential residual: ln_2 sees the residual stream
        AFTER attention output has been added.
        """
        ln = self.model.transformer.h[layer_idx].ln_2
        return ln.weight, ln.bias, ln.eps

    def get_num_layers(self):
        return self.model.config.n_layer

    def get_num_heads(self):
        return self.model.config.n_head

    def get_d_model(self):
        return self.model.config.n_embd

    def get_component_layer(self, name):
        """Sequential residual: MLP at layer L gets L + 0.5.

        This ensures attention heads at layer L are treated as
        'earlier' than the MLP at the same layer, reflecting the
        sequential computation order.
        """
        src_layer = -1
        for part in name.replace("head_", "h").split("_"):
            if part.isdigit():
                src_layer = int(part)
                break
        if name.startswith("mlp_"):
            return src_layer + 0.5
        return src_layer

    def get_mlp_up_params(self, layer_idx):
        """Return (weight, bias) of c_fc in Linear convention.

        Conv1D stores weight as (in, out); we transpose to (out, in)
        to match the Linear convention used by the rest of the framework.
        """
        mlp = self.model.transformer.h[layer_idx].mlp
        return mlp.c_fc.weight.T, mlp.c_fc.bias

    def get_mlp_down_params(self, layer_idx):
        """Return (weight, bias) of MLP c_proj in Linear convention."""
        mlp = self.model.transformer.h[layer_idx].mlp
        return mlp.c_proj.weight.T, mlp.c_proj.bias

    def get_final_norm_params(self):
        """Return (weight, bias, eps) for transformer.ln_f."""
        ln = self.model.transformer.ln_f
        return ln.weight, ln.bias, ln.eps

    def get_unembed_weight(self):
        """Return lm_head.weight (tied to wte.weight)."""
        return self.model.lm_head.weight

    def mlp_up_forward(self, layer_idx, normed_input):
        """GPT-2 uses GELU (new) activation after c_fc."""
        mlp = self.model.transformer.h[layer_idx].mlp
        pre_act = mlp.c_fc(normed_input)
        activated = mlp.act(pre_act)
        return pre_act, activated

    # ==========================================
    #  Logit Lens
    # ==========================================

    def apply_logit_lens(self, components, marginal=False,
                         final_logits=None):
        layer_norm = self.model.transformer.ln_f
        lm_head = self.model.lm_head
        with torch.no_grad():
            if marginal:
                assert final_logits is not None
                mean = components.mean(dim=-1, keepdim=True)
                components_zero_mean = components - mean
                layer_var = final_logits.var(dim=-1, unbiased=False,
                                            keepdim=True)
                layer_std = torch.sqrt(layer_var + layer_norm.eps)
                normalized = (components_zero_mean / layer_std
                              * layer_norm.weight)
            else:
                normalized = layer_norm(components)
            outputs = lm_head(normalized)
        return outputs

    def apply_attn_norm(self, layer_idx, hidden):
        """Apply this layer's pre-attention layer norm (ln_1) to a tensor.
    
        Just delegates to the model's actual nn.LayerNorm module, so we
        don't re-implement LN math (epsilon, dtype handling, etc.) and
        we automatically match whatever the model does in its forward.
    
        Args:
            layer_idx: int
            hidden: tensor of shape ending in (..., d_model)
    
        Returns:
            post-LN hidden, same shape.
        """
        return self.model.transformer.h[layer_idx].ln_1(hidden)