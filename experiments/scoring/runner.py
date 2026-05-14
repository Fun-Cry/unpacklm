import torch
import signal
import sys
import traceback
import numpy as np
from tqdm import tqdm
from utils.state import update_last_processed_index, get_last_processed_index
from utils.storage_manager import StorageManager
from unpack.core import ComponentStreamer, AttentionScorer, MLPScorer


def _masked_std(arr, valid_mask):
    """Compute std over last axis, considering only valid (non-padding) positions."""
    if valid_mask is None:
        return np.std(arr, axis=-1)
    safe = np.where(valid_mask, arr, 0.0)
    count = valid_mask.sum(axis=-1)
    count = np.maximum(count, 1)
    mean = safe.sum(axis=-1) / count
    sq_diff = np.where(valid_mask, (arr - mean[..., None]) ** 2, 0.0)
    return np.sqrt(sq_diff.sum(axis=-1) / count)


class ExperimentRunner:
    def __init__(self, args, sentences, model, tokenizer, hook_manager):
        self.args = args
        self.sentences = sentences
        self.processors = []
        self.last_index = get_last_processed_index(args.state_file)

        # Storage
        self.storage = StorageManager(args.db_path, args.h5_path)

        # Model (provided by caller)
        self.model = model
        self.tokenizer = tokenizer
        self.hook_manager = hook_manager
        hook_manager.register_hooks(model)
        self.streamer = ComponentStreamer(model, tokenizer, hook_manager)
        self.attn_scorer = AttentionScorer(hook_manager)
        self.mlp_scorer = MLPScorer(hook_manager)

        # Model ID for storage
        self.model_id = self.storage.get_model_id(args.model_size, args.step)

        # Rollback any partial HDF5 writes from a previous interrupted run
        self.storage.rollback_to_checkpoint()

    def add_processor(self, processor):
        self.processors.append(processor)

    def _apply_post_mask(self, attention_weights, seq_len):
        if not torch.is_tensor(attention_weights): attention_weights = torch.from_numpy(attention_weights)
        if not torch.is_tensor(seq_len): seq_len = torch.tensor(seq_len)
        batch_size, num_heads, max_len, _ = attention_weights.shape
        seq_len = seq_len.to(attention_weights.device)
        mask = torch.arange(max_len, device=attention_weights.device).expand(batch_size, max_len) < seq_len.unsqueeze(1)
        row_mask = mask.view(batch_size, 1, max_len, 1)
        col_mask = mask.view(batch_size, 1, 1, max_len)
        return attention_weights * (row_mask * col_mask)

    # ==========================================
    #  Streaming Pipeline
    # ==========================================

    def _process_batch_streaming(self, batch, batch_pbar=None):
        if batch_pbar:
            batch_pbar.set_postfix_str("forward pass")
        self.streamer.set_context(batch)
        seq_lens = self.streamer.seq_lens
        batch_sentence_ids = self.storage.register_sentences(batch)

        num_layers = self.hook_manager.get_num_layers()
        num_heads = self.hook_manager.get_num_heads()
        batch_size = len(batch)

        for p in self.processors:
            if hasattr(p, 'on_batch_start'):
                p.on_batch_start(batch_size, num_heads, num_layers, batch, seq_lens)

        raw_sums = [None] * num_layers
        attn_masks_np = [None] * num_layers
        valid_masks = [None] * num_layers
        value_states_cache = [None] * num_layers

        layer_bar = tqdm(
            total=num_layers, desc="  layers", leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        prev_completed = -1

        for target_L, components, names, hidden, is_last_group in self.streamer.stream():

            # Capture-only sentinel for final-layer components; this consumer
            # only does scoring, so there is nothing to do on these events.
            if target_L is None:
                continue

            if batch_pbar:
                batch_pbar.set_postfix_str(f"L{target_L} attn+mlp")

            # ── Attention scoring ──
            attn_names, attn_scores, attn_mask, v_states = self.attn_scorer.score(
                target_L, components, names, hidden, is_last_group,
                comp_batch_size=self.args.comp_batch_size,
            )

            if attn_masks_np[target_L] is None:
                mask_np = attn_mask.detach().cpu().numpy() if isinstance(attn_mask, torch.Tensor) else attn_mask
                if mask_np is not None and len(mask_np.shape) == 2:
                    mask_np = mask_np[:, None, None, :]
                attn_masks_np[target_L] = mask_np
                valid_masks[target_L] = (mask_np == 0) if mask_np is not None else None
                value_states_cache[target_L] = v_states

            vm = valid_masks[target_L]

            for name, score_tensor in zip(attn_names, attn_scores):
                score_np = score_tensor.detach().cpu().numpy()
                
                if vm is not None:
                    vm_count = vm.sum(axis=-1, keepdims=True)
                    vm_count = np.maximum(vm_count, 1)
                    valid_mean = (score_np * vm).sum(axis=-1, keepdims=True) / vm_count
                    centered = score_np - valid_mean
                else:
                    centered = score_np - score_np.mean(axis=-1, keepdims=True)
                
                if raw_sums[target_L] is None:
                    raw_sums[target_L] = centered.copy()
                else:
                    raw_sums[target_L] += centered
                
                comp_std = _masked_std(centered, vm)
                
                for p in self.processors:
                    if hasattr(p, 'on_component'):
                        p.on_component(target_L, name, comp_std, centered,
                                      batch_sentence_ids, batch, seq_lens)

            # ── MLP scoring ──
            mlp_names, mlp_norms_np = self.mlp_scorer.score(
                target_L, components, names, hidden, is_last_group,
            )
            for p in self.processors:
                if hasattr(p, 'on_mlp_norms'):
                    p.on_mlp_norms(target_L, mlp_names, mlp_norms_np,
                                   batch_sentence_ids, batch, seq_lens, is_last_group)

            if is_last_group:
                if batch_pbar:
                    batch_pbar.set_postfix_str(f"L{target_L} finalize")
                self._finalize_layer(
                    target_L, raw_sums[target_L], attn_masks_np[target_L],
                    value_states_cache[target_L], valid_masks[target_L],
                    batch_sentence_ids, batch, seq_lens,
                )
                raw_sums[target_L] = None
                attn_masks_np[target_L] = None
                valid_masks[target_L] = None
                value_states_cache[target_L] = None

                layer_bar.update(target_L - prev_completed)
                prev_completed = target_L

        layer_bar.close()

        if batch_pbar:
            batch_pbar.set_postfix_str("flush")
        for p in self.processors:
            if hasattr(p, 'on_batch_complete'):
                p.on_batch_complete()

    def _finalize_layer(self, layer_idx, raw_sum, attn_mask_np, value_states,
                        valid_mask, sentence_ids, sentences, seq_lens):
        if attn_mask_np is not None:
            masked = raw_sum + attn_mask_np
        else:
            masked = raw_sum
        final_attention = torch.softmax(torch.from_numpy(masked), dim=-1)
        final_attention = self._apply_post_mask(final_attention, seq_lens)
        
        residual_values = self.attn_scorer.project_values(layer_idx, value_states)
        residual_values_np = residual_values.detach().cpu().numpy()
        
        final_attention_dev = final_attention.to(value_states.device).float()
        weighted_values = torch.matmul(final_attention_dev, residual_values)
        weighted_values_np = weighted_values.detach().cpu().numpy()
        
        total_std = _masked_std(raw_sum, valid_mask)
        
        for p in self.processors:
            if hasattr(p, 'on_layer_complete'):
                p.on_layer_complete(
                    layer_idx, final_attention, value_states,
                    residual_values_np, weighted_values_np, total_std,
                    sentence_ids, sentences, seq_lens,
                )

    # ==========================================
    #  Main Loop
    # ==========================================

    def run(self):
        total_len = len(self.sentences)
        bs = self.args.batch_size

        # Count batches that actually need processing
        total_batches = 0
        for i in range(0, total_len, bs):
            end = min(i + bs, total_len)
            if end - 1 > self.last_index:
                total_batches += 1

        batch_pbar = tqdm(
            total=total_batches, desc="batches", unit="batch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        try:
            for i in range(0, total_len, bs):
                current_batch_end_idx = min(i + bs, total_len)
                if current_batch_end_idx - 1 <= self.last_index:
                    continue

                effective_start_idx = max(i, self.last_index + 1)
                batch = self.sentences[effective_start_idx : current_batch_end_idx]
                if not batch:
                    continue

                batch_pbar.set_description(
                    f"batch {effective_start_idx}-{current_batch_end_idx}"
                )
                
                try:
                    self._process_batch_streaming(batch, batch_pbar)
                    
                    batch_pbar.set_postfix_str("checkpoint")
                    self.storage.checkpoint_h5()
                    update_last_processed_index(self.args.state_file, current_batch_end_idx - 1)
                    self.last_index = current_batch_end_idx - 1
                    
                    batch_pbar.update(1)
                except Exception as e:
                    traceback.print_exc()
                    break
        finally:
            batch_pbar.close()
            self._cleanup()

    def _signal_handler(self, sig, frame):
        print("\nInterrupt received. cleaning up...")
        sys.exit(0)

    def _cleanup(self):
        for p in self.processors:
            p.close()
        if hasattr(self, 'storage'):
            self.storage.close()