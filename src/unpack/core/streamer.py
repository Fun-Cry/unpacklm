import torch
from abc import ABC, abstractmethod
from typing import Callable, Dict, List
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np
from unpack.models.base import ModelAdapter

# ==========================================
#  Component Streamer — shared infrastructure
# ==========================================

class ComponentStreamer:
    """
    Forward pass + raw residual stream component iteration.

    Central class that both AttentionScorer and MLPScorer consume.
    Handles the model forward pass and source-group iteration.
    Yields RAW (un-normalized) components plus the hidden state —
    each scorer normalizes with its own layernorm parameters.

    This is critical for models with parallel residual (e.g. Pythia)
    where attention and MLP use different layernorms on the same input.

    Usage::

        streamer = ComponentStreamer(model, tokenizer, hook_manager)
        attn = AttentionScorer(hook_manager)
        mlp  = MLPScorer(hook_manager)

        streamer.set_context(batch)
        for target_L, components, names, hidden, is_last_group in streamer.stream():
            attn_result = attn.score(target_L, components, names, hidden, is_last_group)
            mlp_result  = mlp.score(target_L, components, names, hidden, is_last_group)
    """

    def __init__(self, model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 hook_manager: ModelAdapter):
        self.model = model
        self.tokenizer = tokenizer
        self.hook_manager = hook_manager
        self.outputs = None
        self.seq_lens = None

    # ==========================================
    #  Forward Pass
    # ==========================================

    def set_context(self, model_input, *, interventions=None):
        """Run forward pass, capture hidden states and hook data.

        If `interventions` is given, they're installed on the hook_manager
        for this forward pass and cleared afterward.  This is the single
        entry point for "forward pass that captures everything tracing
        needs" — installing interventions here (rather than in the caller)
        guarantees that whatever the caller later reads from the streamer
        and from the hook_manager's capture buffers reflects the same
        forward pass.
        """
        self.hook_manager.clear()
        if interventions:
            self.hook_manager.clear_interventions()
            self.hook_manager.register_interventions(interventions)
        inputs = self.tokenizer(
            model_input, return_tensors="pt", padding=True, truncation=True,
        )
        self.seq_lens = np.array(
            [sum(mask).item() for mask in inputs["attention_mask"]]
        )
        inputs = inputs.to(self.model.device)
        try:
            with torch.no_grad():
                self.outputs = self.model(
                    **inputs, output_hidden_states=True, output_attentions=True,
                )
        finally:
            if interventions:
                self.hook_manager.clear_interventions()

    # ==========================================
    #  Raw Component Stream
    # ==========================================

    def stream(self):
        """
        Yield raw (un-normalized) components per source-group × target-layer.

        Each scorer applies its own layernorm normalization and bias.
        This is necessary for parallel-residual models where attention
        and MLP use different layernorms.

        Memory lifecycle:
          - One source group's (B, S, C_group, D) tensor alive at a time
          - Hook data freed progressively via iter_source_groups
          - Attention input cache freed after each completed source layer

        Yields:
            (target_L, components, names, hidden, is_last_group)

            target_L:       int, target layer index
            components:     Tensor (B, S, C, D) — raw residual components
            names:          list[str], component names
            hidden:         Tensor (B, S, D) — hidden state at target layer
                            (used by scorers for stream variance)
            is_last_group:  bool, True when all source groups for this target
                            have been yielded
        """
        with torch.no_grad():
            hidden_states = self.outputs.hidden_states
            num_layers = self.hook_manager.get_num_layers()

            for group_tensor, names, src_layer in self.hook_manager.iter_source_groups():
                first_target = max(src_layer + 1, 0)

                any_emitted = False
                for target_L in range(first_target, num_layers):
                    is_last_group = (src_layer == target_L - 1)
                    hidden = hidden_states[target_L].detach().cpu()

                    yield target_L, group_tensor, list(names), hidden, is_last_group
                    any_emitted = True

                # Final-layer source groups have no downstream targets to
                # score against (first_target == num_layers makes the inner
                # loop empty). Emit one capture-only event so callers can
                # still record these components into component_vecs and
                # compute their direct logit-lens score. target_L = None
                # is the sentinel; callers must skip scoring on this event.
                if not any_emitted:
                    hidden = hidden_states[-1].detach().cpu()
                    yield None, group_tensor, list(names), hidden, True

                # Free completed target's attention cache
                completed_target = src_layer + 1
                if 0 <= completed_target < num_layers:
                    self.hook_manager.free_attention_cache(completed_target)

    # ==========================================
    #  Utilities
    # ==========================================

    def get_unnormalized_logits(self):
        """Residual stream before final layer norm (pre-softmax)."""
        return self.hook_manager.pre_final_norm.cpu()

    def get_sentence_part(self, sentence, idx):
        inputs = self.tokenizer(
            sentence, return_tensors="pt", padding=True, truncation=True,
        )
        input_ids = inputs.input_ids[0]
        return self.tokenizer.decode(input_ids[:idx + 1])

    def logit_lens(self, components, top_k=10, marginal=False, final_logits=None):
        with torch.no_grad():
            if marginal:
                assert final_logits is not None
            outputs = self.hook_manager.apply_logit_lens(components, marginal, final_logits)
            top_vals, top_indices = torch.topk(outputs, top_k, dim=-1)
            bot_vals, bot_indices = torch.topk(outputs, top_k, dim=-1, largest=False)
            top_results = []
            bot_results = []
            for t_idx, t_val, b_idx, b_val in zip(top_indices, top_vals, bot_indices, bot_vals):
                top_tokens = self.tokenizer.convert_ids_to_tokens(t_idx)
                bot_tokens = self.tokenizer.convert_ids_to_tokens(b_idx)
                top_results.append(list(zip(top_tokens, t_val.tolist())))
                bot_results.append(list(zip(bot_tokens, b_val.tolist())))
            return top_results, bot_results


# ==========================================
#  Shared Normalization
# ==========================================

def _marginal_normalize(components, hidden_state, norm_weight, norm_eps):
    """Marginal normalization: center, divide by stream std (constant), scale by LN weight.
    
    Args:
        components:   (B, S, C, D) — raw residual components
        hidden_state: (B, S, D) — full hidden state (for variance)
        norm_weight:  (D,) — layernorm weight
        norm_eps:     float
    
    Returns:
        (B, S, C, D) — normalized components
    """
    norm_weight = norm_weight.to(components.device)
    hidden_state = hidden_state.to(components.device)

    comp_mean = components.mean(dim=-1, keepdim=True)
    zero_mean = components - comp_mean

    layer_var = hidden_state.var(dim=-1, unbiased=False)
    layer_std = torch.sqrt(layer_var + norm_eps)[:, :, None, None]

    return zero_mean / layer_std * norm_weight