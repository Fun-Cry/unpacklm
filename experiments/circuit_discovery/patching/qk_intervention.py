"""Score-level path patching at per-head, per-channel granularity.

Splits the path-patching workflow into three phases that the caller
orchestrates:

    capture  →  build (pure analytical, no forwards)  →  run

`capture_path_patch_inputs` runs two forward passes (clean + corrupted)
and extracts every tensor needed for many subsequent build calls
against the same prompt pair.

`build_qk_score_patch` is pure: takes captured inputs and an edge spec,
returns an Intervention. No forwards. Cheap, can be called for many
edges in a sweep.

`run_with_intervention` (existing) executes the patched forward pass
to read out the target logit.

The intervention point is `attn_{L}_pre_dense` — the receiver head's
attention output before W_O. We compute analytically:

    delta_z = modified_attention_pattern @ V  -  clean_attention_pattern @ V

where modified_attention_pattern has one score entry shifted to reflect
the K-derived-from-corrupted-residual at sender_pos, then re-softmaxed.

Key choices:
- LN application via the model's actual norm module (no re-implementation).
  Manager exposes `apply_attn_norm(layer, hidden)`.
- Score computation via the manager's `get_cross_attention_scores`,
  which already handles W_K projection and RoPE per architecture.
- V at sender_pos: caller chooses clean (K-only test) or corrupted
  (K and V together).
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from unpack._interventions import add_at
from experiments.ablation_tracing.core.trace import Intervention


# ──────────────────────────────────────────────────────────────────────
# Hook-point template
# ──────────────────────────────────────────────────────────────────────

def add_at_head_slice(delta: torch.Tensor, positions: List[int],
                      head_idx: int, head_size: int):
    """Intervention template: add `delta` (shape (head_size,)) at the
    receiver-head slice of a `(B, S, n_heads * head_size)` tensor at
    the given positions.

    Used at attn_{L}_pre_dense to inject a delta into one head's
    output without touching other heads at the same layer.
    """
    pos = list(positions)
    head_start = head_idx * head_size
    head_end   = head_start + head_size

    def fn(tensor, hook_name):
        d = delta.to(tensor.device).to(tensor.dtype)
        out = tensor.clone()
        out[:, pos, head_start:head_end] = (
            out[:, pos, head_start:head_end] + d
        )
        return out

    fn.__qualname__ = f"add_at_head_slice(head={head_idx}, positions={pos})"
    return fn


# ──────────────────────────────────────────────────────────────────────
# Captured inputs container
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PathPatchInputs:
    """All tensors captured up-front, reused for many build_qk calls.

    Captured per (clean_prompt, corrupted_prompt) pair. Holds:
      - sender contributions (clean & corrupted) for the senders the
        caller plans to test
      - pre-LN and post-LN residuals at the receiver layers
      - per-head attention scores, patterns, and V at receiver heads
      - corrupted V (optional, for K+V test mode)
    """
    seq_len: int
    head_size: int
    n_heads: int
    d_model: int

    # (sender_name, sender_pos) -> (d_model,) residual contribution
    sender_contribs_clean:     Dict[Tuple[str, int], torch.Tensor]
    sender_contribs_corrupted: Dict[Tuple[str, int], torch.Tensor]

    # layer -> (S, d_model) pre-attention-LN residual (the residual stream)
    clean_pre_LN_residuals:  Dict[int, torch.Tensor]

    # layer -> (S, d_model) post-attention-LN (what attention reads)
    clean_post_LN_residuals: Dict[int, torch.Tensor]

    # (layer, head) -> (S, S) full clean scores  (Q · K^T / sqrt(d_h), pre-softmax)
    clean_attn_scores:   Dict[Tuple[int, int], torch.Tensor]

    # (layer, head) -> (S, S) clean attention pattern (post-softmax, masked)
    clean_attn_patterns: Dict[Tuple[int, int], torch.Tensor]

    # (layer, head) -> (S, head_size) clean V states
    clean_v_per_head: Dict[Tuple[int, int], torch.Tensor]

    # (layer, head) -> (S, head_size) corrupted V (only if requested)
    corrupted_v_per_head: Optional[Dict[Tuple[int, int], torch.Tensor]] = None


# ──────────────────────────────────────────────────────────────────────
# Capture
# ──────────────────────────────────────────────────────────────────────

def _extract_sender_contribs_batch(hook_manager,
                                    sender_specs: List[Tuple[str, int]],
                                    device) -> Dict[Tuple[str, int], torch.Tensor]:
    """Walk source groups exactly once, collecting all requested
    (name, pos) contributions in a single pass.

    `iter_source_groups` consumes its capture buffers as it iterates —
    re-iterating it after a previous walk gives None tensors. So we
    iterate ONCE and grab everything we need.
    """
    needed_names = {name for (name, _pos) in sender_specs}
    found: Dict[Tuple[str, int], torch.Tensor] = {}

    for group_tensor, names, _src_layer in hook_manager.iter_source_groups():
        for needed_name in list(needed_names):
            if needed_name in names:
                idx = names.index(needed_name)
                # Grab every requested position for this name from this group
                for (n, p) in sender_specs:
                    if n == needed_name and (n, p) not in found:
                        found[(n, p)] = group_tensor[0, p, idx, :].detach().clone().to(device)
                # Stop looking for this name after we got it
                needed_names.discard(needed_name)
        if not needed_names:
            break

    if needed_names:
        raise ValueError(
            f"sender components not found in any source group: {needed_names}"
        )
    return found


def _capture_attn_residuals(hook_manager, layers: List[int]) -> Tuple[
    Dict[int, torch.Tensor], Dict[int, torch.Tensor]
]:
    """Pull pre-LN and post-LN residuals at the requested layers from
    hook_manager state set by the most recent forward pass.

    post-LN comes from `attention_input_cache[L]["hidden_states"]`,
    captured by the existing `_wire_attn_input_hook`.

    pre-LN must be captured separately. The `attn_ln_{L}_input` hook
    fires the residual stream but doesn't store it by default. We
    rely on the caller having installed a recording-intervention
    during the forward pass — see `_RecordResidual` and
    `capture_path_patch_inputs`.
    """
    pre_LN  = {}
    post_LN = {}
    for L in layers:
        post_LN[L] = hook_manager.attention_input_cache[L]["hidden_states"][0].detach().clone()
        # pre-LN: filled by recording interventions during the forward pass;
        # see capture_path_patch_inputs().
        pre_LN[L] = hook_manager._captured_pre_LN.get(L)
        if pre_LN[L] is None:
            raise RuntimeError(
                f"pre-LN residual for layer {L} wasn't captured. "
                "Make sure `_RecordResidual` interventions were installed "
                "before the forward pass."
            )
    return pre_LN, post_LN


def _make_recorder(layer: int):
    """Returns an intervention function that records the residual it sees
    into hook_manager._captured_pre_LN[layer], passes it through
    unchanged. Use as an intervention at attn_ln_{layer}_input.
    """
    def fn(tensor, hook_name, _layer=layer):
        # tensor: (B, S, d_model). We grab batch 0.
        # The hook_manager is the closure context; we set the attr directly
        # via the recorder's bound state.
        # Approach: stash on a global dict keyed by id(hook_manager), but
        # cleaner: install via a wrapper that captures the manager.
        return tensor   # passthrough; recording done outside via the install helper
    return fn


def _install_pre_LN_recording(hook_manager, layers: List[int]):
    """Install per-layer interventions that record the residual at
    attn_ln_{L}_input into hook_manager._captured_pre_LN. Returns a
    cleanup callable that clears the recorder dict.
    """
    if not hasattr(hook_manager, "_captured_pre_LN"):
        hook_manager._captured_pre_LN = {}

    for L in layers:
        # Closure binds `L` and `hook_manager` so the recorder writes
        # into the manager's _captured_pre_LN dict.
        def _make(L=L, hm=hook_manager):
            def fn(tensor, hook_name):
                hm._captured_pre_LN[L] = tensor[0].detach().clone()
                return tensor
            return fn
        hook_manager.register_intervention(f"attn_ln_{L}_input", _make())


def _compute_attention_for_head(hook_manager, layer: int, head: int,
                                 device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (full_scores, full_pattern, V) for a (layer, head).

    Uses the manager's `get_cross_attention_scores` with the post-LN
    residual itself as the new_kv_input (so K is computed from the
    actual residual). The bias term is omitted because softmax along
    the key axis is invariant to constant per-row shifts.

    Returns:
        scores:  (S, S)  pre-softmax scores: Q · K^T / sqrt(d_h)
        pattern: (S, S)  post-softmax attention pattern (causal-masked)
        V:       (S, head_size)  clean V states
    """
    # post-LN residual at all positions
    post_LN = hook_manager.attention_input_cache[layer]["hidden_states"]   # (1, S, d)
    new_kv = post_LN.to(device)

    attn_weights, _attention_mask, v_states = \
        hook_manager.get_cross_attention_scores(layer, new_kv, include_bias=False)
    # attn_weights: (1, n_heads, S, S)

    scores  = attn_weights[0, head]                     # (S, S)
    V       = v_states[0, head]                          # (S, head_size)

    # Apply causal mask + softmax to get the pattern
    S = scores.shape[0]
    causal_mask = torch.triu(
        torch.ones(S, S, device=device), diagonal=1
    ).bool()
    masked_scores = scores.masked_fill(causal_mask, float("-inf"))
    pattern = torch.softmax(masked_scores, dim=-1)

    return scores.detach().clone(), pattern.detach().clone(), V.detach().clone()


def capture_path_patch_inputs(
    model, tokenizer, hook_manager,
    clean_prompt: str,
    corrupted_prompt: str,
    sender_specs:   Iterable[Tuple[str, int]],
    receiver_specs: Iterable[Tuple[int, int]],
    capture_corrupted_v: bool = False,
    device=None,
) -> PathPatchInputs:
    """Two forward passes (clean + corrupted). Extract everything
    `build_qk_score_patch` will need.

    Args:
        sender_specs:   iterable of (sender_name, sender_pos) — every
                        sender that will be tested in subsequent builds.
        receiver_specs: iterable of (receiver_layer, receiver_head) —
                        every receiver head that will be tested.
        capture_corrupted_v: if True, also capture V states from the
                        corrupted prompt (for K+V joint testing).
    """
    if device is None:
        device = next(model.parameters()).device

    sender_specs   = list(sender_specs)
    receiver_specs = list(receiver_specs)

    # Re-register hooks if they were removed (e.g., trace_flow.py calls
    # hook_manager.remove_hooks() at the end of trace; if discover ran
    # earlier in the same script, the manager has no live handles).
    if not hook_manager.handles:
        hook_manager.register_hooks(model)

    receiver_layers = sorted({L for (L, _h) in receiver_specs})

    n_heads   = model.config.n_head if hasattr(model.config, "n_head") \
                else model.config.num_attention_heads
    d_model   = model.config.n_embd if hasattr(model.config, "n_embd") \
                else model.config.hidden_size
    head_size = d_model // n_heads

    # ── PASS 1: clean prompt ──────────────────────────────────────────
    hook_manager.clear()
    hook_manager.clear_interventions()
    if hasattr(hook_manager, "_captured_pre_LN"):
        hook_manager._captured_pre_LN.clear()
    _install_pre_LN_recording(hook_manager, receiver_layers)

    inputs = tokenizer(clean_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    hook_manager.clear_interventions()

    # IMPORTANT: order matters. iter_source_groups (used by sender
    # extraction) destroys pre_dense_inputs and mlp_outputs as it
    # iterates. So we capture per-head attention data first (uses
    # attention_input_cache, untouched), then pre-LN/post-LN, then
    # finally walk source groups for sender contributions.

    # per-head attention data (uses attention_input_cache, doesn't
    # touch pre_dense_inputs or mlp_outputs)
    clean_scores   = {}
    clean_patterns = {}
    clean_V        = {}
    for (L, h) in receiver_specs:
        s, p, v = _compute_attention_for_head(hook_manager, L, h, device)
        clean_scores[(L, h)]   = s
        clean_patterns[(L, h)] = p
        clean_V[(L, h)]        = v

    # pre-LN & post-LN at receiver layers
    pre_LN_clean, post_LN_clean = _capture_attn_residuals(
        hook_manager, receiver_layers
    )
    seq_len = post_LN_clean[receiver_layers[0]].shape[0]

    # senders' clean contributions — single-pass walk through source groups
    sender_clean = _extract_sender_contribs_batch(
        hook_manager, sender_specs, device
    )

    # ── PASS 2: corrupted prompt ──────────────────────────────────────
    hook_manager.clear()
    hook_manager.clear_interventions()
    inputs_c = tokenizer(corrupted_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs_c)

    # If the user wants corrupted V, capture it BEFORE walking source groups.
    corrupted_V = None
    if capture_corrupted_v:
        corrupted_V = {}
        for (L, h) in receiver_specs:
            _, _, v = _compute_attention_for_head(hook_manager, L, h, device)
            corrupted_V[(L, h)] = v

    sender_corrupted = _extract_sender_contribs_batch(
        hook_manager, sender_specs, device
    )

    return PathPatchInputs(
        seq_len=seq_len, head_size=head_size,
        n_heads=n_heads, d_model=d_model,
        sender_contribs_clean=sender_clean,
        sender_contribs_corrupted=sender_corrupted,
        clean_pre_LN_residuals=pre_LN_clean,
        clean_post_LN_residuals=post_LN_clean,
        clean_attn_scores=clean_scores,
        clean_attn_patterns=clean_patterns,
        clean_v_per_head=clean_V,
        corrupted_v_per_head=corrupted_V,
    )


# ──────────────────────────────────────────────────────────────────────
# Build (pure analytical, no forwards)
# ──────────────────────────────────────────────────────────────────────

def build_qk_score_patch(
    inputs: PathPatchInputs,
    hook_manager,
    sender_name: str,
    sender_pos: int,
    receiver_layer: int,
    receiver_head: int,
    query_pos: int,
    use_corrupted_v: bool = False,
    device=None,
) -> Intervention:
    """Build a K-channel path-patching intervention from cached inputs.

    Tests: "if K at sender_pos for receiver_head were what it would be
    when sender's contribution to the residual at sender_pos were
    corrupted instead of clean, holding everything else clean, what
    would happen?"

    Pure analytical computation. No forward passes happen here.
    """
    if device is None:
        # infer from one of the cached tensors
        device = next(iter(inputs.sender_contribs_clean.values())).device

    # 1. Sender residual delta
    clean_send = inputs.sender_contribs_clean[(sender_name, sender_pos)]
    corr_send  = inputs.sender_contribs_corrupted[(sender_name, sender_pos)]
    delta_residual = corr_send - clean_send                   # (d_model,)

    # 2. Modified pre-LN residual at sender_pos, then run through the
    #    actual LN module (architecture-agnostic correctness)
    pre_LN_at_sender = inputs.clean_pre_LN_residuals[receiver_layer][sender_pos]
    modified_pre_LN_at_sender = pre_LN_at_sender + delta_residual

    # Apply the model's actual LN at this single position. The LN
    # module operates on the last dim, so position-wise LN is just
    # ln(single_position_residual). We pass shape (1, 1, d_model).
    modified_pre_LN_at_sender = modified_pre_LN_at_sender.unsqueeze(0).unsqueeze(0)
    modified_post_LN_at_sender = hook_manager.apply_attn_norm(
        receiver_layer, modified_pre_LN_at_sender
    ).squeeze(0).squeeze(0)                                   # (d_model,)

    # 3. Score delta from K-projection.
    #
    #    Build a full-sequence "modified post-LN" tensor — clean post-LN
    #    everywhere except at sender_pos, where we use the modified value.
    #    Pass that to get_cross_attention_scores, which projects through
    #    W_K and computes scores against original Q for ALL positions.
    #    This gives us scores for "K computed from a residual where only
    #    sender_pos's contribution differs from clean."
    full_clean_post_LN = inputs.clean_post_LN_residuals[receiver_layer]    # (S, d_model)
    full_modified_post_LN = full_clean_post_LN.clone()
    full_modified_post_LN[sender_pos] = modified_post_LN_at_sender

    # shape (1, S, d_model) — chunk_size = 1
    new_kv = full_modified_post_LN.unsqueeze(0).to(device)
    new_scores, _, _ = hook_manager.get_cross_attention_scores(
        receiver_layer, new_kv, include_bias=False
    )
    # new_scores: (1, n_heads, S, S)

    # Reference: clean post-LN at all positions
    clean_kv = full_clean_post_LN.unsqueeze(0).to(device)
    clean_scores_ref, _, _ = hook_manager.get_cross_attention_scores(
        receiver_layer, clean_kv, include_bias=False
    )

    # We only care about the (head, query_pos, sender_pos) entry — the
    # difference is the score-delta induced by the K-modification at sender_pos.
    new_score_at_target   = new_scores[0,   receiver_head, query_pos, sender_pos]
    clean_score_at_target = clean_scores_ref[0, receiver_head, query_pos, sender_pos]
    score_delta = new_score_at_target - clean_score_at_target  # scalar

    # 4. Rebuild attention pattern at (receiver_layer, receiver_head, query_pos):
    #    swap the score at sender_pos, re-softmax with causal mask
    full_clean_scores  = inputs.clean_attn_scores[(receiver_layer, receiver_head)]
    scores_row = full_clean_scores[query_pos].clone()
    scores_row[sender_pos] = scores_row[sender_pos] + score_delta

    # causal mask for this query position
    S = scores_row.shape[0]
    causal_mask = torch.zeros(S, dtype=torch.bool, device=device)
    causal_mask[query_pos + 1:] = True
    scores_row = scores_row.masked_fill(causal_mask, float("-inf"))
    modified_pattern_row = torch.softmax(scores_row, dim=-1)            # (S,)

    clean_pattern_row = inputs.clean_attn_patterns[(receiver_layer, receiver_head)][query_pos]

    # 5. Compute z deltas
    V = inputs.clean_v_per_head[(receiver_layer, receiver_head)]  # (S, head_size)
    if use_corrupted_v and inputs.corrupted_v_per_head is not None:
        V = V.clone()
        V[sender_pos] = inputs.corrupted_v_per_head[(receiver_layer, receiver_head)][sender_pos]

    clean_z    = clean_pattern_row    @ V                         # (head_size,)
    modified_z = modified_pattern_row @ V                         # (head_size,)
    delta_z    = modified_z - clean_z                             # (head_size,)

    # 6. Build intervention at attn_{L}_pre_dense
    fn = add_at_head_slice(
        delta_z,
        positions=[query_pos],
        head_idx=receiver_head,
        head_size=inputs.head_size,
    )
    return Intervention(
        interventions=[(f"attn_{receiver_layer}_pre_dense", fn)],
        ablated_components={sender_name},
    )