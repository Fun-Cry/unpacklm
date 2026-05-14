"""
core.prep - Forward-pass setup shared by trace_recursive and trace_flow.

`_prepare_trace_inputs` runs a single hooked forward pass and returns
all the precomputed quantities needed for backward attribution:

    component_vecs      per-component residual contributions
    attention_weights   raw attention patterns
    key_decomp          K-side per-component attention logit shares
    query_decomp        Q-side per-component attention logit shares
    value_decomp        V-side per-component value-route shares
    mlp_l2              L2 norms of MLP up-projection contributions
    mlp_principled      d_t-aligned MLP shares (depth 0)
    mlp_geva            optional: activation-weighted MLP shares (depth >= 1)
    mlp_outproj         optional: output-aligned MLP shares (depth >= 1)
    attn_shares         d_t-aligned attention shares (depth 0)
    importance          DLA score per component at target position
    forward_attn_at_t   forward edge scores (attention)
    forward_mlp_at_t    forward edge scores (MLP)

Both trace_flow and trace_recursive consume this dict.
"""

import numpy as np
import torch

from unpack.core.streamer import ComponentStreamer
from unpack.core.scorers import AttentionScorer, MLPScorer
from unpack.core.decomposition import (
    compute_target_direction,
    precompute_attn_shares,
    precompute_attn_shares_outproj,
    compute_mlp_decomp_principled,
    compute_mlp_decomp_geva,
    compute_mlp_decomp_outproj,
)


def _prepare_trace_inputs(
    model, tokenizer, text,
    target_position="last", target_token=None, distractor_token=None,
    hook_manager=None, comp_batch_size=8,
    interventions=None,
    enable_q_side=True,
    enable_v_side=True,
    include_bias=False,
    mlp_geva_enabled=False,
    mlp_outproj_enabled=False,
    attn_outproj_enabled=False,
):
    """Forward pass, hook capture, and all precomputes needed for
    backward attribution. Returns a dict of intermediates that are
    independent of which backward algorithm consumes them.

    `interventions`, if given, are installed for the single hooked
    forward pass that populates streamer + capture buffers + attention
    weights. There is no second forward pass: every quantity below
    comes from one consistent run of the model under whatever
    interventions were installed.
    """
    if hook_manager is None:
        from models.gpt_neox import GPTNeoXAdapter
        hook_manager = GPTNeoXAdapter()
    # Register hooks only if not already registered.  Re-registering on
    # an already-hooked manager would double-fire every PyTorch hook
    # callback per forward pass and corrupt the capture buffers.
    if not hook_manager.handles:
        hook_manager.register_hooks(model)
    streamer = ComponentStreamer(model, tokenizer, hook_manager)
    attn_scorer = AttentionScorer(hook_manager)
    mlp_scorer = MLPScorer(hook_manager)

    streamer.set_context(text, interventions=interventions)

    seq_len = int(streamer.seq_lens[0])

    attention_weights = {}
    if streamer.outputs.attentions is not None:
        for li, at in enumerate(streamer.outputs.attentions):
            attention_weights[li] = at[0].detach().cpu().numpy()

    num_layers = hook_manager.get_num_layers()
    num_heads = hook_manager.get_num_heads()

    tok_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(tok_ids)[:seq_len]
    t_pos = seq_len - 1 if target_position == "last" else int(target_position)

    pre_final = streamer.get_unnormalized_logits()
    logits = streamer.outputs.logits[0, t_pos]
    probs = torch.softmax(logits, dim=-1)
    top_vals, top_ids = torch.topk(probs, k=5)
    predictions = [(tokenizer.decode([tid]), float(p))
                    for tid, p in zip(top_ids.tolist(), top_vals.tolist())]

    if target_token is not None:
        ids = tokenizer.encode(target_token)
        if not ids:
            raise ValueError(f"Could not encode: {target_token!r}")
        top_token_id = ids[0]
        target_token_str = target_token
        target_prob = float(probs[top_token_id].item())
    else:
        top_token_id = top_ids[0].item()
        target_token_str = predictions[0][0]
        target_prob = predictions[0][1]

    target_logit_centered = float((logits[top_token_id] - logits.mean()).item())

    distractor_token_id = None
    if distractor_token is not None:
        d_ids = tokenizer.encode(distractor_token)
        if not d_ids:
            raise ValueError(f"Could not encode distractor: {distractor_token!r}")
        distractor_token_id = d_ids[0]

    d_t = compute_target_direction(
        hook_manager, pre_final, t_pos, top_token_id,
        distractor_token_id=distractor_token_id,
    )

    # ── Streaming pass: component_vecs, key_decomp, mlp_l2, projected_values ──
    component_vecs = {}
    component_layer = {}
    component_order = []
    key_decomp = {}
    query_decomp = {}
    value_decomp = {}
    mlp_l2 = {}
    projected_values = {}
    attn_masks_np = [None] * num_layers
    valid_masks = [None] * num_layers

    for target_L, components, names, hidden, is_last_group in streamer.stream():
        for i, name in enumerate(names):
            if name not in component_vecs:
                component_vecs[name] = components[:, :, i, :].clone().numpy()
                component_layer[name] = hook_manager.get_component_layer(name)
                component_order.append(name)

        if target_L is None:
            continue

        # ── K-side scoring ──────────────────────────────────────
        attn_names, attn_scores, attn_mask, v_states = attn_scorer.score(
            target_L, components, names, hidden, is_last_group,
            side="key", comp_batch_size=comp_batch_size,
        )
        if attn_masks_np[target_L] is None:
            mask_np = (attn_mask.detach().cpu().numpy()
                       if isinstance(attn_mask, torch.Tensor) else attn_mask)
            if mask_np is not None and len(mask_np.shape) == 2:
                mask_np = mask_np[:, None, None, :]
            attn_masks_np[target_L] = mask_np
            valid_masks[target_L] = (mask_np == 0) if mask_np is not None else None
            pv = attn_scorer.project_values(target_L, v_states)
            projected_values[target_L] = pv[0].detach().cpu().numpy()

        vm = valid_masks[target_L]
        for name, score_tensor in zip(attn_names, attn_scores):
            if name in ("norm_bias", "attn_bias") and not include_bias:
                continue
            score_np = score_tensor.detach().cpu().numpy()
            if vm is not None:
                vm_count = np.maximum(vm.sum(axis=-1, keepdims=True), 1)
                valid_mean = (score_np * vm).sum(axis=-1, keepdims=True) / vm_count
                centered = score_np - valid_mean
            else:
                centered = score_np - score_np.mean(axis=-1, keepdims=True)
            if target_L not in key_decomp:
                key_decomp[target_L] = {}
            key_decomp[target_L][name] = centered[0]

        # ── Q-side scoring ──────────────────────────────────────
        if enable_q_side:
            q_attn_names, q_attn_scores, _, _ = attn_scorer.score(
                target_L, components, names, hidden, is_last_group,
                side="query", comp_batch_size=comp_batch_size,
            )
            for name, score_tensor in zip(q_attn_names, q_attn_scores):
                if name in ("norm_bias", "attn_bias") and not include_bias:
                    continue
                score_np = score_tensor.detach().cpu().numpy()
                if vm is not None:
                    vm_count = np.maximum(vm.sum(axis=-1, keepdims=True), 1)
                    valid_mean = (score_np * vm).sum(axis=-1, keepdims=True) / vm_count
                    centered = score_np - valid_mean
                else:
                    centered = score_np - score_np.mean(axis=-1, keepdims=True)
                if target_L not in query_decomp:
                    query_decomp[target_L] = {}
                query_decomp[target_L][name] = centered[0]

        # ── V-side scoring ──────────────────────────────────────
        if enable_v_side:
            v_names, v_scores = attn_scorer.score(
                target_L, components, names, hidden,
                is_last_group, side="value", value_states=v_states,
            )
            for c_idx, name in enumerate(v_names):
                if name in ("norm_bias",) and not include_bias:
                    continue
                score_np = v_scores[0, c_idx].detach().cpu().numpy()  # (H, S)
                if target_L not in value_decomp:
                    value_decomp[target_L] = {}
                value_decomp[target_L][name] = score_np

        mlp_names, mlp_norms_np = mlp_scorer.score(
            target_L, components, names, hidden, is_last_group,
        )
        for i, mname in enumerate(mlp_names):
            if mname in ("mlp_norm_bias", "mlp_up_bias") and not include_bias:
                continue
            if mname in component_layer:
                if target_L not in mlp_l2:
                    mlp_l2[target_L] = {}
                mlp_l2[target_L][mname] = mlp_norms_np[0, :, i].copy()

    # ── Fix for sequential residual: mlp_l2 may be missing components ──
    # The streaming loop scores groups with first_target = src_layer + 1,
    # so attn_L components (component_layer=L) are never scored against
    # MLP_L (component_layer=L+0.5) despite being upstream. Use
    # component_layer to find and score the missing upstream components.
    for L in range(num_layers):
        mlp_cl = component_layer.get(f"mlp_{L}", L)
        if mlp_cl == L:
            continue  # parallel residual — no missing components
        existing = set(mlp_l2.get(L, {}).keys())
        missing = [n for n in component_order
                   if n not in existing
                   and 'bias' not in n
                   and component_layer.get(n, 999) < mlp_cl
                   and n in component_vecs]
        if not missing:
            continue
        tensors = [torch.from_numpy(component_vecs[n]) for n in missing]
        comp_tensor = torch.stack(tensors, dim=2)
        hidden = streamer.outputs.hidden_states[L].detach().cpu()
        with torch.no_grad():
            extra_names, extra_norms = mlp_scorer.score(
                L, comp_tensor, missing, hidden, is_last_group=False,
            )
        if L not in mlp_l2:
            mlp_l2[L] = {}
        for i, mname in enumerate(extra_names):
            if mname in component_layer:
                mlp_l2[L][mname] = extra_norms[0, :, i].copy()

    attn_shares = precompute_attn_shares(
        attention_weights, projected_values, d_t, seq_len,
    )

    attn_shares_outproj = None
    if attn_outproj_enabled:
        attn_shares_outproj = precompute_attn_shares_outproj(
            attention_weights, projected_values, seq_len,
        )

    mlp_principled = compute_mlp_decomp_principled(
        hook_manager, streamer.outputs, component_vecs, d_t,
        num_layers, seq_len,
    )

    mlp_geva = None
    if mlp_geva_enabled:
        mlp_geva = compute_mlp_decomp_geva(
            hook_manager, streamer.outputs, component_vecs,
            num_layers, seq_len,
        )

    mlp_outproj = None
    if mlp_outproj_enabled:
        mlp_outproj = compute_mlp_decomp_outproj(
            hook_manager, streamer.outputs, component_vecs,
            num_layers, seq_len,
        )

    # ── Output importance (marginal logit lens, centered) ──
    model_device = model.device
    final_slice = pre_final[:, t_pos:t_pos+1, :].to(model_device)
    importance = {}
    for name, vec in component_vecs.items():
        comp = torch.from_numpy(vec[:, t_pos:t_pos+1, :]).to(model_device)
        comp_logits = hook_manager.apply_logit_lens(
            comp, marginal=True, final_logits=final_slice
        )
        target_logit = comp_logits[0, 0, top_token_id]
        mean_logit = comp_logits[0, 0, :].mean()
        importance[name] = float((target_logit - mean_logit).item())

    # ── Forward edge scores at q = t_pos ──
    forward_attn = {}
    forward_mlp = {}

    for L, by_name in key_decomp.items():
        vm = valid_masks[L] if L < len(valid_masks) else None
        vm_s = None
        if vm is not None:
            vm_s = np.asarray(vm).reshape(-1).astype(bool)
        for name, arr in by_name.items():
            if arr is None or t_pos >= arr.shape[1]:
                continue
            slc = arr[:, t_pos, :]
            if vm_s is not None and vm_s.shape[0] == slc.shape[1]:
                safe = np.where(vm_s[None, :], slc, 0.0)
                count = max(int(vm_s.sum()), 1)
                mean = safe.sum(axis=-1, keepdims=True) / count
                sq = np.where(vm_s[None, :], (slc - mean) ** 2, 0.0)
                std_h = np.sqrt(sq.sum(axis=-1) / count)
            else:
                std_h = slc.std(axis=-1)
            for h in range(std_h.shape[0]):
                forward_attn[(L, int(h), name)] = float(std_h[h])

    for L, by_name in mlp_l2.items():
        for name, vec in by_name.items():
            if vec is None or t_pos >= vec.shape[0]:
                continue
            forward_mlp[(L, name)] = float(vec[t_pos])

    hook_manager.remove_hooks()

    return {
        "tokens": tokens,
        "seq_len": seq_len,
        "t_pos": t_pos,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "target_token_str": target_token_str,
        "target_prob": target_prob,
        "target_logit_centered": target_logit_centered,
        "predictions": predictions,
        "importance": importance,
        "attn_shares": attn_shares,
        "attn_shares_outproj": attn_shares_outproj,
        "attention_weights": attention_weights,
        "key_decomp": key_decomp,
        "query_decomp": query_decomp,
        "value_decomp": value_decomp,
        "mlp_l2": mlp_l2,
        "mlp_principled": mlp_principled,
        "mlp_geva": mlp_geva,
        "mlp_outproj": mlp_outproj,
        "component_layer": component_layer,
        "component_order": component_order,
        "forward_attn_at_t": forward_attn,
        "forward_mlp_at_t":  forward_mlp,
    }