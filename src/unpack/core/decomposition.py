"""
core.decomposition - Forward-pass decomposition functions.

Each function takes the forward-pass state (hidden states, MLP internals)
and produces per-writer scores. The scores are consumed by the recursion
(core.recursion) or the flow sweep (core.flow).

Functions:
    compute_target_direction       - LN-scaled unembedding direction d_t
    precompute_attn_shares         - d_t-aligned per-head per-source contributions
    compute_mlp_decomp_principled  - d_t-aligned MLP scores (depth 0)
    compute_mlp_decomp_geva        - activation-weighted MLP shares (depth >= 1)
    compute_mlp_decomp_outproj     - output-aligned MLP shares (depth >= 1)

Geva and outproj differ in how they weight each MLP cell:

  geva:    cell weight = phi_j / (sum_k phi_k * pre_j)
           — purely activation-based; cells with high phi dominate
  outproj: cell weight = phi_j * <v_j, h_mlp> / (||h_mlp||^2 * pre_j)
           — also weighted by value-output alignment; cells whose values
             cancel against other cells are downweighted

Both produce per-writer shares that sum to ~1 by construction (or
linearization, in outproj's case).
"""

import numpy as np
import torch


def compute_target_direction(hook_manager, pre_final, t_pos, target_token_id,
                             distractor_token_id=None):
    """d_t: LN-scaled unembedding direction.

    When distractor_token_id is None (default):
        Single-token target. d_t corresponds to the target-token logit
        with mean-centering across the vocabulary applied so that
        attention-sink contributions (which affect all logits uniformly)
        receive zero importance.

    When distractor_token_id is provided:
        Logit-difference target. d_t corresponds to
            logit(target) - logit(distractor)
        This is the standard metric for contrastive tasks such as IOI,
        greater-than, and similar. No mean-centering is applied because
        the W_U.mean() term cancels in the subtraction; attention-sink
        contributions also cancel because they project identically onto
        every vocab logit.
    """
    device = hook_manager.device

    ln_w, _ln_b, ln_eps = hook_manager.get_final_norm_params()
    ln_w = ln_w.detach().float().to(device)

    h = pre_final[0, t_pos, :].float().to(device)
    var = h.var()
    inv_std = 1.0 / torch.sqrt(var + ln_eps)

    W_unembed = hook_manager.get_unembed_weight().detach().float().to(device)

    if distractor_token_id is None:
        target_vec = W_unembed[target_token_id, :]
        target_vec = target_vec - W_unembed.mean(dim=0)
    else:
        target_vec = (W_unembed[target_token_id, :]
                      - W_unembed[distractor_token_id, :])

    d = ln_w * inv_std * target_vec
    return d.cpu().numpy()


def precompute_attn_shares(attention_weights, projected_values, d_t, seq_len):
    """attn_shares[L][H, Q, S] = attn[H,Q,S] * dot(centered(V_s*W_O), d_t)"""
    attn_shares = {}
    for L in attention_weights:
        if L not in projected_values:
            continue
        pv = projected_values[L][:, :seq_len, :]                    # (H, S, D)
        pv_centered = pv - pv.mean(axis=-1, keepdims=True)         # center each value
        pv_proj = pv_centered @ d_t                                 # (H, S)
        attn_shares[L] = attention_weights[L][:, :, :seq_len] * pv_proj[:, None, :]
    return attn_shares


def precompute_attn_shares_outproj(attention_weights, projected_values, seq_len, eps=1e-10):
    """attn_shares_outproj[L][H, Q, S] = output-aligned per-(head, source) shares.

    Each (head, source) cell contributes
        s_{h,q,s} = alpha_{h,q,s} * OV(c_s)
    to the realized attention output at query q,
        o_attn(q) = sum_h sum_s s_{h,q,s}.

    Cell share of realized output direction:
        cell_share(h, q, s) = <s_{h,q,s}, o_attn(q)> / ||o_attn(q)||^2
                            = alpha_{h,q,s} * <OV(c_s), o_attn(q)> / ||o_attn(q)||^2

    These sum to ~1 across (h, s) at each q by construction.

    Used at depth >= 1 in place of raw attention_weights when
    --attn-outproj is enabled. Independent of d_t — credits each
    head-source by alignment with the realized attention output, not
    with the target unembedding direction.
    """
    attn_shares_outproj = {}
    for L in attention_weights:
        if L not in projected_values:
            continue
        pv = projected_values[L][:, :seq_len, :]                    # (H, S, D)
        attn = attention_weights[L][:, :, :seq_len]                 # (H, Q, S)

        # o_attn(q) = sum_h sum_s alpha_{h,q,s} * pv[h, s]   shape (Q, D)
        # einsum: (H, Q, S) x (H, S, D) -> (Q, D)
        o_attn = np.einsum('hqs,hsd->qd', attn, pv)                 # (Q, D)

        # ||o_attn(q)||^2  shape (Q,)
        o_norm_sq = (o_attn * o_attn).sum(axis=-1)                   # (Q,)
        o_norm_sq = np.maximum(o_norm_sq, eps)

        # <pv[h, s], o_attn[q]>  shape (H, Q, S)
        # einsum: (H, S, D) x (Q, D) -> (H, Q, S)
        inner = np.einsum('hsd,qd->hqs', pv, o_attn)                 # (H, Q, S)

        # Cell share = alpha * inner / ||o_attn||^2
        attn_shares_outproj[L] = attn * inner / o_norm_sq[None, :, None]
    return attn_shares_outproj


def compute_mlp_decomp_principled(hook_manager, outputs, component_vecs,
                                   d_t_np, num_layers, seq_len):
    """MLP decomposition via d_t effective direction.

    effective_dir = sum_j [phi(pre_j)/pre_j * <W_down[:,j], d_t>] * W_up[j,:]
    score(c_k) = <LN(c_k), effective_dir>

    Used at depth 0 only (requires d_t alignment).
    """
    mlp_decomp = {}
    device = hook_manager.device
    d_t = torch.from_numpy(d_t_np).float().to(device)

    with torch.no_grad():
        for L in range(num_layers):
            hidden = outputs.hidden_states[L][0, :seq_len, :].to(device)

            norm_w, norm_b, norm_eps = hook_manager.get_mlp_norm_params(L)
            h_mean = hidden.mean(dim=-1, keepdim=True)
            h_var = (hidden - h_mean).pow(2).mean(dim=-1, keepdim=True)
            hidden_normed = (hidden - h_mean) / torch.sqrt(h_var + norm_eps)
            hidden_normed = hidden_normed * norm_w.to(device) + norm_b.to(device)

            pre_act, activated = hook_manager.mlp_up_forward(L, hidden_normed)

            W_down, _b_down = hook_manager.get_mlp_down_params(L)
            W_up, _b_up = hook_manager.get_mlp_up_params(L)

            W_down_T = W_down.T                           # (d_mlp, d_model)
            W_down_centered = W_down_T - W_down_T.mean(dim=-1, keepdim=True)
            target_proj = W_down_centered @ d_t           # (d_mlp,)
            safe_pre = pre_act.clone()
            safe_pre[safe_pre.abs() < 1e-10] = 1e-10
            ratio = activated / safe_pre

            eff_dir = ((ratio * target_proj[None, :]) @ W_up).float()

            ln_w = norm_w.detach().cpu().float()
            var = hidden.float().var(dim=-1, keepdim=True)
            inv_std = (1.0 / torch.sqrt(var + norm_eps)).cpu()
            eff_dir_t = eff_dir.cpu().float()

            eligible_names = []
            eligible_vecs = []
            mlp_threshold = hook_manager.get_component_layer(f'mlp_{L}')
            for comp_name, comp_arr in component_vecs.items():
                comp_layer = hook_manager.get_component_layer(comp_name)
                if comp_layer >= mlp_threshold:
                    continue
                eligible_names.append(comp_name)
                eligible_vecs.append(comp_arr[0, :seq_len, :])

            if not eligible_names:
                continue

            stacked = torch.from_numpy(np.stack(eligible_vecs, axis=0)).float()
            stacked_norm = stacked * ln_w[None, None, :] * inv_std[None, :, :]
            scores_all = (stacked_norm * eff_dir_t[None, :, :]).sum(dim=-1).numpy()

            mlp_decomp[L] = {}
            for idx, comp_name in enumerate(eligible_names):
                mlp_decomp[L][comp_name] = scores_all[idx]

    return mlp_decomp


def compute_mlp_decomp_geva(hook_manager, outputs, component_vecs,
                            num_layers, seq_len, eps=1e-10):
    """MLP decomposition via activation-weighted gating direction (Geva).

    Each MLP cell j has key direction k_j = W_up[j,:], value vector
    v_j = W_down[:,j], pre-activation pre_j, and post-activation phi_j.

    Geva weights each cell by its share of the *active* gates' inputs:
        gating_dir = sum_j [phi_j / (sum_k phi_k * pre_j)] * k_j
        share(c_k) = <LN(c_k), gating_dir>

    Writer-shares sum to 1 by construction (writer's fraction of pre_j
    weighted by gate j's fraction of total activation, summed over j).

    Independent of d_t. Used at depth >= 1 to surface chains that
    L2 misses.
    """
    mlp_decomp = {}
    device = hook_manager.device

    with torch.no_grad():
        for L in range(num_layers):
            hidden = outputs.hidden_states[L][0, :seq_len, :].to(device)

            norm_w, norm_b, norm_eps = hook_manager.get_mlp_norm_params(L)
            h_mean = hidden.mean(dim=-1, keepdim=True)
            h_var = (hidden - h_mean).pow(2).mean(dim=-1, keepdim=True)
            hidden_normed = (hidden - h_mean) / torch.sqrt(h_var + norm_eps)
            hidden_normed = hidden_normed * norm_w.to(device) + norm_b.to(device)

            pre_act, activated = hook_manager.mlp_up_forward(L, hidden_normed)
            # pre_act, activated: (seq_len, d_mlp)

            W_up, _ = hook_manager.get_mlp_up_params(L)        # (d_mlp, d_model)

            sum_phi = activated.sum(dim=-1, keepdim=True).clamp(min=eps)

            safe_pre = pre_act.clone()
            safe_pre[safe_pre.abs() < eps] = eps

            cell_weight = activated / sum_phi / safe_pre        # (seq_len, d_mlp)

            gating_dir = cell_weight @ W_up                     # (seq_len, d_model)
            gating_dir = gating_dir.float().cpu()

            ln_w = norm_w.detach().cpu().float()
            var = hidden.float().var(dim=-1, keepdim=True)
            inv_std = (1.0 / torch.sqrt(var + norm_eps)).cpu()

            eligible_names = []
            eligible_vecs = []
            mlp_threshold = hook_manager.get_component_layer(f'mlp_{L}')
            for comp_name, comp_arr in component_vecs.items():
                comp_layer = hook_manager.get_component_layer(comp_name)
                if comp_layer >= mlp_threshold:
                    continue
                eligible_names.append(comp_name)
                eligible_vecs.append(comp_arr[0, :seq_len, :])

            if not eligible_names:
                continue

            stacked = torch.from_numpy(np.stack(eligible_vecs, axis=0)).float()
            stacked_norm = stacked * ln_w[None, None, :] * inv_std[None, :, :]
            scores_all = (stacked_norm * gating_dir[None, :, :]).sum(dim=-1).numpy()

            mlp_decomp[L] = {}
            for idx, comp_name in enumerate(eligible_names):
                mlp_decomp[L][comp_name] = scores_all[idx]

    return mlp_decomp


def compute_mlp_decomp_outproj(hook_manager, outputs, component_vecs,
                                num_layers, seq_len, eps=1e-10):
    """MLP decomposition via output-direction projection.

    Each MLP cell j has key direction k_j = W_up[j,:], value vector
    v_j = W_down[:,j], pre-activation pre_j, and post-activation phi_j.
    The cell's contribution to MLP output h_mlp is s_j = phi_j * v_j.

    Cell j's share of the realized output:
        cell_share(j) = <s_j, h_mlp> / ||h_mlp||^2
                      = phi_j * <v_j, h_mlp> / ||h_mlp||^2
    These sum to 1 by construction (sum_j s_j = h_mlp).

    A writer c_k contributes to cell j via key score: pre_j is raised
    by <LN(c_k), k_j> / sigma. Combined per-writer share:

        share(c_k) = <LN(c_k), output_dir>
        output_dir = sum_j [phi_j * <v_j, h_mlp> / (||h_mlp||^2 * pre_j)] * k_j

    Writer-shares sum to ~1 (linearization, not exact).

    Difference from Geva: Geva weights cells purely by activation.
    Outproj also weights by value-output alignment; cells whose value
    cancels against other cells get low weight even if highly active.
    """
    mlp_decomp = {}
    device = hook_manager.device

    with torch.no_grad():
        for L in range(num_layers):
            hidden = outputs.hidden_states[L][0, :seq_len, :].to(device)

            norm_w, norm_b, norm_eps = hook_manager.get_mlp_norm_params(L)
            h_mean = hidden.mean(dim=-1, keepdim=True)
            h_var = (hidden - h_mean).pow(2).mean(dim=-1, keepdim=True)
            hidden_normed = (hidden - h_mean) / torch.sqrt(h_var + norm_eps)
            hidden_normed = hidden_normed * norm_w.to(device) + norm_b.to(device)

            pre_act, activated = hook_manager.mlp_up_forward(L, hidden_normed)
            # pre_act, activated: (seq_len, d_mlp)

            W_up, _ = hook_manager.get_mlp_up_params(L)        # (d_mlp, d_model)
            W_down, _ = hook_manager.get_mlp_down_params(L)    # (d_model, d_mlp)

            # h_mlp at each position
            h_mlp = activated @ W_down.T                       # (seq_len, d_model)

            h_mlp_norm_sq = (h_mlp * h_mlp).sum(dim=-1, keepdim=True).clamp(min=eps)

            # cell_align[s, j] = <v_j, h_mlp[s]> / ||h_mlp[s]||^2
            cell_align = (h_mlp @ W_down) / h_mlp_norm_sq      # (seq_len, d_mlp)

            safe_pre = pre_act.clone()
            safe_pre[safe_pre.abs() < eps] = eps

            cell_weight = activated * cell_align / safe_pre    # (seq_len, d_mlp)

            output_dir = cell_weight @ W_up                    # (seq_len, d_model)
            output_dir = output_dir.float().cpu()

            ln_w = norm_w.detach().cpu().float()
            var = hidden.float().var(dim=-1, keepdim=True)
            inv_std = (1.0 / torch.sqrt(var + norm_eps)).cpu()

            eligible_names = []
            eligible_vecs = []
            mlp_threshold = hook_manager.get_component_layer(f'mlp_{L}')
            for comp_name, comp_arr in component_vecs.items():
                comp_layer = hook_manager.get_component_layer(comp_name)
                if comp_layer >= mlp_threshold:
                    continue
                eligible_names.append(comp_name)
                eligible_vecs.append(comp_arr[0, :seq_len, :])

            if not eligible_names:
                continue

            stacked = torch.from_numpy(np.stack(eligible_vecs, axis=0)).float()
            stacked_norm = stacked * ln_w[None, None, :] * inv_std[None, :, :]
            scores_all = (stacked_norm * output_dir[None, :, :]).sum(dim=-1).numpy()

            mlp_decomp[L] = {}
            for idx, comp_name in enumerate(eligible_names):
                mlp_decomp[L][comp_name] = scores_all[idx]

    return mlp_decomp