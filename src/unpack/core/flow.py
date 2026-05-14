"""
core.flow - Top-down flow sweep.

`_run_flow_sweep` is mathematically equivalent (to machine precision) to
`backward_recursive` with no pruning, but runs as a single top-down
tensor sweep with O(K * S) memory instead of O(K * S^2). Used for the
aggregate token attribution; the recursive walker is kept for path
extraction (it can prune; the sweep cannot).
"""

import numpy as np

from unpack.core.recursion import _safe_denom


def _run_flow_sweep(
    importance, attn_shares, attention_weights, key_decomp,
    mlp_principled, mlp_l2,
    component_layer, component_order,
    num_layers, num_heads, seq_len, target_pos,
    ablated_components=None,
    query_decomp=None,
    value_decomp=None,
    branch_weights=None,
    attn_shares_outproj=None,
    mlp_geva=None,
    mlp_outproj=None,
):
    """Top-down propagation of signed credit from d_t to token positions.

    Maintains two parallel flow tensors per component:

        flow_d0[name][q]   credit that reached this component as the
                           root of its path (depth 0). Seeded from
                           `importance[name]` at `target_pos` only.
        flow_dk[name][q]   credit that reached this component via
                           composition from a higher layer (depth >= 1).
                           Starts empty; accumulates during the sweep.

    The two are kept separate because they use *different* rules when
    redistributing downstream:

        Depth 0 (flow_d0): value-side uses d_t-aligned `attn_shares`
                           (signed, post-softmax × value-projection);
                           MLP uses principled per-neuron shares.
        Depth >= 1 (flow_dk): value-side uses raw post-softmax
                              `attention_weights`; MLP uses L2-norm
                              fallback with sign inherited from parent.

    When a component redistributes, it consumes `flow_d0[name]` with
    depth-0 rules and `flow_dk[name]` with depth-k rules. Both sets
    of outgoing credit accumulate into the children's `flow_dk`.

    Returns:
        credit_pct:     (seq_len,) percent-of-positive-total token credit
        suppress:       negative-mass ratio
        component_flow: dict name -> (seq_len,) combined signed flow
    """
    flow_d0 = {}
    flow_dk = {}
    for name in component_order:
        if 'bias' in name:
            continue
        flow_d0[name] = np.zeros(seq_len, dtype=np.float64)
        flow_dk[name] = np.zeros(seq_len, dtype=np.float64)

    for name, I_val in importance.items():
        if 'bias' in name or abs(I_val) < 1e-14:
            continue
        if name not in flow_d0:
            flow_d0[name] = np.zeros(seq_len, dtype=np.float64)
            flow_dk[name] = np.zeros(seq_len, dtype=np.float64)
        flow_d0[name][target_pos] += I_val

    for term in ("embedding", "pos_embedding"):
        if term not in flow_d0:
            flow_d0[term] = np.zeros(seq_len, dtype=np.float64)
            flow_dk[term] = np.zeros(seq_len, dtype=np.float64)

    attn_earlier_cache = {}
    mlp_earlier_cache = {}
    for L in range(num_layers):
        attn_earlier = [name for name in component_order
                        if 'bias' not in name
                        and component_layer.get(name, 999) < L]
        attn_earlier_cache[L] = attn_earlier

        mlp_name = f'mlp_{L}'
        mlp_threshold = component_layer.get(mlp_name, L)
        if mlp_threshold != L:
            mlp_earlier = [name for name in component_order
                           if 'bias' not in name
                           and component_layer.get(name, 999) < mlp_threshold]
        else:
            mlp_earlier = attn_earlier
        mlp_earlier_cache[L] = mlp_earlier

    bw = branch_weights if branch_weights is not None else {"K": 0.3, "Q": 0.3, "V": 0.4}

    def _redistribute_head(L, h, q, I_val, depth0):
        """Redistribute I_val arriving at head (L,h) at query pos q."""
        if abs(I_val) < 1e-14:
            return

        has_shares = (L in attn_shares)
        has_weights = (L in attention_weights)
        has_outproj = (attn_shares_outproj is not None
                       and L in attn_shares_outproj)
        has_kd = (L in key_decomp)
        has_qd = (query_decomp is not None and L in query_decomp)
        has_vd = (value_decomp is not None and L in value_decomp)
        ea_names = attn_earlier_cache[L]

        # Value-side: credit split across source positions
        if depth0 and has_shares:
            shares = attn_shares[L][h, q, :seq_len]
            denom = _safe_denom(shares)
            if denom is None:
                return
            source_factors = np.asarray(shares) / denom
        elif has_outproj:
            # depth >= 1 with output-aligned attention.
            shares = attn_shares_outproj[L][h, q, :seq_len]
            denom = _safe_denom(shares)
            if denom is None:
                return
            source_factors = np.asarray(shares) / denom
        elif has_weights:
            weights = attention_weights[L][h, q, :seq_len]
            total_w = float(weights.sum())
            if total_w <= 1e-10:
                return
            source_factors = np.asarray(weights) / total_w
        else:
            flow_dk["embedding"][q] += I_val
            return

        if not ea_names or (not has_kd and not has_qd and not has_vd):
            for s in range(seq_len):
                flow_dk["embedding"][s] += I_val * source_factors[s]
            return

        for s in range(seq_len):
            imp_s = I_val * source_factors[s]
            if abs(imp_s) < 1e-14:
                continue

            w_K = bw.get("K", 0.3) if has_kd else 0.0
            w_Q = bw.get("Q", 0.3) if has_qd else 0.0
            w_V = bw.get("V", 0.4) if has_vd else 0.0

            # K-branch
            if has_kd and w_K > 0.0:
                logits = np.array([
                    float(key_decomp[L][name][h, q, s])
                    if name in key_decomp[L] else 0.0
                    for name in ea_names
                ], dtype=np.float64)
                kd = _safe_denom(logits)
                if kd is not None:
                    weights = (w_K * imp_s) * logits / kd
                    for name, w in zip(ea_names, weights):
                        if abs(w) > 1e-14:
                            flow_dk[name][s] += w

            # Q-branch
            if has_qd and w_Q > 0.0:
                qlogits = np.array([
                    float(query_decomp[L][name][h, q, s])
                    if name in query_decomp[L] else 0.0
                    for name in ea_names
                ], dtype=np.float64)
                qd = _safe_denom(qlogits)
                if qd is not None:
                    qweights = (w_Q * imp_s) * qlogits / qd
                    for name, w in zip(ea_names, qweights):
                        if abs(w) > 1e-14:
                            flow_dk[name][q] += w

            # V-branch
            if has_vd and w_V > 0.0:
                vlogits = np.array([
                    float(value_decomp[L][name][h, s])
                    if name in value_decomp[L] else 0.0
                    for name in ea_names
                ], dtype=np.float64)
                vd = _safe_denom(vlogits)
                if vd is not None:
                    vweights = (w_V * imp_s) * vlogits / vd
                    for name, w in zip(ea_names, vweights):
                        if abs(w) > 1e-14:
                            flow_dk[name][s] += w

    def _redistribute_mlp(L, q, I_val, depth0):
        """Redistribute MLP flow I_val at query pos q."""
        if abs(I_val) < 1e-14:
            return

        ea_names = mlp_earlier_cache[L]
        has_principled = (mlp_principled is not None and L in mlp_principled)
        has_l2 = (mlp_l2 is not None and L in mlp_l2)

        if depth0 and has_principled and ea_names:
            scores = np.array([
                float(mlp_principled[L][name][q])
                if (name in mlp_principled[L]
                    and q < len(mlp_principled[L][name]))
                else 0.0
                for name in ea_names
            ], dtype=np.float64)
            md = _safe_denom(scores)
            if md is not None:
                weights = I_val * scores / md
                for name, w in zip(ea_names, weights):
                    if abs(w) > 1e-14:
                        flow_dk[name][q] += w
                return

        # Deeper levels: outproj > geva > l2
        has_outproj = (mlp_outproj is not None and L in mlp_outproj)
        has_geva = (mlp_geva is not None and L in mlp_geva)

        if has_outproj and ea_names:
            scores = np.array([
                float(mlp_outproj[L][name][q])
                if (name in mlp_outproj[L]
                    and q < len(mlp_outproj[L][name]))
                else 0.0
                for name in ea_names
            ], dtype=np.float64)
            md = _safe_denom(scores)
            if md is not None:
                weights = I_val * scores / md
                for name, w in zip(ea_names, weights):
                    if abs(w) > 1e-14:
                        flow_dk[name][q] += w
                return

        if has_geva and ea_names:
            scores = np.array([
                float(mlp_geva[L][name][q])
                if (name in mlp_geva[L]
                    and q < len(mlp_geva[L][name]))
                else 0.0
                for name in ea_names
            ], dtype=np.float64)
            md = _safe_denom(scores)
            if md is not None:
                weights = I_val * scores / md
                for name, w in zip(ea_names, weights):
                    if abs(w) > 1e-14:
                        flow_dk[name][q] += w
                return

        if has_l2 and ea_names:
            norms = np.array([
                float(mlp_l2[L][name][q])
                if (name in mlp_l2[L]
                    and q < len(mlp_l2[L][name]))
                else 0.0
                for name in ea_names
            ], dtype=np.float64)
            total = norms.sum()
            if total > 0:
                sign = 1.0 if I_val >= 0 else -1.0
                weights = abs(I_val) * norms / total * sign
                for name, w in zip(ea_names, weights):
                    if abs(w) > 1e-14:
                        flow_dk[name][q] += w
                return

        flow_dk["embedding"][q] += I_val

    ablated_set = set(ablated_components or [])

    # Top-down sweep: MLP before heads at each layer (sequential residual
    # in GPT-2; parallel residual in Pythia gets the same correctness).
    for L in range(num_layers - 1, -1, -1):
        mlp_name = f'mlp_{L}'
        if mlp_name in flow_d0:
            mlp_is_ablated = mlp_name in ablated_set
            v0 = flow_d0[mlp_name]
            vk = flow_dk[mlp_name]
            for q in range(seq_len):
                if abs(v0[q]) > 1e-14 and not (mlp_is_ablated and q == target_pos):
                    _redistribute_mlp(L, q, v0[q], depth0=True)
                if abs(vk[q]) > 1e-14 and not (mlp_is_ablated and q == target_pos):
                    _redistribute_mlp(L, q, vk[q], depth0=False)

        for h in range(num_heads):
            head_name = f'attn_{L}_head_{h}'
            if head_name not in flow_d0:
                continue
            head_is_ablated = head_name in ablated_set
            v0 = flow_d0[head_name]
            vk = flow_dk[head_name]
            for q in range(seq_len):
                if abs(v0[q]) > 1e-14 and not (head_is_ablated and q == target_pos):
                    _redistribute_head(L, h, q, v0[q], depth0=True)
                if abs(vk[q]) > 1e-14 and not (head_is_ablated and q == target_pos):
                    _redistribute_head(L, h, q, vk[q], depth0=False)

    component_flow = {}
    for name in flow_d0:
        combined = flow_d0[name] + flow_dk[name]
        if np.any(np.abs(combined) > 1e-14):
            component_flow[name] = combined

    credit = (flow_d0.get("embedding", np.zeros(seq_len))
              + flow_dk.get("embedding", np.zeros(seq_len))).copy()
    if "pos_embedding" in flow_d0:
        credit += flow_d0["pos_embedding"] + flow_dk["pos_embedding"]

    pos_total = credit[credit > 0].sum()
    neg_total = abs(credit[credit < 0].sum())
    credit_pct = credit / pos_total * 100 if pos_total > 0 else np.zeros(seq_len)
    suppress = neg_total / (pos_total + neg_total) if (pos_total + neg_total) > 0 else 0

    return credit_pct, suppress, component_flow