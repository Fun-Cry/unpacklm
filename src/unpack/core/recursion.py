"""
core.recursion - Backward attribution recursion.

Walks the model in reverse from a target token at a target position,
distributing credit among residual-stream components by a depth-aware
rule. The depth-0 (d_t-aligned) and depth >= 1 (parent-credit-driven)
rules are documented inline on `backward_recursive`.

This module is consumed by:
    trace_recursive.py     (single-prompt CLI)
    trace_flow.py          (full pipeline with flow sweep)
    core.flow              (flow sweep uses _safe_denom)
"""

import numpy as np


# ================================================================
#  SafeDenom beta (module-level, settable for sweeps)
# ================================================================

# Module-level beta for SafeDenom. Set via set_beta() to sweep for the
# sensitivity analysis. trace_tensor calls set_beta() at the start of
# each trace so the recursive path and the tensor path stay in sync.
_BETA = 0.3


def set_beta(beta):
    """Update the global SafeDenom beta for all subsequent _safe_denom calls."""
    global _BETA
    _BETA = float(beta)


def get_beta():
    return _BETA


def _safe_denom(values):
    total = sum(values)
    abs_total = sum(abs(v) for v in values)
    if abs_total < 1e-10:
        return None
    # signed sum, but never smaller than beta * abs_sum
    magnitude = max(abs(total), _BETA * abs_total)
    return magnitude if total >= 0 else -magnitude


def _passes_gate(imp, depth, min_frac, geomean_min):
    """Decide whether a path with current cumulative importance survives
    the pruning gate.

    Two criteria, OR-combined:

    * Raw threshold: |imp| >= min_frac. Cheap shallow paths with strong
      direct credit pass.

    * Geomean threshold: the geometric mean of the (depth+1) factors that
      produced imp exceeds geomean_min, i.e.
          |imp|^(1/(depth+1)) >= geomean_min
      Deep paths whose raw score has been attenuated by many small
      per-step shares but whose *average* per-step magnitude is still
      meaningful pass. With geomean_min=None this branch is disabled
      and behaviour reverts exactly to the raw-only gate.

    Use OR rather than AND: any path strong enough on either axis is
    kept. Recursion termination is then guaranteed by max_depth, not
    by raw-score decay.
    """
    abs_imp = abs(imp)
    if abs_imp >= min_frac:
        return True
    if geomean_min is not None and abs_imp > 0.0:
        if abs_imp ** (1.0 / (depth + 1)) >= geomean_min:
            return True
    return False


def _select_mlp_decomp(layer_idx, mlp_outproj, mlp_geva, mlp_l2):
    """Pick the MLP decomposition to use at depth >= 1.

    Priority: outproj > geva > l2. Returns (decomp_dict, decomp_name)
    or (None, None) if no decomp is available for this layer.
    """
    if mlp_outproj is not None and layer_idx in mlp_outproj:
        return mlp_outproj[layer_idx], 'outproj'
    if mlp_geva is not None and layer_idx in mlp_geva:
        return mlp_geva[layer_idx], 'geva'
    if layer_idx in mlp_l2:
        return mlp_l2[layer_idx], 'l2'
    return None, None


# ================================================================
#  Sentinel for backward_to_tokens / trace ablation_target_pos
# ================================================================

# UNSET means "use default" (the trace's target_pos). Explicit None
# means "short-circuit fires at every query position" (legacy
# full-sequence ablation).
_UNSET = object()


# ================================================================
#  Backward recursion
# ================================================================

def backward_recursive(
    scores, query_pos,
    attn_shares, attention_weights, key_decomp, query_decomp,
    mlp_l2, mlp_principled,
    seq_len, credit, paths,
    path_prefix="", min_frac=1e-4, depth=0, max_depth=50,
    ablated_components=None, ablation_target_pos=None,
    include_bias=False,
    value_decomp=None,
    branch_weights=None,
    geomean_min=None,
    mlp_geva=None,
    mlp_outproj=None,
    attn_shares_outproj=None,
):
    """Recursively attribute importance scores back to input token positions.

    Distribution rules depend on whether we're at depth 0 (credit is
    d_t-aligned, came straight from the target-token logit lens) or
    at depth >= 1 (credit was inherited from a parent component).

    Depth 0:
      attn V-side: signed shares (SafeDenom)
      MLP K-side:  d_t-aligned principled effective direction
      attn K-side: centered logits (SafeDenom)

    Depth >= 1:
      attn V-side: raw attention weights
      MLP K-side:  outproj (if mlp_outproj set) > geva (if mlp_geva set) > L2
      attn K-side: centered logits (SafeDenom)

    Signs can flip at any depth via the attention key-side step.

    mlp_geva (default None): optional dict[layer]->dict[comp]->array of
    activation-weighted writer-shares. When provided AND mlp_outproj is
    None, used at depth >= 1 instead of L2. Writer-shares sum to 1 by
    construction, so SafeDenom signed distribution applies.

    mlp_outproj (default None): optional dict[layer]->dict[comp]->array
    of output-aligned writer-shares. When provided, takes precedence over
    geva and L2 at depth >= 1. Writer-shares sum to ~1 (linearization).

    ablated_components (optional): set of component names whose direct-
    effect contribution at ablation_target_pos was overwritten by the
    forward-pass intervention.
    """
    if depth > max_depth:
        return

    ablated_set = ablated_components if ablated_components else set()

    for name, importance in scores.items():
        if not _passes_gate(importance, depth, min_frac, geomean_min):
            continue

        annotated_name = f"{name}@{query_pos}"
        cur_path = f"{path_prefix}{annotated_name}" if path_prefix else annotated_name

        # Bias: skip (constants carry no input information).
        if "bias" in name:
            if include_bias:
                credit[query_pos] += importance
                if _passes_gate(importance, depth, min_frac, geomean_min):
                    paths.append((cur_path, query_pos, importance))
            continue

        # Ablated component.
        if name in ablated_set:
            if ablation_target_pos is None or query_pos == ablation_target_pos:
                marker = f"[ABLATED:{name}]@{query_pos}"
                marked_path = (f"{path_prefix}{marker}"
                               if path_prefix else marker)
                paths.append((marked_path, query_pos, importance))
                continue

        # Embedding / position embedding: terminal.
        if name in ("embedding", "pos_embedding"):
            credit[query_pos] += importance
            if _passes_gate(importance, depth, min_frac, geomean_min):
                paths.append((cur_path, query_pos, importance))
            continue

        # Attention head.
        if name.startswith("attn_") and "head_" in name:
            parts = name.split("_")
            layer_idx = int(parts[1])
            head_idx = int(parts[3])

            if depth == 0:
                if layer_idx not in attn_shares:
                    # No V-side data for this layer: cannot recurse.
                    # Drop instead of recording truncated chain.
                    continue
                shares = attn_shares[layer_idx][head_idx, query_pos, :seq_len]
                denom = _safe_denom(shares)
                if denom is None:
                    continue
                source_factors = np.asarray(shares) / denom
            elif attn_shares_outproj is not None and layer_idx in attn_shares_outproj:
                # depth >= 1 with output-aligned attention.
                # source factors are signed (alpha * <OV, o_attn> can be
                # negative when OV cancels against other heads). Use
                # SafeDenom for signed normalization.
                shares = attn_shares_outproj[layer_idx][head_idx, query_pos, :seq_len]
                denom = _safe_denom(shares)
                if denom is None:
                    continue
                source_factors = np.asarray(shares) / denom
            else:
                if layer_idx not in attention_weights:
                    # No attention weights for this layer: drop instead of
                    # recording truncated chain.
                    continue
                weights = attention_weights[layer_idx][head_idx, query_pos, :seq_len]
                total_w = float(weights.sum())
                if total_w <= 1e-10:
                    continue
                source_factors = np.asarray(weights) / total_w

            for s in range(seq_len):
                imp_s = importance * float(source_factors[s])
                if not _passes_gate(imp_s, depth, min_frac, geomean_min):
                    continue

                q_available = (query_decomp is not None
                               and layer_idx in query_decomp)
                v_available = (value_decomp is not None
                               and layer_idx in value_decomp)

                bw = (branch_weights if branch_weights is not None
                      else {"K": 0.333, "Q": 0.333, "V": 0.333})
                w_K = bw.get("K", 0.333) if (layer_idx in key_decomp) else 0.0
                w_Q = bw.get("Q", 0.333) if q_available else 0.0
                w_V = bw.get("V", 0.333) if v_available else 0.0

                imp_K = imp_s * w_K
                imp_Q = imp_s * w_Q
                imp_V = imp_s * w_V

                # ── K-branch ───────────────────────────────────────
                k_branched = False
                tagged_K = f"{name}[K]@{query_pos}"
                cur_path_K = (f"{path_prefix}{tagged_K}"
                              if path_prefix else tagged_K)
                if layer_idx in key_decomp and w_K > 0.0:
                    contrib_K = {cn: float(arr[head_idx, query_pos, s])
                                 for cn, arr in key_decomp[layer_idx].items()}
                    kd = _safe_denom(contrib_K.values())
                    if kd is not None:
                        sub_K = {k: imp_K * v / kd
                                 for k, v in contrib_K.items()
                                 if _passes_gate(imp_K * v / kd, depth, min_frac, geomean_min)}
                        if sub_K:
                            backward_recursive(
                                sub_K, s,
                                attn_shares, attention_weights, key_decomp, query_decomp,
                                mlp_l2, mlp_principled,
                                seq_len, credit, paths,
                                path_prefix=cur_path_K + "→",
                                min_frac=min_frac, depth=depth+1, max_depth=max_depth, geomean_min=geomean_min,
                                ablated_components=ablated_components,
                                ablation_target_pos=ablation_target_pos,
                                include_bias=include_bias, value_decomp=value_decomp,
                                branch_weights=branch_weights,
                                mlp_geva=mlp_geva, mlp_outproj=mlp_outproj, attn_shares_outproj=attn_shares_outproj,
                            )
                            k_branched = True

                # ── Q-branch ───────────────────────────────────────
                q_branched = False
                tagged_Q = f"{name}[Q]@{query_pos}"
                cur_path_Q = (f"{path_prefix}{tagged_Q}"
                              if path_prefix else tagged_Q)
                if q_available and w_Q > 0.0:
                    contrib_Q = {cn: float(arr[head_idx, query_pos, s])
                                 for cn, arr in query_decomp[layer_idx].items()}
                    qd = _safe_denom(contrib_Q.values())
                    if qd is not None:
                        sub_Q = {k: imp_Q * v / qd
                                 for k, v in contrib_Q.items()
                                 if _passes_gate(imp_Q * v / qd, depth, min_frac, geomean_min)}
                        if sub_Q:
                            backward_recursive(
                                sub_Q, query_pos,
                                attn_shares, attention_weights, key_decomp, query_decomp,
                                mlp_l2, mlp_principled,
                                seq_len, credit, paths,
                                path_prefix=cur_path_Q + "→",
                                min_frac=min_frac, depth=depth+1, max_depth=max_depth, geomean_min=geomean_min,
                                ablated_components=ablated_components,
                                ablation_target_pos=ablation_target_pos,
                                include_bias=include_bias, value_decomp=value_decomp,
                                branch_weights=branch_weights,
                                mlp_geva=mlp_geva, mlp_outproj=mlp_outproj, attn_shares_outproj=attn_shares_outproj,
                            )
                            q_branched = True

                # ── V-branch ───────────────────────────────────────
                v_branched = False
                tagged_V = f"{name}[V]@{query_pos}"
                cur_path_V = (f"{path_prefix}{tagged_V}"
                              if path_prefix else tagged_V)
                if v_available and w_V > 0.0:
                    contrib_V = {cn: float(arr[head_idx, s])
                                 for cn, arr in value_decomp[layer_idx].items()}
                    vd = _safe_denom(contrib_V.values())
                    if vd is not None:
                        sub_V = {k: imp_V * v / vd
                                 for k, v in contrib_V.items()
                                 if _passes_gate(imp_V * v / vd, depth, min_frac, geomean_min)}
                        if sub_V:
                            backward_recursive(
                                sub_V, s,
                                attn_shares, attention_weights, key_decomp, query_decomp,
                                mlp_l2, mlp_principled,
                                seq_len, credit, paths,
                                path_prefix=cur_path_V + "→",
                                min_frac=min_frac, depth=depth+1, max_depth=max_depth, geomean_min=geomean_min,
                                ablated_components=ablated_components,
                                ablation_target_pos=ablation_target_pos,
                                include_bias=include_bias, value_decomp=value_decomp,
                                branch_weights=branch_weights,
                                mlp_geva=mlp_geva, mlp_outproj=mlp_outproj, attn_shares_outproj=attn_shares_outproj,
                            )
                            v_branched = True

                if not (k_branched or q_branched or v_branched):
                    # All branches' sub-credits failed the gate: chain
                    # would terminate at this attention head rather than
                    # at an embedding. Drop instead of recording a
                    # truncated chain. Credit is no longer accumulated
                    # (it never reached an input token).
                    pass
            continue

        # MLP.
        if name.startswith("mlp_"):
            layer_idx = int(name.split("_")[1])

            if depth == 0 and mlp_principled is not None and layer_idx in mlp_principled:
                contrib = {cn: float(arr[query_pos])
                           for cn, arr in mlp_principled[layer_idx].items()
                           if query_pos < len(arr)}
                md = _safe_denom(contrib.values())
                if md is not None:
                    sub = {k: importance * v / md for k, v in contrib.items()
                           if _passes_gate(importance * v / md, depth, min_frac, geomean_min)}
                    backward_recursive(
                        sub, query_pos,
                        attn_shares, attention_weights, key_decomp, query_decomp,
                        mlp_l2, mlp_principled,
                        seq_len, credit, paths,
                        path_prefix=cur_path + "→",
                        min_frac=min_frac, depth=depth+1, max_depth=max_depth, geomean_min=geomean_min,
                        ablated_components=ablated_components,
                        ablation_target_pos=ablation_target_pos,
                        include_bias=include_bias, value_decomp=value_decomp,
                        branch_weights=branch_weights,
                        mlp_geva=mlp_geva, mlp_outproj=mlp_outproj, attn_shares_outproj=attn_shares_outproj,
                    )
                    continue

            # Depth >= 1: pick best decomposition.
            decomp, decomp_name = _select_mlp_decomp(
                layer_idx, mlp_outproj, mlp_geva, mlp_l2,
            )

            if decomp is not None and decomp_name in ('outproj', 'geva'):
                # Signed shares that sum to ~1 → SafeDenom signed distribution.
                contrib = {cn: float(arr[query_pos])
                           for cn, arr in decomp.items()
                           if query_pos < len(arr)}
                md = _safe_denom(contrib.values())
                if md is not None:
                    sub = {k: importance * v / md
                           for k, v in contrib.items()
                           if _passes_gate(importance * v / md, depth, min_frac, geomean_min)}
                    if sub:
                        backward_recursive(
                            sub, query_pos,
                            attn_shares, attention_weights, key_decomp, query_decomp,
                            mlp_l2, mlp_principled,
                            seq_len, credit, paths,
                            path_prefix=cur_path + "→",
                            min_frac=min_frac, depth=depth+1, max_depth=max_depth, geomean_min=geomean_min,
                            ablated_components=ablated_components,
                            ablation_target_pos=ablation_target_pos,
                            include_bias=include_bias, value_decomp=value_decomp,
                            branch_weights=branch_weights,
                            mlp_geva=mlp_geva, mlp_outproj=mlp_outproj, attn_shares_outproj=attn_shares_outproj,
                        )
                        continue

            elif decomp is not None and decomp_name == 'l2':
                # L2 fallback: magnitude-based, sign inherited from parent.
                norms = {cn: float(arr[query_pos])
                         for cn, arr in decomp.items()
                         if query_pos < len(arr)}
                total_norm = sum(norms.values())
                if total_norm > 0:
                    sub = {}
                    for cn, n in norms.items():
                        val = abs(importance) * (n / total_norm)
                        if importance < 0:
                            val = -val
                        if _passes_gate(val, depth, min_frac, geomean_min):
                            sub[cn] = val
                    backward_recursive(
                        sub, query_pos,
                        attn_shares, attention_weights, key_decomp, query_decomp,
                        mlp_l2, mlp_principled,
                        seq_len, credit, paths,
                        path_prefix=cur_path + "→",
                        min_frac=min_frac, depth=depth+1, max_depth=max_depth, geomean_min=geomean_min,
                        ablated_components=ablated_components,
                        ablation_target_pos=ablation_target_pos,
                        include_bias=include_bias, value_decomp=value_decomp,
                        branch_weights=branch_weights,
                        mlp_geva=mlp_geva, mlp_outproj=mlp_outproj, attn_shares_outproj=attn_shares_outproj,
                    )
                    continue

            # No decomposition available: drop instead of recording
            # truncated chain. Credit not accumulated since it never
            # reached an input token.
            continue

        # Unknown component name: drop. We never expect this in normal
        # operation; recording would just clutter the path data.
        continue


def backward_to_tokens(importance,
                       attn_shares, attention_weights, key_decomp, query_decomp,
                       mlp_l2, mlp_principled,
                       target_pos, seq_len,
                       min_frac=1e-4, ablated_components=None,
                       ablation_target_pos=None, include_bias=False,
                       value_decomp=None, branch_weights=None,
                       geomean_min=None,
                       mlp_geva=None, mlp_outproj=None,
                       attn_shares_outproj=None):
    credit = np.zeros(seq_len)
    paths = []

    if ablated_components and ablation_target_pos is None:
        ablation_target_pos = target_pos

    backward_recursive(
        importance, target_pos,
        attn_shares, attention_weights, key_decomp, query_decomp,
        mlp_l2, mlp_principled,
        seq_len, credit, paths,
        min_frac=min_frac,
        ablated_components=ablated_components,
        ablation_target_pos=ablation_target_pos,
        include_bias=include_bias, value_decomp=value_decomp,
        branch_weights=branch_weights,
        geomean_min=geomean_min,
        mlp_geva=mlp_geva, mlp_outproj=mlp_outproj, attn_shares_outproj=attn_shares_outproj,
    )

    pos_total = credit[credit > 0].sum()
    neg_total = abs(credit[credit < 0].sum())
    credit_pct = credit / pos_total * 100 if pos_total > 0 else np.zeros(seq_len)

    if pos_total > 0:
        paths = [(p, pos, val / pos_total * 100) for p, pos, val in paths]
    paths.sort(key=lambda x: abs(x[2]), reverse=True)

    suppress_ratio = neg_total / (pos_total + neg_total) if (pos_total + neg_total) > 0 else 0
    return credit_pct, paths, suppress_ratio, float(pos_total)