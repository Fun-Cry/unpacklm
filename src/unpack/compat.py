"""
unpack.compat - Backward-compatible function API for experiment scripts.

Provides trace_flow() returning a raw dict, as used by the experiment
pipeline. Internally delegates to unpack.core primitives.

New code should use unpack.Tracer instead.
"""

import time
import numpy as np

from unpack.core.recursion import backward_recursive, set_beta
from unpack.core.prep import _prepare_trace_inputs
from unpack.core.flow import _run_flow_sweep


def trace_flow(model, tokenizer, text, device="cpu",
               target_position="last", comp_batch_size=8,
               target_token=None, distractor_token=None,
               hook_manager=None, beta=0.3,
               top_paths_k=20, path_min_frac=1e-3,
               ablated_components=None,
               interventions=None,
               enable_q_side=True,
               enable_v_side=True,
               include_bias=False,
               branch_weights=None,
               geomean_min=None,
               mlp_geva_enabled=False,
               mlp_outproj_enabled=False,
               attn_outproj_enabled=False,
               roots=None):
    """Signed backward attribution via a top-down flow sweep.

    Returns a raw dict compatible with the experiment pipeline.
    New code should use unpack.Tracer.trace() instead.
    """
    set_beta(beta)

    prep = _prepare_trace_inputs(
        model, tokenizer, text,
        target_position=target_position,
        target_token=target_token,
        distractor_token=distractor_token,
        hook_manager=hook_manager,
        comp_batch_size=comp_batch_size,
        interventions=interventions,
        enable_q_side=enable_q_side,
        enable_v_side=enable_v_side,
        include_bias=include_bias,
        mlp_geva_enabled=mlp_geva_enabled,
        mlp_outproj_enabled=mlp_outproj_enabled,
        attn_outproj_enabled=attn_outproj_enabled,
    )

    importance = prep["importance"]
    query_pos = prep["t_pos"]

    # Aggregate: top-down flow sweep
    credit_pct, suppress_ratio, component_flow = _run_flow_sweep(
        importance, prep["attn_shares"], prep["attention_weights"],
        prep["key_decomp"], prep["mlp_principled"], prep["mlp_l2"],
        prep["component_layer"], prep["component_order"],
        prep["num_layers"], prep["num_heads"], prep["seq_len"],
        query_pos,
        ablated_components=ablated_components,
        query_decomp=prep["query_decomp"] if enable_q_side else None,
        value_decomp=prep["value_decomp"] if enable_v_side else None,
        branch_weights=branch_weights,
        attn_shares_outproj=prep.get("attn_shares_outproj"),
    )

    # Paths: secondary recursive pass with pruning
    credit_raw_rec = np.zeros(prep["seq_len"])
    paths_raw = []
    backward_recursive(
        importance, query_pos,
        prep["attn_shares"], prep["attention_weights"],
        prep["key_decomp"], prep["query_decomp"],
        prep["mlp_l2"], prep["mlp_principled"],
        prep["seq_len"], credit_raw_rec, paths_raw,
        min_frac=path_min_frac,
        ablated_components=ablated_components,
        ablation_target_pos=query_pos,
        include_bias=include_bias,
        value_decomp=prep["value_decomp"] if enable_v_side else None,
        branch_weights=branch_weights,
        geomean_min=geomean_min,
        mlp_geva=prep.get("mlp_geva"),
        mlp_outproj=prep.get("mlp_outproj"),
        attn_shares_outproj=prep.get("attn_shares_outproj"),
    )

    pos_total_flow = credit_pct[credit_pct > 0].sum()
    rec_pos = credit_raw_rec[credit_raw_rec > 0].sum()
    if rec_pos > 0 and pos_total_flow > 0:
        top_paths = [(p, pos, val / rec_pos * 100)
                     for p, pos, val in paths_raw]
    else:
        top_paths = [(p, pos, 0.0) for p, pos, val in paths_raw]
    top_paths.sort(key=lambda x: abs(x[2]), reverse=True)
    top_paths = top_paths[:top_paths_k]

    paths_raw_sorted = sorted(paths_raw, key=lambda x: abs(x[2]),
                              reverse=True)[:top_paths_k]

    return {
        "root": "target",
        "root_pos": query_pos,
        "tokens": prep["tokens"],
        "target_token": prep["target_token_str"],
        "target_prob": prep["target_prob"],
        "target_logit_centered": prep["target_logit_centered"],
        "predictions": prep["predictions"],
        "token_attribution": credit_pct,
        "token_attribution_raw": credit_raw_rec,
        "top_paths": top_paths,
        "top_paths_raw": paths_raw_sorted,
        "suppress_ratio": suppress_ratio,
        "importance": importance,
        "component_flow": {n: v.copy() for n, v in component_flow.items()},
        "component_layer": dict(prep["component_layer"]),
        "forward_attn_at_t": prep.get("forward_attn_at_t", {}),
        "forward_mlp_at_t": prep.get("forward_mlp_at_t", {}),
    }
