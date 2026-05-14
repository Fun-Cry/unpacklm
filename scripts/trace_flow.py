"""
trace_flow.py - Top-down flow sweep CLI for backward attribution.

Thin wrapper around core.prep + core.flow + core.recursion.

The flow sweep is mathematically equivalent to backward_recursive
without pruning, but uses O(K * S) memory instead of O(K * S^2).
A secondary recursive pass with pruning is run only to extract named
paths for display.

`roots` parameter (single string, list of strings, or None) controls
where attribution begins:
  None                     → standard target-rooted trace.
  'attn_L_head_H'          → re-root at this component, at t_pos.
  'attn_L_head_H@p'        → re-root at this component, at position p.
  list of the above        → multi-root trace, sharing one forward pass.
"""

import argparse
import time
import json

import numpy as np
import torch

from unpack.core import (
    backward_recursive,
    set_beta,
)
from unpack.core.prep import _prepare_trace_inputs
from unpack.core.flow import _run_flow_sweep


# ================================================================
#  Root parsing
# ================================================================

def _parse_root(root, default_pos, seq_len, component_layer):
    """Parse a root spec into (component_name_or_None, query_pos).

    None              → (None, default_pos)
    'name'            → ('name', default_pos)
    'name@p'          → ('name', p)

    Validates name and pos.
    """
    if root is None:
        return None, default_pos
    if "@" in root:
        comp_name, pos_str = root.rsplit("@", 1)
        try:
            query_pos = int(pos_str)
        except ValueError:
            raise ValueError(
                f"Bad position in root={root!r}; expected integer after '@'."
            )
    else:
        comp_name = root
        query_pos = default_pos
    if comp_name not in component_layer:
        examples = list(component_layer.keys())[:6]
        raise ValueError(
            f"root component={comp_name!r} not in model components. "
            f"Examples: {examples}"
        )
    if query_pos < 0 or query_pos >= seq_len:
        raise ValueError(
            f"root position {query_pos} out of range [0, {seq_len})."
        )
    return comp_name, query_pos


# ================================================================
#  Top-level entry point
# ================================================================

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

    `roots`:
      None | 'name' | 'name@p' | list of str   See module docstring.
    """
    set_beta(beta)

    # ── One expensive prep, shared across all roots ──
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

    return_single = False
    if roots is None:
        root_list = [None]
        return_single = True
    elif isinstance(roots, str):
        root_list = [roots]
        return_single = True
    else:
        root_list = list(roots)

    results = {}
    for root in root_list:
        result = _run_one_root(
            prep, root,
            top_paths_k=top_paths_k,
            path_min_frac=path_min_frac,
            ablated_components=ablated_components,
            include_bias=include_bias,
            enable_q_side=enable_q_side,
            enable_v_side=enable_v_side,
            branch_weights=branch_weights,
            geomean_min=geomean_min,
        )
        results[result["root"]] = result

    if return_single:
        return next(iter(results.values()))
    return results


def _run_one_root(prep, root, *,
                  top_paths_k, path_min_frac, ablated_components,
                  include_bias, enable_q_side, enable_v_side,
                  branch_weights, geomean_min):
    """Run one backward attribution starting from `root`.

    root=None             → target-rooted (uses prep['importance'] at t_pos).
    root='name'           → re-rooted at component 'name', at t_pos.
    root='name@p'         → re-rooted at component 'name', at position p.
    """
    comp_name, query_pos = _parse_root(
        root,
        default_pos=prep["t_pos"],
        seq_len=prep["seq_len"],
        component_layer=prep["component_layer"],
    )

    if comp_name is None:
        importance = prep["importance"]
        root_label = "target"
    else:
        importance = {comp_name: 1.0}
        root_label = f"{comp_name}@{query_pos}"

    # ── Aggregate: top-down flow sweep ──
    t0 = time.time()
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
    t_sweep = time.time() - t0

    # ── Paths: secondary recursive pass with pruning ──
    t0 = time.time()
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

    paths_raw_sorted = sorted(paths_raw, key=lambda x: abs(x[2]), reverse=True)
    top_paths_raw = paths_raw_sorted[:top_paths_k]

    t_paths = time.time() - t0

    return {
        "root": root_label,
        "root_pos": query_pos,
        "tokens": prep["tokens"],
        "target_token": prep["target_token_str"],
        "target_prob": prep["target_prob"],
        "target_logit_centered": prep["target_logit_centered"],
        "predictions": prep["predictions"],
        "token_attribution": credit_pct,
        "token_attribution_raw": credit_raw_rec,
        "top_paths": top_paths,
        "top_paths_raw": top_paths_raw,
        "suppress_ratio": suppress_ratio,
        "importance": importance,
        "component_flow": {n: v.copy() for n, v in component_flow.items()},
        "component_layer": dict(prep["component_layer"]),
        "forward_attn_at_t": prep.get("forward_attn_at_t", {}),
        "forward_mlp_at_t":  prep.get("forward_mlp_at_t",  {}),
        "t_sweep": t_sweep,
        "t_paths": t_paths,
    }


# ================================================================
#  Comparison sentences for diagnostic mode
# ================================================================

SENTENCES = [
    ("The capital of France is", " Paris"),
    ("Barack Obama was born in", " Hawaii"),
    ("Paris is the capital of", " France"),
    ("The largest planet in the solar system is", " Jupiter"),
    ("The color of the sky is", " blue"),
    ("The color of grass is", " green"),
    ("The opposite of hot is", " cold"),
    ("The opposite of small is", " big"),
    ("The quick brown fox jumps over the lazy", " dog"),
    ("Once upon a", " time"),
    ("The capital of the state containing Dallas is", " Austin"),
    ("Einstein was famous for", " his"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_family", type=str, default="pythia",
                        choices=["pythia", "gpt2"])
    parser.add_argument("--model_size", type=str, default="410m")
    parser.add_argument("--step", type=int, default=143000)
    parser.add_argument("--deduped", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--target-token", type=str, default=None)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--mlp-geva", action="store_true", default=False)
    parser.add_argument("--mlp-outproj", action="store_true", default=False)
    parser.add_argument("--attn-outproj", action="store_true", default=False)
    parser.add_argument("--outproj", action="store_true", default=False)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--root", action="append", default=None,
                        help="Re-root at this component. Format: "
                             "'attn_L_head_H' (defaults to t_pos) or "
                             "'attn_L_head_H@P' (specific position). "
                             "Repeatable; multiple roots share one prep.")
    args = parser.parse_args()

    from unpack.models import load_model, get_adapter

    if args.model_family == "pythia":
        deduped_suffix = "-deduped" if args.deduped else ""
        model_name = f"EleutherAI/pythia-{args.model_size}{deduped_suffix}"
        print(f"Loading {model_name}...")
        model, tokenizer = load_model(model_name, device=args.device,
                                      cache_dir=args.cache_dir,
                                      step=args.step)
    elif args.model_family == "gpt2":
        name_map = {"small": "gpt2", "medium": "gpt2-medium",
                    "large": "gpt2-large", "xl": "gpt2-xl"}
        model_name = name_map.get(args.model_size, args.model_size)
        print(f"Loading {model_name}...")
        model, tokenizer = load_model(model_name, device=args.device,
                                      cache_dir=args.cache_dir)

    hook_manager = get_adapter(model)
    hook_manager.register_hooks(model)

    if args.text:
        sentences = [(args.text, args.target_token)]
    else:
        sentences = SENTENCES

    if args.root is None:
        roots_arg = None
    elif len(args.root) == 1:
        roots_arg = args.root[0]
    else:
        roots_arg = args.root

    results = []
    for text, target in sentences:
        print(f"\n{'=' * 70}")
        label = f"  \"{text}\"" + (f" -> {target}" if target else "")
        print(label)
        print(f"{'=' * 70}")

        t0 = time.time()
        data = trace_flow(model, tokenizer, text, device=args.device,
                          target_token=target, hook_manager=hook_manager,
                          mlp_geva_enabled=args.mlp_geva,
                          mlp_outproj_enabled=args.mlp_outproj or args.outproj,
                          attn_outproj_enabled=args.attn_outproj or args.outproj,
                          roots=roots_arg)
        elapsed = time.time() - t0

        if isinstance(roots_arg, list):
            root_results = list(data.items())
        else:
            root_results = [(data["root"], data)]

        for root_label, rdata in root_results:
            print(f"\n--- root: {root_label} ---")
            tokens = rdata["tokens"]
            attr = rdata["token_attribution"]
            seq_len = len(tokens)

            print(f"  Tracing for: {rdata['target_token']!r}"
                  f" (p={rdata['target_prob']:.4f})")
            print(f"  Top-1 prediction: {rdata['predictions'][0][0]!r}"
                  f" (p={rdata['predictions'][0][1]:.4f})")
            print(f"  Time: {elapsed:.1f}s total"
                  f"  (sweep={rdata['t_sweep']:.2f}s"
                  f"  paths={rdata['t_paths']:.2f}s)")
            print(f"  Suppress: {rdata['suppress_ratio']:.1%}")

            print(f"\n  {'pos':>4}  {'token':<16} {'%':>8}  bar")
            print(f"  {'─'*4}  {'─'*16} {'─'*8}  {'─'*30}")
            for s in range(seq_len):
                if attr[s] >= 0:
                    bar = "█" * int(attr[s] / 2)
                else:
                    bar = "▒" * int(abs(attr[s]) / 2) + " (suppresses)"
                print(f"  {s:>4}  {tokens[s]:<16} {attr[s]:>+7.1f}%  {bar}")

            top_paths = rdata["top_paths"]
            print(f"\n  {'#':>3}  {'path':<50} {'pos':>4} {'tok':<8} {'%':>6}")
            print(f"  {'─'*3}  {'─'*50} {'─'*4} {'─'*8} {'─'*6}")
            for i, (path, pos, pct) in enumerate(top_paths[:args.top]):
                tok = tokens[pos] if pos < seq_len else "?"
                disp = path if len(path) <= 50 else "…" + path[-49:]
                print(f"  {i+1:>3}  {disp:<50} {pos:>4} {tok:<8} {pct:>+5.1f}%")

        results.append(data)

    if args.save:
        def _jsonify(obj):
            if isinstance(obj, dict):
                return {str(k): _jsonify(v) for k, v in obj.items()}
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, list):
                return [_jsonify(x) for x in obj]
            if isinstance(obj, tuple):
                return [_jsonify(x) for x in obj]
            return obj
        with open(args.save, "w") as f:
            json.dump([_jsonify(r) for r in results], f, indent=2)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()