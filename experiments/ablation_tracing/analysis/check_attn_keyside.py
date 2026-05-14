"""Per-head key-side input-layer distribution.

Re-runs a clean trace over a sample of prompts and aggregates the raw
key_decomp slices per attention head — bypassing the std summary stored
in clean_edges_attn. Outputs a layer-share-of-|key_decomp| table per
attention head, optionally restricted to particular receivers.

For each (receiver_head=(L, h), source_layer L'), the value is

    sum over prompts of  sum over s |key_decomp[L][src][h, t_pos, s]|
    summed over all components `src` whose layer == L'

then normalized to a layer-share within each receiver head. This is
the signal the recursive walker actually distributes credit by at
depth ≥1, so it directly answers "where does this head pull credit
from in the flow attribution."

Compensators-vs-non-compensators framing: pass --compensators a comma-
separated list (matched against receiver_head names like attn_10_head_2),
and the table marks each head with [C] for compensator, [.] otherwise.

Usage examples:
    # all attn heads, clean trace, one prompt
    python check_attn_keyside.py --prompt-idx 0 --condition nm_joint

    # restrict to layers 9-11 receivers, mark known compensators
    python check_attn_keyside.py --condition nm_joint \\
        --receiver-layers 9,10,11 \\
        --compensators attn_10_head_2,attn_10_head_10,attn_11_head_2,attn_10_head_7,attn_10_head_6 \\
        --n-prompts 20

    # also run the ablated side and print clean - ablated diff per head
    python check_attn_keyside.py --condition nm_joint \\
        --receiver-layers 10,11 \\
        --side both \\
        --n-prompts 20

Note: requires re-running trace, so it loads the model. Uses the
config from the experiment folder.
"""

import argparse
import os
import sys
import importlib.util
import json
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

# Project root on path

from experiments.ablation_tracing import (
    load_runs, build_intervention, AblationConfig
)
from unpack.core.prep import _prepare_trace_inputs


# --------------------------------------------------------------------
def src_layer(name: str) -> int:
    """embedding / pos_embedding -> -1; mlp_L -> L; attn_L_head_h -> L."""
    if name in ("embedding", "pos_embedding"):
        return -1
    if name.startswith("mlp_"):
        return int(name.split("_")[1])
    if name.startswith("attn_"):
        return int(name.split("_")[1])
    return -2


def load_experiment_folder(folder):
    """Load config.py and prompts.py from an experiment folder."""
    cfg_spec = importlib.util.spec_from_file_location(
        "ioi_cfg", os.path.join(folder, "config.py"))
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    cfg_spec.loader.exec_module(cfg_mod)

    pr_spec = importlib.util.spec_from_file_location(
        "ioi_prompts", os.path.join(folder, "prompts.py"))
    pr_mod = importlib.util.module_from_spec(pr_spec)
    pr_spec.loader.exec_module(pr_mod)

    cd_spec = importlib.util.spec_from_file_location(
        "ioi_conditions", os.path.join(folder, "conditions.py"))
    cd_mod = importlib.util.module_from_spec(cd_spec)
    cd_spec.loader.exec_module(cd_mod)

    return cfg_mod.CONFIG, pr_mod.build_prompts, cd_mod.CONDITIONS


def load_model(cfg):
    """Load the model from CONFIG['model']."""
    family = cfg["model"]["family"]
    size = cfg["model"]["size"]
    device = cfg["model"]["device"]

    if family == "gpt2":
        from unpack.models import load_model, get_adapter
        name_map = {"small": "gpt2", "medium": "gpt2-medium",
                    "large": "gpt2-large", "xl": "gpt2-xl"}
        model_name = name_map.get(size, size)
        model, tokenizer = load_model(model_name, device=device,
                                      cache_dir=cfg["model"].get("cache_dir"))
        hook_manager = get_adapter(model)
    else:
        raise ValueError(f"unsupported family: {family}")

    hook_manager.register_hooks(model)
    return model, tokenizer, hook_manager, device


# --------------------------------------------------------------------
def aggregate_keydecomp(key_decomp, t_pos):
    """For one trace's key_decomp output, compute per-(L, h, src_L) abs sums.

    key_decomp[L][src_name] has shape (H, Q, S). We slice at q=t_pos and
    sum |.| over the S axis to get per-source-position contribution
    summed (the quantity the depth-1 walker distributes credit by, before
    applying signed normalization).

    Returns:
      out[(L, h, src_layer)] = sum over (sources at that layer) of
                               sum_s |key_decomp[L][src][h, t_pos, s]|
    """
    out: Dict[Tuple[int, int, int], float] = defaultdict(float)
    for L, by_name in key_decomp.items():
        for name, arr in by_name.items():
            # arr shape: (H, Q, S)
            if arr is None or t_pos >= arr.shape[1]:
                continue
            slc = arr[:, t_pos, :]                        # (H, S)
            mag = np.abs(slc).sum(axis=-1)                # (H,)
            srcL = src_layer(name)
            for h in range(mag.shape[0]):
                out[(L, h, srcL)] += float(mag[h])
    return out


# --------------------------------------------------------------------
def run_trace_one(model, tokenizer, hook_manager, device, beta,
                  prompt_dict, intervention=None):
    """Run only the prep half of trace_flow (forward + key_decomp build)
    on one prompt, optionally with an intervention. We don't need the
    backward flow sweep for this analysis.

    Returns (key_decomp, t_pos).
    """
    interventions = intervention.interventions if intervention is not None else None
    prep = _prepare_trace_inputs(
        model, tokenizer, prompt_dict["prompt"],
        target_token=prompt_dict["target_token"],
        distractor_token=prompt_dict.get("distractor_token"),
        hook_manager=hook_manager,
        interventions=interventions,
    )
    return prep["key_decomp"], prep["t_pos"]


# --------------------------------------------------------------------
def print_table(agg, n_prompts, label, receivers_filter=None,
                compensator_set=None):
    """Print per-head layer-share table.

    agg[(L, h, srcL)] = absolute key-side magnitude
    """
    # All receiver heads we have data for
    all_heads = sorted({(L, h) for (L, h, _) in agg.keys()})
    if receivers_filter is not None:
        all_heads = [(L, h) for (L, h) in all_heads if L in receivers_filter]

    # Per-head total
    per_head_total: Dict[Tuple[int, int], float] = defaultdict(float)
    for (L, h, _), v in agg.items():
        per_head_total[(L, h)] += v

    # Source layers seen, sorted
    src_layers = sorted({sl for (_, _, sl) in agg.keys()})

    print(f"\n{'=' * 78}")
    print(f"  {label}")
    print(f"  ({n_prompts} prompts; per-head layer-share of "
          f"|key_decomp[L][src][h,t,:]|)")
    print(f"{'=' * 78}")
    print(f"  [C] = compensator (per --compensators)" if compensator_set else "")

    header = f"  {'receiver':<22}" + "".join(f"L{sl:>4}" for sl in src_layers) \
             + f"   {'total':>9}"
    print(header)
    print("-" * len(header))

    for (L, h) in all_heads:
        name = f"attn_{L}_head_{h}"
        flag = "[C]" if compensator_set and name in compensator_set else "[.]"
        total = per_head_total[(L, h)]
        row = f"  {flag} {name:<18}"
        for sl in src_layers:
            v = agg.get((L, h, sl), 0.0)
            share = v / total if total > 0 else 0.0
            row += f"{'·':>5}" if share == 0.0 else f"{share*100:>4.0f}%"
        row += f"   {total:>9.3f}"
        print(row)


# --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="experiments/ablation_tracing/ioi",
                        help="experiment folder with config.py / prompts.py / conditions.py")
    parser.add_argument("--results-dir", default=None,
                        help="results dir to read clean_target_prob/canonical "
                             "filter from. Default: <folder>/results")
    parser.add_argument("--condition", default="nm_joint",
                        help="condition label whose ablated_components to use "
                             "for the ablated side (default: nm_joint)")
    parser.add_argument("--side", choices=("clean", "ablated", "both"),
                        default="clean")
    parser.add_argument("--n-prompts", type=int, default=10,
                        help="how many canonical-correct prompts to trace "
                             "(default: 10). More = lower variance.")
    parser.add_argument("--prompt-idx", type=int, default=None,
                        help="trace just this single prompt index instead of "
                             "the first --n-prompts canonical-correct.")
    parser.add_argument("--receiver-layers", default=None,
                        help="comma-separated receiver layers to show, "
                             "e.g. '9,10,11'. Default: all.")
    parser.add_argument("--compensators", default=None,
                        help="comma-separated compensator names to mark with "
                             "[C] in the table.")
    parser.add_argument("--p-min", type=float, default=0.1)
    args = parser.parse_args()

    # ── Setup ──
    cfg, build_prompts, conditions = load_experiment_folder(args.folder)
    results_dir = args.results_dir or os.path.join(args.folder, "results")

    print(f"loading model: {cfg['model']['family']}-{cfg['model']['size']}")
    model, tokenizer, hook_manager, device = load_model(cfg)
    beta = cfg["trace"]["beta"]

    # ── Pick the prompts ──
    print("building prompt list")
    prompts = build_prompts(tokenizer)

    # Filter to canonical-correct prompts via the existing run results.
    runs = load_runs(results_dir)
    runs_for_label = [r for r in runs if r.label == args.condition]
    canonical_idx = [
        r.prompt_idx for r in runs_for_label
        if r.clean_target_prob >= args.p_min and r.clean_target_logit > 0
    ]

    if args.prompt_idx is not None:
        chosen = [args.prompt_idx]
    else:
        chosen = canonical_idx[:args.n_prompts]
    print(f"tracing {len(chosen)} prompts: {chosen[:10]}{'...' if len(chosen) > 10 else ''}")

    # ── Compute clean and ablated aggregates ──
    receivers_filter = None
    if args.receiver_layers:
        receivers_filter = set(int(x) for x in args.receiver_layers.split(","))

    compensator_set = None
    if args.compensators:
        compensator_set = {x.strip() for x in args.compensators.split(",")}

    agg_clean: Dict[Tuple[int, int, int], float] = defaultdict(float)
    agg_ablated: Dict[Tuple[int, int, int], float] = defaultdict(float)

    # Find ablation set for the chosen condition.
    ablated_components = None
    if args.side in ("ablated", "both"):
        for label, comps in conditions:
            if label == args.condition:
                ablated_components = comps
                break
        if ablated_components is None:
            raise SystemExit(f"condition '{args.condition}' not in conditions.py")
        print(f"ablation set: {ablated_components}")

    for i, pidx in enumerate(chosen):
        p = prompts[pidx]
        print(f"  [{i+1}/{len(chosen)}] prompt {pidx}: {p['prompt'][:60]}...")

        if args.side in ("clean", "both"):
            kd, tp = run_trace_one(
                model, tokenizer, hook_manager, device, beta, p,
                intervention=None,
            )
            for k, v in aggregate_keydecomp(kd, tp).items():
                agg_clean[k] += v

        if args.side in ("ablated", "both"):
            ab_cfg = AblationConfig(
                components = ablated_components,
                mode       = cfg["ablation"]["mode"],
                references = p["references"],
                positions  = cfg["ablation"]["positions"],
            )
            iv = build_intervention(
                model, tokenizer, hook_manager,
                p["prompt"], ab_cfg,
            )
            kd, tp = run_trace_one(
                model, tokenizer, hook_manager, device, beta, p,
                intervention=iv,
            )
            for k, v in aggregate_keydecomp(kd, tp).items():
                agg_ablated[k] += v

    # ── Print ──
    if args.side in ("clean", "both"):
        print_table(agg_clean, len(chosen),
                    label=f"CLEAN  (condition '{args.condition}')",
                    receivers_filter=receivers_filter,
                    compensator_set=compensator_set)

    if args.side in ("ablated", "both"):
        print_table(agg_ablated, len(chosen),
                    label=f"ABLATED  (condition '{args.condition}', "
                          f"set={ablated_components})",
                    receivers_filter=receivers_filter,
                    compensator_set=compensator_set)


if __name__ == "__main__":
    main()