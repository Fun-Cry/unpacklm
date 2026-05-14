"""End-to-end: select a circuit, then verify it.

Wires the selection method (currently partition_coverage) into the
verification pipeline in one process so the model loads once for an
entire partition-threshold sweep.

Example — single run:

    python -m experiments.circuit_discovery.run \\
        --results-dir /data/.../pythia_1_4b_step143000 \\
        --task ioi \\
        --partition-threshold 0.05 \\
        --min-prompt-fraction 0.25 \\
        --always-on mlp_0 \\
        --p-min 0.3 \\
        --device cuda:0 \\
        --out-dir /data/.../verify_runs/

Example — sweep over partition_threshold (model loaded once):

    python -m experiments.circuit_discovery.run \\
        --results-dir /data/.../pythia_1_4b_step143000 \\
        --task ioi \\
        --partition-threshold-sweep 0.01 0.05 0.10 0.20 \\
        --min-prompt-fraction 0.25 \\
        --always-on mlp_0 --p-min 0.3 --device cuda:0 \\
        --out-dir /data/.../verify_sweep/
"""

import argparse
import json
import os
import sys
from typing import Iterable, List


from experiments.circuit_discovery.selection import METHODS, SelectionResult
from experiments.circuit_discovery.selection._common import (
    DEFAULT_EXCLUDE, filter_correct, load_run,
    resolve_role_keys,
)
from experiments.circuit_discovery.utils import load_model
from experiments.circuit_discovery.ioi.abc_prep import add_abc_references
from experiments.circuit_discovery.verification.__main__ import (
    add_ioi_length_matched_references, verify, print_summary,
)


# ──────────────────────────────────────────────────────────────────────
# Selection
# ──────────────────────────────────────────────────────────────────────
def run_selection(prompts, strategy: str, *, role_keys, exclude,
                  partition_threshold, min_prompt_fraction,
                  elbow_rescue, elbow_floor_k) -> SelectionResult:
    if strategy not in METHODS:
        raise ValueError(
            f"unknown strategy {strategy!r}; choices: {sorted(METHODS)}"
        )
    return METHODS[strategy](
        prompts,
        role_keys=role_keys,
        exclude=set(exclude),
        partition_threshold=partition_threshold,
        min_prompt_fraction=min_prompt_fraction,
        elbow_rescue=elbow_rescue,
        elbow_floor_k=elbow_floor_k,
    )


# ──────────────────────────────────────────────────────────────────────
# Verification (model held by caller so sweeps don't reload)
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# Verification (helpers and core function imported from verification.__main__)
# ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="circuit_discovery output dir with prompt_*.json")
    ap.add_argument("--strategy", default="partition_coverage",
                    choices=sorted(METHODS),
                    help="selection strategy (default: partition_coverage)")

    # Task / role mapping
    ap.add_argument("--task", default=None,
                    help="imports experiments.circuit_discovery.<task>.roles")
    ap.add_argument("--positions-from-metadata", default=None,
                    help="inline 'field=LABEL,...' role mapping")

    # Selection params
    ap.add_argument("--partition-threshold", type=float, default=0.05,
                    help="keep partitions whose share of total kept "
                         "mass is >= this")
    ap.add_argument("--partition-threshold-sweep", type=float, nargs="+",
                    default=None,
                    help="if given, sweep partition_threshold over these "
                         "values (model loaded once)")
    ap.add_argument("--min-prompt-fraction", type=float, default=0.25,
                    help="within each partition, drop normalized chains "
                         "appearing in fewer than this fraction of kept "
                         "prompts (default 0.25)")
    ap.add_argument("--no-elbow-rescue", action="store_true",
                    help="disable elbow-based rescue of below-threshold "
                         "partitions (default: rescue is on)")
    ap.add_argument("--elbow-floor-k", type=int, default=1,
                    help="within each kept partition, keep at least "
                         "this many chains even if the within-partition "
                         "elbow returns 0 (default 1)")
    ap.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE),
                    help="components excluded from selection's denominator")

    # Verification params
    ap.add_argument("--always-on", nargs="*", default=[],
                    help="components NEVER ablated during verification")
    ap.add_argument("--p-min", type=float, default=0.3,
                    help="clean_target_prob threshold for kept prompts")
    ap.add_argument("--max-prompts", type=int, default=None)

    # Reference source for mean ablation
    rgrp = ap.add_mutually_exclusive_group()
    rgrp.add_argument("--abc", action="store_true",
                      help="use ABC corruptions as mean-ablation refs "
                           "(Wang-faithful; default)")
    rgrp.add_argument("--ioi-length-matched", action="store_true",
                      help="use length-matched IOI prompts as refs "
                           "(available for comparison)")
    ap.add_argument("--n-abc-refs", type=int, default=10)
    ap.add_argument("--abc-seed-offset", type=int, default=10000)

    # Model
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default="/data/s4283341")

    # Output
    ap.add_argument("--out-dir", default=None,
                    help="if set, writes per-threshold circuit + verify "
                         "JSONs and a sweep_summary.json")

    args = ap.parse_args()

    # Default to ABC if neither reference flag was given
    if not args.abc and not args.ioi_length_matched:
        args.abc = True

    role_keys = resolve_role_keys(args.task, args.positions_from_metadata)
    if not role_keys:
        sys.exit(
            "partition_coverage needs a role mapping; "
            "pass --task or --positions-from-metadata"
        )

    thresholds = (args.partition_threshold_sweep
                  if args.partition_threshold_sweep
                  else [args.partition_threshold])

    # ── Load run + prompts ─────────────────────────────────────────
    cfg, prompts_all = load_run(args.results_dir)
    prompts_kept = filter_correct(prompts_all, args.p_min)
    print(f"=== {args.results_dir} ===")
    print(f"  model: {cfg.get('family')}-{cfg.get('size')}  "
          f"step={cfg.get('step')}")
    print(f"  prompts: {len(prompts_kept)}/{len(prompts_all)} "
          f"(target_prob >= {args.p_min})")
    print(f"  strategy: {args.strategy}")
    print(f"  partition thresholds: {thresholds}")
    print(f"  min_prompt_fraction:  {args.min_prompt_fraction}")
    print(f"  elbow_floor_k:        {args.elbow_floor_k}")

    # ── Load model once for the whole sweep ────────────────────────
    model_cfg = {"model": {**cfg, "device": args.device,
                           "cache_dir": args.cache_dir}}
    model, tokenizer, hook_manager, device = load_model(model_cfg)
    num_layers = model.config.num_hidden_layers
    num_heads  = model.config.num_attention_heads
    print(f"  loaded {cfg.get('family')}-{cfg.get('size')}  "
          f"L={num_layers} H={num_heads}")

    # ── Build prepared prompts (with references) ────────────────────
    if args.abc:
        prepared = add_abc_references(
            prompts_kept, tokenizer,
            n_abc_refs=args.n_abc_refs,
            abc_seed_offset=args.abc_seed_offset,
        )
        ref_kind = "ABC corruptions"
    else:
        prepared = add_ioi_length_matched_references(
            prompts_kept, tokenizer,
        )
        ref_kind = "IOI length-matched"
    print(f"  prepared: {len(prepared)} prompts (refs: {ref_kind})")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    sweep_summary = []
    for thr in thresholds:
        print(f"\n{'='*60}\npartition_threshold = {thr}\n{'='*60}")

        # Select
        sel = run_selection(
            prompts_kept, args.strategy,
            role_keys=role_keys,
            exclude=args.exclude,
            partition_threshold=thr,
            min_prompt_fraction=args.min_prompt_fraction,
            elbow_rescue=not args.no_elbow_rescue,
            elbow_floor_k=args.elbow_floor_k,
        )
        print(f"  selected {sel.n_components} components: {sel.components}")

        if not sel.components:
            print("  (empty selection — skipping verification)")
            sweep_summary.append({
                "threshold": thr, "n_components": 0,
                "summary": None, "circuit": [],
            })
            continue

        # Verify
        verify_summary = verify(
            model, tokenizer, hook_manager, device,
            prepared,
            circuit=set(sel.components),
            always_on=set(args.always_on),
            num_layers=num_layers, num_heads=num_heads,
            max_prompts=args.max_prompts,
            verbose=(len(thresholds) == 1),
        )
        print_summary(verify_summary)

        sweep_summary.append({
            "threshold":     thr,
            "n_components":  sel.n_components,
            "circuit":       sel.components,
            "selection":     sel.to_json(),
            "summary":       {k: v for k, v in verify_summary.items()
                              if k != "rows"},
        })

        if args.out_dir:
            tag = f"{args.strategy}_t{thr}"
            with open(os.path.join(args.out_dir, f"circuit_{tag}.json"), "w") as f:
                payload = sel.to_json()
                payload["results_dir"] = args.results_dir
                json.dump(payload, f, indent=2)
            with open(os.path.join(args.out_dir, f"verify_{tag}.json"), "w") as f:
                json.dump(verify_summary, f, indent=2)

    # Sweep summary
    if len(sweep_summary) > 1:
        print(f"\n{'='*60}\nsweep summary\n{'='*60}")
        print(f"  {'thr':>5}  {'|C|':>4}  {'faith':>7}  {'comp':>7}")
        for row in sweep_summary:
            s = row["summary"]
            if s is None:
                print(f"  {row['threshold']:>5.2f}  {row['n_components']:>4}  "
                      f"   --       --")
            else:
                print(f"  {row['threshold']:>5.2f}  {row['n_components']:>4}  "
                      f"{s['faith_ratio']:>+7.2f}  {s['comp_drop']:>+7.2f}")

    if args.out_dir:
        with open(os.path.join(args.out_dir, "sweep_summary.json"), "w") as f:
            json.dump({
                "results_dir":           args.results_dir,
                "strategy":              args.strategy,
                "p_min":                 args.p_min,
                "always_on":             args.always_on,
                "partition_thresholds":  thresholds,
                "min_prompt_fraction":   args.min_prompt_fraction,
                "elbow_floor_k":         args.elbow_floor_k,
                "elbow_rescue":          not args.no_elbow_rescue,
                "rows":                  sweep_summary,
            }, f, indent=2)
        print(f"\n  wrote {args.out_dir}/sweep_summary.json")


if __name__ == "__main__":
    main()