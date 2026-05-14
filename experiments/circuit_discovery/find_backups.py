"""Iterative backup detection: ablate, retrace, find compensators, repeat.

Reuses experiments.ablation_tracing.runner: it does (clean trace,
ablate, ablated trace, compare → DiffResult) per (prompt, condition).
Compare classifies each non-ablated component as
    compensator | doubler | breakage | unclear
where 'compensator' is Wang's "backup" (a head that picks up credit
in the same direction the ablated set was supplying).

This driver iterates:

  iter 1: ablate the positive name movers from the labelled circuit.
          Find compensators above (Δdirect, prompt-fraction) thresholds.
          Add them to the cumulative ablation set.
  iter 2: ablate primary movers + iter-1 backups. Find compensators
          again, deduplicating against everything already ablated.
  iter t: ablate everything from prior iterations. Find new
          compensators. Stop early when no new backups pass criteria.

ABC corruption references are generated ONCE before iteration starts,
so iter-to-iter changes only reflect the ablation set, not different
references.

Usage:
    python -m experiments.circuit_discovery.find_backups \\
        /data/.../pythia_1_4b_step143000 \\
        --labels /data/.../labels_1_4b.json \\
        --p-min 0.3 \\
        --high-comp-from /data/.../verify_*.json \\
        --high-comp-cutoff 3.0 \\
        --iterations 3 \\
        --backup-min-delta 0.2 \\
        --backup-min-prompt-fraction 0.5 \\
        --out-dir /data/.../backups_1_4b/ \\
        --device cuda:0
"""

import argparse
import glob
import json
import os
import sys


from collections import defaultdict
from typing import Dict, List, Optional

from experiments.circuit_discovery.utils import load_model
from experiments.circuit_discovery.selection._common import (
    filter_correct, load_run,
)
from experiments.circuit_discovery.ioi.abc_prep import add_abc_references
from experiments.ablation_tracing import ExperimentConfig, run, load_runs


# ──────────────────────────────────────────────────────────────────────
def load_labels(path) -> dict:
    with open(path) as f:
        return json.load(f)


def positive_name_movers(labels_json: dict) -> List[str]:
    """Components labeled `name_mover` (positive sign in {IO, END})."""
    by_role = labels_json.get("by_primary_role", {})
    return sorted(by_role.get("name_mover", []))


def load_high_comp_prompts(verify_path: str, cutoff: float) -> List[str]:
    """Pull prompt strings from a verify_*.json (the file with per-prompt
    rows: ld_clean, ld_faith, ld_comp). Keeps prompts where the comp
    LD is above `cutoff` — the circuit didn't matter much there, so
    they're the prompts most likely to expose backup pathways."""
    with open(verify_path) as f:
        data = json.load(f)
    rows = data.get("rows", [])
    if not rows:
        return []
    return [r["prompt"] for r in rows if r["ld_comp"] >= cutoff]


# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
def aggregate_compensators(out_dir: str) -> List[dict]:
    """Across all saved DiffResults in `out_dir`, sum the delta_direct
    for each component classified as 'compensator', counting how many
    prompts each compensator appears in.

    Returns a ranking sorted by mean delta_direct descending.
    """
    runs = load_runs(out_dir)
    sum_delta: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int]      = defaultdict(int)

    for r in runs:
        if r.diff is None:
            continue
        for cd in r.diff.components:
            if cd.role == "compensator":
                sum_delta[cd.name] += cd.delta_direct
                counts[cd.name]    += 1

    rows = [{
        "component":          name,
        "mean_delta_direct":  sum_delta[name] / counts[name],
        "total_delta_direct": sum_delta[name],
        "n_prompts":          counts[name],
    } for name in sum_delta]
    rows.sort(key=lambda x: -abs(x["mean_delta_direct"]))
    return rows


# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir",
                    help="trace-sweep results dir with prompt_*.json + run_config.json")
    ap.add_argument("--labels", required=True,
                    help="labels JSON from label_circuit.py")
    ap.add_argument("--p-min", type=float, default=0.3)

    # Prompt subset
    ap.add_argument("--high-comp-from", default=None,
                    help="path to verify_*.json (the per-prompt rows file "
                         "from run.py output); restricts to prompts where "
                         "the circuit didn't matter under verification")
    ap.add_argument("--high-comp-cutoff", type=float, default=3.0,
                    help="ld_comp threshold for 'circuit-bypassing' prompts")

    # ABC
    ap.add_argument("--n-abc-refs", type=int, default=10,
                    help="number of ABC corruptions per prompt")
    ap.add_argument("--abc-seed-offset", type=int, default=10000)

    # Ablation set override
    ap.add_argument("--ablate-set", nargs="+", default=None,
                    help="explicit components to ablate; overrides label-based "
                         "positive-name-mover selection")
    ap.add_argument("--condition-label", default="ablate_pos_name_movers")

    # Iteration
    ap.add_argument("--iterations", type=int, default=1,
                    help="how many iterations of (ablate, retrace, find new "
                         "backups, add to ablation set) to run. Stops early "
                         "when no new backups pass criteria.")
    ap.add_argument("--backup-min-delta", type=float, default=0.2,
                    help="mean Δdirect threshold for a compensator to count "
                         "as a backup (default 0.2)")
    ap.add_argument("--backup-min-prompt-fraction", type=float, default=0.5,
                    help="fraction of prompts a compensator must appear in "
                         "to count as a backup (default 0.5 = majority)")

    # Model
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default="/data/s4283341")

    # Output
    ap.add_argument("--out-dir", required=True)

    args = ap.parse_args()

    # ── 1. Load labels, decide initial ablation set ────────────────
    labels = load_labels(args.labels)
    if args.ablate_set is not None:
        primary_set = list(args.ablate_set)
        print(f"  initial ablate set (from --ablate-set): {primary_set}")
    else:
        primary_set = positive_name_movers(labels)
        print(f"  initial ablate set (positive name movers): {primary_set}")
    if not primary_set:
        sys.exit("no components to ablate; check --labels or --ablate-set")

    # ── 2. Load saved trace JSONs, filter by p_min ──────────────────
    cfg, prompts_all = load_run(args.results_dir)
    prompts = filter_correct(prompts_all, args.p_min)
    print(f"  prompts (target_prob >= {args.p_min}): "
          f"{len(prompts)}/{len(prompts_all)}")

    # Optional high-comp filter
    keep_only = None
    if args.high_comp_from:
        high = set(load_high_comp_prompts(args.high_comp_from,
                                          args.high_comp_cutoff))
        if not high:
            print("  warning: --high-comp-from given but no matching rows; "
                  "using all p_min prompts")
        else:
            keep_only = high
            print(f"  restricting to {len(keep_only)} high-comp prompts "
                  f"(comp >= {args.high_comp_cutoff})")

    # ── 3. Load model ───────────────────────────────────────────────
    model_cfg = {"model": {**cfg, "device": args.device,
                           "cache_dir": args.cache_dir}}
    model, tokenizer, hook_manager, device = load_model(model_cfg)

    # ── 4. Build experiment prompts with ABC refs ───────────────────
    # Generated ONCE — stable across iterations so that iter-to-iter
    # changes only reflect the ablation set, not different references.
    exp_prompts = add_abc_references(
        prompts, tokenizer,
        n_abc_refs=args.n_abc_refs,
        abc_seed_offset=args.abc_seed_offset,
        keep_only=keep_only,
    )
    print(f"  built {len(exp_prompts)} prompts with ABC refs "
          f"({args.n_abc_refs} per prompt)")
    if not exp_prompts:
        sys.exit("no prompts after ABC generation; check metadata")

    # ── 5. Iterative backup detection ───────────────────────────────
    n_prompts = len(exp_prompts)
    min_n_prompts = max(1, int(round(args.backup_min_prompt_fraction * n_prompts)))

    cumulative_ablate = list(primary_set)   # everything ablated so far
    backups_per_iter = []                    # per-iteration NEW backups
    iter_summaries = []
    in_circuit = set(labels.get("circuit", []))

    for it in range(1, args.iterations + 1):
        iter_label = f"iter{it}_ablate_{len(cumulative_ablate)}_components"
        iter_dir = os.path.join(args.out_dir, f"iter_{it}")
        os.makedirs(iter_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"ITERATION {it}/{args.iterations}")
        print(f"  ablating {len(cumulative_ablate)} components: "
              f"{cumulative_ablate}")
        print('='*60)

        cond = (iter_label, list(cumulative_ablate))
        expcfg = ExperimentConfig(
            prompts    = exp_prompts,
            conditions = [cond],
            out_dir    = iter_dir,
            mode       = "mean",
            positions  = "target",
            storage    = "full",
            verbose    = True,
        )
        print(f"  running experiment → {iter_dir}")
        run(model, tokenizer, hook_manager, expcfg)

        # Aggregate compensators (excluding things we've already
        # ablated, since those can't be compensators by definition).
        rows = aggregate_compensators(iter_dir)
        rows = [r for r in rows if r["component"] not in set(cumulative_ablate)]
        if not rows:
            print(f"\n  iter {it}: no compensators found")
            iter_summaries.append({
                "iteration":     it,
                "ablated":       list(cumulative_ablate),
                "n_compensators": 0,
                "new_backups":   [],
                "ranking":       [],
            })
            break

        print(f"\n  iter {it}: top compensators (after dedupe)")
        print(f"    {'component':<22} {'mean Δdirect':>13}  "
              f"{'n_prompts':>9}")
        for r in rows[:15]:
            print(f"    {r['component']:<22} "
                  f"{r['mean_delta_direct']:>+13.3f}  "
                  f"{r['n_prompts']:>9}")

        # New backups: above magnitude floor AND prevalence floor AND
        # not already in the cumulative ablation set.
        new_backups = [
            r for r in rows
            if abs(r["mean_delta_direct"]) >= args.backup_min_delta
               and r["n_prompts"] >= min_n_prompts
        ]

        # Mark which of the new backups are also "true Wang backups"
        # (i.e. NOT in the labelled circuit). This is just a label,
        # not a filter — we add to ablation regardless.
        not_in_circuit = [r["component"] for r in new_backups
                          if r["component"] not in in_circuit]

        print(f"\n  iter {it}: {len(new_backups)} new backups passing "
              f"criteria (Δdirect ≥ {args.backup_min_delta}, "
              f"≥ {min_n_prompts}/{n_prompts} prompts)")
        for r in new_backups:
            tag = "Wang-backup" if r["component"] not in in_circuit else "in-circuit"
            print(f"    [{tag}] {r['component']:<22} "
                  f"{r['mean_delta_direct']:>+8.3f}  "
                  f"({r['n_prompts']} prompts)")

        iter_summaries.append({
            "iteration":      it,
            "ablated":        list(cumulative_ablate),
            "n_compensators": len(rows),
            "new_backups":    [r["component"] for r in new_backups],
            "new_backups_not_in_circuit": not_in_circuit,
            "ranking":        rows,
        })

        if not new_backups:
            print(f"\n  iter {it}: no new backups → stopping")
            break

        # Add new backups to cumulative ablation set for the next iter.
        cumulative_ablate = cumulative_ablate + [
            r["component"] for r in new_backups
        ]
        backups_per_iter.append([r["component"] for r in new_backups])

    # ── 6. Final summary ───────────────────────────────────────────
    all_backups = sorted({c for tier in backups_per_iter for c in tier})
    wang_backups = sorted({
        c for tier in backups_per_iter for c in tier
        if c not in in_circuit
    })

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print('='*60)
    print(f"  iterations run:           {len(iter_summaries)}")
    print(f"  primary ablation set:     {len(primary_set)} components")
    print(f"  total new backups:        {len(all_backups)}")
    print(f"  backups not in circuit:   {len(wang_backups)}")
    print(f"  final ablation set size:  {len(cumulative_ablate)}")
    if backups_per_iter:
        print(f"\n  backups per iteration:")
        for it, tier in enumerate(backups_per_iter, 1):
            print(f"    iter {it}: {tier}")
    if wang_backups:
        print(f"\n  Wang-style backups (not in original circuit):")
        for c in wang_backups:
            print(f"    {c}")

    summary_path = os.path.join(args.out_dir, "iterative_backup_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "primary_set":        primary_set,
            "n_prompts":          n_prompts,
            "n_iterations_run":   len(iter_summaries),
            "n_iterations_max":   args.iterations,
            "backup_min_delta":   args.backup_min_delta,
            "backup_min_prompt_fraction": args.backup_min_prompt_fraction,
            "all_new_backups":    all_backups,
            "wang_backups":       wang_backups,
            "final_ablation_set": cumulative_ablate,
            "extended_circuit":   sorted(set(labels.get("circuit", []))
                                         | set(all_backups)),
            "per_iteration":      iter_summaries,
        }, f, indent=2)
    print(f"\n  wrote {summary_path}")


if __name__ == "__main__":
    main()