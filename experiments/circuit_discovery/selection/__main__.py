"""Run a selection method from the CLI.

Selection itself is task-agnostic. The role mapping (which metadata
fields denote task-relevant token positions) is supplied via:

    --task <name>
        Imports experiments.circuit_discovery.<name>.roles and reads
        ROLE_KEYS. For IOI: --task ioi.

    --positions-from-metadata 'field=LABEL,field=LABEL,...'
        Inline mapping for ad-hoc tasks.
        Example: --positions-from-metadata subject_position=SUBJECT

partition_coverage requires one or the other.

Usage:
    python -m experiments.circuit_discovery.selection METHOD <results_dir>
        [--task ioi | --positions-from-metadata 'a=A,b=B']
        [--p-min 0.3]
        [--exclude embedding pos_embedding mlp_0]
        [--partition-threshold 0.05] [--path-threshold 0.9]
        [--out circuit.json]

Currently registered methods: see selection.METHODS.
"""

import argparse
import json
import sys

from . import METHODS
from ._common import (
    DEFAULT_EXCLUDE, filter_correct, load_run, resolve_role_keys,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=sorted(METHODS.keys()))
    ap.add_argument("results_dir")
    ap.add_argument("--p-min", type=float, default=0.3)
    ap.add_argument("--task", default=None,
                    help="task module name (e.g. 'ioi'); imports "
                         "experiments.circuit_discovery.<task>.roles "
                         "for the role mapping")
    ap.add_argument("--positions-from-metadata", default=None,
                    help="inline role mapping, "
                         "'field=LABEL,field=LABEL,...'")
    ap.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE),
                    help="components removed from selection AND from "
                         "the total-mass denominator")
    ap.add_argument("--partition-threshold", type=float, default=0.05,
                    help="keep partitions whose share of total kept "
                         "mass is >= this")
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
    ap.add_argument("--role-union", action="store_true",
                    help="component_flow only: union the elbow-picked "
                         "set with components touching any role position "
                         "(rescues role-active components below elbow)")
    ap.add_argument("--out", default=None,
                    help="write SelectionResult JSON consumable by "
                         "verification via --circuit-file")
    args = ap.parse_args()

    role_keys = resolve_role_keys(args.task, args.positions_from_metadata)
    if not role_keys:
        sys.exit(
            "partition_coverage needs a role mapping; "
            "pass --task or --positions-from-metadata"
        )

    cfg, prompts_all = load_run(args.results_dir)
    prompts = filter_correct(prompts_all, args.p_min)

    print(f"=== {args.results_dir} ===")
    print(f"  model:   {cfg.get('family')}-{cfg.get('size')}  "
          f"step={cfg.get('step')}")
    print(f"  prompts: {len(prompts)}/{len(prompts_all)} "
          f"(target_prob >= {args.p_min})")
    print(f"  method:  {args.method}  exclude={sorted(args.exclude)}")
    print(f"  roles:   {dict(role_keys)}")

    fn = METHODS[args.method]
    result = fn(
        prompts,
        role_keys=role_keys,
        exclude=set(args.exclude),
        partition_threshold=args.partition_threshold,
        min_prompt_fraction=args.min_prompt_fraction,
        elbow_rescue=not args.no_elbow_rescue,
        elbow_floor_k=args.elbow_floor_k,
        role_union=args.role_union,
    )

    print(f"\n  selected: {result.n_components} components")

    diag = result.diagnostics
    if args.method == "partition_coverage":
        print(f"\n  partition table (sorted by mass):")
        print(f"  {'fingerprint':<18} {'#paths':>7}  {'mass':>9}  "
              f"{'%kept':>6}  {'status':<13}")
        for row in diag["all_partitions"]:
            print(f"  {row['fingerprint']:<18} {row['n_paths']:>7}  "
                  f"{row['total_mass']:>9.3f}  "
                  f"{row['share_of_kept']*100:>5.1f}%  "
                  f"{row['status']:<13}")

        elbow = diag.get("elbow", {})
        if not elbow.get("enabled"):
            if elbow.get("n_candidates", 0) > 0:
                print(f"\n  elbow rescue disabled; "
                      f"{elbow['n_candidates']} below-threshold partitions ignored")
        elif elbow.get("n_candidates", 0) > 0:
            rescued = elbow.get("rescued_fingerprints", [])
            if rescued:
                print(f"\n  elbow rescued {len(rescued)}/"
                      f"{elbow['n_candidates']} below-threshold partitions: "
                      f"{rescued}")
            else:
                print(f"\n  elbow examined {elbow['n_candidates']} "
                      f"below-threshold partitions; no clear elbow, "
                      f"none rescued")

        print()
        for part in diag["kept_partitions"]:
            print(f"  ── kept: {part['fingerprint']}  "
                  f"({part['share_of_kept']*100:.1f}% of kept mass) ──")
            print(f"     chains: {part['n_chains_total']} total → "
                  f"{part['n_chains_after_occ']} after occurrence filter → "
                  f"{part['n_chains_kept']} kept by elbow")
            if part['kept_chains']:
                for c in part['kept_chains'][:10]:
                    share = c['share'] * 100
                    print(f"     {share:>5.1f}%  {c['n_prompts']:>3} prompts  "
                          f"{c['chain']}")
                if len(part['kept_chains']) > 10:
                    print(f"     ... ({len(part['kept_chains']) - 10} more)")
            print(f"     {part['n_components']} components: {part['components']}")

    elif args.method == "component_flow":
        ranking = diag.get("ranking", [])
        elbow_rank = diag.get("elbow_rank", 0)
        print(f"\n  component ranking (top 40):")
        print(f"  {'rank':>4}  {'cum_score':>10}  {'share':>6}  "
              f"{'cum%':>6}  {'n_pr':>4}  {'status':<13}  component")
        for row in ranking[:40]:
            mark = "►" if row["rank"] == elbow_rank else " "
            print(f"  {mark}{row['rank']:>3}  {row['cum_score']:>10.3f}  "
                  f"{row['share']*100:>5.1f}%  "
                  f"{row['share_cum']*100:>5.1f}%  "
                  f"{row['n_prompts']:>4}  "
                  f"{row['status']:<13}  {row['component']}")
        print(f"\n  elbow at rank {elbow_rank} of {len(ranking)} components")
        n_role = diag.get("n_components_role_only", 0)
        if n_role:
            print(f"  + {n_role} additional components rescued via role-touch")

    print(f"\n  circuit components ({result.n_components}):")
    for name in result.components:
        print(f"    {name}")

    if args.out:
        payload = result.to_json()
        payload["results_dir"] = args.results_dir
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()