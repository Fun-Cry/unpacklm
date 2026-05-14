"""Quick check: layer-share of |incoming edge| into each MLP receiver.

Reads either clean or ablated edges per cell, controlled by --side.
Includes a sanity-check that prints one specific edge's clean and
ablated values side by side, so you can confirm the ablation actually
shifted the data before reading the aggregate table.

Usage:
    python check_mlp_incoming_layers.py <results_dir>
    python check_mlp_incoming_layers.py <results_dir> --side ablated
    python check_mlp_incoming_layers.py <results_dir> --condition nm_joint --side ablated
    python check_mlp_incoming_layers.py <results_dir> --condition nm_joint \\
        --sanity-edge attn_9_head_9,mlp_10
"""

import argparse
import os
import sys
from collections import defaultdict

# Make the project root importable.

from experiments.ablation_tracing import load_runs


def src_layer(name: str) -> int:
    """embedding / pos_embedding -> -1; mlp_L -> L; attn_L_head_h -> L."""
    if name in ("embedding", "pos_embedding"):
        return -1
    if name.startswith("mlp_"):
        return int(name.split("_")[1])
    if name.startswith("attn_"):
        return int(name.split("_")[1])
    return -2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--side", choices=("clean", "ablated"), default="clean",
                        help="which edge dict to aggregate (default: clean)")
    parser.add_argument("--condition", default=None,
                        help="restrict to this condition label "
                             "(default: pick first one with cells)")
    parser.add_argument("--p-min", type=float, default=0.1,
                        help="canonical-correct filter (default: 0.1)")
    parser.add_argument("--sanity-edge", default=None,
                        help="src,tgt pair to print as a sanity check, "
                             "e.g. 'attn_9_head_9,mlp_10'. Prints clean & "
                             "ablated values for the first 5 matching cells.")
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    if args.condition:
        runs = [r for r in runs if r.label == args.condition]
    else:
        labels = sorted({r.label for r in runs})
        if not labels:
            print("no cells")
            return
        chosen = labels[0]
        runs = [r for r in runs if r.label == chosen]
        print(f"using condition '{chosen}' (one cell per prompt)")

    runs = [r for r in runs
            if r.clean_target_prob >= args.p_min and r.clean_target_logit > 0]
    print(f"{len(runs)} cells after canonical-correct filter")
    if not runs:
        return

    # ── Sanity check: confirm clean and ablated edges genuinely differ ──
    if args.sanity_edge:
        try:
            sk_src, sk_tgt = args.sanity_edge.split(",")
        except ValueError:
            print(f"--sanity-edge expects 'src,tgt'; got {args.sanity_edge!r}")
            return
        key = (sk_src.strip(), sk_tgt.strip())
        print(f"\nSanity check on edge {key}:")
        for r in runs[:5]:
            ec = (r.clean_edges_mlp   or {}).get(key, None)
            ea = (r.ablated_edges_mlp or {}).get(key, None)
            ec_s = "    None" if ec is None else f"{ec:+.4f}"
            ea_s = "    None" if ea is None else f"{ea:+.4f}"
            print(f"  {r.label:<14} prompt {r.prompt_idx:>3}: "
                  f"clean = {ec_s:>10}    ablated = {ea_s:>10}")
        print()

    # ── Pick which edge dict to read ──
    edge_attr = "clean_edges_mlp" if args.side == "clean" else "ablated_edges_mlp"
    print(f"aggregating: {edge_attr}")

    incoming = defaultdict(float)
    per_target_total = defaultdict(float)
    n_with_edges = 0

    for r in runs:
        edges = getattr(r, edge_attr, None)
        if not edges:
            continue
        n_with_edges += 1
        for (src, tgt), val in edges.items():
            if not tgt.startswith("mlp_"):
                continue
            sl = src_layer(src)
            incoming[(tgt, sl)] += abs(val)
            per_target_total[tgt] += abs(val)

    print(f"({n_with_edges} cells contributed edges)\n")
    if not per_target_total:
        print(f"no MLP-receiver edges found in {edge_attr}")
        return

    targets = sorted(per_target_total.keys(), key=lambda t: int(t.split("_")[1]))
    src_layers = sorted({sl for (_, sl) in incoming.keys()})

    print(f"Layer-share of |incoming flow| into each MLP receiver  ({args.side}):")
    print("(rows: MLP receiver; cols: source layer; -1 = embedding/pos_embedding)")
    print()
    header = f"{'tgt':<10}" + "".join(f"L{sl:>4}" for sl in src_layers) + "    total"
    print(header)
    print("-" * len(header))
    for tgt in targets:
        total = per_target_total[tgt]
        row = f"{tgt:<10}"
        for sl in src_layers:
            v = incoming.get((tgt, sl), 0.0)
            share = v / total if total > 0 else 0.0
            row += f"{'·':>5}" if share == 0.0 else f"{share*100:>4.0f}%"
        row += f"   {total:>8.2f}"
        print(row)

    print()
    print(f"Overall layer-share, summed across all MLP receivers  ({args.side}):")
    overall = defaultdict(float)
    grand_total = 0.0
    for (tgt, sl), v in incoming.items():
        overall[sl] += v
        grand_total += v
    print(f"{'src layer':<12}{'share':>10}{'total':>14}")
    print("-" * 36)
    for sl in src_layers:
        share = overall[sl] / grand_total if grand_total > 0 else 0.0
        print(f"L{sl:<11}{share*100:>9.1f}%   {overall[sl]:>10.2f}")
    print(f"{'GRAND':<12}{'':<10}{grand_total:>14.2f}")


if __name__ == "__main__":
    main()