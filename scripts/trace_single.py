"""Trace a single sentence — show top paths and token-level attribution.

Minimal CLI: take a model family/size and a sentence, run trace_flow
with K+Q+V decomposition enabled, print the top paths sorted by |%|
and the per-token credit at the prediction position.

Usage examples:
    python -m experiments.circuit_discovery.trace_single \\
        --family gpt2 --size small \\
        --sentence "Mary and John went to the store. John gave the bag to"

    python -m experiments.circuit_discovery.trace_single \\
        --family pythia --size 1.4b --cache-dir /data/s4283341 \\
        --sentence "The Eiffel Tower is located in" \\
        --target " Paris"

    python -m experiments.circuit_discovery.trace_single \\
        --family gpt2 --size small \\
        --sentence "Mary and John went to the store. John gave the bag to" \\
        --target " Mary" --distractor " John" \\
        --top-k 30
"""

import argparse
import os
import sys


import numpy as np
import torch

from experiments.circuit_discovery.utils import load_model
from unpack.compat import trace_flow


def main():
    ap = argparse.ArgumentParser(
        description="Trace a single sentence and show top paths + token credit.")
    ap.add_argument("--family", required=True, choices=["gpt2", "pythia"],
                    help="Model family.")
    ap.add_argument("--size", required=True,
                    help="Model size (e.g. 'small', 'medium' for gpt2; "
                         "'70m', '410m', '1.4b', '2.8b' for pythia).")
    ap.add_argument("--sentence", required=True,
                    help="The input prompt to trace.")
    ap.add_argument("--target", default=None,
                    help="Target token (with leading space if needed). "
                         "Defaults to the model's top prediction.")
    ap.add_argument("--distractor", default=None,
                    help="Optional distractor for logit-difference target "
                         "direction (target - distractor instead of "
                         "target - mean).")
    ap.add_argument("--cache-dir", default=None,
                    help="HuggingFace cache directory.")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available()
                                       else "cpu")
    ap.add_argument("--step", type=int, default=143000,
                    help="Pythia checkpoint step (only for --family pythia).")
    ap.add_argument("--no-q-side", dest="enable_q_side", action="store_false",
                    default=True)
    ap.add_argument("--no-v-side", dest="enable_v_side", action="store_false",
                    default=True)
    ap.add_argument("--branch-weights", default=None,
                    help="Branch weights as 'K=w,Q=w,V=w' (default 1,1,1).")
    ap.add_argument("--beta", type=float, default=0.8,
                    help="SafeDenom β for credit propagation (default 0.8).")
    ap.add_argument("--top-paths-k", type=int, default=2000,
                    help="How many top paths to compute (default 2000).")
    ap.add_argument("--top-k", type=int, default=20,
                    help="How many top paths to print (default 20).")
    ap.add_argument("--path-min-frac", type=float, default=1e-4,
                    help="Drop paths below this fraction of importance "
                         "during recursion (default 1e-4).")
    ap.add_argument("--geomean-min", type=float, default=None,
                    help="If set, OR the raw threshold with a depth-aware "
                         "gate: keep paths whose |score|^(1/(depth+1)) "
                         "is at least this value. Surfaces deep mediator "
                         "paths whose raw score is attenuated by fan-out "
                         "but whose average per-step factor is meaningful. "
                         "Try 0.05 to start.")
    args = ap.parse_args()

    cfg = {
        "model": {
            "family":    args.family,
            "size":      args.size,
            "device":    args.device,
            "cache_dir": args.cache_dir,
            "step":      args.step,
            "deduped":   True,
        },
    }

    branch_weights = None
    if args.branch_weights:
        branch_weights = {}
        for chunk in args.branch_weights.split(","):
            k, v = chunk.split("=")
            branch_weights[k.strip()] = float(v.strip())

    print(f"loading {args.family}/{args.size}")
    model, tokenizer, hook_manager, device = load_model(cfg)

    print(f"\nsentence: {args.sentence!r}")
    if args.target:
        print(f"target:   {args.target!r}")
    if args.distractor:
        print(f"distractor: {args.distractor!r}")

    res = trace_flow(
        model, tokenizer, args.sentence,
        target_token=args.target,
        distractor_token=args.distractor,
        hook_manager=hook_manager,
        beta=args.beta,
        top_paths_k=args.top_paths_k,
        path_min_frac=args.path_min_frac,
        enable_q_side=args.enable_q_side,
        enable_v_side=args.enable_v_side,
        branch_weights=branch_weights,
        geomean_min=args.geomean_min,
    )

    # ── Header ──
    print("\n" + "=" * 70)
    print(f"  target token:   {res['target_token']!r}")
    print(f"  target prob:    {res['target_prob']:.4f}")
    print(f"  target logit:   {res['target_logit_centered']:+.3f} (centered)")
    print("=" * 70)

    # ── Token-level attribution at the prediction position ──
    tokens = res["tokens"]
    credit = np.asarray(res.get("token_attribution", []))
    if credit.size:
        print("\nToken-level attribution (sequential):")
        # Bar width scaled to max |credit|. Positive = right, negative = left.
        max_abs = max(np.abs(credit).max(), 1e-9)
        BAR_WIDTH = 10
        for i, (tok, c) in enumerate(zip(tokens, credit)):
            tok_repr = repr(tok)[:20]
            n_bar = int(round(abs(c) / max_abs * BAR_WIDTH))
            if c >= 0:
                bar = " " * BAR_WIDTH + "│" + "█" * n_bar
            else:
                bar = " " * (BAR_WIDTH - n_bar) + "█" * n_bar + "│"
            print(f"  {i:>3}  {tok_repr:<20}  {c:>+6.2f}%  {bar}")

    # ── Top paths ──
    top_paths = res.get("top_paths", [])
    top_paths_raw = res.get("top_paths_raw", [])

    # Build augmented rows so we can sort two ways without re-tracing.
    # Each row: (path, src_pos, pct, raw, depth, geomean_raw)
    augmented = []
    for (path, src_pos, pct), (_, _, raw) in zip(top_paths, top_paths_raw):
        n_hops = max(0, path.count("→"))
        abs_raw = abs(raw)
        if abs_raw > 0.0:
            sign = 1.0 if raw >= 0.0 else -1.0
            geo = sign * abs_raw ** (1.0 / (n_hops + 1))
        else:
            geo = 0.0
        augmented.append((path, src_pos, pct, raw, n_hops, geo))

    def _print_table(rows, title):
        print(f"\n{title}:")
        print(f"  {'rank':>4}  {'pct':>7}  {'raw':>9}  "
              f"{'d':>2}  {'geomean':>8}  {'src':>15}  path")
        for rank, (path, src_pos, pct, raw, depth, geo) in enumerate(
                rows[:args.top_k], start=1):
            src_tok = tokens[src_pos] if 0 <= src_pos < len(tokens) else "?"
            src_label = f"{src_pos}({src_tok!r})"
            print(f"  {rank:>4}  {pct:>+6.2f}%  {raw:>+9.4f}  "
                  f"{depth:>2}  {geo:>+8.4f}  {src_label:>15}  {path}")

    print(f"\nReturned {len(augmented)} paths.")

    # By raw |pct| — the original ranking, unchanged.
    by_pct = sorted(augmented, key=lambda r: abs(r[2]), reverse=True)
    _print_table(by_pct, f"Top {args.top_k} paths by raw |pct| (existing)")

    # By |geomean| — surfaces deep mediators whose per-step factors
    # are strong even if the path-product is small.
    by_geo = sorted(augmented, key=lambda r: abs(r[5]), reverse=True)
    _print_table(by_geo, f"Top {args.top_k} paths by |geomean| (depth-corrected)")

    # Diagnostic: depth distribution of returned paths.
    from collections import Counter
    depth_hist = Counter(r[4] for r in augmented)
    print(f"\nDepth distribution of returned paths:")
    for d in sorted(depth_hist):
        bar = "█" * min(40, depth_hist[d])
        print(f"  depth {d:>2}:  {depth_hist[d]:>5}  {bar}")

    # ── Suppression diagnostic ──
    suppress = res.get("suppress_ratio", None)
    if suppress is not None:
        print(f"\nsuppression ratio: {suppress:.3f}  "
              f"(0=all positive, 0.5=balanced, 1=all negative)")

    # ── Component flow ──
    # Per-component aggregate flow, independent of path enumeration.
    # If induction / prev-token / etc. heads receive any credit at all,
    # they appear here with non-zero flow regardless of whether their
    # paths survived top_paths_k.
    cflow = res.get("component_flow", {})
    if cflow:
        # Sum over positions to get per-component signed scalar.
        per_comp = {}
        for name, arr in cflow.items():
            if "bias" in name:
                continue
            per_comp[name] = float(np.asarray(arr).sum())

        # Wang's IOI roles, for inline annotation.
        WANG_ROLES = {
            "attn_9_head_6":   "NM",
            "attn_9_head_9":   "NM",
            "attn_10_head_0":  "NM",
            "attn_10_head_10": "BackupNM",
            "attn_11_head_2":  "BackupNM",
            "attn_9_head_0":   "BackupNM",
            "attn_9_head_7":   "BackupNM",
            "attn_10_head_7":  "NegNM",
            "attn_11_head_10": "NegNM",
            "attn_7_head_3":   "S-Inh",
            "attn_7_head_9":   "S-Inh",
            "attn_8_head_6":   "S-Inh",
            "attn_8_head_10":  "S-Inh",
            "attn_5_head_5":   "Induction",
            "attn_5_head_8":   "Induction",
            "attn_5_head_9":   "Induction",
            "attn_6_head_9":   "Induction",
            "attn_0_head_1":   "DupTok",
            "attn_0_head_10":  "DupTok",
            "attn_3_head_0":   "DupTok",
            "attn_2_head_2":   "PrevTok",
            "attn_4_head_11":  "PrevTok",
        }

        # Top-K by |flow|.
        top = sorted(per_comp.items(), key=lambda kv: abs(kv[1]),
                     reverse=True)[:30]
        print(f"\nTop 30 components by |flow|:")
        print(f"  {'rank':>4}  {'flow':>9}  {'role':<10}  component")
        for rank, (name, val) in enumerate(top, start=1):
            role = WANG_ROLES.get(name, "")
            tag = f"[{role}]" if role else ""
            print(f"  {rank:>4}  {val:>+9.4f}  {tag:<10}  {name}")

        # Wang-head report: full status for each canonical head.
        print(f"\nWang canonical heads — flow + rank:")
        rank_by_abs = {n: i + 1 for i, (n, _) in enumerate(
            sorted(per_comp.items(), key=lambda kv: abs(kv[1]),
                   reverse=True))}
        # Group by role for readability.
        from collections import defaultdict
        by_role = defaultdict(list)
        for h, role in WANG_ROLES.items():
            by_role[role].append(h)
        for role in ["NM", "BackupNM", "NegNM", "S-Inh",
                     "Induction", "DupTok", "PrevTok"]:
            for h in by_role[role]:
                v = per_comp.get(h, 0.0)
                r = rank_by_abs.get(h, 9999)
                print(f"  {h:<18} {role:<10} flow={v:>+9.4f}  rank={r:>4}")


if __name__ == "__main__":
    main()