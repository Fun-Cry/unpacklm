"""
Check whether per-prompt attribution values are truly identical across configs.

Usage:
    python -m experiments.ioi_analysis.check_detail results/ioi_analysis/ioi_analysis.json
"""

import json
import sys
import numpy as np


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/ioi_analysis/ioi_analysis.json"
    with open(path) as f:
        data = json.load(f)

    tok = data.get("token_attribution", {})
    configs = list(tok.keys())
    n = len(tok[configs[0]])

    print(f"Configs: {configs}")
    print(f"Prompts: {n}")

    # Compare every pair of configs
    print(f"\n{'config_A':<18s}  {'config_B':<18s}  {'io_max_diff':>10s}  {'s1_max_diff':>10s}  {'s2_max_diff':>10s}  {'identical':>9s}")
    print("-" * 90)

    for i, a in enumerate(configs):
        for b in configs[i+1:]:
            io_diffs = []
            s1_diffs = []
            s2_diffs = []
            for j in range(n):
                io_diffs.append(abs(tok[a][j]["io_attr"] - tok[b][j]["io_attr"]))
                s1_diffs.append(abs(tok[a][j]["s1_attr"] - tok[b][j]["s1_attr"]))
                s2_diffs.append(abs(tok[a][j]["s2_attr"] - tok[b][j]["s2_attr"]))

            io_max = max(io_diffs)
            s1_max = max(s1_diffs)
            s2_max = max(s2_diffs)
            identical = io_max < 1e-6 and s1_max < 1e-6 and s2_max < 1e-6

            print(f"{a:<18s}  {b:<18s}  {io_max:>10.2e}  {s1_max:>10.2e}  {s2_max:>10.2e}  {'YES' if identical else 'NO':>9s}")

    # Show a few per-prompt values for the first 5 prompts
    print(f"\nFirst 5 prompts, io_attr per config:")
    print(f"  {'prompt':>6s}", end="")
    for c in configs:
        print(f"  {c:>14s}", end="")
    print()
    for j in range(min(5, n)):
        print(f"  {j:>6d}", end="")
        for c in configs:
            print(f"  {tok[c][j]['io_attr']:>+14.6f}", end="")
        print()

    # Also check composition: are paths identical across mlp variants?
    comp = data.get("composition", {})
    if len(comp) >= 2:
        print(f"\nComposition: checking upstream scores across configs")
        pairs = [("kqv_weighted", "kqv_l2"), ("kqv_weighted", "kqv_aligned"),
                 ("default", "k_only_l2")]
        for a, b in pairs:
            if a not in comp or b not in comp:
                continue
            recs_a = comp[a]
            recs_b = comp[b]
            if len(recs_a) != len(recs_b):
                print(f"  {a} vs {b}: different record count ({len(recs_a)} vs {len(recs_b)})")
                continue
            diffs = []
            for ra, rb in zip(recs_a, recs_b):
                # Compare top upstream abs_scores
                names_a = {u["name"]: u["abs_score"] for u in ra["upstream"]}
                names_b = {u["name"]: u["abs_score"] for u in rb["upstream"]}
                shared = set(names_a) & set(names_b)
                for name in shared:
                    diffs.append(abs(names_a[name] - names_b[name]))
            if diffs:
                print(f"  {a} vs {b}: max_diff={max(diffs):.2e}  "
                      f"mean_diff={np.mean(diffs):.2e}  "
                      f"identical={max(diffs) < 1e-6}")
            else:
                print(f"  {a} vs {b}: no shared upstream to compare")


if __name__ == "__main__":
    main()
