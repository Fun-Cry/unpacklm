"""
Merge ioi_analysis JSON results from multiple runs.

Takes corrected default + kqv_weighted from v3,
kqv_aligned from v2, and drops the duplicate k_only_l2 + kqv_l2.

Usage:
    python -m experiments.ioi_analysis.merge \
        --v2 results/ioi_analysis_v2/ioi_analysis.json \
        --v3 results/ioi_analysis_v3/ioi_analysis.json \
        --out results/ioi_analysis_final/ioi_analysis.json
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="results/ioi_analysis_v2/ioi_analysis.json")
    ap.add_argument("--v3", default="results/ioi_analysis_v3/ioi_analysis.json")
    ap.add_argument("--out", default="results/ioi_analysis_final/ioi_analysis.json")
    args = ap.parse_args()

    with open(args.v2) as f:
        v2 = json.load(f)
    with open(args.v3) as f:
        v3 = json.load(f)

    # Correct mapping:
    # v3 ran WITH geva fix → use for "weighted" configs
    # v2 ran WITHOUT geva (= L2 fallback) → use for "l2" configs
    # v2 kqv_aligned already had outproj fix
    keep = {
        "k_only_weighted":      ("v3", "default"),       # K-only + geva (weighted)
        "k_only_l2":    ("v2", "default"),        # K-only + L2 (v2 had no geva = correct L2)
        "kqv_weighted": ("v3", "kqv_weighted"),   # KQV + geva (weighted)
        "kqv_l2":       ("v2", "kqv_weighted"),   # KQV + L2 (v2 had no geva = correct L2)
        "kqv_aligned":  ("v2", "kqv_aligned"),    # KQV + outproj
    }

    sources = {"v2": v2, "v3": v3}

    merged = {
        "n_prompts": v3["n_prompts"],
        "seed": v3["seed"],
        "configs": list(keep.keys()),
    }

    for section in ["token_attribution", "composition"]:
        merged[section] = {}
        for config, (src_name, src_key) in keep.items():
            src = sources[src_name]
            if section in src and src_key in src[section]:
                merged[section][config] = src[section][src_key]
            else:
                print(f"  WARNING: {src_key} not found in {src_name}/{section}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2, default=str)

    print(f"Merged: {list(merged['configs'])}")
    for section in ["token_attribution", "composition"]:
        configs = list(merged.get(section, {}).keys())
        print(f"  {section}: {configs}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()