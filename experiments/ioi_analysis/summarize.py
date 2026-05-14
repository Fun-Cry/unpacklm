"""
Read ioi_analysis.json and print summary tables.

Usage:
    python -m experiments.ioi_analysis.summarize
    python -m experiments.ioi_analysis.summarize results/ioi_analysis/ioi_analysis.json
"""

import argparse
import json
import sys
from collections import defaultdict

import numpy as np

from experiments.ioi_analysis.run import (
    COMPOSITION_CLAIMS, WANG_CIRCUIT_HEADS,
    print_token_summary, print_composition_summary,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results/ioi_analysis/ioi_analysis.json")
    args = ap.parse_args()

    with open(args.path) as f:
        data = json.load(f)

    print(f"Loaded {args.path}")
    print(f"  n_prompts={data['n_prompts']}  seed={data['seed']}  configs={data['configs']}")

    if "token_attribution" in data:
        print_token_summary(data["token_attribution"])

    if "composition" in data:
        print_composition_summary(data["composition"])


if __name__ == "__main__":
    main()
