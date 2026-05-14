"""First-pass summary of a circuit_discovery run.

Reads every prompt_*.json from a results directory and produces:
  - Coverage stats: target_prob distribution, fraction above thresholds.
  - Position-conditioned signed component rankings:
      * all paths
      * src_pos == IO position
      * src_pos == S2 position
      * src_pos != IO and != S2
  - "any step touches IO/S2" rankings (the existing membership-lens
    semantics) for comparison with the stricter src_pos filter.

Hypothesis being tested: the bulk of IOI-relevant signal sits in paths
whose credit lands at the IO token. If true, the IO-restricted ranking
should be markedly sharper than the unrestricted one and the
"neither" bucket should look like noise. If false, restricting by
position doesn't help.

Sign is recovered from `ranked_paths[*].score`, not from the
pre-computed `ranked_components` (which dropped the sign via abs()).

Usage:
    python -m experiments.circuit_discovery.summarize <results_dir>
    python -m experiments.circuit_discovery.summarize <results_dir> \\
        --p-min 0.3 --top-k 20
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


from experiments.circuit_discovery.utils import (
    chain_components, chain_positions,
)


# ──────────────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────────────
def load_run(results_dir):
    cfg_path = os.path.join(results_dir, "run_config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}

    prompt_paths = sorted(glob.glob(os.path.join(results_dir, "prompt_*.json")))
    prompts = []
    for path in prompt_paths:
        with open(path) as f:
            prompts.append(json.load(f))
    return cfg, prompts


# ──────────────────────────────────────────────────────────────────────
# Aggregation under a path-level filter
# ──────────────────────────────────────────────────────────────────────
def aggregate_signed(prompts, path_filter=None):
    """Sum signed path scores per component over all prompts.

    `path_filter` is callable(prompt_dict, chain_list, src_pos) -> bool;
    None means keep every path.

    Returns:
        signed:  dict name -> signed sum of path.score (over surviving paths)
        n_paths: dict name -> count of surviving paths the component appears in
        n_kept:  total number of paths that survived the filter
    """
    signed = defaultdict(float)
    n_paths = defaultdict(int)
    n_kept = 0
    for p in prompts:
        for path in p["ranked_paths"]:
            chain = path["chain"]
            src_pos = path["src_pos"]
            if path_filter is not None and not path_filter(p, chain, src_pos):
                continue
            n_kept += 1
            score = float(path["score"])
            for name in chain_components(chain):
                if name in ("embedding", "pos_embedding"):
                    continue
                signed[name] += score
                n_paths[name] += 1
    return dict(signed), dict(n_paths), n_kept


# ──────────────────────────────────────────────────────────────────────
# Filter factories
# ──────────────────────────────────────────────────────────────────────
def src_pos_in(role_keys):
    """Keep paths whose src_pos matches one of these metadata role keys."""
    def _f(prompt, chain, src_pos):
        meta = prompt.get("metadata", {})
        targets = {meta[k] for k in role_keys if meta.get(k) is not None}
        return src_pos in targets
    return _f


def src_pos_not_in(role_keys):
    """Keep paths whose src_pos is NONE of these role positions."""
    def _f(prompt, chain, src_pos):
        meta = prompt.get("metadata", {})
        targets = {meta[k] for k in role_keys if meta.get(k) is not None}
        return src_pos not in targets
    return _f


def chain_touches(role_keys):
    """Keep paths where any step's @position matches one of these roles."""
    def _f(prompt, chain, src_pos):
        meta = prompt.get("metadata", {})
        targets = {meta[k] for k in role_keys if meta.get(k) is not None}
        if not targets:
            return False
        return any(p in targets for p in chain_positions(chain))
    return _f


# ──────────────────────────────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────────────────────────────
def print_ranking(title, signed, n_paths, n_kept, k=20):
    print()
    print(f"── {title}  (paths kept: {n_kept})")
    if not signed:
        print("  (empty)")
        return
    items = sorted(signed.items(), key=lambda kv: -abs(kv[1]))[:k]
    print(f"  {'rank':>4}  {'component':<20} {'signed':>10}  {'n_paths':>8}")
    for i, (name, score) in enumerate(items, 1):
        print(f"  {i:>4}  {name:<20} {score:>+10.3f}  {n_paths.get(name, 0):>8}")


# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--p-min", type=float, default=0.3,
                    help="exclude prompts with clean_target_prob below this "
                         "from the ranking aggregation (coverage stats use all)")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    cfg, prompts = load_run(args.results_dir)
    if not prompts:
        print(f"no prompt_*.json found under {args.results_dir}")
        return

    print(f"=== {args.results_dir} ===")
    if cfg:
        print(f"  model: {cfg.get('family')}-{cfg.get('size')}  step: {cfg.get('step')}")
        print(f"  trace: {cfg.get('trace')}")
        print(f"  lens:  {cfg.get('lens')}")

    # Coverage
    probs = np.array([p["clean_target_prob"] for p in prompts])
    print(f"\n  prompts loaded: {len(prompts)}")
    print(f"  target_prob:  mean={probs.mean():.3f}  median={np.median(probs):.3f}"
          f"  p25={np.percentile(probs, 25):.3f}  p75={np.percentile(probs, 75):.3f}")
    for tau in [0.1, 0.3, 0.5]:
        n = int((probs >= tau).sum())
        print(f"    >= {tau}:  {n}/{len(prompts)}  ({100*n/len(prompts):.0f}%)")

    kept = [p for p in prompts if p["clean_target_prob"] >= args.p_min]
    print(f"\n  using {len(kept)} prompts (target_prob >= {args.p_min}) for rankings")
    if not kept:
        print("  no prompts passed threshold; nothing to rank")
        return

    # Rankings under different position filters.
    # Two semantics for "touching IO":
    #   src_pos == io_position   — credit LANDED at IO
    #   any chain step at IO     — looser; matches existing membership lens
    filters = [
        ("all paths",                        None),
        ("src_pos == IO",                    src_pos_in(["io_position"])),
        ("src_pos == S2",                    src_pos_in(["s2_position"])),
        ("src_pos == END",                   src_pos_in(["end_position"])),
        ("src_pos NOT in {IO, S2}",          src_pos_not_in(["io_position", "s2_position"])),
        ("any step touches IO  (membership)", chain_touches(["io_position"])),
        ("any step touches S2  (membership)", chain_touches(["s2_position"])),
    ]
    for label, f in filters:
        signed, n_paths, n_kept = aggregate_signed(kept, path_filter=f)
        print_ranking(label, signed, n_paths, n_kept, k=args.top_k)


if __name__ == "__main__":
    main()