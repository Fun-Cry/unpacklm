"""Top-path inspector for circuit_discovery runs.

The summarize.py script aggregates per-component. This one groups
per-chain: it normalizes each path's @position labels to their
semantic roles (IO / S1 / S2 / END / raw int) using the prompt's
metadata, then groups identical normalized chains across prompts
and ranks them by summed signed score.

This tells you which actual computational paths carry the most
credit, not just which components are involved. Two heads can both
appear in the per-component table, but the per-chain table reveals
whether they show up in the *same* path (composition) or in
different paths (parallel routes).

Sign comes from ranked_paths[*].score, never |score|.

Usage:
    # Top paths landing at IO (the canonical name-mover signature)
    python -m experiments.circuit_discovery.top_paths <results_dir> \\
        --p-min 0.3 --top-k 30 --src IO

    # Top paths overall (all src positions)
    python -m experiments.circuit_discovery.top_paths <results_dir> \\
        --p-min 0.3 --top-k 30

    # Top paths landing at S2 (S-inhibition / duplicate-token signature)
    python -m experiments.circuit_discovery.top_paths <results_dir> \\
        --p-min 0.3 --top-k 30 --src S2
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict


from experiments.circuit_discovery.utils import parse_step


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
# Role normalization
# ──────────────────────────────────────────────────────────────────────
# Order matters only for collision-handling; IOI prompts won't have
# colliding positions in practice. Roles checked in this order; first
# match wins.
_ROLE_KEYS = [
    ("io_position",  "IO"),
    ("s1_position",  "S1"),
    ("s2_position",  "S2"),
    ("end_position", "END"),
]


def role_table(prompt):
    """Build int(token_pos) -> role label for one prompt's metadata."""
    meta = prompt.get("metadata", {})
    table = {}
    for key, label in _ROLE_KEYS:
        v = meta.get(key)
        if v is not None and int(v) not in table:
            table[int(v)] = label
    return table


def normalize_chain(chain, role_map):
    """Rewrite each step's @position as @ROLE when the position is
    a recognized role; otherwise keep the raw integer."""
    out = []
    for step in chain:
        name, pos = parse_step(step)
        label = role_map.get(int(pos), str(pos))
        out.append(f"{name}@{label}")
    return tuple(out)


# ──────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────
def aggregate_paths(prompts, src_role=None):
    """Group paths by normalized chain across prompts.

    src_role:
      None       — keep all paths
      "IO"|"S1"|"S2"|"END" — keep paths whose terminal step lands at this role
                             (i.e. src_pos == that role's position in the prompt)

    Returns: list of dicts sorted by |sum_score|, each with
        chain (tuple of step strings), sum_score, n_paths, n_prompts.
    """
    bucket_score   = defaultdict(float)
    bucket_count   = defaultdict(int)
    bucket_prompts = defaultdict(set)

    for p_idx, p in enumerate(prompts):
        rmap = role_table(p)
        for path in p["ranked_paths"]:
            norm = normalize_chain(path["chain"], rmap)
            if src_role is not None:
                terminal = norm[-1].rsplit("@", 1)[-1]
                if terminal != src_role:
                    continue
            bucket_score[norm]   += float(path["score"])
            bucket_count[norm]   += 1
            bucket_prompts[norm].add(p_idx)

    rows = [{
        "chain":     chain,
        "sum_score": score,
        "n_paths":   bucket_count[chain],
        "n_prompts": len(bucket_prompts[chain]),
    } for chain, score in bucket_score.items()]
    rows.sort(key=lambda r: -abs(r["sum_score"]))
    return rows


def fmt_chain(chain, max_len=90):
    s = " → ".join(chain)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--p-min", type=float, default=0.3)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--src", default=None,
                    help="terminal-role filter (IO|S1|S2|END). "
                         "Default: all paths.")
    ap.add_argument("--max-chain-len", type=int, default=110,
                    help="truncate displayed chain to this many characters")
    args = ap.parse_args()

    cfg, prompts = load_run(args.results_dir)
    if not prompts:
        print(f"no prompt_*.json under {args.results_dir}")
        return

    print(f"=== {args.results_dir} ===")
    if cfg:
        print(f"  model: {cfg.get('family')}-{cfg.get('size')}  "
              f"step: {cfg.get('step')}")

    kept = [p for p in prompts if p["clean_target_prob"] >= args.p_min]
    print(f"  prompts kept: {len(kept)}/{len(prompts)} "
          f"(target_prob >= {args.p_min})")
    if not kept:
        return

    rows = aggregate_paths(kept, src_role=args.src)
    label = f"terminal=={args.src}" if args.src else "all paths"
    print(f"\n  unique normalized chains: {len(rows)}")
    print(f"\n── top {args.top_k} chains ({label}) ──")
    print(f"  {'rank':>4}  {'sum':>9}  {'n_paths':>8}  {'n_prompts':>9}  chain")
    print(f"  {'-'*4}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*40}")
    for i, r in enumerate(rows[:args.top_k], 1):
        chain_str = fmt_chain(r["chain"], max_len=args.max_chain_len)
        print(f"  {i:>4}  {r['sum_score']:>+9.3f}  {r['n_paths']:>8}  "
              f"{r['n_prompts']:>9}  {chain_str}")


if __name__ == "__main__":
    main()
