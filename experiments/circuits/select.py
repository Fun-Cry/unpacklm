"""Step 2: Select — aggregate component scores, pick circuit via elbow.

Reads per-prompt JSONs from discover step. Uses `component_flow`
(dense flow sweep) for ranking. Elbow detection on cumulative share
curve picks the circuit boundary.

Usage:
    python -m experiments.circuits.select \
        results/circuits/gpt2_default \
        --p-min 0.3 --out results/circuits/gpt2_default/circuit.json
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple


DEFAULT_EXCLUDE = {"embedding", "pos_embedding"}


def load_prompts(results_dir: str) -> Tuple[dict, List[dict]]:
    """Load run_config and all per-prompt JSONs from a results dir."""
    cfg_path = os.path.join(results_dir, "run_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    prompts = []
    for path in sorted(glob.glob(os.path.join(results_dir, "prompt_*.json"))):
        with open(path) as f:
            prompts.append(json.load(f))
    return cfg, prompts


def filter_correct(prompts, p_min=0.3):
    """Keep prompts where clean P(target) >= p_min."""
    return [p for p in prompts
            if p.get("clean_target_prob", 0) >= p_min]


def elbow_index(shares: List[float]) -> int:
    """Max-perpendicular-distance elbow on concave cumulative curve."""
    if not shares:
        return 0
    n = len(shares)
    cum = [0.0]
    s = 0.0
    for v in shares:
        s += v
        cum.append(s)
    if cum[-1] <= 0:
        return 0
    total = float(cum[-1])
    best_i, best_d = 0, -1.0
    for i in range(1, n + 1):
        d = abs(total * i - n * cum[i])
        if d > best_d:
            best_d = d
            best_i = i
    return best_i


def select_circuit(prompts: List[dict],
                   exclude: Set[str] = DEFAULT_EXCLUDE,
                   role_union: bool = True) -> dict:
    """Aggregate component_flow, pick circuit via elbow.

    Returns dict with 'circuit' (list of names), 'ranking' (full table),
    'elbow_rank', and diagnostics.
    """
    # Aggregate |flow| across prompts
    cum_score: Dict[str, float] = defaultdict(float)
    n_prompts_seen: Dict[str, int] = defaultdict(int)

    for p in prompts:
        flow = p.get("component_flow", {})
        for name, score in flow.items():
            if name in exclude:
                continue
            cum_score[name] += abs(score)
            n_prompts_seen[name] += 1

    # Sort descending by cum_score
    ranking = sorted(cum_score.items(), key=lambda x: -x[1])
    total = sum(v for _, v in ranking)
    if total <= 0:
        return {"circuit": [], "ranking": [], "elbow_rank": 0}

    # Build ranking table with shares
    shares = [v / total for _, v in ranking]
    cum_share = 0.0
    rows = []
    for i, ((name, score), share) in enumerate(zip(ranking, shares)):
        cum_share += share
        rows.append({
            "rank": i + 1,
            "component": name,
            "cum_score": score,
            "share": share,
            "share_cum": cum_share,
            "n_prompts": n_prompts_seen[name],
        })

    # Elbow detection
    eidx = elbow_index(shares)

    # Components above elbow
    circuit = set()
    for row in rows[:eidx]:
        circuit.add(row["component"])

    # Optional: union with role-touching components
    n_role_rescue = 0
    if role_union:
        for p in prompts:
            md = p.get("metadata", {})
            # Role positions for THIS prompt
            prompt_roles = set()
            for key in ["io_position", "s1_position", "s2_position",
                        "end_position"]:
                pos = md.get(key)
                if pos is not None:
                    prompt_roles.add(pos)

            flow = p.get("component_flow", {})
            for name, score in flow.items():
                if name in exclude or name in circuit:
                    continue
                # Check if this component appears in a path
                # that terminates at a role position in THIS prompt
                for rp in p.get("ranked_paths", []):
                    chain_comps = []
                    for step in rp.get("chain", []):
                        # Strip mode tags and @pos: "attn_9_head_9[K]@14" → "attn_9_head_9"
                        clean = re.sub(r"\[.*?\]", "", step)
                        clean = re.sub(r"@-?\d+$", "", clean).strip()
                        chain_comps.append(clean)
                    if name in chain_comps:
                        src = rp.get("src_pos")
                        if src in prompt_roles:
                            circuit.add(name)
                            n_role_rescue += 1
                            break

    circuit_list = sorted(circuit)

    return {
        "circuit": circuit_list,
        "n_components": len(circuit_list),
        "elbow_rank": eidx,
        "n_role_rescued": n_role_rescue,
        "ranking": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--p-min", type=float, default=0.3)
    ap.add_argument("--no-role-union", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg, all_prompts = load_prompts(args.results_dir)
    prompts = filter_correct(all_prompts, args.p_min)

    print(f"=== {args.results_dir} ===")
    print(f"  config:  {cfg.get('config_preset', '?')}")
    print(f"  prompts: {len(prompts)}/{len(all_prompts)} "
          f"(P(target) >= {args.p_min})")

    result = select_circuit(prompts, role_union=not args.no_role_union)

    print(f"  elbow at rank {result['elbow_rank']} of "
          f"{len(result['ranking'])} components")
    if result['n_role_rescued']:
        print(f"  + {result['n_role_rescued']} via role-union rescue")
    print(f"\n  circuit ({result['n_components']} components):")
    for name in result["circuit"]:
        print(f"    {name}")

    # Print top-20 ranking
    print(f"\n  ranking (top 20):")
    print(f"  {'rank':>4}  {'score':>8}  {'share':>6}  {'cum%':>6}  component")
    for row in result["ranking"][:20]:
        mark = ">" if row["rank"] == result["elbow_rank"] else " "
        print(f"  {mark}{row['rank']:>3}  {row['cum_score']:>8.1f}  "
              f"{row['share']*100:>5.1f}%  {row['share_cum']*100:>5.1f}%  "
              f"{row['component']}")

    out = args.out or os.path.join(args.results_dir, "circuit.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  saved: {out}")


if __name__ == "__main__":
    main()
