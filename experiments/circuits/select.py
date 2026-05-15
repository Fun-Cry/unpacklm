"""Step 2: Select — partition-coverage selection (Wang-faithful).

Groups paths by fingerprint (role-set × branch-profile), within each
kept partition filters by occurrence and elbows on cumulative score.

Defaults: partition_threshold=0 (keep all), min_prompt_fraction=0.15.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, FrozenSet, List, Set, Tuple

from experiments.circuits.ioi_utils import parse_step

IOI_ROLE_KEYS = [
    ("io_position", "IO"),
    ("s1_position", "S1"),
    ("s2_position", "S2"),
    ("end_position", "END"),
]

DEFAULT_EXCLUDE = {"embedding", "pos_embedding"}
SKIP_ROLE_SETS = {frozenset(), frozenset(["END"])}

_BRANCH_RE = re.compile(r"\[([KQV])\]")


def _strip_tag(name):
    return _BRANCH_RE.sub("", name)


def _branch_profile(chain) -> Tuple[str, ...]:
    tags = []
    for step in chain[:-1]:
        m = _BRANCH_RE.search(step)
        if m:
            tags.append(m.group(1))
    return tuple(tags)


def _role_for_pos(metadata, pos, role_keys):
    for field, label in role_keys:
        v = metadata.get(field)
        if v is not None and int(v) == int(pos):
            return label
    return None


def _fingerprint(chain, metadata, role_keys):
    roles = set()
    for step in chain:
        _, pos = parse_step(step)
        role = _role_for_pos(metadata, pos, role_keys)
        if role:
            roles.add(role)
    return (frozenset(roles), _branch_profile(chain))


def _fmt_fp(fp):
    roles, profile = fp
    r = "{}" if not roles else "{" + ",".join(sorted(roles)) + "}"
    p = "()" if not profile else "(" + ",".join(profile) + ")"
    return f"{r}|{p}"


def _normalized_chain(chain, role_map):
    out = []
    for step in chain:
        name, pos = parse_step(step)
        label = role_map.get(int(pos), str(pos))
        out.append(f"{name}@{label}")
    return tuple(out)


def _elbow_index(shares):
    if len(shares) < 3:
        return len(shares)
    cum = []
    s = 0.0
    for v in shares:
        s += v
        cum.append(s)
    n = len(cum)
    xN, yN = float(n), cum[-1]
    denom = (xN ** 2 + yN ** 2) ** 0.5
    if denom == 0:
        return 0
    best_d, best_k = -1.0, 0
    for i, c in enumerate(cum, start=1):
        d = abs(yN * i - xN * c) / denom
        if d > best_d:
            best_d, best_k = d, i
    if best_d < 0.02:
        return 0
    return best_k


def load_prompts(results_dir):
    with open(os.path.join(results_dir, "run_config.json")) as f:
        cfg = json.load(f)
    prompts = []
    for path in sorted(glob.glob(os.path.join(results_dir, "prompt_*.json"))):
        with open(path) as f:
            prompts.append(json.load(f))
    return cfg, prompts


def filter_correct(prompts, p_min=0.3):
    return [p for p in prompts if p.get("clean_target_prob", 0) >= p_min]


def select_circuit(prompts, role_keys=IOI_ROLE_KEYS,
                   exclude=DEFAULT_EXCLUDE,
                   partition_threshold=0.0,
                   min_prompt_fraction=0.15,
                   elbow_floor_k=1):

    n_prompts = len(prompts)
    min_prompts_floor = max(1, int(round(min_prompt_fraction * n_prompts)))

    # Aggregate paths by fingerprint
    by_fp = defaultdict(list)
    for prompt in prompts:
        meta = prompt.get("metadata", {})
        role_map = {}
        for field, label in role_keys:
            v = meta.get(field)
            if v is not None and int(v) not in role_map:
                role_map[int(v)] = label

        for path in prompt.get("ranked_paths", []):
            chain = path["chain"]
            fp = _fingerprint(chain, meta, role_keys)
            comps = tuple(_strip_tag(parse_step(s)[0]) for s in chain
                          if _strip_tag(parse_step(s)[0]) not in exclude)
            norm = _normalized_chain(chain, role_map)
            by_fp[fp].append({
                "prompt": prompt["prompt"],
                "score": float(path["score"]),
                "normalized_chain": norm,
                "components": comps,
            })

    fp_total = {fp: sum(abs(r["score"]) for r in paths)
                for fp, paths in by_fp.items()}
    kept_fps = [fp for fp in by_fp if fp[0] not in SKIP_ROLE_SETS]
    kept_total = sum(fp_total[fp] for fp in kept_fps)
    fp_share = {fp: (fp_total[fp] / kept_total if kept_total > 0 else 0)
                for fp in kept_fps}

    above = sorted([fp for fp in kept_fps if fp_share[fp] >= partition_threshold],
                   key=lambda fp: -fp_share[fp])

    circuit = set()
    partition_info = []

    for fp in above:
        paths = by_fp[fp]
        buckets = {}
        for p in paths:
            norm = p["normalized_chain"]
            b = buckets.setdefault(norm, {
                "chain": norm, "abs_score": 0.0, "n_paths": 0,
                "prompts": set(), "components": ()})
            b["abs_score"] += abs(p["score"])
            b["n_paths"] += 1
            b["prompts"].add(p["prompt"])
            if not b["components"]:
                b["components"] = p["components"]

        chain_list = list(buckets.values())
        n_total = len(chain_list)
        after_occ = [c for c in chain_list
                     if len(c["prompts"]) >= min_prompts_floor]

        if not after_occ:
            partition_info.append({
                "fingerprint": _fmt_fp(fp), "share": fp_share[fp],
                "n_chains_total": n_total, "n_chains_after_occ": 0,
                "n_chains_kept": 0, "components": []})
            continue

        after_occ.sort(key=lambda c: -c["abs_score"])
        total_score = sum(c["abs_score"] for c in after_occ)
        shares = [c["abs_score"] / total_score for c in after_occ] if total_score > 0 else []
        eidx = _elbow_index(shares) if shares else 0
        eidx = max(eidx, min(elbow_floor_k, len(after_occ)))
        kept = after_occ[:eidx]

        for c in kept:
            for comp in c["components"]:
                circuit.add(comp)

        partition_info.append({
            "fingerprint": _fmt_fp(fp),
            "share": fp_share[fp],
            "n_chains_total": n_total,
            "n_chains_after_occ": len(after_occ),
            "n_chains_kept": len(kept),
            "components": sorted(set().union(*(set(c["components"]) for c in kept))),
            "kept_chains": [{
                "chain": " → ".join(c["chain"]),
                "abs_score": c["abs_score"],
                "share": c["abs_score"] / total_score if total_score > 0 else 0,
                "n_prompts": len(c["prompts"]),
            } for c in kept[:10]],
        })

    circuit_list = sorted(circuit)
    return {
        "circuit": circuit_list,
        "n_components": len(circuit_list),
        "method": "partition_coverage",
        "params": {
            "partition_threshold": partition_threshold,
            "min_prompt_fraction": min_prompt_fraction,
            "min_prompts_floor": min_prompts_floor,
        },
        "kept_partitions": partition_info,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--p-min", type=float, default=0.3)
    ap.add_argument("--partition-threshold", type=float, default=0.0)
    ap.add_argument("--min-prompt-fraction", type=float, default=0.15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg, all_prompts = load_prompts(args.results_dir)
    prompts = filter_correct(all_prompts, args.p_min)

    print(f"=== {args.results_dir} ===")
    print(f"  config:  {cfg.get('config_preset', '?')}")
    print(f"  prompts: {len(prompts)}/{len(all_prompts)} "
          f"(P(target) >= {args.p_min})")

    result = select_circuit(
        prompts,
        partition_threshold=args.partition_threshold,
        min_prompt_fraction=args.min_prompt_fraction,
    )

    for part in result["kept_partitions"]:
        if part["n_chains_kept"] == 0:
            continue
        print(f"\n  ── {part['fingerprint']} ({part['share']*100:.1f}%) "
              f"{part['n_chains_total']} → "
              f"{part['n_chains_after_occ']} → "
              f"{part['n_chains_kept']} ──")
        for c in part.get("kept_chains", [])[:5]:
            print(f"     {c['share']*100:>5.1f}%  {c['n_prompts']:>3} prompts  {c['chain']}")
        print(f"     components: {part['components']}")

    print(f"\n  circuit ({result['n_components']} components):")
    for name in result["circuit"]:
        print(f"    {name}")

    out = args.out or os.path.join(args.results_dir, "circuit.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  saved: {out}")


if __name__ == "__main__":
    main()