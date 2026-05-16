"""Circuit selection via partition coverage.

Paths are fingerprinted by (role-set, branch-profile). Selection keeps
partitions above a mass threshold (with elbow rescue for borderline ones),
then within each partition applies occurrence filtering and elbow on
cumulative score to pick representative chains. The circuit is the
union of components across kept chains.
"""

import re
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .ioi_utils import (
    ROLE_KEYS, DEFAULT_EXCLUDE, parse_step, chain_components,
    chain_branch_profile, strip_branch_tag, role_for_pos,
)

Fingerprint = Tuple[FrozenSet[str], Tuple[str, ...]]

SKIP_ROLE_SETS = frozenset([
    frozenset(),
    frozenset(["END"]),
])


def _elbow_index(shares):
    """Maximum-perpendicular-distance elbow on cumulative curve."""
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


def _chain_fingerprint(metadata, chain, role_keys=ROLE_KEYS):
    """(role-set, branch-profile) for a chain."""
    roles = set()
    for step in chain:
        _, pos = parse_step(step)
        role = role_for_pos(metadata, pos, role_keys)
        if role is not None:
            roles.add(role)
    return (frozenset(roles), chain_branch_profile(chain))


def _normalized_chain(chain, role_map):
    """Replace @positions with role labels for cross-prompt grouping."""
    out = []
    for step in chain:
        name, pos = parse_step(step)
        label = role_map.get(int(pos), str(pos))
        out.append(f"{name}@{label}")
    return tuple(out)


def _build_role_map(metadata, role_keys=ROLE_KEYS):
    """int(pos) → role_label for this prompt."""
    role_map = {}
    for field, label in role_keys:
        v = metadata.get(field)
        if v is not None and int(v) not in role_map:
            role_map[int(v)] = label
    return role_map


def _select_within_partition(paths, *, min_prompts_floor, elbow_floor_k=1):
    """Within-partition: occurrence filter → elbow → union components."""
    if not paths:
        return [], {}

    chain_buckets = {}
    for p in paths:
        norm = p["normalized_chain"]
        bucket = chain_buckets.setdefault(norm, {
            "chain": norm, "abs_score": 0.0, "n_paths": 0,
            "prompts": set(), "components": tuple(),
        })
        bucket["abs_score"] += abs(p["score"])
        bucket["n_paths"] += 1
        bucket["prompts"].add(p["prompt_idx"])
        if not bucket["components"]:
            bucket["components"] = p["components"]

    chain_list = list(chain_buckets.values())
    n_total = len(chain_list)

    after_occ = [c for c in chain_list
                 if len(c["prompts"]) >= min_prompts_floor]
    if not after_occ:
        return [], {"n_chains_total": n_total, "n_chains_kept": 0}

    after_occ.sort(key=lambda c: -c["abs_score"])
    total_score = sum(c["abs_score"] for c in after_occ)
    shares = [c["abs_score"] / total_score for c in after_occ] if total_score > 0 else []

    elbow_rank = _elbow_index(shares) if shares else 0
    elbow_rank = max(elbow_rank, min(elbow_floor_k, len(after_occ)))
    kept = after_occ[:elbow_rank]

    union = set()
    for c in kept:
        for comp in c["components"]:
            union.add(comp)

    info = {
        "n_chains_total": n_total,
        "n_chains_after_occ": len(after_occ),
        "n_chains_kept": len(kept),
        "kept_chains": [{
            "chain": " → ".join(c["chain"]),
            "abs_score": c["abs_score"],
            "n_prompts": len(c["prompts"]),
            "components": sorted(c["components"]),
        } for c in kept[:10]],
    }
    return sorted(union), info


def select(prompt_results, *,
           role_keys=ROLE_KEYS,
           exclude=DEFAULT_EXCLUDE,
           partition_threshold=0.05,
           min_prompt_fraction=0.25,
           elbow_rescue=True,
           elbow_floor_k=1,
           verbose=True):
    """Run partition coverage selection.

    Args:
        prompt_results: list of per-prompt dicts, each with 'ranked_paths',
            'metadata', 'prompt' fields.
        role_keys: list of (metadata_field, role_label) tuples.
        exclude: component names to exclude.
        partition_threshold: min share of kept mass for a partition.
        min_prompt_fraction: min fraction of prompts a chain must appear in.
        elbow_rescue: rescue below-threshold partitions via elbow.
        elbow_floor_k: min chains to keep per partition.

    Returns: (components: list[str], diagnostics: dict)
    """
    n_prompts = len(prompt_results)
    min_prompts_floor = max(1, int(round(min_prompt_fraction * n_prompts)))
    exclude_set = set(exclude)

    # Collect paths grouped by fingerprint
    by_fp = defaultdict(list)
    for pi, prompt in enumerate(prompt_results):
        meta = prompt.get("metadata", {})
        role_map = _build_role_map(meta, role_keys)

        for path in prompt.get("ranked_paths", []):
            chain = path["chain"]
            fp = _chain_fingerprint(meta, chain, role_keys)
            comps = tuple(strip_branch_tag(c)
                          for c in chain_components(chain)
                          if strip_branch_tag(c) not in exclude_set)
            norm = _normalized_chain(chain, role_map)
            by_fp[fp].append({
                "prompt_idx": pi,
                "score": float(path["score"]),
                "normalized_chain": norm,
                "src_pos": path.get("src_pos"),
                "components": comps,
            })

    # Mass per partition
    fp_total = {fp: sum(abs(r["score"]) for r in paths)
                for fp, paths in by_fp.items()}
    kept_fps = [fp for fp in by_fp if fp[0] not in SKIP_ROLE_SETS]
    kept_total = sum(fp_total[fp] for fp in kept_fps)
    fp_share = {fp: (fp_total[fp] / kept_total if kept_total > 0 else 0)
                for fp in kept_fps}

    # Above threshold
    above = sorted([fp for fp in kept_fps if fp_share[fp] >= partition_threshold],
                   key=lambda fp: -fp_share[fp])

    # Elbow rescue for below-threshold
    below = sorted([fp for fp in kept_fps if fp_share[fp] < partition_threshold],
                   key=lambda fp: -fp_share[fp])
    rescued = []
    if elbow_rescue and below:
        below_shares = [fp_share[fp] for fp in below]
        k = _elbow_index(below_shares)
        rescued = below[:k]

    all_kept = above + rescued

    # Per-partition selection
    union = set()
    partition_info = []
    for fp in all_kept:
        roles_str = "{" + ",".join(sorted(fp[0])) + "}" if fp[0] else "{}"
        prof_str = "(" + ",".join(fp[1]) + ")" if fp[1] else "()"
        fp_label = f"{roles_str}|{prof_str}"

        comps, info = _select_within_partition(
            by_fp[fp],
            min_prompts_floor=min_prompts_floor,
            elbow_floor_k=elbow_floor_k,
        )
        union.update(comps)

        status = "kept" if fp in above else "rescued"
        partition_info.append({
            "fingerprint": fp_label,
            "status": status,
            "share": fp_share[fp],
            "n_components": len(comps),
            "components": comps,
            **info,
        })

    if verbose:
        print(f"\n  Partitions:")
        for p in partition_info:
            print(f"    {p['fingerprint']:<25s} {p['share']*100:>5.1f}%  "
                  f"{p['status']:<8s} {p['n_chains_kept']:>3} chains → "
                  f"{p['n_components']} components")
        print(f"\n  Total: {len(union)} components")

    return sorted(union), {"partitions": partition_info}
