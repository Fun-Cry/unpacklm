"""
Comprehensive composition analysis.

Outputs:
  1. Role x Role ranking matrices:
     - Unfiltered (all components)
     - No MLP
     - Attn only, layer>=1
     - Per mode (K, Q, V), attn only, layer>=1
  2. Mode breakdown: avg credit per role pair with K/Q/V split
  3. Per-head detail: top-10 feeders with mode breakdown

Usage: python experiments/ioi_analysis/analyze_composition.py results/ioi_analysis_topk100_full/ioi_analysis.json [config]
"""
import json, sys
import numpy as np
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "results/ioi_analysis_topk100_full/ioi_analysis.json"
config = sys.argv[2] if len(sys.argv) > 2 else "kqv_aligned"

with open(path) as f:
    d = json.load(f)

recs = d["composition"].get(config, [])
if not recs:
    print(f"No data for '{config}'. Available: {list(d['composition'].keys())}")
    sys.exit(1)

HEAD_TIERS = {
    "attn_9_head_9": "NM", "attn_9_head_6": "NM", "attn_10_head_0": "NM",
    "attn_10_head_7": "Neg-NM", "attn_11_head_10": "Neg-NM",
    "attn_7_head_3": "S-Inh", "attn_7_head_9": "S-Inh",
    "attn_8_head_6": "S-Inh", "attn_8_head_10": "S-Inh",
    "attn_4_head_11": "Ind", "attn_5_head_5": "Ind",
    "attn_5_head_8": "Ind", "attn_5_head_9": "Ind",
    "attn_6_head_2": "Ind", "attn_6_head_6": "Ind",
    "attn_6_head_9": "Ind",
    "attn_0_head_10": "Dup", "attn_3_head_0": "Dup",
    "attn_0_head_1": "Prev", "attn_2_head_2": "Prev",
    "attn_4_head_3": "Prev",
}

ROLES = ["NM", "S-Inh", "Ind", "Dup", "Prev"]


def get_layer(name):
    parts = name.split("_")
    try:
        return int(parts[1])
    except (IndexError, ValueError):
        return -1


def find_rank(upstream, target_roles, mode=None, attn_only=False, no_mlp=False, min_layer=0):
    """1-indexed rank of first component with tier in target_roles, or None."""
    scored = []
    for u in upstream:
        name = u["name"]
        if attn_only and not name.startswith("attn_"):
            continue
        if no_mlp and name.startswith("mlp_"):
            continue
        if get_layer(name) < min_layer:
            continue
        score = u["modes"].get(mode, 0) if mode else u["abs_score"]
        scored.append((name, u.get("tier"), score))
    scored.sort(key=lambda x: x[2], reverse=True)
    for rank, (_, tier, sc) in enumerate(scored):
        if tier in target_roles and sc > 0:
            return rank + 1
    return None


# Group by downstream role
by_down = defaultdict(list)
for r in recs:
    role = HEAD_TIERS.get(r["root_head"])
    if role:
        by_down[role].append(r)


# ================================================================
# 1. Role x Role ranking matrices
# ================================================================
def print_matrix(title, mode=None, attn_only=False, no_mlp=False, min_layer=0):
    filt = []
    if mode:
        filt.append(f"{mode}-mode")
    if attn_only:
        filt.append("attn only")
    elif no_mlp:
        filt.append("no MLP")
    if min_layer > 0:
        filt.append(f"layer>={min_layer}")
    filt_str = ", ".join(filt) if filt else "all components"

    print(f"\n{'=' * 75}")
    print(f"  {title} ({filt_str})")
    print(f"  Cell: median [mean] (found%)")
    print(f"{'=' * 75}")

    header = f"{'':>8}"
    for up in ROLES:
        header += f"{up:>14}"
    print(header)
    print("-" * (8 + 14 * len(ROLES)))

    for down in ROLES:
        if not by_down[down]:
            continue
        row = f"{down:<8}"
        for up in ROLES:
            ranks = []
            for r in by_down[down]:
                rank = find_rank(
                    r["upstream"], {up},
                    mode=mode, attn_only=attn_only, no_mlp=no_mlp, min_layer=min_layer,
                )
                if rank is not None:
                    ranks.append(rank)
            if ranks:
                n_total = len(by_down[down])
                med = int(np.median(ranks))
                mean = np.mean(ranks)
                found_pct = len(ranks) / n_total * 100
                row += f"  {med:>2} [{mean:>4.1f}] {found_pct:>3.0f}%"
            else:
                row += f"{'--':>14}"
        print(row + f"  (n={len(by_down[down])})")


print(f"Config: {config}, records: {len(recs)}, upstream/rec: {len(recs[0]['upstream'])}\n")

print_matrix("Unfiltered")
print_matrix("No MLP", no_mlp=True)
print_matrix("Attn only, layer>=1", attn_only=True, min_layer=1)
for m in ["K", "Q", "V"]:
    print_matrix(f"{m}-mode ranking", mode=m, attn_only=True, min_layer=1)


# ================================================================
# 2. Mode breakdown
# ================================================================
print(f"\n{'=' * 75}")
print(f"  Mode breakdown: avg credit per rerooting, all role pairs")
print(f"{'=' * 75}")
print(f"{'down':<8} {'up':<8} {'total':>7} {'K':>7} {'Q':>7} {'V':>7} {'K%':>5} {'Q%':>5} {'V%':>5}")
print("-" * 62)

for down in ROLES:
    for up in ROLES:
        totals, ks, qs, vs = [], [], [], []
        for r in by_down[down]:
            t = k = q = v = 0
            for u in r["upstream"]:
                if HEAD_TIERS.get(u["name"]) != up:
                    continue
                t += u["abs_score"]
                k += u["modes"].get("K", 0)
                q += u["modes"].get("Q", 0)
                v += u["modes"].get("V", 0)
            totals.append(t)
            ks.append(k)
            qs.append(q)
            vs.append(v)
        if not totals or np.mean(totals) < 0.01:
            continue
        tm, km, qm, vm = np.mean(totals), np.mean(ks), np.mean(qs), np.mean(vs)
        s = km + qm + vm
        kp = km / s * 100 if s > 0 else 0
        qp = qm / s * 100 if s > 0 else 0
        vp = vm / s * 100 if s > 0 else 0
        print(f"{down:<8} {up:<8} {tm:>7.2f} {km:>7.2f} {qm:>7.2f} {vm:>7.2f} {kp:>4.0f}% {qp:>4.0f}% {vp:>4.0f}%")


# ================================================================
# 3. Per-head detail
# ================================================================
print(f"\n{'=' * 75}")
print(f"  Per-head top-10 feeders (attn, layer>=1)")
print(f"{'=' * 75}")

by_root = defaultdict(list)
for r in recs:
    by_root[r["root_head"]].append(r)

for root_head in sorted(by_root.keys()):
    root_recs = by_root[root_head]
    root_tier = HEAD_TIERS.get(root_head, "?")
    root_pos = root_recs[0]["root_pos_key"]

    print(f"\n  {root_head} [{root_tier}] @ {root_pos} (n={len(root_recs)})")

    comp = defaultdict(lambda: {"total": 0, "count": 0, "K": 0, "Q": 0, "V": 0})
    for r in root_recs:
        for u in r["upstream"]:
            name = u["name"]
            if not name.startswith("attn_") or get_layer(name) < 1:
                continue
            cs = comp[name]
            cs["total"] += u["abs_score"]
            cs["count"] += 1
            for m in ["K", "Q", "V"]:
                cs[m] += u["modes"].get(m, 0)

    ranked = sorted(comp.items(), key=lambda x: x[1]["total"], reverse=True)

    print(f"  {'Component':<22} {'Tier':<6} {'Avg':>7} {'Pres':>5} {'K%':>5} {'Q%':>5} {'V%':>5}")
    for name, cs in ranked[:10]:
        tier = HEAD_TIERS.get(name, "")
        n = len(root_recs)
        avg = cs["total"] / n
        pct = cs["count"] / n * 100
        s = cs["K"] + cs["Q"] + cs["V"]
        kp = cs["K"] / s * 100 if s > 0 else 0
        qp = cs["Q"] / s * 100 if s > 0 else 0
        vp = cs["V"] / s * 100 if s > 0 else 0
        print(f"  {name:<22} {tier:<6} {avg:>7.2f} {pct:>4.0f}% {kp:>4.0f}% {qp:>4.0f}% {vp:>4.0f}%")