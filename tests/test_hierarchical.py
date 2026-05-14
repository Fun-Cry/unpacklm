"""
Hierarchical circuit discovery via DLA + iterative rerooting.

Tier 0: DLA → keep heads with |score| >= 50% of top |score|.
Tier 1+: reroot → path scores by upstream head → same filter.

Usage:
    python tests/test_hierarchical.py --device cuda:0 --config kqv_aligned
    python tests/test_hierarchical.py --device cuda:0 --config kqv_aligned --ratio 0.3
"""

import argparse
import re
from collections import defaultdict

import numpy as np
import unpack
from unpack.tasks.ioi import WANG_CIRCUIT_HEADS


def _clean_name(raw):
    name = re.sub(r"\[[KQV]\]", "", raw)
    name = re.sub(r"@\d+$", "", name)
    return name.strip()


def is_head(name):
    return name.startswith("attn_")


def wang_tag(name):
    role = WANG_CIRCUIT_HEADS.get(name, "")
    return f"  [{role}]" if role else ""


def half_max_filter(items, ratio=0.5, min_k=1):
    """Keep items with |score| >= ratio * top |score|.
    
    items: list of (name, score, ...) sorted by |score| descending.
    """
    if not items:
        return []
    top = abs(items[0][1])
    threshold = top * ratio
    kept = [x for x in items if abs(x[1]) >= threshold]
    return kept if len(kept) >= min_k else items[:min_k]


def dla_select(importance, ratio=0.5):
    """Tier 0: heads by DLA, half-max filter."""
    items = [(name, val) for name, val in importance.items()
             if is_head(name)]
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return half_max_filter(items, ratio=ratio)


def path_select(paths, root_names, exclude, ratio=0.5):
    """Tier 1+: group paths by first upstream head, half-max filter."""
    groups = defaultdict(lambda: {"abs": 0.0, "signed": 0.0, "n": 0})

    for p in paths:
        for comp in p.components:
            clean = _clean_name(comp)
            if clean in root_names or clean in exclude:
                continue
            if not is_head(clean):
                continue
            groups[clean]["abs"] += abs(p.score)
            groups[clean]["signed"] += p.score
            groups[clean]["n"] += 1
            break

    items = [(name, d["abs"], d["signed"], d["n"])
             for name, d in groups.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return half_max_filter(items, ratio=ratio)


def discover_single(tracer, text, target, distractor, config="default",
                    max_tiers=6, ratio=0.5, verbose=True):
    tiers = []
    edges = []
    all_found = set()

    kw = dict(target=target, distractor=distractor, config=config)

    # Tier 0: DLA
    result = tracer.trace(text, **kw)
    if verbose:
        print(f"\n  target={target!r}  prob={result.target_prob:.4f}")
        print(f"\n  Tier 0 (DLA, ratio={ratio}):")

    selected = dla_select(result.importance, ratio=ratio)
    tier_0 = set()
    for name, score in selected:
        tier_0.add(name)
        edges.append(("target", name, 0, score))
        if verbose:
            print(f"    {score:>+8.4f}  {name}{wang_tag(name)}")

    tiers.append(tier_0)
    all_found.update(tier_0)
    frontier = set(tier_0)

    # Tier 1+: reroot → path scores
    for tier_idx in range(1, max_tiers):
        if not frontier:
            break

        tier_new = set()
        if verbose:
            print(f"\n  Tier {tier_idx} (reroot at {len(frontier)} heads, ratio={ratio}):")

        for comp in sorted(frontier):
            r = tracer.trace(text, **kw, root=comp)
            selected = path_select(r.paths, root_names={comp},
                                   exclude=all_found, ratio=ratio)

            for name, abs_score, signed_score, n_paths in selected:
                tier_new.add(name)
                edges.append((comp, name, tier_idx, signed_score))
                if verbose:
                    print(f"    {comp}{wang_tag(comp)} ← {name}{wang_tag(name)}  "
                          f"|score|={abs_score:.4f}  n={n_paths}")

        new = tier_new - all_found
        if not new:
            if verbose:
                print(f"    (no new components)")
            break

        tiers.append(new)
        all_found.update(new)
        frontier = new

    return tiers, edges, all_found, result


def print_summary(tiers, edges, all_found):
    heads = sorted(c for c in all_found if c.startswith("attn_"))

    print(f"\n  {'=' * 55}")
    print(f"  Circuit: {len(all_found)} heads in {len(tiers)} tiers")
    print(f"  {'=' * 55}")

    for i, tier in enumerate(tiers):
        parts = [f"{c}{wang_tag(c)}" for c in sorted(tier)]
        print(f"  Tier {i}: {', '.join(parts)}")

    print(f"\n  Heads ({len(heads)}): {heads}")

    wang_found = set(heads) & set(WANG_CIRCUIT_HEADS)
    wang_total = set(WANG_CIRCUIT_HEADS)
    print(f"\n  Wang overlap: {len(wang_found)}/{len(wang_total)} heads")
    if wang_found:
        by_role = defaultdict(list)
        for h in wang_found:
            by_role[WANG_CIRCUIT_HEADS[h]].append(h)
        for role in ["NM", "BackupNM", "NegNM", "S-Inh", "Induction", "DupTok", "PrevTok"]:
            if role in by_role:
                print(f"    [{role}]: {sorted(by_role[role])}")

    wang_missed = wang_total - set(heads)
    if wang_missed:
        print(f"  Wang missed:")
        by_role = defaultdict(list)
        for h in wang_missed:
            by_role[WANG_CIRCUIT_HEADS[h]].append(h)
        for role, hs in sorted(by_role.items()):
            print(f"    [{role}]: {sorted(hs)}")

    print(f"\n  Edges ({len(edges)}):")
    for src, dst, tier, score in edges:
        print(f"    T{tier}  {src}{wang_tag(src)} ← {dst}{wang_tag(dst)}  "
              f"score={score:>+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--config", default="default")
    ap.add_argument("--max-tiers", type=int, default=6)
    ap.add_argument("--ratio", type=float, default=0.5,
                    help="Keep components with |score| >= ratio * top |score|")
    args = ap.parse_args()

    print("Loading GPT-2 small...")
    tracer = unpack.Tracer("gpt2", device=args.device, cache_dir=args.cache_dir)

    text = "When Mary and John went to the store, John gave a drink to"

    print(f"\n{'=' * 60}")
    print(f"  Hierarchical discovery (config={args.config}, ratio={args.ratio})")
    print(f"  '{text}'")
    print(f"{'=' * 60}")

    tiers, edges, all_found, result = discover_single(
        tracer, text, " Mary", " John",
        config=args.config, max_tiers=args.max_tiers, ratio=args.ratio,
    )

    print_summary(tiers, edges, all_found)

    c = unpack.Circuit.from_components(all_found, model_name="gpt2",
                                       config_name=args.config)
    print(f"\n  {c}")


if __name__ == "__main__":
    main()