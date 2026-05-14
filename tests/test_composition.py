"""
Quick test: token attribution roles + composition via rerooting.

Usage:
    python tests/test_composition.py --device cuda:0
    python tests/test_composition.py --device cuda:0 --config kqv_aligned
"""

import argparse
from collections import defaultdict
import unpack
from unpack.tasks.ioi import WANG_CIRCUIT_HEADS


# Hand-picked IOI prompt with known token positions
# Tokenization: When| Mary| and| John| went| to| the| store|,| John| gave| a| drink| to
#               0     1     2    3     4     5   6    7      8  9     10    11 12     13
PROMPT = {
    "text": "When Mary and John went to the store, John gave a drink to",
    "target": " Mary",
    "distractor": " John",
    "positions": {
        1: "IO",    # Mary
        3: "S1",    # John (first mention)
        9: "S2",    # John (second mention)
        13: "END",  # to (prediction position)
    },
}

# Wang's role tiers
NAME_MOVERS = ["attn_9_head_9", "attn_9_head_6", "attn_10_head_0"]
S_INHIBITION = ["attn_7_head_3", "attn_7_head_9", "attn_8_head_6", "attn_8_head_10"]
INDUCTION = ["attn_5_head_5", "attn_5_head_8", "attn_5_head_9", "attn_6_head_9"]
DUP_TOKEN = ["attn_0_head_1", "attn_0_head_10", "attn_3_head_0"]
PREV_TOKEN = ["attn_2_head_2", "attn_4_head_11"]

ALL_TIERS = {
    "NM": NAME_MOVERS, "S-Inh": S_INHIBITION,
    "Ind": INDUCTION, "Dup": DUP_TOKEN, "Prev": PREV_TOKEN,
}


def pos_label(pos, positions):
    role = positions.get(pos, "")
    return f"{pos}({role})" if role else f"{pos}"


def print_token_attribution(result, positions, tokens):
    attr = result.token_attribution

    role_credit = defaultdict(float)
    for i, c in enumerate(attr):
        role = positions.get(i, "other")
        role_credit[role] += c

    print(f"  By role:")
    for role in ["IO", "S1", "S2", "END", "other"]:
        if role in role_credit:
            print(f"    {role:<6s} {role_credit[role]:>+7.1f}%")

    print(f"  Per-token (|credit| > 1%):")
    for i, (tok, c) in enumerate(zip(tokens, attr)):
        if abs(c) > 1.0:
            role = positions.get(i, "")
            tag = f"  [{role}]" if role else ""
            print(f"    {i:>3} {tok!r:<15} {c:>+6.1f}%{tag}")


def print_reroot_analysis(result, positions, tokens, label):
    print(f"\n  -- Rerooted at {label} --")

    flow = result.component_flow
    tier_flow = defaultdict(float)
    for name, val in flow.items():
        found = False
        for tier, members in ALL_TIERS.items():
            if name in members:
                tier_flow[tier] += val
                found = True
                break
        if not found:
            if name.startswith("mlp_"):
                tier_flow["MLP"] += val
            elif name.startswith("embedding") or name.startswith("pos_"):
                tier_flow["embed"] += val
            else:
                tier_flow["other_attn"] += val

    print(f"  Flow by tier:")
    for tier in ["NM", "S-Inh", "Ind", "Dup", "Prev", "MLP", "embed", "other_attn"]:
        if tier in tier_flow and abs(tier_flow[tier]) > 0.001:
            print(f"    {tier:<12s} {tier_flow[tier]:>+8.4f}")

    wang_flow = [(name, val) for name, val in flow.items()
                 if name in WANG_CIRCUIT_HEADS]
    wang_flow.sort(key=lambda kv: abs(kv[1]), reverse=True)
    if wang_flow:
        print(f"  Wang circuit heads (by |flow|):")
        for name, val in wang_flow[:10]:
            print(f"    {val:>+8.4f}  {name}  [{WANG_CIRCUIT_HEADS[name]}]")

    pos_paths = defaultdict(list)
    for p in result.paths:
        pos_paths[p.source_pos].append(p)

    print(f"  Paths by source position:")
    for pos in sorted(pos_paths.keys()):
        paths = pos_paths[pos]
        total = sum(p.score for p in paths)
        label_str = pos_label(pos, positions)
        tok = tokens[pos] if 0 <= pos < len(tokens) else "?"
        print(f"    pos {label_str:<10s} tok={tok!r:<10s}  "
              f"n={len(paths):<3d}  total={total:>+7.2f}%")
        paths.sort(key=lambda p: abs(p.score), reverse=True)
        for p in paths[:2]:
            print(f"      {p.score:>+6.2f}%  {p.chain}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--config", default="default",
                    help="Trace config preset (default, kqv_aligned, etc.)")
    args = ap.parse_args()

    print("Loading GPT-2 small...")
    tracer = unpack.Tracer("gpt2", device=args.device, cache_dir=args.cache_dir)

    p = PROMPT
    positions = p["positions"]
    trace_kwargs = dict(target=p["target"], distractor=p["distractor"],
                        config=args.config)

    result = tracer.trace(p["text"], **trace_kwargs)
    tokens = result.tokens
    print(f"\nTokenization ({len(tokens)} tokens):")
    for i, tok in enumerate(tokens):
        role = positions.get(i, "")
        tag = f"  <- {role}" if role else ""
        print(f"  {i:>3} {tok!r}{tag}")

    # -- Test 1: Token attribution --
    print("\n" + "=" * 60)
    print(f"  TEST 1: Token attribution (config={args.config})")
    print("=" * 60)
    print(f"  target={p['target']!r}  prob={result.target_prob:.4f}")
    print_token_attribution(result, positions, tokens)

    # -- Test 2: Reroot at name movers --
    print("\n" + "=" * 60)
    print(f"  TEST 2: Reroot at name movers (config={args.config})")
    print("=" * 60)
    for nm in NAME_MOVERS:
        r = tracer.trace(p["text"], **trace_kwargs, root=nm)
        print_reroot_analysis(r, positions, tokens,
                              f"{nm} [{WANG_CIRCUIT_HEADS[nm]}]")

    # -- Test 3: Reroot at S-inhibition heads --
    print("\n" + "=" * 60)
    print(f"  TEST 3: Reroot at S-inhibition heads (config={args.config})")
    print("=" * 60)
    for sh in S_INHIBITION[:2]:
        r = tracer.trace(p["text"], **trace_kwargs, root=sh)
        print_reroot_analysis(r, positions, tokens,
                              f"{sh} [{WANG_CIRCUIT_HEADS[sh]}]")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()