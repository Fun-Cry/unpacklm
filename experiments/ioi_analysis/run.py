"""
IOI token attribution + composition analysis.

Efficient: one forward pass per (prompt, config), many backward passes.

Usage:
    python -m experiments.ioi_analysis.run --device cuda:0
    python -m experiments.ioi_analysis.run --device cuda:0 --n-prompts 5 --configs default kqv_aligned
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import unpack
from unpack.tasks.ioi import WANG_CIRCUIT_HEADS


# ── Wang's tiers ──

NAME_MOVERS = {"attn_9_head_9", "attn_9_head_6", "attn_10_head_0"}
NEG_NAME_MOVERS = {"attn_10_head_7", "attn_11_head_10"}
S_INHIBITION = {"attn_7_head_3", "attn_7_head_9", "attn_8_head_6", "attn_8_head_10"}
INDUCTION = {"attn_5_head_5", "attn_5_head_8", "attn_5_head_9", "attn_6_head_9"}
DUP_TOKEN = {"attn_0_head_1", "attn_0_head_10", "attn_3_head_0"}
PREV_TOKEN = {"attn_2_head_2", "attn_4_head_11"}

TIER_MAP = {
    "NM": NAME_MOVERS, "NegNM": NEG_NAME_MOVERS,
    "S-Inh": S_INHIBITION, "Ind": INDUCTION,
    "Dup": DUP_TOKEN, "Prev": PREV_TOKEN,
}

ALL_CONFIGS = ["default", "k_only_l2", "k_only_aligned", "kqv_weighted", "kqv_l2", "kqv_aligned"]

MODE_RE = re.compile(r"\[([KQV])\]")

# ── Wang Sec 3.2-3.3 composition claims ──
COMPOSITION_CLAIMS = [
    {
        "name": "NM_from_SInh",
        "description": "S-Inh modulates NM attention via K/Q (Wang Sec 3.2)",
        "root_heads": list(NAME_MOVERS),
        "root_pos": "END",
        "expected_tiers": {"S-Inh"},
        "expected_modes": {"K", "Q"},
    },
    {
        "name": "SInh_from_DupInd",
        "description": "DupTok/Induction write position signal into S-Inh values at S2 (Wang Sec 3.3)",
        "root_heads": list(S_INHIBITION),
        "root_pos": "END",
        "expected_tiers": {"Ind", "Dup"},
        "expected_modes": {"V"},
    },
    {
        "name": "Ind_from_Prev",
        "description": "PrevTok feeds Induction keys at S1+1 (Wang Sec 3.3)",
        "root_heads": list(INDUCTION),
        "root_pos": "S2",
        "expected_tiers": {"Prev", "Dup"},
        "expected_modes": {"K"},
    },
]


# ── IOI prompt loading ──

def load_ioi_prompts(tokenizer, n_prompts=100, seed=42):
    from utils.load_data import load_ioi_dataset
    from experiments.circuits.ioi_utils import resolve_positions

    # Over-generate: ~50% of names are multi-token and get filtered
    ds = load_ioi_dataset(target=n_prompts * 3, seed=seed)
    raw = ds.metadata
    eos = tokenizer.eos_token or "<|endoftext|>"
    prompts = []
    for d in raw:
        roles = resolve_positions(d["prompt"], d["IO"], d["S"], tokenizer)
        if roles is None or "IO" not in roles or "S2" not in roles:
            continue
        positions = {k: v + 1 for k, v in roles.items()}  # +1 for <eos> prefix
        prompts.append({
            "text": eos + d["prompt"],
            "target": d["IO"],
            "distractor": d["S"],
            "positions": positions,
            "metadata": {
                "io_token": d["IO"],
                "s_token": d["S"],
                "template_type": d.get("template_type", ""),
            },
        })
    return prompts[:n_prompts]


# ── Helpers ──

def _clean_name(raw):
    name = re.sub(r"\[[KQV]\]", "", raw)
    name = re.sub(r"@\d+$", "", name)
    return name.strip()


def tier_of(name):
    for tier, members in TIER_MAP.items():
        if name in members:
            return tier
    return None


def extract_upstream(paths, root_name, top_k=15):
    """Extract upstream components with K/Q/V mode breakdown and full paths."""
    groups = defaultdict(lambda: {
        "K": 0.0, "Q": 0.0, "V": 0.0, "?": 0.0,
        "total": 0.0, "n": 0, "paths": [],
    })

    for p in paths:
        root_mode = "?"
        if p.chain:
            first_sep = p.chain.find("→")
            root_part = p.chain[:first_sep] if first_sep >= 0 else p.chain
            m = MODE_RE.search(root_part)
            if m:
                root_mode = m.group(1)

        for comp in p.components:
            clean = _clean_name(comp)
            if clean == root_name or clean in ("embedding", "pos_embedding"):
                continue
            groups[clean]["total"] += abs(p.score)
            groups[clean][root_mode] += abs(p.score)
            groups[clean]["n"] += 1
            groups[clean]["paths"].append((p.chain, p.score))
            break

    items = []
    for name, d in groups.items():
        mode_scores = {m: d[m] for m in ("K", "Q", "V", "?") if d[m] > 0}
        top_paths = sorted(d["paths"], key=lambda x: abs(x[1]), reverse=True)[:3]
        items.append({
            "name": name, "tier": tier_of(name),
            "abs_score": d["total"], "modes": mode_scores,
            "n_paths": d["n"], "top_paths": top_paths,
        })
    items.sort(key=lambda x: x["abs_score"], reverse=True)
    return items[:top_k]


# ── Per-prompt processing (one forward pass, many backward passes) ──

def process_prompt(tracer, prompt, config, top_k=15,
                   do_tokens=True, do_composition=True,
                   verbose_first=False):
    """Process one prompt: one prepare(), many trace_from_prep().
    
    Returns (token_record, composition_records).
    """
    prep, cfg = tracer.prepare(
        prompt["text"], target=prompt["target"],
        distractor=prompt["distractor"], config=config,
    )

    if verbose_first:
        print(f"    [prep] q_decomp={prep.get('query_decomp') is not None}  "
              f"v_decomp={prep.get('value_decomp') is not None}  "
              f"attn_outproj={prep.get('attn_shares_outproj') is not None}  "
              f"mlp_outproj={prep.get('mlp_outproj') is not None}  "
              f"mlp_geva={prep.get('mlp_geva') is not None}")

    positions = prompt["positions"]

    # ── Token attribution (backward from target) ──
    token_record = None
    if do_tokens:
        result = tracer.trace_from_prep(prep, cfg, root=None)
        attr = result.token_attribution
        io_attr = float(attr[positions["IO"]])
        s1_attr = float(attr[positions["S1"]])
        s2_attr = float(attr[positions["S2"]])
        top1_pos = int(np.argmax(attr))
        attr_no_bos = attr.copy()
        attr_no_bos[0] = -float('inf')
        top1_no_bos = int(np.argmax(attr_no_bos))

        token_record = {
            "io_attr": io_attr, "s1_attr": s1_attr, "s2_attr": s2_attr,
            "io_gt_s1": io_attr > s1_attr,
            "io_gt_s2": io_attr > s2_attr,
            "io_is_top1": top1_pos == positions["IO"],
            "io_is_top1_no_bos": top1_no_bos == positions["IO"],
            "target_prob": result.target_prob,
            "io_token": prompt["metadata"]["io_token"],
            "s_token": prompt["metadata"]["s_token"],
        }

    # ── Composition (reroot at each claim's heads) ──
    comp_records = []
    if do_composition:
        for claim in COMPOSITION_CLAIMS:
            root_pos = positions.get(claim["root_pos"])
            if root_pos is None:
                continue

            for head in claim["root_heads"]:
                root_str = f"{head}@{root_pos}"
                result = tracer.trace_from_prep(prep, cfg, root=root_str)

                upstream = extract_upstream(result.paths, head, top_k=top_k)

                upstream_tiers = set()
                for u in upstream:
                    if u["tier"]:
                        upstream_tiers.add(u["tier"])

                mode_hit = False
                for u in upstream:
                    if u["tier"] in claim["expected_tiers"]:
                        for mode in claim["expected_modes"]:
                            if u["modes"].get(mode, 0) > 0:
                                mode_hit = True
                                break

                comp_records.append({
                    "claim": claim["name"],
                    "root_head": head,
                    "root_pos_key": claim["root_pos"],
                    "root_pos": root_pos,
                    "upstream": upstream,
                    "upstream_tiers_found": sorted(upstream_tiers),
                    "expected_tier_hit": bool(upstream_tiers & claim["expected_tiers"]),
                    "expected_mode_hit": mode_hit,
                    "target_prob": result.target_prob,
                    "positions": positions,
                })

    return token_record, comp_records


# ── Summaries ──

def print_token_summary(all_results):
    print("\n" + "=" * 70)
    print("  TOKEN ATTRIBUTION RESULTS")
    print("=" * 70)
    header = f"  {'config':<18s}  {'IO>S1':>6s}  {'IO>S2':>6s}  {'IO top1':>8s}  {'top1*':>6s}  {'mean IO':>8s}  {'mean S1':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for config, records in all_results.items():
        n = len(records)
        io_gt_s1 = sum(1 for r in records if r["io_gt_s1"]) / n * 100
        io_gt_s2 = sum(1 for r in records if r["io_gt_s2"]) / n * 100
        io_top1 = sum(1 for r in records if r["io_is_top1"]) / n * 100
        io_top1_nb = sum(1 for r in records if r.get("io_is_top1_no_bos", r["io_is_top1"])) / n * 100
        mean_io = np.mean([r["io_attr"] for r in records])
        mean_s1 = np.mean([r["s1_attr"] for r in records])
        print(f"  {config:<18s}  {io_gt_s1:>5.1f}%  {io_gt_s2:>5.1f}%  "
              f"{io_top1:>7.1f}%  {io_top1_nb:>5.1f}%  {mean_io:>+7.1f}%  {mean_s1:>+7.1f}%")
    print("  (* top-1 excluding BOS position)")


def print_composition_summary(all_results):
    print("\n" + "=" * 70)
    print("  COMPOSITION ANALYSIS RESULTS")
    print("=" * 70)
    for config, records in all_results.items():
        print(f"\n  Config: {config}")
        print(f"  {'─' * 65}")
        by_claim = defaultdict(list)
        for r in records:
            by_claim[r["claim"]].append(r)

        for claim_info in COMPOSITION_CLAIMS:
            claim_name = claim_info["name"]
            recs = by_claim.get(claim_name, [])
            if not recs:
                continue
            n = len(recs)
            tier_hit = sum(1 for r in recs if r["expected_tier_hit"]) / n * 100
            mode_hit = sum(1 for r in recs if r["expected_mode_hit"]) / n * 100

            tier_presence = defaultdict(int)
            tier_modes = defaultdict(lambda: defaultdict(float))
            for r in recs:
                seen = set()
                for u in r["upstream"]:
                    t = u["tier"]
                    if t and t not in seen:
                        tier_presence[t] += 1
                        seen.add(t)
                    if t:
                        for mode, score in u["modes"].items():
                            tier_modes[t][mode] += score

            print(f"\n    {claim_name}: {claim_info['description']}")
            print(f"    root@{claim_info['root_pos']}  "
                  f"expect={claim_info['expected_tiers']} via {claim_info['expected_modes']}")
            print(f"    tier_hit={tier_hit:.1f}%  mode_hit={mode_hit:.1f}%  (n={n})")
            print(f"    {'upstream':<10s}  {'presence':>9s}  {'K':>8s}  {'Q':>8s}  {'V':>8s}")
            print(f"    {'─'*10}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*8}")
            for t in ["NM", "NegNM", "S-Inh", "Ind", "Dup", "Prev"]:
                if tier_presence.get(t, 0) == 0:
                    continue
                pct = tier_presence[t] / n * 100
                modes = tier_modes[t]
                total = sum(modes.values()) or 1
                k = modes.get("K", 0) / total * 100
                q = modes.get("Q", 0) / total * 100
                v = modes.get("V", 0) / total * 100
                print(f"    {t:<10s}  {pct:>8.1f}%  {k:>7.1f}%  {q:>7.1f}%  {v:>7.1f}%")

            by_head = defaultdict(list)
            for r in recs:
                by_head[r["root_head"]].append(r)
            for head in sorted(by_head.keys()):
                h_recs = by_head[head]
                n_h = len(h_recs)
                t_hit = sum(1 for r in h_recs if r["expected_tier_hit"]) / n_h * 100
                m_hit = sum(1 for r in h_recs if r["expected_mode_hit"]) / n_h * 100
                role = WANG_CIRCUIT_HEADS.get(head, "")
                print(f"      {head} [{role}]: tier={t_hit:.0f}% mode={m_hit:.0f}% (n={n_h})")


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", nargs="+", default=ALL_CONFIGS)
    ap.add_argument("--out-dir", default="results/ioi_analysis")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--skip-composition", action="store_true")
    ap.add_argument("--skip-tokens", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    do_tok = not args.skip_tokens
    do_comp = not args.skip_composition

    print("Loading GPT-2 small...")
    tracer = unpack.Tracer("gpt2", device=args.device, cache_dir=args.cache_dir)

    print(f"Loading {args.n_prompts} IOI prompts (seed={args.seed})...")
    prompts = load_ioi_prompts(tracer.tokenizer, n_prompts=args.n_prompts,
                               seed=args.seed)
    print(f"  Got {len(prompts)} prompts")

    token_results = {}
    comp_results = {}

    for config in args.configs:
        # Resolve config to see actual flags
        from unpack.config import get_config
        cfg = get_config(config)
        print(f"\n{'─' * 60}")
        print(f"  Config: {config}")
        print(f"    branches={cfg.branches}  mlp_rule={cfg.mlp_rule}  "
              f"aligned={cfg.aligned}")
        print(f"    enable_q={cfg.enable_q_side}  enable_v={cfg.enable_v_side}  "
              f"outproj={cfg.aligned}")
        print(f"{'─' * 60}")

        tok_records = []
        comp_records = []

        for i, p in enumerate(tqdm(prompts, desc=f"  {config}")):
            tok_rec, comp_recs = process_prompt(
                tracer, p, config, top_k=args.top_k,
                do_tokens=do_tok, do_composition=do_comp,
                verbose_first=(i == 0),
            )
            if tok_rec:
                tok_records.append(tok_rec)
            comp_records.extend(comp_recs)

        if tok_records:
            token_results[config] = tok_records
        if comp_records:
            comp_results[config] = comp_records

    if token_results:
        print_token_summary(token_results)
    if comp_results:
        print_composition_summary(comp_results)

    # Save
    out_path = os.path.join(args.out_dir, "ioi_analysis.json")
    save_data = {"n_prompts": len(prompts), "seed": args.seed, "configs": args.configs}
    if token_results:
        save_data["token_attribution"] = token_results
    if comp_results:
        for config in comp_results:
            for r in comp_results[config]:
                r["upstream_tiers_found"] = list(r["upstream_tiers_found"])
                for u in r["upstream"]:
                    u["top_paths"] = [(c, float(s)) for c, s in u["top_paths"]]
        save_data["composition"] = comp_results

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()