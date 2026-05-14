"""Discover circuits via path-level credit attribution.

For each prompt:
  1. Run a clean trace (no ablation) — single forward pass.
  2. Filter the resulting paths by the configured lens.
  3. Accumulate a component ranking (sum |path score| per component
     appearance in surviving paths).
  4. Save per-prompt JSON to <output_dir>/p{idx:04d}.json.

Universal: the discover step has no task knowledge. Anything task-
specific (which positions are "the IO" etc.) is carried in the prompt's
metadata, populated by prompts.py.

Resume-friendly: skips prompts whose JSON already exists in output_dir.

Usage:
    python -m experiments.circuit_discovery.discover <experiment_folder>
    python -m experiments.circuit_discovery.discover \\
        experiments/circuit_discovery/ioi --output-dir runs/ioi_discover_v1
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

# Make project root importable when invoked via `-m` or directly.

from experiments.ablation_tracing import trace
from experiments.circuit_discovery.utils  import (
    load_experiment_folder, load_model, parse_step, chain_components,
)
from experiments.circuit_discovery.inspect.lenses import make_lens_filter


# ──────────────────────────────────────────────────────────────────────
def derive_component_ranking(kept_paths) -> List[Dict]:
    """Sum |path score| per unique component over the kept-path set.

    Returns sorted list of dicts:
        [{"name": str, "cum_score": float, "n_paths": int}, ...]
    """
    cum_score: Dict[str, float] = defaultdict(float)
    n_paths:   Dict[str, int]   = defaultdict(int)
    for path in kept_paths:
        score = abs(path["score"])
        # Each component appears at most once per chain in practice; we
        # count once per path-occurrence (a component appearing twice in
        # the same chain — rare, but possible if a head shows up at two
        # positions — gets counted twice, which is the right call:
        # higher rank reflects appearing-more-strongly).
        for name in chain_components(path["chain"]):
            if name in ("embedding", "pos_embedding"):
                continue   # terminals; not patchable
            cum_score[name] += score
            n_paths[name]   += 1

    rows = [
        {"name": n, "cum_score": float(s), "n_paths": int(n_paths[n])}
        for n, s in cum_score.items()
    ]
    rows.sort(key=lambda r: -r["cum_score"])
    return rows


def discover_one(model, tokenizer, hook_manager, beta, top_paths_k,
                 path_min_frac, lens_cfg, prompt_dict,
                 enable_q_side: bool = True,
                 enable_v_side: bool = True,
                 branch_weights = None,
                 geomean_min = None,
                 mlp_geva_enabled: bool = False,
                 mlp_outproj_enabled: bool = False,
                 attn_outproj_enabled: bool = False) -> Dict:
    """Run trace + lens filter + ranking for one prompt. Returns JSON-able dict."""
    res = trace(
        model, tokenizer, prompt_dict["prompt"],
        target_token=prompt_dict["target_token"],
        distractor_token=prompt_dict.get("distractor_token"),
        hook_manager=hook_manager,
        beta=beta,
        top_paths_k=top_paths_k,
        path_min_frac=path_min_frac,
        edges_top_k_per_node=0,    # we don't need edges for discover
        enable_q_side=enable_q_side,
        enable_v_side=enable_v_side,
        branch_weights=branch_weights,
        geomean_min=geomean_min,
        mlp_geva_enabled=mlp_geva_enabled,
        mlp_outproj_enabled=mlp_outproj_enabled,
        attn_outproj_enabled=attn_outproj_enabled,
    )

    lens_fn = make_lens_filter(lens_cfg, prompt_dict)
    kept_paths = []
    for chain, src_pos, score in res.paths:
        if not lens_fn(chain, src_pos):
            continue
        kept_paths.append({
            "chain":   list(chain),
            "src_pos": int(src_pos),
            "score":   float(score),
        })

    ranked_components = derive_component_ranking(kept_paths)

    # ── Component flow: dense top-down sweep, no min_frac truncation ──
    # Each value is the per-component signed flow summed across positions.
    # Selection strategies (component_flow, partition_coverage) can read
    # this for ranking that's independent of path enumeration / pruning.
    component_flow = {
        name: float(score)
        for name, score in res.flow.items()
        if "bias" not in name  # drop norm_bias / attn_bias / value_bias
    }

    return {
        "prompt":                      prompt_dict["prompt"],
        "target_token":                prompt_dict["target_token"],
        "distractor_token":            prompt_dict.get("distractor_token"),
        "n_tokens":                    res.seq_len,
        "clean_target_prob":           float(res.target_prob),
        "clean_target_logit_centered": float(res.target_logit_centered),

        "metadata":                    prompt_dict.get("metadata", {}),

        "lens":         lens_cfg["type"],
        "lens_params":  {k: v for k, v in lens_cfg.items() if k != "type"},

        "ranked_paths":      kept_paths,
        "ranked_components": ranked_components,
        "component_flow":    component_flow,
    }


# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder",
                        help="experiment folder with config.py + prompts.py")
    parser.add_argument("--output-dir", default=None,
                        help="default: <folder>/discoveries/")
    parser.add_argument("--p-min", type=float, default=0.0,
                        help="skip prompts with clean P(target) < this. "
                             "Default 0 (keep all).")
    parser.add_argument("--limit", type=int, default=None,
                        help="trace at most this many prompts (debugging)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing per-prompt JSONs")
    args = parser.parse_args()

    cfg, build_prompts = load_experiment_folder(args.folder)
    out_dir = args.output_dir or os.path.join(args.folder, "discoveries")
    os.makedirs(out_dir, exist_ok=True)

    # Snapshot the config alongside the discoveries for reproducibility.
    spec_path = os.path.join(out_dir, "_spec.json")
    with open(spec_path, "w") as f:
        json.dump({"config": cfg, "folder": args.folder}, f, indent=2,
                  default=str)

    print(f"loading model: {cfg['model']['family']}-{cfg['model']['size']}")
    model, tokenizer, hook_manager, device = load_model(cfg)

    print("building prompts")
    prompts = build_prompts(tokenizer)
    if args.limit:
        prompts = prompts[:args.limit]

    beta          = cfg["trace"]["beta"]
    top_paths_k   = cfg["trace"]["top_paths_k"]
    path_min_frac = cfg["trace"]["path_min_frac"]
    lens_cfg      = cfg["lens"]

    print(f"discovering on {len(prompts)} prompts "
          f"(β={beta}, top_paths_k={top_paths_k}, "
          f"path_min_frac={path_min_frac}, lens={lens_cfg['type']})")

    n_done = 0
    n_skipped = 0
    for idx, p in enumerate(prompts):
        path = os.path.join(out_dir, f"p{idx:04d}.json")
        if os.path.exists(path) and not args.force:
            n_skipped += 1
            continue
        if idx % 10 == 0:
            print(f"  [{idx+1}/{len(prompts)}] {p['prompt'][:60]}...")

        try:
            d = discover_one(model, tokenizer, hook_manager,
                             beta, top_paths_k, path_min_frac,
                             lens_cfg, p)
        except Exception as e:
            print(f"    skip prompt {idx}: {e}")
            continue

        if d["clean_target_prob"] < args.p_min:
            # Save anyway — analyses can filter later — but flag in JSON.
            d["below_p_min"] = True

        d["prompt_idx"] = idx
        with open(path, "w") as f:
            json.dump(d, f)
        n_done += 1

    print(f"\nfinished: {n_done} new, {n_skipped} resumed")
    print(f"output:   {out_dir}")


if __name__ == "__main__":
    main()