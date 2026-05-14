"""Step 1: Discover — trace each prompt, save per-prompt JSONs.

Usage:
    python -m experiments.circuits.discover \
        --model gpt2 --config default --n-prompts 100 \
        --results-dir results/circuits/gpt2_default
"""

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List

import unpack
from experiments.circuits.ioi_utils import resolve_positions, make_lens
from utils.load_data import load_ioi_with_abc


CONFIGS = list(unpack.PRESETS.keys())


def _derive_model_info(model_name):
    """Extract family/size/step/deduped from model name string."""
    m = model_name.lower()
    if "pythia" in m:
        family = "pythia"
        size = next((s for s in ["160m", "410m", "1.4b", "2.8b", "6.9b"]
                     if s in m), m.split("-")[-1])
        deduped = "deduped" in m
        return family, size, 143000, deduped
    return "gpt2", "small", "final", False


def discover_one(tracer, prompt_dict, config, top_paths_k, lens_cfg):
    """Trace one prompt → lens filter → component ranking."""
    result = tracer.trace(
        prompt_dict["prompt"],
        target=prompt_dict["target_token"],
        distractor=prompt_dict.get("distractor_token"),
        config=config,
    )

    # Lens filter on paths
    lens_fn = make_lens(lens_cfg, prompt_dict.get("metadata", {}))
    kept = []
    for p in result.paths[:top_paths_k * 5]:
        steps = [s.strip() for s in p.chain.split("\u2192")]
        if not lens_fn(steps, p.source_pos):
            continue
        kept.append({"chain": steps, "src_pos": p.source_pos,
                     "score": p.score})
    kept.sort(key=lambda x: abs(x["score"]), reverse=True)
    kept = kept[:top_paths_k]

    # Component ranking from all paths (pre-filter)
    cum = defaultdict(float)
    npaths = defaultdict(int)
    for p in result.paths:
        s = abs(p.score)
        for name in p.components:
            if name in ("embedding", "pos_embedding"):
                continue
            cum[name] += s
            npaths[name] += 1
    ranked = [{"name": n, "cum_score": float(cum[n]),
               "n_paths": int(npaths[n])} for n in cum]
    ranked.sort(key=lambda r: -r["cum_score"])

    # Dense component flow (no pruning)
    flow = {n: float(v) for n, v in result.component_flow.items()
            if "bias" not in n}

    return {
        "prompt": prompt_dict["prompt"],
        "target_token": prompt_dict["target_token"],
        "distractor_token": prompt_dict.get("distractor_token"),
        "n_tokens": len(result.tokens),
        "clean_target_prob": result.target_prob,
        "clean_target_logit_centered": result.target_logit_centered,
        "metadata": prompt_dict.get("metadata", {}),
        "lens": lens_cfg["type"],
        "lens_params": {k: v for k, v in lens_cfg.items() if k != "type"},
        "ranked_paths": kept,
        "ranked_components": ranked,
        "component_flow": flow,
    }


def build_ioi_prompts(tracer, n_prompts, seed):
    """Generate IOI prompts with resolved positions."""
    raw = load_ioi_with_abc(n_prompts=n_prompts, n_abc_refs=0,
                            tokenizer=tracer.tokenizer, seed=seed)
    prompts = []
    for p in raw:
        try:
            roles = resolve_positions(
                p["prompt"], p["IO"], p["S"], tracer.tokenizer)
        except Exception:
            continue
        if roles is None or "IO" not in roles:
            continue
        prompts.append({
            "prompt": p["prompt"],
            "target_token": p["target_token"],
            "distractor_token": p["distractor_token"],
            "metadata": {
                "io_position": roles["IO"],
                "s1_position": roles.get("S1"),
                "s2_position": roles.get("S2"),
                "end_position": roles["END"],
                "target_positions": [roles["IO"],
                                     roles.get("S1", roles["IO"]),
                                     roles.get("S2", roles["IO"])],
                "template_type": p["template_type"],
                "IO": p["IO"], "S": p["S"],
            },
        })
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--config", default="default", choices=CONFIGS)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-paths-k", type=int, default=200)
    ap.add_argument("--beta", type=float, default=0.8)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    lens_cfg = {"type": "diversity", "min_positions": 2}

    os.makedirs(args.results_dir, exist_ok=True)
    family, size, step, deduped = _derive_model_info(args.model)

    with open(os.path.join(args.results_dir, "run_config.json"), "w") as f:
        json.dump({"family": family, "size": size, "step": step,
                   "deduped": deduped,
                   "config_preset": args.config,
                   "n_prompts": args.n_prompts,
                   "trace": {"beta": args.beta,
                             "top_paths_k": args.top_paths_k,
                             "path_min_frac": 1e-4},
                   "lens": lens_cfg, "seed": args.seed}, f, indent=2)

    print(f"Loading {args.model}...")
    load_kwargs = {}
    if family == "pythia" and step != "final":
        load_kwargs["step"] = step
    tracer = unpack.Tracer(args.model, device=args.device,
                           cache_dir=args.cache_dir, **load_kwargs)

    prompts = build_ioi_prompts(tracer, args.n_prompts, args.seed)
    print(f"Generated {len(prompts)} IOI prompts")

    n_saved = n_skip = 0
    t0 = time.time()
    for pi, p in enumerate(prompts):
        path = os.path.join(args.results_dir, f"prompt_{pi:04d}.json")
        if os.path.exists(path) and not args.force:
            n_skip += 1; continue

        try:
            d = discover_one(tracer, p, args.config,
                             args.top_paths_k, lens_cfg)
        except Exception as e:
            print(f"  [{pi}] error: {e}"); continue

        d["prompt_idx"] = pi
        with open(path, "w") as f:
            json.dump(d, f, indent=2, default=str)
        n_saved += 1

        if n_saved <= 3 or n_saved % 10 == 0:
            rate = (pi + 1) / (time.time() - t0)
            print(f"  [{pi}] prob={d['clean_target_prob']:.3f}  "
                  f"logit={d['clean_target_logit_centered']:+.2f}  "
                  f"{rate:.2f} prompt/s")

    print(f"\nDone in {time.time()-t0:.0f}s  saved={n_saved} skip={n_skip}")


if __name__ == "__main__":
    main()
