"""Run discover on many IOI prompts for one (scale, checkpoint).

Saves one JSON per prompt under `results_dir/`. Each JSON has the
prompt metadata, target/distractor, target probability, and the
top-K paths after lens filtering.

Filtering by target probability (canonical-correct vs. failure case)
happens at analysis time, not capture time. We save every prompt,
including failures — failure-case traces sometimes show induction-
style fallback mechanisms or other interesting behavior, and keeping
them lets us re-analyze with different thresholds without re-running
the model.

The script is model-agnostic — the model is read from the experiment
folder's CONFIG, with optional CLI overrides.

Usage:
    # Use the model in ioi/config.py
    python -m experiments.circuit_discovery.run_ioi_sweep \\
        --n-prompts 100 \\
        --results-dir results/run

    # Override model from CLI (no config edit needed)
    python -m experiments.circuit_discovery.run_ioi_sweep \\
        --family pythia --size 1.4b --step 143000 --deduped 1 \\
        --cache-dir /data/s4283341 \\
        --n-prompts 100 \\
        --results-dir results/pythia_1.4b_step143000

For cross-scale and cross-checkpoint sweeps, see the bash driver
run_pythia_sweeps.sh which iterates the CLI overrides.
"""

import argparse
import json
import os
import sys
import time


from experiments.circuit_discovery.utils import (
    load_experiment_folder, load_model,
)
from experiments.circuit_discovery.discover import discover_one
from experiments.circuit_discovery.ioi.prompts import _resolve_positions
from utils.load_data import load_ioi_with_abc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="experiments/circuit_discovery/ioi",
                    help="experiment folder with config.py")
    ap.add_argument("--n-prompts", type=int, default=100,
                    help="how many IOI prompts to generate")
    ap.add_argument("--n-abc-refs", type=int, default=0,
                    help="ABC corruptions to attach per prompt (0 if not patching)")
    ap.add_argument("--results-dir", required=True,
                    help="output directory; per-prompt JSONs saved here")
    ap.add_argument("--seed", type=int, default=42)

    # Optional: override the model section of the config from the CLI.
    # Useful for cross-scale / cross-checkpoint sweeps without editing
    # config.py between runs.
    ap.add_argument("--family", default=None,
                    help="override: model family (e.g., 'pythia')")
    ap.add_argument("--size", default=None,
                    help="override: model size (e.g., '1.4b')")
    ap.add_argument("--step", type=int, default=None,
                    help="override: training step (Pythia only)")
    ap.add_argument("--deduped", type=int, default=None,
                    choices=[0, 1],
                    help="override: 1 for deduped Pythia, 0 for not")
    ap.add_argument("--cache-dir", default=None,
                    help="override: model cache dir")
    ap.add_argument("--device", default=None,
                    help="override: cuda:0, cpu, etc.")
    ap.add_argument("--no-q-side", dest="enable_q_side", action="store_false",
                    default=True,
                    help="disable Q-side decomposition; reproduces the "
                         "original K-only path attribution. Used for "
                         "ablation comparison: shows what the method "
                         "captures without query-side recursion.")
    ap.add_argument("--no-v-side", dest="enable_v_side", action="store_false",
                    default=True,
                    help="disable V-side decomposition; pair with "
                         "--no-q-side for the original K-only baseline, "
                         "or use alone for K+Q-only (without V).")
    ap.add_argument("--branch-weights", default=None,
                    help="branch weights as 'K=w,Q=w,V=w' (default 1,1,1 "
                         "i.e. no-split mode where each enabled branch "
                         "consumes full credit). Use K=0.33,Q=0.33,V=0.33 "
                         "for an equal partition.")
    ap.add_argument("--geomean-min", type=float, default=None,
                    help="Geomean threshold for path pruning gate. "
                         "Paths with raw < min_frac but geomean >= this "
                         "threshold survive pruning. None disables.")
    ap.add_argument("--mlp-geva", action="store_true", default=False,
                    help="Use Geva activation-weighted MLP decomposition "
                         "at depth >= 1.")
    ap.add_argument("--mlp-outproj", action="store_true", default=False,
                    help="Use output-aligned MLP decomposition at depth >= 1. "
                         "Takes precedence over --mlp-geva.")
    ap.add_argument("--attn-outproj", action="store_true", default=False,
                    help="Use output-aligned attention value-side dispatch "
                         "at depth >= 1.")
    ap.add_argument("--outproj", action="store_true", default=False,
                    help="Convenience: enable both --mlp-outproj and "
                         "--attn-outproj together.")
    ap.add_argument("--beta", type=float, default=0.8,
                help="SafeDenom soft-floor amplification cap.")
    args = ap.parse_args()

    # Load model + config
    print(f"loading config + model from {args.folder}")
    cfg, _ = load_experiment_folder(args.folder)

    # Apply CLI overrides to the model block
    overrides = {
        "family":    args.family,
        "size":      args.size,
        "step":      args.step,
        "deduped":   bool(args.deduped) if args.deduped is not None else None,
        "cache_dir": args.cache_dir,
        "device":    args.device,
    }
    for key, val in overrides.items():
        if val is not None:
            cfg["model"][key] = val

    model, tokenizer, hook_manager, device = load_model(cfg)

    fam  = cfg["model"]["family"]
    size = cfg["model"]["size"]
    step = cfg["model"].get("step", "final")
    print(f"\nmodel: {fam} {size}  step: {step}")
    print(f"n_layers={model.config.num_hidden_layers}  "
          f"n_heads={model.config.num_attention_heads}  "
          f"d_model={model.config.hidden_size}")

    # Discover settings
    beta          = args.beta
    top_paths_k   = cfg.get("trace", {}).get("top_paths_k", 200)
    path_min_frac = cfg.get("trace", {}).get("path_min_frac", 1e-4)
    lens_cfg      = cfg.get("lens", {"type": "diversity", "min_diversity": 2})
    print(f"trace: beta={beta}  top_paths_k={top_paths_k}  "
          f"path_min_frac={path_min_frac}")
    print(f"lens: {lens_cfg}")

    # Generate prompts
    print(f"\nloading {args.n_prompts} IOI prompts (seed={args.seed})")
    raw = load_ioi_with_abc(
        n_prompts=args.n_prompts, n_abc_refs=args.n_abc_refs,
        tokenizer=tokenizer, seed=args.seed,
    )

    # Output dir
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"output: {args.results_dir}")
    print()

    # Save the run config alongside the per-prompt JSONs
    with open(os.path.join(args.results_dir, "run_config.json"), "w") as f:
        json.dump({
            "family":       fam,
            "size":         size,
            "step":         step,
            "n_prompts":    args.n_prompts,
            "n_abc_refs":   args.n_abc_refs,
            "trace": {
                "beta":          beta,
                "top_paths_k":   top_paths_k,
                "path_min_frac": path_min_frac,
                "enable_q_side": args.enable_q_side,
                "enable_v_side": args.enable_v_side,
                "branch_weights": args.branch_weights,
            },
            "lens":         lens_cfg,
            "seed":         args.seed,
        }, f, indent=2)

    # Parse branch_weights string -> dict
    branch_weights = None
    if args.branch_weights:
        branch_weights = {}
        for chunk in args.branch_weights.split(","):
            k, v = chunk.split("=")
            branch_weights[k.strip()] = float(v.strip())

    # Run discover per prompt, save every result. Filtering by target
    # probability happens at analysis time so failure cases stay
    # available for inspection (they can show induction-style fallback
    # mechanisms or just noise — useful to see).
    n_saved  = 0
    n_failed = 0
    t_start = time.time()

    for pi, p in enumerate(raw):
        clean_prompt     = p["prompt"]
        target_token     = p["target_token"]
        distractor_token = p["distractor_token"]

        try:
            roles = _resolve_positions(clean_prompt, p["IO"], p["S"], tokenizer)
        except Exception as e:
            print(f"  prompt {pi}: position resolution error: {e}")
            n_failed += 1
            continue

        prompt_dict = {
            "prompt":           clean_prompt,
            "target_token":     target_token,
            "distractor_token": distractor_token,
            "metadata": {
                "io_position":      roles["IO"],
                "s1_position":      roles.get("S1"),
                "s2_position":      roles.get("S2"),
                "end_position":     roles["END"],
                "target_positions": [
                    roles["IO"],
                    roles.get("S1", roles["IO"]),
                    roles.get("S2", roles["IO"]),
                ],
                "template_type":    p["template_type"],
                "IO":               p["IO"],
                "S":                p["S"],
            },
        }

        try:
            disc = discover_one(
                model, tokenizer, hook_manager,
                beta=beta, top_paths_k=top_paths_k,
                path_min_frac=path_min_frac, lens_cfg=lens_cfg,
                prompt_dict=prompt_dict,
                enable_q_side=args.enable_q_side,
                enable_v_side=args.enable_v_side,
                branch_weights=branch_weights,
                geomean_min=args.geomean_min,
                mlp_geva_enabled=args.mlp_geva,
                mlp_outproj_enabled=args.mlp_outproj or args.outproj,
                attn_outproj_enabled=args.attn_outproj or args.outproj,
            )
        except Exception as e:
            print(f"  prompt {pi}: discover error: {e}")
            n_failed += 1
            continue

        target_prob = disc.get("clean_target_prob")

        # Save: every prompt, including failure cases. Filter at
        # analysis time using clean_target_prob from the JSON.
        out_path = os.path.join(args.results_dir, f"prompt_{pi:04d}.json")
        with open(out_path, "w") as f:
            json.dump(disc, f, indent=2)
        n_saved += 1

        if n_saved <= 3 or n_saved % 10 == 0:
            elapsed = time.time() - t_start
            rate = (pi + 1) / elapsed
            tp_str = f"{target_prob:.3f}" if target_prob is not None else "n/a"
            logit = disc.get("clean_target_logit_centered", 0)
            print(f"  [{pi}] target_prob={tp_str}  logit={logit:+.2f}  "
                  f"saved={n_saved}  rate={rate:.2f} prompt/s")

    elapsed = time.time() - t_start
    print()
    print(f"done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  saved:  {n_saved}")
    print(f"  failed: {n_failed}")


if __name__ == "__main__":
    main()