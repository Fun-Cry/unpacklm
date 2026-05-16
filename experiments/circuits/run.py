"""Circuit discovery pipeline: discover → select → verify.

Uses unpack.Tracer for path extraction, partition_coverage for selection,
mean-ablation for verification.

Usage:
    python -m experiments.circuits.run --device cuda:0
    python -m experiments.circuits.run --device cuda:0 --models gpt2
    python -m experiments.circuits.run --device cuda:0 --models gpt2 --configs kqv_aligned
"""

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List

import unpack
from .ioi_utils import (
    load_ioi_prompts, diversity_filter, DEFAULT_EXCLUDE,
    chain_components, strip_branch_tag,
)
from .select import select
from .verify import verify, prepare_verify_prompts


MODELS = {
    "gpt2": {"name": "gpt2", "tag": "gpt2_small"},
    "pythia-160m": {"name": "EleutherAI/pythia-160m-deduped", "tag": "pythia_160m"},
    "pythia-410m": {"name": "EleutherAI/pythia-410m-deduped", "tag": "pythia_410m"},
    "pythia-1.4b": {"name": "EleutherAI/pythia-1.4b-deduped", "tag": "pythia_1.4b"},
    "pythia-2.8b": {"name": "EleutherAI/pythia-2.8b-deduped", "tag": "pythia_2.8b"},
    "pythia-6.9b": {"name": "EleutherAI/pythia-6.9b-deduped", "tag": "pythia_6.9b"},
}

DEFAULT_CONFIGS = ["kqv_weighted", "kqv_l2", "kqv_aligned"]


# ── Discovery ──

def _derive_component_ranking(paths) -> List[Dict]:
    """Sum |path.score| per unique component."""
    cum_score: Dict[str, float] = defaultdict(float)
    n_paths: Dict[str, int] = defaultdict(int)
    for p in paths:
        score = abs(p.score)
        for name in p.components:
            if name in ("embedding", "pos_embedding"):
                continue
            cum_score[name] += score
            n_paths[name] += 1
    rows = [
        {"name": n, "cum_score": float(s), "n_paths": int(n_paths[n])}
        for n, s in cum_score.items()
    ]
    rows.sort(key=lambda r: -r["cum_score"])
    return rows


def discover_one(tracer, prompt, config, top_paths_k=200,
                 min_positions=2):
    """Trace one prompt, apply diversity lens, return result dict."""
    result = tracer.trace(
        prompt["prompt"],
        target=prompt["target_token"],
        distractor=prompt.get("distractor_token"),
        config=config,
    )

    # Apply diversity lens
    kept_paths = []
    for p in result.paths[:top_paths_k * 5]:
        # Parse chain string into steps: "a[K]@3→b@5→c@5" → ["a[K]@3","b@5","c@5"]
        chain_steps = p.chain.split("→")
        if not diversity_filter(chain_steps, p.source_pos, min_positions):
            continue
        kept_paths.append({
            "chain": chain_steps,
            "src_pos": p.source_pos,
            "score": p.score,
        })
    kept_paths.sort(key=lambda x: abs(x["score"]), reverse=True)
    kept_paths = kept_paths[:top_paths_k]

    component_flow = {
        name: float(score)
        for name, score in result.component_flow.items()
        if "bias" not in name
    }

    return {
        "prompt": prompt["prompt"],
        "target_token": prompt["target_token"],
        "distractor_token": prompt.get("distractor_token"),
        "clean_target_prob": result.target_prob,
        "metadata": prompt.get("metadata", {}),
        "ranked_paths": kept_paths,
        "component_flow": component_flow,
    }


# ── Main pipeline ──

def run_one(tracer, model_tag, config, disc_prompts, verify_prompts,
            out_dir="results/circuits", device="cuda:0",
            n_top_paths=200, p_min=0.0, force=False,
            partition_threshold=0.0, min_prompt_fraction=0.0):
    """Discover + select + verify for one model/config."""

    tag = f"{model_tag}_{config}"
    run_dir = os.path.join(out_dir, tag)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"{'='*60}")

    # ── Step 1: Discover ──
    print(f"\n  [1/3] Discovering ({len(disc_prompts)} prompts, diversity≥2)...")
    t0 = time.time()

    prompt_results = []
    n_ok = 0
    n_cached = 0
    for pi, p in enumerate(disc_prompts):
        out_path = os.path.join(run_dir, f"prompt_{pi:04d}.json")

        # Resume: load cached result if exists
        if not force and os.path.exists(out_path):
            with open(out_path) as f:
                disc = json.load(f)
            # Patch s1p1 into cached metadata
            meta = disc.get("metadata", {})
            s1 = meta.get("s1_position")
            if s1 is not None and "s1p1_position" not in meta:
                meta["s1p1_position"] = s1 + 1
            if disc.get("clean_target_prob", 0) >= p_min:
                prompt_results.append(disc)
                n_ok += 1
            n_cached += 1
            continue

        try:
            disc = discover_one(tracer, p, config, top_paths_k=n_top_paths)
        except Exception as e:
            print(f"    prompt {pi}: error {e}")
            continue

        # Save per-prompt JSON
        disc["prompt_idx"] = pi
        with open(out_path, "w") as f:
            json.dump(disc, f, indent=2, default=str)

        if disc["clean_target_prob"] < p_min:
            continue

        prompt_results.append(disc)
        n_ok += 1
        if n_ok <= 3 or n_ok % 20 == 0:
            print(f"    [{n_ok}] prob={disc['clean_target_prob']:.3f} "
                  f"paths={len(disc['ranked_paths'])}")

    dt_disc = time.time() - t0
    if n_cached:
        print(f"    {n_cached} cached, {n_ok - min(n_cached, n_ok)} new")
    print(f"    {n_ok} prompts kept (prob≥{p_min}) in {dt_disc:.1f}s")

    # ── Step 2: Select ──
    print(f"\n  [2/3] Selecting (partition_coverage)...")
    circuit_components, diag = select(
        prompt_results, verbose=True,
        partition_threshold=partition_threshold,
        min_prompt_fraction=min_prompt_fraction,
    )

    # Save circuit
    circuit_data = {
        "components": circuit_components,
        "n_components": len(circuit_components),
        "model": model_tag,
        "config": config,
        "n_discovery_prompts": n_ok,
    }
    with open(os.path.join(run_dir, "circuit.json"), "w") as f:
        json.dump(circuit_data, f, indent=2)
    print(f"    Circuit: {len(circuit_components)} components")

    # ── Step 3: Verify ──
    print(f"\n  [3/3] Verifying ({len(verify_prompts)} prompts)...")
    summary = verify(
        tracer.model, tracer.tokenizer, tracer.adapter, device,
        set(circuit_components), verify_prompts,
    )

    # Save
    with open(os.path.join(run_dir, "verification.json"), "w") as f:
        json.dump({**summary, "circuit": circuit_components}, f, indent=2)

    m = summary.get("mean_ld", {})
    sem = summary.get("sem_ld", {})
    fr = summary.get("faith_ratio", 0) or 0
    ko = summary.get("comp_drop", 0) or 0
    print(f"\n    |C|={summary.get('circuit_size', 0)}  n={summary.get('n_prompts', 0)}")
    print(f"    clean  {m.get('clean',0):+.3f} ± {sem.get('clean',0):.3f}")
    print(f"    faith  {m.get('faith',0):+.3f} ± {sem.get('faith',0):.3f}  (ratio: {fr:+.3f})")
    print(f"    comp   {m.get('comp',0):+.3f} ± {sem.get('comp',0):.3f}  (drop:  {ko:+.3f})")

    return circuit_components, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                    choices=list(MODELS.keys()))
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--n-discovery", type=int, default=100)
    ap.add_argument("--n-verify", type=int, default=100)
    ap.add_argument("--discovery-seed", type=int, default=42)
    ap.add_argument("--verify-seed", type=int, default=7)
    ap.add_argument("--out-dir", default="results/circuits")
    ap.add_argument("--force", action="store_true",
                    help="Re-trace even if per-prompt JSONs exist")
    ap.add_argument("--verify-p-min", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}

    for model_key in args.models:
        info = MODELS[model_key]
        print(f"\nLoading {info['name']}...")
        tracer = unpack.Tracer(info["name"], device=args.device,
                               cache_dir=args.cache_dir)

        # Load prompts once per model
        disc_prompts = load_ioi_prompts(tracer.tokenizer,
                                        n_prompts=args.n_discovery,
                                        seed=args.discovery_seed)
        print(f"  {len(disc_prompts)} discovery prompts")

        verify_prompts = prepare_verify_prompts(
            tracer.tokenizer, tracer.model, args.device,
            n_prompts=args.n_verify, seed=args.verify_seed,
            p_min=args.verify_p_min)
        print(f"  {len(verify_prompts)} verification prompts")

        for config in args.configs:
            circuit, summary = run_one(
                tracer, info["tag"], config,
                disc_prompts=disc_prompts,
                verify_prompts=verify_prompts,
                out_dir=args.out_dir,
                device=args.device,
                force=args.force,
            )
            results[(info["tag"], config)] = (circuit, summary)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  {'Model':<16s} {'Config':<18s} {'|C|':>4s} {'Faith':>8s} {'KO':>8s}")
    print(f"  {'-'*58}")
    for (tag, config), (circuit, summary) in results.items():
        fr = summary.get("faith_ratio") or 0
        ko = summary.get("comp_drop") or 0
        sz = summary.get("circuit_size", 0)
        print(f"  {tag:<16s} {config:<18s} {sz:>4d} {fr:>+.3f}   {ko:>+.3f}")


if __name__ == "__main__":
    main()