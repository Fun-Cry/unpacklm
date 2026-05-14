"""Re-verify saved circuits with a universal test set per model.

For each model under <parent-dir>:
  1. Generate a fresh pool of IOI prompts at --prompt-seed.
  2. Filter to prompts where clean_target_prob >= p_min for that family.
  3. Take the first --n-prompts of those — that's the *universal test
     set* for this model.
  4. For every (variant, abc_seed) pair, build ABC references at that
     abc_seed and verify the variant's saved circuit on the universal
     test set.

Every variant of the same model is therefore evaluated on identical
inputs; the only thing that changes between rows is the ABC corruption
seed (variance) and the variant's circuit (the comparison target).

The model is loaded ONCE per family+size+step key, so a sweep over
e.g. six models × four variants × five abc-seeds (= 120 verifies) costs
only six model loads.

The saved prompt JSONs in each variant's run dir are NOT used for
verification under this design — only the variant's circuit_pc.json
is. Discovery prompt seed and verify prompt seed can therefore be
the same (in-sample) or different (out-of-sample); both are supported.

Usage examples
--------------
All variants under results/paper_sweep/, default 5 abc seeds, OOS
prompts (seed=7, different from discovery seed=42, n=100 prompts):

    python -m experiments.circuit_discovery.run_seed_variance \\
        --parent-dir results/paper_sweep/ \\
        --device cuda:0 --cache-dir .

In-sample verify (uses the same prompts discovery saw, seed=42):

    python -m experiments.circuit_discovery.run_seed_variance \\
        --parent-dir results/paper_sweep/ \\
        --prompt-seeds 42 \\
        --device cuda:0

Multi-prompt-seed × multi-abc-seed (full variance grid, in-sample +
two OOS sets):

    python -m experiments.circuit_discovery.run_seed_variance \\
        --parent-dir results/paper_sweep/ \\
        --prompt-seeds 42 7 17 \\
        --abc-seeds 10000 20000 30000 \\
        --device cuda:0

Restrict to one model:

    python -m experiments.circuit_discovery.run_seed_variance \\
        --parent-dir results/paper_sweep/ \\
        --models gpt2_small \\
        --device cuda:0

By default geo15 variants are skipped (handoff: byte-identical to nogeo
with the clean pipeline). Pass --include-geo to include them.

Output
------
Per (variant, prompt_seed, abc_seed):
    <run_dir>/verify_p{prompt_seed}_abc{abc_seed}.json
A combined summary is written to:
    <parent-dir>/seed_variance_summary.json
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# Re-use the existing verifier + helpers; no fork.
from experiments.circuit_discovery.utils import load_model
from experiments.circuit_discovery.ioi.abc_prep import add_abc_references
from experiments.circuit_discovery.ioi.prompts import _resolve_positions
from experiments.circuit_discovery.selection._common import (
    filter_correct, load_run,
)
from experiments.circuit_discovery.verification.__main__ import verify
from utils.load_data import load_ioi_with_abc


# ──────────────────────────────────────────────────────────────────────
# Run-dir parsing
# ──────────────────────────────────────────────────────────────────────

# Variant suffixes to test by default. geo15 is dropped because the
# handoff confirmed it is byte-identical to nogeo with the clean
# pipeline; --include-geo restores it.
DEFAULT_VARIANTS = ("k_only_geva", "k_only", "geva_nogeo", "l2_nogeo", "outproj_full")
GEO_VARIANTS     = ("l2_geo15", "geva_geo15")


# Match GPT-2 (small/medium/large/xl) and Pythia (160m/410m/1b/1.4b/2.8b/6.9b/12b).
_DIR_RE = re.compile(
    r"^(?P<model>"
    r"gpt2_(?:small|medium|large|xl)"
    r"|pythia_\d+\.?\d*[mb]"
    r")_(?P<variant>.+)$"
)


def parse_run_dir(name: str) -> Optional[Tuple[str, str]]:
    m = _DIR_RE.match(name)
    if m is None:
        return None
    return m.group("model"), m.group("variant")


def discover_runs(
    parent_dir: str,
    *,
    models_filter: Optional[List[str]],
    variants_filter: Optional[List[str]],
    include_geo: bool,
    circuit_filename: str,
) -> List[Tuple[str, str, str]]:
    """Scan parent_dir, return list of (model, variant, run_dir) tuples
    that have circuit_filename present."""
    out = []
    if not os.path.isdir(parent_dir):
        sys.exit(f"--parent-dir not a directory: {parent_dir}")

    keep_variants = set(variants_filter) if variants_filter \
                    else set(DEFAULT_VARIANTS)
    if include_geo:
        keep_variants |= set(GEO_VARIANTS)

    for name in sorted(os.listdir(parent_dir)):
        path = os.path.join(parent_dir, name)
        if not os.path.isdir(path):
            continue
        parsed = parse_run_dir(name)
        if parsed is None:
            continue
        model, variant = parsed
        if variant not in keep_variants:
            continue
        if models_filter and model not in models_filter:
            continue
        if not os.path.exists(os.path.join(path, circuit_filename)):
            print(f"  [skip] no {circuit_filename} in {name}")
            continue
        out.append((model, variant, path))
    return out


def load_circuit(run_dir: str, circuit_filename: str) -> List[str]:
    with open(os.path.join(run_dir, circuit_filename)) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data["circuit"])
    return list(data)


def p_min_for_model(model: str, p_min_gpt2: float, p_min_pythia: float) -> float:
    return p_min_gpt2 if model.startswith("gpt2") else p_min_pythia


# ──────────────────────────────────────────────────────────────────────
# Universal prompt set for one (model, prompt_seed)
# ──────────────────────────────────────────────────────────────────────

def build_universal_prompts(
    *, model, tokenizer, hook_manager, device,
    prompt_seed: int,
    n_prompts: int,
    pool_size: int,
    p_min: float,
    chunk_size: int = 25,
) -> List[Dict[str, Any]]:
    """Generate up to `pool_size` IOI prompts at `prompt_seed`, run the
    model forward to score `clean_target_prob`, filter by `p_min`,
    return the first `n_prompts` that pass.

    Streams the pool one prompt at a time and clears
    `hook_manager.clear()` + `torch.cuda.empty_cache()` every
    `chunk_size` forward passes. This is necessary on large models
    (Pythia 6.9b, etc.) because the hook manager appends a tensor per
    layer per forward, and a 300-prompt pool on a 32-layer model
    accumulates ~10k cached tensors before any cleanup, which OOMs.

    Stops early once `n_prompts` passing prompts are collected, so the
    full `pool_size` is only consumed if the threshold is strict.
    """
    import torch
    print(f"  generating prompt pool (target {n_prompts}, pool cap "
          f"{pool_size}) at seed={prompt_seed}")
    raw = load_ioi_with_abc(
        n_prompts=pool_size, n_abc_refs=0,
        tokenizer=tokenizer, seed=prompt_seed,
        abc_seed_offset=10000,   # unused at this stage
    )

    kept: List[Dict[str, Any]] = []
    n_seen      = 0
    n_pos_fail  = 0
    n_below     = 0
    for p in raw:
        if len(kept) >= n_prompts:
            break

        try:
            roles = _resolve_positions(p["prompt"], p["IO"], p["S"], tokenizer)
        except Exception:
            n_pos_fail += 1
            continue

        ids = tokenizer(p["prompt"], return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        tgt_id = tokenizer.encode(
            p["target_token"], add_special_tokens=False
        )[0]
        target_prob = float(probs[tgt_id])
        target_logit_centered = float(logits[tgt_id] - logits.mean())

        del ids, logits, probs

        if target_prob >= p_min:
            kept.append({
                "prompt":                      p["prompt"],
                "target_token":                p["target_token"],
                "distractor_token":            p["distractor_token"],
                "clean_target_prob":           target_prob,
                "clean_target_logit_centered": target_logit_centered,
                "metadata": {
                    "io_position":      roles["IO"],
                    "s1_position":      roles.get("S1"),
                    "s2_position":      roles.get("S2"),
                    "end_position":     roles["END"],
                    "target_positions": [roles["IO"],
                                         roles.get("S2", roles["IO"])],
                    "template_type":    p["template_type"],
                    "IO":               p["IO"],
                    "S":                p["S"],
                },
            })
        else:
            n_below += 1

        n_seen += 1
        # Drain accumulated activations every chunk_size forwards.
        if n_seen % chunk_size == 0:
            hook_manager.clear()
            torch.cuda.empty_cache()

    # Final cleanup before verify starts hammering the model.
    hook_manager.clear()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"  pool consumed: {n_seen}/{len(raw)}  "
          f"position-fail: {n_pos_fail}  below p_min: {n_below}  "
          f"kept: {len(kept)}/{n_prompts}")
    if len(kept) < n_prompts:
        print(f"  warning: only {len(kept)} prompts pass threshold "
              f"(asked for {n_prompts}); using all of them. "
              f"Consider raising --prompt-pool-size.")
    return kept


# ──────────────────────────────────────────────────────────────────────
# Single-circuit verify on a fixed prompt set
# ──────────────────────────────────────────────────────────────────────

def verify_circuit_on_prompts(
    *,
    model, tokenizer, hook_manager, device,
    num_layers: int, num_heads: int,
    universal_prompts: List[Dict[str, Any]],
    circuit: set,
    abc_seed: int,
    n_abc_refs: int,
) -> Dict[str, Any]:
    prepared = add_abc_references(
        universal_prompts, tokenizer,
        n_abc_refs=n_abc_refs,
        abc_seed_offset=abc_seed,
    )
    if not prepared:
        return {"n_prompts": 0}

    summary = verify(
        model, tokenizer, hook_manager, device, prepared,
        circuit=set(circuit), always_on=set(),
        num_layers=num_layers, num_heads=num_heads,
        verbose=False,
    )
    summary["circuit_size_input"] = len(circuit)
    summary["abc_seed"]           = abc_seed
    summary["n_abc_refs"]         = n_abc_refs
    return summary


# ──────────────────────────────────────────────────────────────────────
# Aggregation / printing
# ──────────────────────────────────────────────────────────────────────

def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per (model, variant), reduce list of (prompt_seed, abc_seed)
    summaries to mean ± std of faith and comp_drop."""
    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_key[(r["model"], r["variant"])].append(r)

    agg = []
    for (model, variant), seed_rows in sorted(by_key.items()):
        faith = [s["faith_ratio"]
                 for s in seed_rows if s.get("faith_ratio") is not None]
        comp = [s["comp_drop"]
                for s in seed_rows if s.get("comp_drop") is not None]
        sizes = [s["circuit_size"]
                 for s in seed_rows if s.get("circuit_size") is not None]
        n_pr = [s["n_prompts"]
                for s in seed_rows if s.get("n_prompts") is not None]

        def _ms(xs):
            if not xs:
                return (None, None)
            mean = statistics.mean(xs)
            std  = statistics.pstdev(xs) if len(xs) == 1 \
                                          else statistics.stdev(xs)
            return (mean, std)

        faith_mean, faith_std = _ms(faith)
        comp_mean,  comp_std  = _ms(comp)

        agg.append({
            "model":            model,
            "variant":          variant,
            "n_runs":           len(seed_rows),
            "circuit_size":     sizes[0] if sizes else None,
            "n_test_prompts":   n_pr[0] if n_pr else None,
            "faith_mean":       faith_mean,
            "faith_std":        faith_std,
            "comp_mean":        comp_mean,
            "comp_std":         comp_std,
            "per_run":          [
                {
                    "prompt_seed":  r.get("prompt_seed"),
                    "abc_seed":     r.get("abc_seed"),
                    "faith":        r.get("faith_ratio"),
                    "comp":         r.get("comp_drop"),
                }
                for r in seed_rows
            ],
        })
    return {"rows": agg}


def print_table(agg_rows: List[Dict[str, Any]]):
    print()
    print(f"  {'model':<14} {'variant':<14} {'|C|':>4}  "
          f"{'faith':>16}  {'comp':>16}  runs")
    print("  " + "-" * 76)
    for r in agg_rows:
        if r["faith_mean"] is None:
            faith_str = "    n/a"
        else:
            faith_str = f"{r['faith_mean']:+.3f} ± {r['faith_std']:.3f}"
        if r["comp_mean"] is None:
            comp_str = "    n/a"
        else:
            comp_str = f"{r['comp_mean']:+.3f} ± {r['comp_std']:.3f}"
        size = r["circuit_size"] if r["circuit_size"] is not None else "?"
        print(f"  {r['model']:<14} {r['variant']:<14} {str(size):>4}  "
              f"{faith_str:>16}  {comp_str:>16}  {r['n_runs']}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Re-verify saved circuits with a universal test "
                    "set per model, varying ABC seeds.",
    )
    ap.add_argument("--parent-dir", required=True,
                    help="dir containing <model>_<variant>/ subdirs")
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict to these model prefixes (e.g. "
                         "'gpt2_small pythia_1.4b'). Default: all.")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="restrict to these variant suffixes "
                         f"(default: {' '.join(DEFAULT_VARIANTS)})")
    ap.add_argument("--include-geo", action="store_true",
                    help="also include *_geo15 variants (off by default)")

    # Universal test set
    ap.add_argument("--n-prompts", type=int, default=100,
                    help="size of the universal test set (default 100)")
    ap.add_argument("--prompt-pool-size", type=int, default=300,
                    help="how many prompts to generate before filtering "
                         "by p_min. Should be 2-4x --n-prompts so enough "
                         "survive the filter (default 300).")
    ap.add_argument("--prompt-seeds", type=int, nargs="+",
                    default=[7],
                    help="prompt-generation seeds. seed=42 matches the "
                         "discovery seed used by run_ioi_sweep (in-sample); "
                         "any other value gives an OOS test set. Default "
                         "is [7] so verification is OOS by default.")

    # ABC corruption seeds
    ap.add_argument("--abc-seeds", type=int, nargs="+",
                    default=[10000, 20000, 30000, 40000, 50000],
                    help="ABC seed offsets (default: 5 seeds for variance)")
    ap.add_argument("--n-abc-refs", type=int, default=10)

    # Thresholds
    ap.add_argument("--p-min-gpt2", type=float, default=0.3)
    ap.add_argument("--p-min-pythia", type=float, default=0.1)

    # Model loading
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default=".")

    ap.add_argument("--circuit-filename", default="circuit_pc.json",
                    help="JSON file in each run dir containing the "
                         "circuit (default: circuit_pc.json)")

    # Output / control
    ap.add_argument("--out", default=None,
                    help="combined summary path (default: "
                         "<parent-dir>/seed_variance_summary.json)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip (run, prompt_seed, abc_seed) triples whose "
                         "verify file is already on disk")
    ap.add_argument("--dry-run", action="store_true",
                    help="discover runs and print the plan; don't load "
                         "models or verify")

    args = ap.parse_args()

    runs = discover_runs(
        args.parent_dir,
        models_filter=args.models,
        variants_filter=args.variants,
        include_geo=args.include_geo,
        circuit_filename=args.circuit_filename,
    )
    if not runs:
        sys.exit(f"no runs found under {args.parent_dir}")

    # Group by model so each model loads only once.
    by_model: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for model, variant, run_dir in runs:
        by_model[model].append((variant, run_dir))

    n_runs_total = sum(len(v) for v in by_model.values())
    n_combos = (n_runs_total
                * len(args.prompt_seeds)
                * len(args.abc_seeds))
    print(f"discovered {n_runs_total} runs across {len(by_model)} models")
    for model, runs_for_model in by_model.items():
        print(f"  {model:<14}: {len(runs_for_model)} variants "
              f"-> {[v for v, _ in runs_for_model]}")
    print(f"prompt seeds: {args.prompt_seeds}    "
          f"(n_prompts={args.n_prompts}, pool={args.prompt_pool_size})")
    print(f"abc seeds:    {args.abc_seeds}")
    print(f"total verifies: {n_combos}")

    if args.dry_run:
        return

    all_rows: List[Dict[str, Any]] = []

    for model, runs_for_model in by_model.items():
        # Load model once. Pull family/size/step from the first
        # variant's run_config.json.
        first_dir = runs_for_model[0][1]
        cfg, _ = load_run(first_dir)
        model_cfg = {"model": {**cfg, "device": args.device,
                               "cache_dir": args.cache_dir}}
        print()
        print("=" * 70)
        print(f"loading {model}  (family={cfg.get('family')} "
              f"size={cfg.get('size')} step={cfg.get('step')})")
        t0 = time.time()
        loaded_model, tokenizer, hook_manager, device = \
            load_model(model_cfg)
        num_layers = loaded_model.config.num_hidden_layers
        num_heads  = loaded_model.config.num_attention_heads
        print(f"  loaded in {time.time()-t0:.1f}s  "
              f"L={num_layers} H={num_heads}")

        p_min = p_min_for_model(model, args.p_min_gpt2, args.p_min_pythia)

        # ── Per prompt seed: build the universal test set, then
        #    iterate (variant, abc_seed) over it. ──────────────────
        for prompt_seed in args.prompt_seeds:
            print(f"\n--- prompt seed = {prompt_seed} ---")
            universal = build_universal_prompts(
                model=loaded_model, tokenizer=tokenizer,
                hook_manager=hook_manager, device=device,
                prompt_seed=prompt_seed,
                n_prompts=args.n_prompts,
                pool_size=args.prompt_pool_size,
                p_min=p_min,
            )
            if not universal:
                print(f"  no prompts in universal set; skipping prompt "
                      f"seed {prompt_seed}")
                continue

            for variant, run_dir in runs_for_model:
                print(f"\n  [{model}/{variant}]  {os.path.basename(run_dir)}")
                circuit = load_circuit(run_dir, args.circuit_filename)

                for abc_seed in args.abc_seeds:
                    out_path = os.path.join(
                        run_dir,
                        f"verify_p{prompt_seed}_abc{abc_seed}.json"
                    )
                    if args.skip_existing and os.path.exists(out_path):
                        with open(out_path) as f:
                            s = json.load(f)
                        print(f"    [cached]  prompt={prompt_seed} "
                              f"abc={abc_seed}  "
                              f"faith={s.get('faith_ratio')}")
                    else:
                        t = time.time()
                        s = verify_circuit_on_prompts(
                            model=loaded_model, tokenizer=tokenizer,
                            hook_manager=hook_manager, device=device,
                            num_layers=num_layers, num_heads=num_heads,
                            universal_prompts=universal,
                            circuit=set(circuit),
                            abc_seed=abc_seed,
                            n_abc_refs=args.n_abc_refs,
                        )
                        s["prompt_seed"] = prompt_seed
                        elapsed = time.time() - t
                        print(f"    prompt={prompt_seed} abc={abc_seed}  "
                              f"faith={s.get('faith_ratio')}  "
                              f"comp={s.get('comp_drop')}  "
                              f"|C|={s.get('circuit_size')}  "
                              f"({elapsed:.0f}s)")
                        with open(out_path, "w") as f:
                            json.dump(s, f, indent=2)

                    row = {**s,
                           "model":       model,
                           "variant":     variant,
                           "run_dir":     run_dir,
                           "prompt_seed": prompt_seed}
                    all_rows.append(row)

        # Drop the model from GPU before the next family loads.
        del loaded_model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ── Aggregate ────────────────────────────────────────────────────
    agg = aggregate(all_rows)

    print()
    print("=" * 70)
    print("Aggregate over (prompt_seed, abc_seed) per (model, variant)")
    print("=" * 70)
    print_table(agg["rows"])

    # If multiple prompt seeds, also break out per-prompt-seed for OOS
    # vs in-sample comparison.
    if len(args.prompt_seeds) > 1:
        print()
        print("=" * 70)
        print("Per prompt seed")
        print("=" * 70)
        for ps in args.prompt_seeds:
            sub_rows = [r for r in all_rows if r.get("prompt_seed") == ps]
            sub_agg = aggregate(sub_rows)
            tag = "in-sample" if ps == 42 else "OOS"
            print(f"\n--- prompt seed = {ps}  ({tag}) ---")
            print_table(sub_agg["rows"])

    out = {
        "parent_dir":   args.parent_dir,
        "prompt_seeds": args.prompt_seeds,
        "abc_seeds":    args.abc_seeds,
        "n_prompts":    args.n_prompts,
        "n_abc_refs":   args.n_abc_refs,
        "p_min_gpt2":   args.p_min_gpt2,
        "p_min_pythia": args.p_min_pythia,
        "aggregate":    agg,
        "all_rows":     all_rows,
    }
    out_path = args.out or os.path.join(args.parent_dir,
                                        "seed_variance_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()