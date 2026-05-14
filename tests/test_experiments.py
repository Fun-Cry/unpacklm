"""
Test the full experiment pipeline on GPT-2 small with 5 IOI prompts.

Steps:
  1. Discovery: trace 5 prompts, save per-prompt JSONs
  2. Selection: run partition_coverage to extract a circuit
  3. Verification: ABC-mean ablation faith & knockout (best-effort)

Usage:
    python tests/test_experiments.py --device cuda:0 --cache-dir /data/models

This takes ~1-2 minutes on GPU.
"""

import argparse
import json
import os
import sys
import tempfile
import shutil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--n-prompts", type=int, default=5)
    ap.add_argument("--keep-output", action="store_true",
                    help="don't delete temp output dir")
    args = ap.parse_args()

    out_dir = tempfile.mkdtemp(prefix="unpack_test_")
    discover_dir = os.path.join(out_dir, "discoveries")
    os.makedirs(discover_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # ================================================================
    #  Step 1: Discovery
    # ================================================================
    print("\n" + "=" * 60)
    print("  STEP 1: Discovery")
    print("=" * 60)

    from experiments.circuit_discovery.utils import load_model
    from experiments.circuit_discovery.discover import discover_one

    cfg = {
        "model": {
            "family": "gpt2",
            "size": "small",
            "device": args.device,
            "cache_dir": args.cache_dir,
        },
        "trace": {
            "beta": 0.8,
            "top_paths_k": 200,
            "path_min_frac": 1e-4,
        },
        "lens": {
            "type": "membership",
            "positions_from_metadata": "target_positions",
        },
    }

    print(f"Loading GPT-2 small...")
    model, tokenizer, adapter, device = load_model(cfg)

    print(f"Building {args.n_prompts} IOI prompts...")
    from experiments.circuit_discovery.ioi.prompts import build_prompts
    prompts = build_prompts(tokenizer)[:args.n_prompts]
    print(f"  Got {len(prompts)} prompts")

    n_ok = 0
    for idx, p in enumerate(prompts):
        print(f"  [{idx+1}/{len(prompts)}] {p['prompt'][:50]}...")
        try:
            d = discover_one(
                model, tokenizer, adapter,
                beta=cfg["trace"]["beta"],
                top_paths_k=cfg["trace"]["top_paths_k"],
                path_min_frac=cfg["trace"]["path_min_frac"],
                lens_cfg=cfg["lens"],
                prompt_dict=p,
            )
            d["prompt_idx"] = idx
            path = os.path.join(discover_dir, f"p{idx:04d}.json")
            with open(path, "w") as f:
                json.dump(d, f)
            print(f"    target_prob={d['clean_target_prob']:.4f}  "
                  f"paths={len(d['ranked_paths'])}  "
                  f"components={len(d['ranked_components'])}")
            n_ok += 1
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback; traceback.print_exc()

    assert n_ok > 0, "All discovery prompts failed!"
    print(f"\n  Discovery: {n_ok}/{len(prompts)} prompts OK")

    # ================================================================
    #  Step 2: Selection
    # ================================================================
    print("\n" + "=" * 60)
    print("  STEP 2: Selection (partition_coverage)")
    print("=" * 60)

    from experiments.circuit_discovery.selection._common import (
        load_run, resolve_role_keys, filter_correct, DEFAULT_EXCLUDE,
    )
    from experiments.circuit_discovery.selection import METHODS

    _cfg, run_prompts = load_run(discover_dir)
    print(f"  Loaded {len(run_prompts)} prompt JSONs")

    role_keys = resolve_role_keys("ioi")
    run_prompts = filter_correct(run_prompts, p_min=0.0)
    print(f"  After filtering: {len(run_prompts)} prompts")

    if len(run_prompts) == 0:
        print("  No prompts survived filtering — skipping selection")
        if not args.keep_output:
            shutil.rmtree(out_dir)
        return

    result = METHODS["partition_coverage"](
        run_prompts,
        role_keys=role_keys,
        exclude=set(DEFAULT_EXCLUDE),
        partition_threshold=0.05,
        min_prompt_fraction=0.0,  # relaxed for small N
        elbow_rescue=True,
        elbow_floor_k=1,
    )

    circuit = result.circuit
    print(f"\n  Selected circuit: {len(circuit)} components")
    heads = sorted(c for c in circuit if c.startswith('attn_'))
    mlps = sorted(c for c in circuit if c.startswith('mlp_'))
    print(f"  Heads ({len(heads)}): {heads}")
    print(f"  MLPs  ({len(mlps)}):  {mlps}")

    # Save circuit
    circuit_path = os.path.join(out_dir, "circuit.json")
    with open(circuit_path, "w") as f:
        json.dump(sorted(circuit), f, indent=2)

    # Test unpack.Circuit wrapper
    from unpack import Circuit
    c = Circuit.from_components(circuit, model_name="gpt2")
    print(f"  As Circuit object: {c}")

    assert len(circuit) > 0, "Empty circuit!"

    # ================================================================
    #  Step 3: Verification (best-effort)
    # ================================================================
    print("\n" + "=" * 60)
    print("  STEP 3: Verification (ABC-mean ablation)")
    print("=" * 60)

    try:
        from experiments.circuit_discovery.verification.__main__ import (
            verify, print_summary,
        )
        from experiments.circuit_discovery.ioi.abc_prep import add_abc_references
        from experiments.circuit_discovery.ioi.prompts import _resolve_positions
        from utils.load_data import load_ioi_with_abc

        # Build verification prompts (seed=7, different from discovery seed=42)
        verify_raw = load_ioi_with_abc(
            tokenizer, n_prompts=args.n_prompts, seed=7, n_abc_refs=3,
        )

        verify_prompts = []
        for vp in verify_raw:
            roles = _resolve_positions(
                vp["prompt"], vp["IO"], vp["S"], tokenizer)
            if roles is None:
                continue
            verify_prompts.append({
                "prompt": vp["prompt"],
                "target_token": vp["IO"],
                "distractor_token": vp["S"],
                "references": vp.get("abc_prompts", []),
            })

        print(f"  Built {len(verify_prompts)} verification prompts")
        if verify_prompts:
            num_layers = adapter.get_num_layers()
            num_heads = adapter.get_num_heads()
            summary = verify(
                model, tokenizer, adapter, device,
                verify_prompts,
                circuit=circuit,
                always_on=set(),
                num_layers=num_layers,
                num_heads=num_heads,
                max_prompts=args.n_prompts,
            )
            print_summary(summary)
        else:
            print("  Could not build verification prompts — skipping")

    except Exception as e:
        print(f"  Verification error: {e}")
        import traceback; traceback.print_exc()
        print("  (Steps 1-2 passed; verification is best-effort here)")

    # ================================================================
    print("\n" + "=" * 60)
    if not args.keep_output:
        shutil.rmtree(out_dir)
        print(f"  Cleaned up {out_dir}")
    else:
        print(f"  Output kept at {out_dir}")
    print("  EXPERIMENT PIPELINE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
