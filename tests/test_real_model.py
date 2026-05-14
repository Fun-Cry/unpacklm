"""
Real model test: full pipeline on GPT-2 small.

Requires model download (~500MB). Tests:
  1. Tracer API with factual recall
  2. Tracer API with IOI (contrastive logit-diff)
  3. All five config presets
  4. Re-rooting
  5. Circuit save/load roundtrip
  6. Adapter validation

Usage:
    python tests/test_real_model.py [--device cuda]
"""

import argparse
import sys
import tempfile
import os

import numpy as np
import unpack


def test_factual(tracer):
    """Factual recall: "The capital of France is" → " Paris"."""
    print("\n── Factual recall ──")
    result = tracer.trace("The capital of France is", target=" Paris")

    print(f"  target: {result.target_token!r}  prob: {result.target_prob:.4f}")
    assert result.target_prob > 0.001, f"target prob suspiciously low: {result.target_prob}"
    assert len(result.paths) > 0, "no paths extracted"
    assert np.isfinite(result.token_attribution).all(), "non-finite attribution"

    pos_mass = result.token_attribution[result.token_attribution > 0].sum()
    print(f"  +mass: {pos_mass:.1f}%  paths: {len(result.paths)}  "
          f"comps: {len(result.component_flow)}")

    result.print_tokens()
    result.print_paths(top_k=10)
    print("  [OK]")
    return True


def test_ioi(tracer):
    """IOI contrastive: Mary vs John."""
    print("\n── IOI (contrastive) ──")
    result = tracer.trace(
        "Mary and John went to the store. John gave the bag to",
        target=" Mary",
        distractor=" John",
    )

    print(f"  target: {result.target_token!r}  prob: {result.target_prob:.4f}")
    assert len(result.paths) > 0, "no paths"

    # Check that known IOI heads show up in component flow
    flow = result.component_flow
    top_comps = sorted(flow.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
    print(f"  Top 10 components by |flow|:")
    for name, val in top_comps:
        wang_role = unpack.WANG_CIRCUIT_HEADS.get(name, "")
        tag = f"  [{wang_role}]" if wang_role else ""
        print(f"    {val:>+8.4f}  {name}{tag}")

    result.print_paths(top_k=10)
    print("  [OK]")
    return True


def test_all_configs(tracer):
    """All five paper configs run without error."""
    print("\n── Config presets ──")
    text = "Mary and John went to the store. John gave the bag to"
    ok = True
    for name in unpack.PRESETS:
        try:
            result = tracer.trace(text, target=" Mary", distractor=" John",
                                  config=name)
            print(f"  [OK]   {name:<20} paths={len(result.paths)}")
        except Exception as e:
            print(f"  [FAIL] {name:<20} {e}")
            ok = False
    return ok


def test_reroot(tracer):
    """Re-root attribution at a specific component."""
    print("\n── Re-root ──")
    result = tracer.trace(
        "Mary and John went to the store. John gave the bag to",
        target=" Mary", distractor=" John",
        root="attn_9_head_9",
    )
    assert result.root.startswith("attn_9_head_9"), f"wrong root: {result.root}"
    assert len(result.paths) > 0, "no paths from re-rooted trace"
    print(f"  root: {result.root}  paths: {len(result.paths)}")
    result.print_paths(top_k=5)
    print("  [OK]")
    return True


def test_circuit_roundtrip():
    """Circuit save/load/set-ops."""
    print("\n── Circuit roundtrip ──")
    c = unpack.Circuit.from_components(
        {"attn_9_head_9", "attn_7_head_3", "mlp_5"},
        model_name="gpt2", faith=0.84, knockout=0.93,
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    c.save(path)
    c2 = unpack.Circuit.load(path)
    os.unlink(path)

    assert c2.components == c.components
    assert c2.faith == 0.84
    assert c2.model_name == "gpt2"
    print(f"  {c}")
    print(f"  heads: {c.heads}")
    print(f"  mlps: {c.mlps}")
    print("  [OK]")
    return True


def test_validate(tracer):
    """Run adapter validation checks."""
    print("\n── Adapter validation ──")
    report = unpack.validate(tracer, verbose=True)
    return report.all_passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  REAL MODEL TEST: GPT-2 small")
    print("=" * 60)

    print("\nLoading GPT-2 small...")
    tracer = unpack.Tracer("gpt2", device=args.device,
                           cache_dir=args.cache_dir)
    print(f"  {tracer}")

    all_ok = True
    all_ok &= test_factual(tracer)
    all_ok &= test_ioi(tracer)
    all_ok &= test_all_configs(tracer)
    all_ok &= test_reroot(tracer)
    all_ok &= test_circuit_roundtrip()
    all_ok &= test_validate(tracer)

    print("\n" + "=" * 60)
    if all_ok:
        print("  ALL REAL MODEL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()