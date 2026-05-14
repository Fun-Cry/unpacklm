"""
Smoke test: verify trace runs end-to-end on tiny random-init models.

Tests both architectures (GPT-2 sequential, Pythia parallel) through
both the Tracer API and the compat trace_flow function. No model
downloads needed — uses random-init tiny models.

Usage:
    python tests/test_smoke.py
"""

import sys
import numpy as np

from tests.helpers import build_tiny_gpt2, build_tiny_pythia

import unpack
from unpack.models import get_adapter
from unpack.compat import trace_flow


PROMPTS = [
    ("the quick brown fox", None, None),
    ("a b c d e f g",       None, None),
    ("when mary and john",  None, None),
]


def test_tracer_api(arch_name, model, tokenizer):
    """Test the Tracer class API."""
    print(f"\n  Tracer API on {arch_name}:")
    tracer = unpack.Tracer(model=model, tokenizer=tokenizer)
    ok = True

    for text, target, distractor in PROMPTS:
        result = tracer.trace(text, target=target, distractor=distractor,
                              config="default")

        ta = result.token_attribution
        finite = np.isfinite(ta).all()
        pos_mass = ta[ta > 0].sum()
        n_paths = len(result.paths)
        n_flow = sum(1 for v in result.component_flow.values() if abs(v) > 1e-10)

        status = "OK" if finite and pos_mass > 50 and n_paths > 0 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"    [{status}] '{text[:30]}'  "
              f"+mass={pos_mass:.1f}%  paths={n_paths}  comps={n_flow}")

        # Check Path objects are well-formed
        if n_paths > 0:
            p = result.paths[0]
            assert isinstance(p.chain, str), f"chain should be str, got {type(p.chain)}"
            assert isinstance(p.score, float), f"score should be float"
            assert p.depth >= 0, f"depth should be >= 0"

    # Test print methods don't crash
    result.print_tokens()
    result.print_paths(top_k=5)
    result.print_components(top_k=5)

    return ok


def test_compat_api(arch_name, model, tokenizer):
    """Test the compat trace_flow function."""
    print(f"\n  compat.trace_flow on {arch_name}:")
    adapter = get_adapter(model)
    adapter.register_hooks(model)
    ok = True

    for text, target, distractor in PROMPTS:
        res = trace_flow(
            model, tokenizer, text,
            target_token=target, distractor_token=distractor,
            hook_manager=adapter, beta=0.8, top_paths_k=10,
        )

        ta = np.asarray(res["token_attribution"], dtype=np.float64)
        cf = res["component_flow"]
        paths = res["top_paths"]

        finite = np.isfinite(ta).all()
        pos_mass = ta[ta > 0].sum()
        n_flow = sum(1 for v in cf.values() if np.any(np.abs(v) > 1e-10))

        status = "OK" if finite and pos_mass > 50 and len(paths) > 0 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"    [{status}] '{text[:30]}'  "
              f"+mass={pos_mass:.1f}%  paths={len(paths)}  comps={n_flow}")

    adapter.remove_hooks()
    return ok


def test_configs(model, tokenizer):
    """Test all five config presets run without error."""
    print(f"\n  Config presets:")
    tracer = unpack.Tracer(model=model, tokenizer=tokenizer)
    text = "the quick brown fox"
    ok = True

    for name in unpack.PRESETS:
        try:
            result = tracer.trace(text, config=name)
            n_paths = len(result.paths)
            print(f"    [OK]   {name:<20} paths={n_paths}")
        except Exception as e:
            print(f"    [FAIL] {name:<20} {e}")
            ok = False
    return ok


def main():
    print("=" * 60)
    print("  SMOKE TEST: tiny random-init models")
    print("=" * 60)

    all_ok = True

    # GPT-2 (sequential residual)
    print("\n── GPT-2 (sequential residual) ──")
    model_gpt2, tok_gpt2 = build_tiny_gpt2()
    all_ok &= test_tracer_api("gpt2", model_gpt2, tok_gpt2)
    all_ok &= test_compat_api("gpt2", model_gpt2, tok_gpt2)
    all_ok &= test_configs(model_gpt2, tok_gpt2)

    # Pythia (parallel residual)
    print("\n── Pythia (parallel residual) ──")
    model_pythia, tok_pythia = build_tiny_pythia()
    all_ok &= test_tracer_api("pythia", model_pythia, tok_pythia)
    all_ok &= test_compat_api("pythia", model_pythia, tok_pythia)
    all_ok &= test_configs(model_pythia, tok_pythia)

    print("\n" + "=" * 60)
    if all_ok:
        print("  ALL SMOKE TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
