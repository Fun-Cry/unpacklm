"""
Validate test: run all 6 adapter contract checks on tiny models.

Verifies that the mathematical invariants hold for both GPT-2
(sequential residual) and Pythia (parallel residual) adapters.

Usage:
    python tests/test_validate.py
"""

import sys
from tests.helpers import build_tiny_gpt2, build_tiny_pythia
import unpack


def test_validate_arch(arch_name, model, tokenizer):
    print(f"\n── Validating {arch_name} adapter ──")
    tracer = unpack.Tracer(model=model, tokenizer=tokenizer)
    report = unpack.validate(tracer, verbose=True)
    return report.all_passed


def main():
    print("=" * 60)
    print("  VALIDATE TEST: adapter contract checks")
    print("=" * 60)

    all_ok = True

    model_gpt2, tok_gpt2 = build_tiny_gpt2()
    all_ok &= test_validate_arch("GPT-2", model_gpt2, tok_gpt2)

    model_pythia, tok_pythia = build_tiny_pythia()
    all_ok &= test_validate_arch("Pythia", model_pythia, tok_pythia)

    print("\n" + "=" * 60)
    if all_ok:
        print("  ALL VALIDATION CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
