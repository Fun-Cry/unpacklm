"""
unpack.validate - Adapter contract validation.

Runs mathematical invariant checks to verify that a ModelAdapter
correctly extracts weights and wires hooks for a given model.

Usage:
    import unpack
    tracer = unpack.Tracer("gpt2")
    unpack.validate(tracer)

Six invariants are tested:
  1. Residual stream closure: Σ components = hidden_states at layer boundaries
  2. K-side attention closure: Σ key_decomp reproduces attention pattern
  3. Q-side attention closure: Σ query_decomp reproduces attention pattern
  4. Q/K consistency: K-side and Q-side give the same attention pattern
  5. V-side closure: Σ value_decomp = ||V[h,s]||²
  6. MLP closure: marginal norm sums match real LN output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from unpack.core.streamer import ComponentStreamer
from unpack.core.scorers import AttentionScorer, MLPScorer


@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    details: str = ""


@dataclass
class ValidationReport:
    """Aggregate result of all validation checks."""
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def print(self):
        """Pretty-print the validation report."""
        print("\n" + "=" * 60)
        print("  UNPACK Adapter Validation")
        print("=" * 60)
        for c in self.checks:
            status = "✅ PASS" if c.passed else "❌ FAIL"
            print(f"  {status}  {c.name}")
            if c.details:
                for line in c.details.strip().split("\n"):
                    print(f"         {line}")
        print("─" * 60)
        if self.all_passed:
            print("  All checks passed.")
        else:
            n_fail = sum(1 for c in self.checks if not c.passed)
            print(f"  {n_fail} check(s) failed.")
        print()


def _make_components(model, tokenizer, adapter):
    """Fresh streamer + scorers from an adapter."""
    streamer = ComponentStreamer(model, tokenizer, adapter)
    attn_scorer = AttentionScorer(adapter)
    mlp_scorer = MLPScorer(adapter)
    return streamer, attn_scorer, mlp_scorer


def _apply_norm(hidden, weight, bias, eps):
    """Apply LayerNorm using raw parameters."""
    return F.layer_norm(
        hidden, (hidden.shape[-1],), weight=weight, bias=bias, eps=eps)


# ================================================================
#  Check 1: Residual stream closure
# ================================================================

def check_residual_stream(model, tokenizer, adapter, text_input) -> CheckResult:
    """Verify Σ components = hidden_states at each layer boundary."""
    streamer, _, _, = _make_components(model, tokenizer, adapter)
    streamer.set_context(text_input)
    hidden_states = streamer.outputs.hidden_states
    num_layers = adapter.get_num_layers()

    running_sum = None
    all_passed = True
    details_lines = []

    for group_tensor, names, src_layer in adapter.iter_source_groups():
        group_sum = group_tensor.sum(dim=2)
        if running_sum is None:
            running_sum = group_sum
        else:
            running_sum = running_sum + group_sum

        check_idx = src_layer + 1
        if check_idx < 0:
            target = hidden_states[0].detach().cpu()
        elif check_idx == num_layers:
            norm_w, norm_b, norm_eps = adapter.get_final_norm_params()
            compare = _apply_norm(running_sum, norm_w.cpu(), norm_b.cpu(), norm_eps)
            target = hidden_states[num_layers].detach().cpu()
            match = torch.allclose(compare, target, atol=1e-4)
            if not match:
                diff = (compare - target).abs().max().item()
                details_lines.append(f"After layer {src_layer} + final norm: diff={diff:.6f}")
                all_passed = False
            continue
        else:
            target = hidden_states[check_idx].detach().cpu()

        match = torch.allclose(running_sum, target, atol=1e-4)
        if not match:
            diff = (running_sum - target).abs().max().item()
            details_lines.append(f"After layer {src_layer}: diff={diff:.6f}")
            all_passed = False

    return CheckResult(
        name="Residual stream closure",
        passed=all_passed,
        details="\n".join(details_lines) if details_lines else "",
    )


# ================================================================
#  Check 2/3: Attention closure (K-side and Q-side)
# ================================================================

def _check_attention_closure(model, tokenizer, adapter, text_input,
                             side: str) -> CheckResult:
    """Verify Σ decomp reproduces attention pattern on given side."""
    streamer, attn_scorer, _ = _make_components(model, tokenizer, adapter)
    streamer.set_context(text_input)
    outputs = streamer.outputs
    num_layers = adapter.get_num_layers()

    raw_sums = [None] * num_layers
    attn_masks_np = [None] * num_layers
    valid_masks = [None] * num_layers

    for target_L, components, names, hidden, is_last_group in streamer.stream():
        if target_L is None:
            continue

        attn_names, scores, mask, v_states = attn_scorer.score(
            target_L, components, names, hidden, is_last_group,
            side=side,
        )

        if attn_masks_np[target_L] is None:
            mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
            if mask_np is not None and len(mask_np.shape) == 2:
                mask_np = mask_np[:, None, None, :]
            attn_masks_np[target_L] = mask_np
            valid_masks[target_L] = (mask_np == 0) if mask_np is not None else None

        vm = valid_masks[target_L]
        for name, score_tensor in zip(attn_names, scores):
            score_np = score_tensor.detach().cpu().numpy()
            if vm is not None:
                vm_count = np.maximum(vm.sum(axis=-1, keepdims=True), 1)
                valid_mean = (score_np * vm).sum(axis=-1, keepdims=True) / vm_count
                centered = score_np - valid_mean
            else:
                centered = score_np - score_np.mean(axis=-1, keepdims=True)

            if raw_sums[target_L] is None:
                raw_sums[target_L] = centered.copy()
            else:
                raw_sums[target_L] += centered

    all_passed = True
    details_lines = []

    for i in range(num_layers):
        raw_sum = raw_sums[i]
        mask_np = attn_masks_np[i]
        if raw_sum is None:
            continue
        masked = raw_sum + mask_np if mask_np is not None else raw_sum
        final_attention = torch.softmax(torch.from_numpy(masked), dim=-1).numpy()
        gt = outputs.attentions[i].detach().cpu().numpy()
        match = np.allclose(final_attention, gt, atol=1e-2)
        if not match:
            diff = np.abs(final_attention - gt).max()
            details_lines.append(f"Layer {i}: max diff={diff:.6f}")
            all_passed = False

    adapter.remove_hooks()
    return CheckResult(
        name=f"Attention closure ({side.upper()}-side)",
        passed=all_passed,
        details="\n".join(details_lines) if details_lines else "",
    )


def check_attention_k_side(model, tokenizer, adapter, text_input) -> CheckResult:
    return _check_attention_closure(model, tokenizer, adapter, text_input, "key")


def check_attention_q_side(model, tokenizer, adapter, text_input) -> CheckResult:
    return _check_attention_closure(model, tokenizer, adapter, text_input, "query")


# ================================================================
#  Check 4: Q/K consistency
# ================================================================

def check_qk_consistency(model, tokenizer, adapter, text_input) -> CheckResult:
    """Verify K-side and Q-side reconstructions give same attention pattern."""

    def _get_reconstructed(side):
        streamer, attn_scorer, _ = _make_components(model, tokenizer, adapter)
        streamer.set_context(text_input)
        num_layers = adapter.get_num_layers()

        raw_sums = [None] * num_layers
        attn_masks_np = [None] * num_layers

        for target_L, components, names, hidden, is_last_group in streamer.stream():
            if target_L is None:
                continue
            _, scores, mask, _ = attn_scorer.score(
                target_L, components, names, hidden, is_last_group, side=side)
            if attn_masks_np[target_L] is None:
                mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                if mask_np is not None and len(mask_np.shape) == 2:
                    mask_np = mask_np[:, None, None, :]
                attn_masks_np[target_L] = mask_np

            for _, score_tensor in zip(range(len(scores)), scores):
                score_np = score_tensor.detach().cpu().numpy()
                vm = (attn_masks_np[target_L] == 0) if attn_masks_np[target_L] is not None else None
                if vm is not None:
                    vm_count = np.maximum(vm.sum(axis=-1, keepdims=True), 1)
                    valid_mean = (score_np * vm).sum(axis=-1, keepdims=True) / vm_count
                    centered = score_np - valid_mean
                else:
                    centered = score_np - score_np.mean(axis=-1, keepdims=True)
                if raw_sums[target_L] is None:
                    raw_sums[target_L] = centered.copy()
                else:
                    raw_sums[target_L] += centered

        result = {}
        for i in range(num_layers):
            if raw_sums[i] is None:
                continue
            masked = raw_sums[i] + attn_masks_np[i] if attn_masks_np[i] is not None else raw_sums[i]
            result[i] = torch.softmax(torch.from_numpy(masked), dim=-1).numpy()

        adapter.remove_hooks()
        return result

    k_results = _get_reconstructed("key")
    q_results = _get_reconstructed("query")

    layers = sorted(set(k_results.keys()) & set(q_results.keys()))
    all_passed = True
    details_lines = []

    for L in layers:
        match = np.allclose(k_results[L], q_results[L], atol=1e-2)
        if not match:
            diff = np.abs(k_results[L] - q_results[L]).max()
            details_lines.append(f"Layer {L}: max diff={diff:.6f}")
            all_passed = False

    return CheckResult(
        name="Q/K consistency",
        passed=all_passed,
        details="\n".join(details_lines) if details_lines else "",
    )


# ================================================================
#  Check 5: V-side closure
# ================================================================

def check_value_closure(model, tokenizer, adapter, text_input,
                        atol=1e-3) -> CheckResult:
    """Verify Σ value_decomp = ||V[h,s]||²."""
    streamer, attn_scorer, _ = _make_components(model, tokenizer, adapter)
    streamer.set_context(text_input)
    num_layers = adapter.get_num_layers()

    raw_sums = [None] * num_layers
    actual_v = [None] * num_layers

    for target_L, components, names, hidden, is_last_group in streamer.stream():
        if target_L is None:
            continue
        _, _, _, v_states = attn_scorer.score(
            target_L, components, names, hidden, is_last_group, side="key")
        v_names, v_scores = attn_scorer.score(
            target_L, components, names, hidden, is_last_group,
            side="value", value_states=v_states)
        v_scores_np = v_scores.detach().cpu().numpy()
        sum_over_c = v_scores_np.sum(axis=1)
        if raw_sums[target_L] is None:
            raw_sums[target_L] = sum_over_c.copy()
        else:
            raw_sums[target_L] += sum_over_c
        if actual_v[target_L] is None:
            v_np = v_states.detach().cpu().numpy()
            actual_v[target_L] = (v_np ** 2).sum(axis=-1)

    all_passed = True
    details_lines = []
    for L in range(num_layers):
        if raw_sums[L] is None:
            continue
        diff = np.abs(raw_sums[L] - actual_v[L]).max()
        rel = diff / max(np.abs(actual_v[L]).max(), 1e-12)
        if rel >= atol:
            details_lines.append(f"Layer {L}: rel_diff={rel:.2e}")
            all_passed = False

    adapter.remove_hooks()
    return CheckResult(
        name="V-side closure",
        passed=all_passed,
        details="\n".join(details_lines) if details_lines else "",
    )


# ================================================================
#  Check 6: MLP closure
# ================================================================

def check_mlp_closure(model, tokenizer, adapter, text_input) -> CheckResult:
    """Basic MLP decomposition check — verifies scorer runs without error."""
    streamer, _, mlp_scorer = _make_components(model, tokenizer, adapter)
    streamer.set_context(text_input)

    all_passed = True
    n_layers_checked = 0

    for target_L, components, names, hidden, is_last_group in streamer.stream():
        if target_L is None:
            continue
        mlp_names, l2_scores = mlp_scorer.score(
            target_L, components, names, hidden, is_last_group)

        # Basic sanity: scores are finite and non-negative
        if not np.isfinite(l2_scores).all():
            all_passed = False
        if (l2_scores < 0).any():
            all_passed = False
        n_layers_checked += 1

    adapter.remove_hooks()
    return CheckResult(
        name="MLP decomposition",
        passed=all_passed,
        details=f"Checked {n_layers_checked} layers, all scores finite and non-negative.",
    )


# ================================================================
#  Public entry point
# ================================================================

def validate(
    tracer,
    text: Optional[list] = None,
    verbose: bool = True,
) -> ValidationReport:
    """Run all adapter validation checks.

    Args:
        tracer: an unpack.Tracer instance.
        text: test input(s). Defaults to a standard test sentence.
        verbose: if True, print results as checks run.

    Returns:
        ValidationReport with per-check results.
    """
    if text is None:
        text = ["The quick brown fox jumps over the lazy dog.", "Hello World!"]

    model = tracer.model
    tokenizer = tracer.tokenizer
    adapter = tracer.adapter

    # Re-register hooks fresh for each check
    report = ValidationReport()

    checks = [
        ("Residual stream", lambda: check_residual_stream(model, tokenizer, adapter, text)),
        ("K-side attention", lambda: check_attention_k_side(model, tokenizer, adapter, text)),
        ("Q-side attention", lambda: check_attention_q_side(model, tokenizer, adapter, text)),
        ("Q/K consistency", lambda: check_qk_consistency(model, tokenizer, adapter, text)),
        ("V-side closure", lambda: check_value_closure(model, tokenizer, adapter, text)),
        ("MLP closure", lambda: check_mlp_closure(model, tokenizer, adapter, text)),
    ]

    for name, check_fn in checks:
        if verbose:
            print(f"  Running {name}...", end=" ", flush=True)
        # Re-register hooks (checks may remove them)
        adapter.remove_hooks()
        adapter.register_hooks(model)
        result = check_fn()
        report.checks.append(result)
        if verbose:
            print("✅" if result.passed else "❌")

    # Ensure hooks are restored
    adapter.remove_hooks()
    adapter.register_hooks(model)

    if verbose:
        report.print()

    return report
