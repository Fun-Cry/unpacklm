"""Run a forward pass under an intervention and report the target logit.

The path-patching workflow is:

    iv = build_path_patch_intervention(...)
    delta = run_with_intervention(model, tok, hm, prompt, target,
                                  distractor, iv, baseline_logit_centered)

`baseline_logit_centered` is the clean target_logit_centered — produce
once for the prompt (without intervention) before running many edge
patches against it; we just compute deltas relative to that.

This is intentionally minimal: no trace, no decomposition, no path
extraction. One forward pass, two logit reads, one subtraction.
"""

from typing import Optional

import torch

from experiments.ablation_tracing.core.trace import Intervention


# ──────────────────────────────────────────────────────────────────────
def _target_logit_centered(model, tokenizer, prompt, target_token,
                            distractor_token, device):
    """Run a forward pass and return target_logit - distractor_logit
    at the final position (or just target_logit when no distractor).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :]   # (V,)

    target_id = tokenizer(target_token, add_special_tokens=False)["input_ids"][0]
    target_logit = float(logits[target_id])

    if distractor_token is not None:
        distractor_id = tokenizer(distractor_token,
                                  add_special_tokens=False)["input_ids"][0]
        return target_logit - float(logits[distractor_id])
    return target_logit


def baseline_logit(model, tokenizer, hook_manager, prompt: str,
                   target_token: str, distractor_token: Optional[str] = None,
                   device=None) -> float:
    """Compute the centered target logit on `prompt` under no
    intervention. Run this once per prompt; pass into `run_with_intervention`
    as the reference for all edge patches against this prompt.
    """
    if device is None:
        device = next(model.parameters()).device
    hook_manager.clear()
    hook_manager.clear_interventions()
    return _target_logit_centered(model, tokenizer, prompt,
                                   target_token, distractor_token, device)


def run_with_intervention(
    model,
    tokenizer,
    hook_manager,
    prompt: str,
    target_token: str,
    distractor_token: Optional[str],
    intervention: Intervention,
    baseline: float,
    device=None,
) -> float:
    """Run one forward pass with `intervention` installed and return
    the centered Δ-logit relative to `baseline`.

    Δ = (centered logit under patch) − (baseline centered logit)

    Negative Δ means patching the edge degraded the target's
    advantage over the distractor — i.e., the edge was load-bearing.
    """
    if device is None:
        device = next(model.parameters()).device

    hook_manager.clear()
    hook_manager.clear_interventions()
    for hook_name, fn in intervention.interventions:
        hook_manager.register_intervention(hook_name, fn)
    try:
        patched = _target_logit_centered(model, tokenizer, prompt,
                                          target_token, distractor_token,
                                          device)
    finally:
        hook_manager.clear_interventions()

    return patched - baseline