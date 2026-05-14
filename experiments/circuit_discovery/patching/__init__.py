"""Path patching for circuit validation.

Two intervention styles:

  1. SIMPLE (build_path_patch_intervention): single receiver-side hook
     adding (corrupted_sender - clean_sender) at the receiver layer's
     ln_input. Equivalent to McDougall's direct_includes_mlps=False.
     Layer-level granularity — same-layer receivers give identical Δ.

  2. QK SCORE-LEVEL (capture_path_patch_inputs + build_qk_score_patch):
     per-head, per-channel granularity. Captures clean+corrupted
     activations once, then builds many interventions analytically
     (no extra forward passes). Tests K-channel claims specifically.

Common runtime utilities:
  baseline_logit, run_with_intervention.

Style 2 usage::

    inputs = capture_path_patch_inputs(
        model, tok, hm, clean_prompt, corrupted_prompt,
        sender_specs   = [("mlp_0", io_pos), ("mlp_0", s2_pos), ...],
        receiver_specs = [(9, 9), (9, 6), (10, 7), ...],
    )
    base = baseline_logit(model, tok, hm, clean_prompt, target, distractor)
    for spec in edges:
        iv = build_qk_score_patch(inputs, hm, ...spec...)
        delta = run_with_intervention(model, tok, hm, clean_prompt,
                                      target, distractor, iv, base)
"""

from .intervention      import build_path_patch_intervention
from .run               import baseline_logit, run_with_intervention
from .qk_intervention   import (
    PathPatchInputs,
    capture_path_patch_inputs,
    build_qk_score_patch,
    add_at_head_slice,
    replace_at_head_slice,
)

__all__ = [
    "build_path_patch_intervention",
    "baseline_logit", "run_with_intervention",
    "PathPatchInputs",
    "capture_path_patch_inputs",
    "build_qk_score_patch",
    "add_at_head_slice",
    "replace_at_head_slice",
]