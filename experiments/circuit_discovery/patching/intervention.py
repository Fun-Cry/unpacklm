"""Build a path-patching Intervention.

Path patching tests whether information flowing along a *specific*
sender->receiver edge matters for the target. Unlike full ablation
(which replaces the sender's output everywhere), path patching only
modifies what the receiver reads from the sender, leaving the rest
of the network's view of the sender's output intact.

Implementation: at the receiver's layer-norm input, modify the
residual stream at `receiver_pos` by `corrupted_sender - clean_sender`.
This is mathematically equivalent to "let the sender produce its
clean output, but show the receiver the corrupted version." The
sender's output to all *other* downstream consumers is unchanged.

Sender contributions are extracted via the hook manager's
`iter_source_groups`, the same machinery the trace uses to enumerate
per-component residual contributions. This works uniformly across
component types (attention heads, MLPs, embeddings) without
component-specific extraction code.

The receiver-side hook uses `core.add_at`, the same intervention
template family used for ablation (`replace_at`, `mute`, etc.).
"""

import torch

from unpack._interventions import add_at
from experiments.ablation_tracing.core.trace import Intervention


# ──────────────────────────────────────────────────────────────────────
# Sender contribution extraction
# ──────────────────────────────────────────────────────────────────────

def _capture_sender_contribution(model, tokenizer, hook_manager,
                                  prompt: str, sender_name: str,
                                  sender_pos: int, device) -> torch.Tensor:
    """Run a forward pass on `prompt` and extract the sender's residual
    stream contribution at `sender_pos`.

    Returns a (d_model,) tensor.

    Implementation: walks the hook manager's source groups looking for
    `sender_name`. Each group yielded by `iter_source_groups` contains
    a (B, S, C, D) tensor and a `names` list. The slice at the matching
    name index, batch 0, position sender_pos is the d_model contribution
    we want.
    """
    # Re-register hooks if they were removed (trace_flow.py calls
    # remove_hooks at the end of each trace).
    if not hook_manager.handles:
        hook_manager.register_hooks(model)

    hook_manager.clear()
    hook_manager.clear_interventions()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)

    for group_tensor, names, _src_layer in hook_manager.iter_source_groups():
        if sender_name in names:
            idx = names.index(sender_name)
            return group_tensor[0, sender_pos, idx, :].detach().clone().to(device)

    raise ValueError(f"sender component {sender_name!r} not found in any "
                     f"source group on prompt {prompt!r}")


# ──────────────────────────────────────────────────────────────────────
# Receiver hook
# ──────────────────────────────────────────────────────────────────────

def _receiver_hook_name(receiver_name: str) -> str:
    """The hook installed at the receiver's layer-norm input.

    The hook fires on the residual tensor read by the receiver's
    layer norm; modifying it modifies what the receiver computes
    without affecting any other consumer.
    """
    if receiver_name.startswith("mlp_"):
        L = int(receiver_name.split("_")[1])
        return f"mlp_ln_{L}_input"
    if receiver_name.startswith("attn_"):
        L = int(receiver_name.split("_")[1])
        return f"attn_ln_{L}_input"
    raise ValueError(
        f"receiver must be attn or mlp; got {receiver_name}"
    )


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def build_path_patch_intervention(
    model,
    tokenizer,
    hook_manager,
    clean_prompt: str,
    corrupted_prompt: str,
    sender_name: str,
    sender_pos: int,
    receiver_name: str,
    receiver_pos: int,
    device=None,
) -> Intervention:
    """Build an Intervention that path-patches the (sender, receiver) edge.

    Captures the sender's residual contribution at `sender_pos` on
    both the clean and corrupted prompts (two forward passes). The
    returned Intervention installs a single hook at the receiver's
    layer-norm input that, when run on the clean prompt, adds
    (corrupted_sender − clean_sender) to the residual stream at
    `receiver_pos` — replacing what the receiver reads from the
    sender at that position with the corrupted-prompt value.

    Notes:
      - The clean and corrupted prompts must tokenize to the same
        length (so sender_pos has the same meaning in both).
      - Composing multiple path-patch Interventions is supported by
        Intervention's hook-list semantics, but the math of
        simultaneously patching multiple senders into the same
        receiver is whatever the additive deltas imply — usually
        what you want, but worth checking if you do it.
    """
    if device is None:
        device = next(model.parameters()).device

    # Capture sender contribution under both prompts.
    clean_contrib = _capture_sender_contribution(
        model, tokenizer, hook_manager, clean_prompt,
        sender_name, sender_pos, device,
    )
    corrupted_contrib = _capture_sender_contribution(
        model, tokenizer, hook_manager, corrupted_prompt,
        sender_name, sender_pos, device,
    )

    delta = corrupted_contrib - clean_contrib   # (D,)
    hook_name = _receiver_hook_name(receiver_name)
    fn = add_at(delta, positions=[receiver_pos])

    return Intervention(
        interventions=[(hook_name, fn)],
        ablated_components={sender_name},
    )