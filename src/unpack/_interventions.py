import torch
from abc import ABC, abstractmethod
from typing import Callable, Dict, List
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np
from unpack.models.base import ModelAdapter

def add(value):
    """Add ``value`` to every position of the tensor.

    ``value`` may be a Python scalar (broadcasts across positions) or
    a tensor matching the hook's tensor shape.

    Common uses: scalar perturbations for sanity checks, vector-valued
    steering offsets.
    """
    def fn(tensor, name):
        return tensor + value
    fn.__qualname__ = f"add({value!r})" if isinstance(value, (int, float)) else "add(...)"
    return fn


def mute(component_vec):
    """Subtract ``component_vec`` from the tensor.

    Use this to remove a captured component's contribution from a
    residual stream — e.g. extract head 5's contribution at layer L,
    then ``mute(c_5)`` at any downstream hook to zero out its
    influence.

    ``component_vec`` shape must broadcast to the hook tensor's shape.
    """
    def fn(tensor, name):
        return tensor - component_vec.to(tensor.device)
    fn.__qualname__ = "mute(...)"
    return fn


def scale(component_vec, alpha):
    """Replace a component's contribution with an ``alpha``-scaled version.

    Equivalent to ``tensor - component_vec + alpha * component_vec``,
    i.e. ``tensor + (alpha - 1) * component_vec``.

    ``alpha=0`` -> full mute, ``alpha=1`` -> no-op, ``alpha=2`` -> double.
    Useful for sweeping a component's effect strength to detect
    self-repair (Hydra effect): a missing head's effect is partially
    compensated by other heads scaling up.
    """
    def fn(tensor, name):
        v = component_vec.to(tensor.device)
        return tensor + (alpha - 1.0) * v
    fn.__qualname__ = f"scale(alpha={alpha})"
    return fn


def replace_with(replacement):
    """Overwrite the entire tensor with ``replacement``.

    ``replacement`` must broadcast to the hook's tensor shape.  Common
    use: replace a captured component's contribution with a reference
    value (mean ablation, resample patching).
    """
    rep = replacement
    def fn(tensor, name):
        return rep.to(tensor.device).expand_as(tensor).clone()
    fn.__qualname__ = "replace_with(...)"
    return fn


def replace_at(replacement, positions):
    """Overwrite specific positions only, leave others untouched.

    ``replacement`` must have the hook's tensor shape (B, S, D) — the
    same as ``tensor`` at fire time.  ``positions`` is a list of int
    indices along the sequence axis; only those positions are
    overwritten.

    Common use: ablate at the target query position only (McGrath's
    ``do(A^l_t = ã^l_t)`` convention) without touching upstream
    positions that other heads in later layers may attend to.
    """
    pos = list(positions)
    def fn(tensor, name):
        out = tensor.clone()
        rep = replacement.to(tensor.device)
        # broadcast the replacement to match `out`'s batch dim if needed
        if rep.dim() == out.dim() - 1:
            rep = rep.unsqueeze(0).expand_as(out)
        elif rep.shape[0] != out.shape[0]:
            rep = rep.expand_as(out)
        out[:, pos, :] = rep[:, pos, :]
        return out
    fn.__qualname__ = f"replace_at(positions={pos})"
    return fn


def add_at(delta, positions):
    """Add ``delta`` to specific positions only, leave others untouched.

    ``delta`` must broadcast to the hook's tensor shape at the active
    positions: ``(D,)``, ``(len(positions), D)``, or ``(B, len(positions), D)``.
    ``positions`` is a list of int sequence-axis indices.

    Path patching uses this template at the receiver's layer-norm
    input: ``delta = corrupted_sender_contribution - clean_sender_contribution``
    added at ``[receiver_pos]`` modifies what the receiver reads from
    the sender without disturbing other positions or other sources at
    the same position.
    """
    pos = list(positions)
    def fn(tensor, name):
        out = tensor.clone()
        d = delta.to(tensor.device).to(tensor.dtype)
        if d.dim() == 1:
            # (D,) → broadcast across batch and selected positions
            out[:, pos, :] = out[:, pos, :] + d
        elif d.dim() == 2:
            # (len(pos), D) → broadcast across batch
            out[:, pos, :] = out[:, pos, :] + d.unsqueeze(0)
        elif d.dim() == 3:
            # (B, len(pos), D)
            out[:, pos, :] = out[:, pos, :] + d
        else:
            raise ValueError(f"add_at: delta has unexpected ndim={d.dim()}")
        return out
    fn.__qualname__ = f"add_at(positions={pos})"
    return fn


def replace_head_slice(replacement_z, head_idx, head_size, positions):
    """Overwrite one head's z-slice in a pre_dense tensor.

    For use at ``attn_{L}_pre_dense`` hooks.  The pre_dense tensor has
    shape ``(B, S, d_model)`` where the last dim is laid out as the
    concatenation of per-head z vectors of size ``head_size``.  Head
    ``head_idx`` lives at ``[..., head_idx*head_size : (head_idx+1)*head_size]``;
    this factory writes ``replacement_z`` into that slice at the active
    ``positions`` and leaves everything else untouched.

    By W_O's linearity, this produces an equivalent change to the
    layer's post-W_O attention output as if you had directly replaced
    head ``head_idx``'s post-W_O contribution — but installed at the
    pre_dense capture hook, the modification flows through the
    streamer's per-head reconstruction consistently.

    ``replacement_z`` shape: ``(S, head_size)`` or ``(B, S, head_size)``.
    ``positions``: list of int sequence indices to overwrite.
    """
    j_lo, j_hi = head_idx * head_size, (head_idx + 1) * head_size
    pos = list(positions)
    def fn(tensor, name):
        out = tensor.clone()
        rep = replacement_z.to(tensor.device)
        if rep.dim() == out.dim() - 1:
            rep = rep.unsqueeze(0).expand(out.shape[0], -1, -1)
        elif rep.shape[0] != out.shape[0]:
            rep = rep.expand(out.shape[0], -1, -1)
        out[:, pos, j_lo:j_hi] = rep[:, pos, :]
        return out
    fn.__qualname__ = f"replace_head_slice(h={head_idx}, pos={pos})"
    return fn


def project_out(direction):
    """Remove a direction from the tensor by orthogonal projection.

    For each position, subtract the component along the (normalized)
    ``direction`` from the tensor.  ``direction`` is a 1-D tensor of
    size ``d_model``; it's normalized internally.

    Use this for direction-specific knockout — e.g. project out the
    "is-an-IO-token" direction without ablating any one component.
    """
    d = direction.float()
    d = d / d.norm()
    def fn(tensor, name):
        dd = d.to(tensor.device)
        coeff = (tensor.float() @ dd).unsqueeze(-1)
        return tensor - coeff * dd
    fn.__qualname__ = "project_out(...)"
    return fn


def clamp_norm(max_norm):
    """Clamp the L2 norm of each position's hidden state to ``max_norm``.

    Per-position scaling: positions whose norm exceeds ``max_norm`` are
    rescaled to ``max_norm``; positions under the bound are untouched.
    Useful for norm-controlled ablation studies.
    """
    def fn(tensor, name):
        norms = tensor.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        s = torch.clamp(max_norm / norms, max=1.0)
        return tensor * s
    fn.__qualname__ = f"clamp_norm({max_norm})"
    return fn


def noise(std, seed=None):
    """Add Gaussian noise with std ``std``.

    If ``seed`` is given, the noise is deterministic across forward
    passes.  Useful for testing the model's robustness to small
    perturbations or for noise-based knockout.
    """
    def fn(tensor, name):
        gen = torch.Generator(device=tensor.device)
        if seed is not None:
            gen.manual_seed(seed)
        return tensor + torch.randn_like(tensor, generator=gen) * std
    fn.__qualname__ = f"noise(std={std}, seed={seed})"
    return fn


class InterventionRunner:
    """
    Lightweight wrapper for running forward passes with causal interventions.
    Separate from tracing — no tracing, just modify-and-observe.
    """
 
    class Result:
        __slots__ = ['logits', 'attentions']
        def __init__(self, logits, attentions):
            self.logits = logits
            self.attentions = attentions
 
    def __init__(self, model: 'PreTrainedModel', tokenizer: 'PreTrainedTokenizer',
                 hook_manager: 'HookManager'):
        self.model = model
        self.tokenizer = tokenizer
        self.hook_manager = hook_manager
 
    def run(self, model_input, interventions=None,
            output_attentions=True, baseline_hidden_states=None):
        """Forward pass with optional interventions.
 
        Args:
            model_input:    str or batch of strings.
            interventions:  {hook_name: fn} dict or None.
            output_attentions: capture attention weights.
            baseline_hidden_states: list[Tensor] indexed by layer.
                When provided, every layer touched by an intervention
                gets its input pinned to the baseline hidden state
                before the intervention fires (marginal mode).
        """
        if baseline_hidden_states is not None and interventions:
            interventions = self._pin_to_baseline(interventions, baseline_hidden_states)
 
        if interventions:
            self.hook_manager.register_interventions(interventions)
 
        try:
            self.hook_manager.clear()
            inputs = self.tokenizer(model_input, return_tensors="pt",
                                    padding=True, truncation=True)
            inputs = inputs.to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=output_attentions)
 
            attentions = None
            if output_attentions and outputs.attentions is not None:
                attentions = [a.detach().cpu() for a in outputs.attentions]
 
            return self.Result(
                logits=outputs.logits.detach().cpu(),
                attentions=attentions,
            )
        finally:
            self.hook_manager.clear_interventions()
 
    @staticmethod
    def _pin_to_baseline(interventions, baseline_hidden_states):
        """Wrap interventions to pin affected layers to baseline first."""
        pinned = dict(interventions)
 
        affected = set()
        for name in interventions:
            for part in name.split('_'):
                if part.isdigit():
                    affected.add(int(part))
                    break
 
        for L in affected:
            pin_name = f'layer_{L}_input'
            bh = baseline_hidden_states[L]
            pin_fn = lambda t, n, _b=bh: _b.to(t.device)
 
            if pin_name in pinned:
                user_fn = pinned[pin_name]
                pinned[pin_name] = lambda t, n, _p=pin_fn, _f=user_fn: _f(_p(t, n), n)
            else:
                pinned[pin_name] = pin_fn
 
        return pinned
 
