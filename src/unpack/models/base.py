"""
unpack.models.base - Abstract base for model-specific adapters.

A ModelAdapter teaches UNPACK how to interact with a specific model
architecture. It handles:

1. Hook wiring — which submodules to hook and what to capture
2. Weight extraction — how to get Q/K/V/MLP weights per layer
3. Cross-attention scoring — how to compute decomposed attention logits
4. Residual topology — sequential vs parallel, component ordering
5. Embedding structure — token-only vs token+position

Subclass responsibilities:
    * register_hooks(model) — wire all hook points
    * iter_source_groups() — yield captured component residuals
    * get_cross_attention_scores() — decomposed attention logits
    * All get_*_params() methods for weight/norm extraction
"""

import torch
from abc import ABC, abstractmethod
from typing import Callable, Dict, List
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np


class ModelAdapter(ABC):
    """Abstract base for model-specific adapters.

    Two state attributes with distinct lifetimes:

    ``self.handles`` — list of PyTorch hook handles installed by
    ``register_hooks(model)``. Held until ``remove_hooks()``.

    ``self._interventions`` — ``{hook_name: [Callable, ...]}`` registry
    consulted by every hook fire. Populated and cleared between forward
    passes via ``register_intervention(s)`` and ``clear_interventions``.
    """

    def __init__(self):
        self.handles = []
        self._interventions: Dict[str, List[Callable]] = {}

    @abstractmethod
    def register_hooks(self, model: PreTrainedModel):
        pass

    @abstractmethod
    def get_cross_attention_scores(self, layer_idx: int,
                                   new_input_states: torch.Tensor,
                                   include_bias: bool = False,
                                   side: str = "key") -> torch.Tensor:
        pass

    @abstractmethod
    def get_value_weight(self, layer_idx: int) -> torch.Tensor:
        """Return W_V for this layer, shape (num_heads, d_model, head_dim)."""
        pass

    @abstractmethod
    def get_value_bias(self, layer_idx: int) -> torch.Tensor:
        """Return b_V for this layer, shape (num_heads, head_dim)."""
        pass

    @abstractmethod
    def get_value_at_position(self, layer_idx: int,
                              hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute V[b, h, s, :] from a hidden state. Shape:
        (B, num_heads, S, head_dim)."""
        pass

    # ==========================================
    #  Core Hook Dispatch
    # ==========================================

    def _run_hook(self, name: str, tensor: torch.Tensor, capture_to: str = None):
        """Single chokepoint for every hook fire."""
        fns = self._interventions.get(name)
        modified = bool(fns)
        if modified:
            for fn in fns:
                tensor = fn(tensor, name)
        if capture_to is not None:
            getattr(self, capture_to).append(tensor.detach().clone())
        return tensor, modified

    def _wire_hook(self, module, name: str, hook_type: str, capture_to: str = None):
        """Register one PyTorch hook on ``module``, named ``name``."""
        if hook_type == 'output':
            def hook(mod, inp, out, _n=name, _c=capture_to):
                result, changed = self._run_hook(_n, out, _c)
                if changed:
                    return result
            self.handles.append(module.register_forward_hook(hook))

        elif hook_type == 'tuple_output':
            def hook(mod, inp, out, _n=name, _c=capture_to):
                first = out[0] if isinstance(out, tuple) else out
                result, changed = self._run_hook(_n, first, _c)
                if changed:
                    return (result,) + out[1:] if isinstance(out, tuple) else result
            self.handles.append(module.register_forward_hook(hook))

        elif hook_type == 'input':
            def hook(mod, inp, _n=name, _c=capture_to):
                result, changed = self._run_hook(_n, inp[0], _c)
                if changed:
                    return (result,) + inp[1:]
            self.handles.append(module.register_forward_pre_hook(hook))

    # ==========================================
    #  Intervention API
    # ==========================================

    def register_intervention(self, name: str, fn: Callable):
        """Append `fn` to the chain of interventions at `name`."""
        self._interventions.setdefault(name, []).append(fn)

    def register_interventions(self, interventions: Dict[str, Callable]):
        """Bulk version of register_intervention."""
        for name, fn in interventions.items():
            self.register_intervention(name, fn)

    def clear_interventions(self):
        """Drop all registered interventions at every hook."""
        self._interventions.clear()

    def describe_interventions(self) -> str:
        """Return a human-readable dump of current interventions."""
        if not self._interventions:
            return "ModelAdapter: no interventions registered."
        n_hooks = len(self._interventions)
        n_fns = sum(len(v) for v in self._interventions.values())
        lines = [f"ModelAdapter: {n_fns} intervention(s) at {n_hooks} hook(s)"]
        for name in sorted(self._interventions):
            fns = self._interventions[name]
            lines.append(f"  {name}  [{len(fns)} fn{'s' if len(fns) > 1 else ''}]")
            for i, fn in enumerate(fns, 1):
                qn = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", repr(fn))
                lines.append(f"    {i}. {qn}")
        return "\n".join(lines)

    def clear(self):
        pass

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    # ==========================================
    #  Subclass API — override in model-specific adapters
    # ==========================================

    def get_norm_params(self, layer_idx):
        """Return (weight, bias, eps) for the attention-side layernorm."""
        raise NotImplementedError

    def get_mlp_norm_params(self, layer_idx):
        """Return (weight, bias, eps) for the MLP-side layernorm."""
        raise NotImplementedError

    def get_num_layers(self) -> int:
        raise NotImplementedError

    def get_num_heads(self) -> int:
        raise NotImplementedError

    def get_head_size(self) -> int:
        """Per-head hidden dimension ``d_model // num_heads``."""
        return self.get_d_model() // self.get_num_heads()

    def get_d_model(self) -> int:
        """Residual-stream dimensionality."""
        raise NotImplementedError

    def apply_logit_lens(self, components, marginal=False, final_logits=None):
        raise NotImplementedError

    def project_values(self, layer_idx, values_states):
        raise NotImplementedError

    def iter_source_groups(self):
        """Yield (group_tensor, names, src_layer_idx) one source layer at a time."""
        raise NotImplementedError

    def free_attention_cache(self, layer_idx):
        pass

    def get_mlp_up_params(self, layer_idx):
        """Return (weight, bias) of the MLP up-projection."""
        raise NotImplementedError

    def get_component_layer(self, name):
        """Return the effective layer index for ordering purposes.

        For parallel-residual models, attention heads and MLP at the
        same layer have the same index. For sequential-residual models,
        MLP returns L + 0.5.
        """
        src_layer = -1
        for part in name.replace("head_", "h").split("_"):
            if part.isdigit():
                src_layer = int(part)
                break
        return src_layer

    def get_mlp_down_params(self, layer_idx):
        """Return (weight, bias) of the MLP down-projection."""
        raise NotImplementedError

    def get_final_norm_params(self):
        """Return (weight, bias, eps) for the final layernorm."""
        raise NotImplementedError

    def get_unembed_weight(self):
        """Return the unembedding weight matrix, shape (vocab_size, d_model)."""
        raise NotImplementedError

    def mlp_up_forward(self, layer_idx, normed_input):
        """Run MLP up-projection + activation on normed_input.
        Returns (pre_activation, post_activation)."""
        raise NotImplementedError

    def apply_attn_norm(self, layer_idx, hidden):
        """Apply this layer's pre-attention layer norm to a tensor."""
        raise NotImplementedError


# Backward compatibility alias for internal use
HookManager = ModelAdapter
