"""
interventions.py — What to perturb.

Intervention factories (mute, add, scale, ...), the Intervention spec,
hook point templates, and component extraction from the residual stream.

No model execution happens here except in extract_component(s), which
runs a single forward pass via ComponentStreamer.
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Union
from transformers import PreTrainedModel, PreTrainedTokenizer
from unpack.models.base import ModelAdapter
from unpack.core import ComponentStreamer


# ================================================================
#  Factories — (tensor, name) -> tensor
# ================================================================

def mute(component_vec: torch.Tensor) -> Callable:
    """Subtract a component vector from the hidden state.

    Args:
        component_vec: (B, S, D) tensor to subtract.
    """
    def fn(tensor, name):
        return tensor - component_vec.to(tensor.device)
    return fn


def add(component_vec: torch.Tensor) -> Callable:
    """Add a component vector to the hidden state."""
    def fn(tensor, name):
        return tensor + component_vec.to(tensor.device)
    return fn


def scale(component_vec: torch.Tensor, alpha: float) -> Callable:
    """Replace component's contribution with a scaled version.

    Equivalent to: tensor - component_vec + alpha * component_vec
    alpha=0 -> full mute, alpha=1 -> no-op, alpha=2 -> double.
    """
    def fn(tensor, name):
        v = component_vec.to(tensor.device)
        return tensor + (alpha - 1.0) * v
    return fn


def replace_with(replacement: torch.Tensor) -> Callable:
    """Replace the entire tensor with a fixed value."""
    def fn(tensor, name):
        return replacement.to(tensor.device)
    return fn


def project_out(direction: torch.Tensor) -> Callable:
    """Remove a direction from the hidden state (projection).

    Args:
        direction: (D,) vector. Will be normalized internally.
    """
    d = direction.float()
    d = d / d.norm()

    def fn(tensor, name):
        dd = d.to(tensor.device)
        coeff = (tensor.float() @ dd).unsqueeze(-1)
        return tensor - coeff * dd
    return fn


def clamp_norm(max_norm: float) -> Callable:
    """Clamp the L2 norm of each position's hidden state."""
    def fn(tensor, name):
        norms = tensor.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        s = torch.clamp(max_norm / norms, max=1.0)
        return tensor * s
    return fn


def noise(std: float, seed: Optional[int] = None) -> Callable:
    """Add Gaussian noise."""
    def fn(tensor, name):
        gen = torch.Generator(device=tensor.device)
        if seed is not None:
            gen.manual_seed(seed)
        return tensor + torch.randn_like(tensor, generator=gen) * std
    return fn


# ================================================================
#  Hook Templates
# ================================================================

HOOK_TEMPLATES = {
    'layer_input':    'layer_{layer}_input',    # before BOTH attn and MLP
    'attn_input':     'attn_{layer}_input',     # before attention only
    'attn_output':    'attn_{layer}',
    'attn_pre_dense': 'attn_{layer}_pre_dense',
    'mlp_output':     'mlp_{layer}',
    'residual':       'resid_{layer}',
    'attn_ln_input':  'attn_ln_{layer}_input',  # before attention layernorm
    'mlp_ln_input':   'mlp_ln_{layer}_input',   # before MLP layernorm
}


# ================================================================
#  Intervention Spec
# ================================================================

@dataclass
class Intervention:
    """A single intervention specification.

    Args:
        fn:     Intervention function (tensor, name) -> tensor.
                Use the factories: mute(), add(), scale(), etc.
        layers: Which layers to apply to. int, list, or range.
        hook:   Where in each layer to intervene.
                One of: 'layer_input'  -- before BOTH attn and MLP (full layer input),
                        'attn_input'   -- before attention only,
                        'attn_output', 'attn_pre_dense',
                        'mlp_output', 'residual'.
                For marginal mode, 'layer_input' perturbs both sublayers;
                'attn_input' perturbs attention only (MLP sees clean baseline).
    """
    fn: Callable
    layers: Union[int, List[int], range]
    hook: str = 'attn_input'

    def __post_init__(self):
        if isinstance(self.layers, int):
            self.layers = [self.layers]
        elif isinstance(self.layers, range):
            self.layers = list(self.layers)
        if self.hook not in HOOK_TEMPLATES:
            raise ValueError(
                f"Unknown hook point '{self.hook}'. "
                f"Choose from: {list(HOOK_TEMPLATES.keys())}"
            )

    def resolve_hook_names(self) -> Dict[str, Callable]:
        """Expand to {hook_name: fn} dict."""
        template = HOOK_TEMPLATES[self.hook]
        return {template.format(layer=l): self.fn for l in self.layers}


# ================================================================
#  Component Extraction
# ================================================================

@dataclass
class Component:
    """A single named residual stream component.

    Attributes:
        name:         e.g. 'mlp_4', 'attn_3_head_7'
        vec:          (B, S, D) tensor -- the component's contribution
        source_layer: which layer wrote this component
    """
    name: str
    vec: torch.Tensor       # (B, S, D)
    source_layer: int


def extract_components(model: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer,
                       hook_manager: ModelAdapter,
                       text: str,
                       names: Optional[List[str]] = None,
                       ) -> List[Component]:
    """Run a forward pass and extract named residual stream components.

    Uses ComponentStreamer + iter_source_groups to decompose the residual
    stream into per-head, per-MLP, embedding, and bias components.

    Args:
        names: If provided, only return components with matching names.
               If None, return all components.

    Returns:
        List of Component objects.
    """
    streamer = ComponentStreamer(model, tokenizer, hook_manager)
    streamer.set_context(text)

    results = []
    name_set = set(names) if names else None

    for group_tensor, group_names, src_layer in hook_manager.iter_source_groups():
        for i, comp_name in enumerate(group_names):
            if name_set is not None and comp_name not in name_set:
                continue
            results.append(Component(
                name=comp_name,
                vec=group_tensor[:, :, i, :].clone(),
                source_layer=src_layer,
            ))

    return results


def extract_component(model: PreTrainedModel,
                      tokenizer: PreTrainedTokenizer,
                      hook_manager: ModelAdapter,
                      text: str,
                      name: str) -> Component:
    """Extract a single named component. Convenience wrapper.

    Raises KeyError if the component name is not found.
    """
    results = extract_components(model, tokenizer, hook_manager, text, names=[name])
    if not results:
        raise KeyError(f"Component '{name}' not found in residual stream.")
    return results[0]