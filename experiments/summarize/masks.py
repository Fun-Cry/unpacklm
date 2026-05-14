"""Architectural masks — determine which edges are structurally valid.

An edge from component C to target layer L is valid iff the source layer
of C is strictly less than L. This is derived from the component name
(e.g. 'attn_3_head_7' → source layer 3), never from data.
"""

import re
import numpy as np


def build_valid_mask_attn(num_layers, num_heads, component_names):
    """(L, H, C) bool mask — True where source_layer < target_layer."""
    C = len(component_names)
    source_layers = _parse_source_layers(component_names)
    mask = np.zeros((num_layers, num_heads, C), dtype=bool)
    for c, src in enumerate(source_layers):
        if src is not None:
            for l in range(num_layers):
                if src < l:
                    mask[l, :, c] = True
        else:
            mask[:, :, c] = True
    return mask


def build_valid_mask_mlp(num_layers, component_names):
    """(L, C) bool mask — True where source_layer < target_layer."""
    C = len(component_names)
    source_layers = _parse_source_layers(component_names)
    mask = np.zeros((num_layers, C), dtype=bool)
    for c, src in enumerate(source_layers):
        if src is not None:
            for l in range(num_layers):
                if src < l:
                    mask[l, c] = True
        else:
            mask[:, c] = True
    return mask


def _parse_source_layers(component_names):
    """Extract source layer index from component names.

    Examples:
        'embed' → None (always valid)
        'attn_3_head_7' → 3
        'mlp_5' → 5
        'ln_f' → None
    """
    pattern = re.compile(r'(?:attn|mlp)_(\d+)')
    result = []
    for name in component_names:
        m = pattern.search(name)
        if m:
            result.append(int(m.group(1)))
        else:
            if any(kw in name for kw in ['embed', 'ln_f', 'pos']):
                result.append(None)
            else:
                print(f"Warning: could not parse source layer from '{name}', treating as always valid")
                result.append(None)
    return result
