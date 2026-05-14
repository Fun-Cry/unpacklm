# UNPACK

**Unified Path Attribution through Component Keys**

*Every component is a lookup: tokens, paths, and circuits from a single decomposition.*

UNPACK is a non-interventional attribution method for transformer models. It observes that both attention and MLP can be written in a unified key-value form φ(S)U, and uses this to build a single backward decomposition that produces signed component scores, named end-to-end paths, and per-token attribution from one forward pass — without intervention, gradients, or auxiliary training.

## Installation

```bash
pip install unpacklm
```

## Quick Start

```python
import unpack

# Load a model
tracer = unpack.Tracer("gpt2")

# Trace a prediction
result = tracer.trace(
    "Mary and John went to the store. John gave the bag to",
    target=" Mary",
    distractor=" John",
)

# Level 1: Token attribution
result.print_tokens()

# Level 2: Named paths with K/Q/V composition modes
result.print_paths(top_k=20)

# Bridge: Per-component flow
result.print_components()

# Or print everything at once
result.print()
```

## Three Levels of Output

### Tokens
Per-token signed credit: which input tokens drove this prediction?

```python
result.token_attribution   # np.ndarray of signed percentages
```

### Paths
Named end-to-end routes: how did each token's influence reach the target?

```python
result.paths[0].chain        # "attn_9_head_9[V] → attn_7_head_3[K] → embedding"
result.paths[0].source_token # " Mary"
result.paths[0].score        # +12.3 (%)
result.paths[0].modes        # ["V", "K"] — composition modes at each hop
```

### Circuits
Selected component sets verified by ablation (multi-prompt):

```python
circuit = tracer.discover(prompts, roles=unpack.ioi_roles)
scores = tracer.verify(circuit, test_prompts)
print(f"Faith: {scores.faith:.3f}, Knockout: {scores.knockout:.3f}")
```

## Configuration

Five named presets spanning the paper's variant design space:

```python
# Default: K-only attention + weighted MLP + raw V-side
result = tracer.trace(text, target=t, config="default")

# All five presets
result = tracer.trace(text, target=t, config="kqv_aligned")
# Available: "default", "k_only_l2", "kqv_weighted", "kqv_l2", "kqv_aligned"

# Or fine-grained control
result = tracer.trace(text, target=t, beta=0.8, branches="kqv", aligned=True)
```

## Re-rooting

Trace attribution from a specific component instead of the target:

```python
# What feeds into attention head 9.9?
sub = tracer.trace(text, target=t, root="attn_9_head_9")
sub.print_paths()
```

## Adapter Validation

When implementing support for a new model architecture:

```python
unpack.validate(tracer)  # runs 6 mathematical invariant checks
```

## Supported Models

- GPT-2 (all sizes)
- Pythia / GPT-NeoX (all sizes)

Adding a new architecture requires implementing a `ModelAdapter` subclass (~15 methods for weight extraction and hook wiring).

## Low-Level Access

Power users can access the decomposition primitives directly:

```python
from unpack.core import (
    backward_recursive,
    compute_target_direction,
    precompute_attn_shares,
    ComponentStreamer,
)
```

## Citation

```bibtex
@article{chen2026unpack,
  title={Every Component is a Lookup: Tokens, Paths, and Circuits from a Single Decomposition with {UNPACK}},
  author={Chen, Po-Kai and van Stein, Niki and Plaat, Aske},
  year={2026}
}
```

## License

MIT
