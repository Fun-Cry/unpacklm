# UNPACK

**Unified Path Attribution through Component Keys**

*Every component is a lookup: token attribution and composition from a single decomposition.*

UNPACK is a non-interventional attribution method for transformer models. Both attention and MLP follow a shared key-value template φ(S)U. UNPACK exploits this structure to run a single backward recursion that produces three levels of output from one forward pass — without intervention, gradients, or auxiliary training:

1. **Token attribution** — signed per-token credit showing which input tokens drove a prediction
2. **Named paths** — end-to-end routes through the model with K/Q/V composition labels at each hop
3. **Component interactions** — communication strengths between any two model components

## Installation

```bash
git clone https://github.com/Fun-Cry/unpacklm.git
cd unpacklm
pip install -e .
```

## Quick Start

```python
import unpack

# Load a model
tracer = unpack.Tracer("gpt2")

# Optional: set device and HuggingFace cache directory
tracer = unpack.Tracer("gpt2", device="cuda", cache_dir="/path/to/cache")

# Or bring your own model and tokenizer
tracer = unpack.Tracer(model=my_model, tokenizer=my_tokenizer)

# Trace a prediction
result = tracer.trace(
    "Mary and John went to the store. John gave the bag to",
    target=" Mary",
    distractor=" John",
)

# Per-token attribution
result.print_tokens()

# Named paths with K/Q/V composition modes
result.print_paths(top_k=20)

# Per-component aggregate flow
result.print_components()

# Or print everything at once
result.print()
```

## Token Attribution

Summing all credit that reaches the embedding layer at each position gives a signed attribution vector:

```python
result.token_attribution   # np.ndarray of signed percentages per position
result.tokens              # list of token strings
```

## Named Paths

Each path is a named end-to-end route from an input token through intermediate components to the target, with composition modes (K, Q, or V) labeled at each hop:

```python
path = result.paths[0]
path.chain        # "attn_9_head_9[V]@13 → mlp_0@1 → embedding@1"
path.source_token # " Mary"
path.source_pos   # 1
path.score        # +12.3 (% of total attribution)
path.modes        # ["V"] — composition modes at attention hops
path.depth        # 2
```

By default 20 paths are stored. To trace more, pass `top_paths_k`:

```python
result = tracer.trace(text, target=t, top_paths_k=2000)
```

## Rerooting

The backward recursion can start from any component, not just the target token. **Rerooting** traces backward from a specific component's output, revealing which upstream components compose into it and through which pathway (K, Q, or V):

```python
# What feeds into attention head 8.6 (an S-Inhibition head)?
prep, cfg = tracer.prepare(
    "Mary and John went to the store. John gave the bag to",
    target=" Mary",
    distractor=" John",
)

# Reroot at a specific head and position
sub = tracer.trace_from_prep(prep, cfg, root="attn_8_head_6@12")
sub.print_paths(top_k=10)

# The prep step (forward pass) is expensive; trace_from_prep is cheap.
# Reroot at many components without re-running the forward pass:
for head in ["attn_9_head_9", "attn_9_head_6", "attn_10_head_0"]:
    r = tracer.trace_from_prep(prep, cfg, root=head)
    print(f"\n{head}:")
    r.print_paths(top_k=5)
```

## Path Visualization

Paths can be rendered as interactive HTML or TikZ for papers:

```python
from unpack.viz import CircuitGraph

# Build a graph from the tracer's architecture
graph = CircuitGraph.from_tracer(tracer)
graph.tokens = result.tokens
graph.target_token = result.target_token

# Add the top paths
graph.add_paths(result.paths[:5])

# Render to different formats
graph.save_html("paths.html")   # interactive D3 visualization (open in browser)
graph.save_tikz("paths.tex")    # TikZ for LaTeX papers

# Or get raw strings
html_str = graph.render_html()
tikz_str = graph.render_tikz()

# JSON-safe dict for custom frontends
data = graph.to_dict()
```

Rerooted traces visualize the same way — just feed rerooted paths into the graph:

```python
graph.clear_paths()
graph.root = "attn_8_head_6"
graph.add_paths(sub.paths[:3])
graph.save_html("rerooted_paths.html")
```

## Configuration

Six named presets span the paper's three-axis design space:

| Preset | Attn key-side | MLP key-side | V-side dispatch |
|---|---|---|---|
| `kqv_aligned` | K+Q+V | weighted | aligned |
| `k_only_weighted` | K only | weighted | raw |
| `k_only_l2` | K only | L₂ norm | raw |
| `k_only_aligned` | K only | weighted | aligned |
| `kqv_weighted` | K+Q+V | weighted | raw |
| `kqv_l2` | K+Q+V | L₂ norm | raw |

`kqv_aligned` is used by default when no `config` is specified.

```python
# Default is kqv_aligned — no config needed
result = tracer.trace(text, target=t)

# Use a different preset
result = tracer.trace(text, target=t, config="k_only_weighted")

# Or fine-grained control
result = tracer.trace(text, target=t, branches="kqv", aligned=True, beta=0.8)

# Access all presets
from unpack import PRESETS
print(list(PRESETS.keys()))
```

The default configuration (`kqv_aligned`) traces all three composition branches and uses output-aligned value dispatch, matching the best-performing setup from the paper.

## Adapter Validation

When implementing support for a new model architecture, run the built-in mathematical invariant checks:

```python
unpack.validate(tracer)  # runs 6 checks: residual closure, K/Q/V-side closure, etc.
```

## Supported Models

- **GPT-2** (all sizes)
- **Pythia / GPT-NeoX** (all sizes, including deduped variants)

Adding a new architecture requires implementing a `ModelAdapter` subclass. See `unpack.models.base` for the interface.

## Low-Level Access

The decomposition primitives are available for custom analyses:

```python
from unpack.core import (
    backward_recursive,        # the recursive attribution algorithm
    ComponentStreamer,          # streams residual components layer by layer
)
from unpack.core.scorers import AttentionScorer, MLPScorer
from unpack.core.prep import _prepare_trace_inputs
```

## Citation

```bibtex
@article{chen2026unpack,
  title={Every Component is a Lookup: Token Attribution and Composition from a Single Decomposition},
  author={Chen, Po-Kai and van Stein, Niki and Plaat, Aske},
  year={2026}
}
```
