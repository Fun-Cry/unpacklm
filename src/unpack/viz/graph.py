"""
unpack.viz.graph - Data model for circuit path visualization.

Separates the data model from rendering so the same structure
can drive HTML, TikZ, or custom visualizations.

Usage:
    from unpack.viz import CircuitGraph
    
    graph = CircuitGraph.from_model(tracer.adapter)
    graph.add_paths(result.paths[:10])
    graph.save_html("paths.html")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


MODE_RE = re.compile(r"\[([KQV])\]")
POS_RE = re.compile(r"@(-?\d+)$")
COMP_RE = re.compile(r"^(attn_\d+_head_\d+|mlp_\d+|embedding|pos_embedding)")


@dataclass
class Hop:
    """One step in a path through the model."""
    component: str       # "attn_9_head_9", "mlp_0", "embedding"
    mode: Optional[str]  # "K", "Q", "V", or None
    position: int        # token position
    layer: float         # layer index (-1 for embed, L for attn, L+0.5 for seq MLP)

    @property
    def is_embedding(self) -> bool:
        return self.component in ("embedding", "pos_embedding")

    @property
    def is_attn(self) -> bool:
        return self.component.startswith("attn_")

    @property
    def is_mlp(self) -> bool:
        return self.component.startswith("mlp_")


@dataclass
class VisPath:
    """A single path to visualize, parsed into hops."""
    chain: str              # raw chain string
    hops: List[Hop]         # parsed hops
    score: float            # signed score (%)
    color: Optional[str] = None
    label: Optional[str] = None

    @property
    def positions(self) -> Set[int]:
        return {h.position for h in self.hops}

    @property
    def components(self) -> List[str]:
        return [h.component for h in self.hops]


@dataclass
class CircuitGraph:
    """Data model for visualizing paths through a transformer.

    Holds architecture info and a list of paths to render.
    Independent of rendering backend.
    """
    num_layers: int
    num_heads: int
    parallel_residual: bool  # True for Pythia, False for GPT-2
    paths: List[VisPath] = field(default_factory=list)
    tokens: Optional[List[str]] = None
    target_token: Optional[str] = None
    root: Optional[str] = None  # None = target prediction, "attn_L_head_H" = rerooted

    # ── Construction ──

    @classmethod
    def from_adapter(cls, adapter) -> "CircuitGraph":
        """Build from a ModelAdapter (reads architecture info)."""
        return cls(
            num_layers=adapter.get_num_layers(),
            num_heads=adapter.get_num_heads(),
            parallel_residual=getattr(adapter, 'parallel_residual', False),
        )

    @classmethod
    def from_tracer(cls, tracer) -> "CircuitGraph":
        """Build from a Tracer instance."""
        return cls.from_adapter(tracer.adapter)

    # ── Path management ──

    # Shared palette for auto-coloring
    _palette = [
        "#4a7fb5",  # steel blue
        "#d4804e",  # warm terra
        "#6aab73",  # sage green
        "#8b6dae",  # soft purple
        "#c75a5a",  # muted rose
        "#4aada1",  # teal
        "#c4953a",  # golden
        "#7a8b99",  # slate
    ]

    def _next_color(self):
        return self._palette[len(self.paths) % len(self._palette)]

    def add_path(self, path, color: str = None, label: str = None):
        """Add a path (unpack.Path or raw chain string).

        Args:
            path: an unpack.Path object, or a raw chain string.
            color: CSS color for highlighting (auto-assigned if None).
            label: optional label for the path.
        """
        if isinstance(path, str):
            chain_str = path
            score = 0.0
        else:
            chain_str = path.chain
            score = path.score

        if color is None:
            color = self._next_color()

        hops = self._parse_chain(chain_str)
        vp = VisPath(
            chain=chain_str, hops=hops, score=score,
            color=color, label=label,
        )
        self.paths.append(vp)
        return self

    def add_paths(self, paths, colors=None):
        """Add multiple paths."""
        for i, p in enumerate(paths):
            color = colors[i] if colors else None
            self.add_path(p, color=color)
        return self

    def clear_paths(self):
        self.paths = []
        return self

    # ── Queries ──

    @property
    def active_positions(self) -> List[int]:
        """All token positions that appear in any path, sorted."""
        positions = set()
        for vp in self.paths:
            positions.update(vp.positions)
        return sorted(positions)

    @property
    def active_layers(self) -> List[int]:
        """All layers that contain path components."""
        layers = set()
        for vp in self.paths:
            for h in vp.hops:
                if not h.is_embedding:
                    layers.add(int(h.layer))
        return sorted(layers)

    @property
    def active_components(self) -> Set[str]:
        """All components that appear in any path."""
        comps = set()
        for vp in self.paths:
            for h in vp.hops:
                comps.add(h.component)
        return comps

    # ── Serialization (for future web integration) ──

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict for web rendering."""
        return {
            "architecture": {
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "parallel_residual": self.parallel_residual,
            },
            "tokens": self.tokens,
            "target_token": self.target_token,
            "root": self.root,
            "active_positions": self.active_positions,
            "paths": [
                {
                    "chain": vp.chain,
                    "score": vp.score,
                    "color": vp.color,
                    "label": vp.label,
                    "hops": [
                        {
                            "component": h.component,
                            "mode": h.mode,
                            "position": h.position,
                            "layer": h.layer,
                        }
                        for h in vp.hops
                    ],
                }
                for vp in self.paths
            ],
        }

    # ── Rendering ──

    def render_html(self, width=900, height=None) -> str:
        """Render paths as an interactive HTML string with D3.js.

        Features: parallel path lanes, hover highlighting, click to toggle.
        Requires internet (loads D3 from CDN).
        """
        from unpack.viz.render_html import render_html
        return render_html(self, width=width, height=height)

    def save_html(self, path: str, **kwargs):
        """Render and save to an HTML file. Open in browser for interaction."""
        html = self.render_html(**kwargs)
        with open(path, "w") as f:
            f.write(html)

    def render_tikz(self, standalone: bool = True, **kwargs) -> str:
        """Render paths as TikZ.
        
        Args:
            standalone: if True, full compilable .tex document.
                       if False, just pc colors + tikzpicture for \\input{}.
        """
        from unpack.viz.render_tikz import render_tikz
        return render_tikz(self.to_dict(), standalone=standalone, **kwargs)

    def save_tikz(self, path: str, **kwargs):
        """Render and save to a .tex file."""
        tikz = self.render_tikz(**kwargs)
        with open(path, "w") as f:
            f.write(tikz)

    # ── Internal ──

    def _parse_chain(self, chain_str: str) -> List[Hop]:
        """Parse 'attn_9_head_9[K]@13→mlp_0@1→embedding@1' into Hops."""
        parts = re.split(r"→|->", chain_str)
        hops = []
        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Extract mode [K/Q/V]
            mode_match = MODE_RE.search(part)
            mode = mode_match.group(1) if mode_match else None

            # Extract position @N
            pos_match = POS_RE.search(part)
            position = int(pos_match.group(1)) if pos_match else -1

            # Extract component name
            comp_match = COMP_RE.match(part)
            component = comp_match.group(1) if comp_match else part.split("[")[0].split("@")[0]

            # Determine layer
            layer = self._component_layer(component)

            hops.append(Hop(
                component=component, mode=mode,
                position=position, layer=layer,
            ))

        return hops

    def _component_layer(self, name: str) -> float:
        """Map component name to layer index."""
        if name in ("embedding", "pos_embedding"):
            return -1
        # Extract layer number
        m = re.search(r"(\d+)", name)
        if m is None:
            return -1
        layer = int(m.group(1))
        if name.startswith("mlp_") and not self.parallel_residual:
            return layer + 0.5
        return float(layer)