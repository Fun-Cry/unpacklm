"""
unpack.result - Data classes for trace output.

TraceResult wraps the three levels of UNPACK output:
  Level 1 (Tokens): per-token signed attribution
  Level 2 (Paths):  named end-to-end routes with composition modes
  Bridge:           per-component aggregate flow
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from unpack.core.path_views import (
    parse_chain,
    chain_source_position,
    chain_modes,
    dominant_attn_mode,
    render_grouped,
)


# ================================================================
#  Path — a single named end-to-end route
# ================================================================

_ROLE_RE = re.compile(r"\[([KQV])\]")


@dataclass
class Path:
    """A single named attribution path from input token to target.

    Attributes:
        chain: Full path string, e.g.
            "attn_9_head_9[V] → attn_7_head_3[K] → embedding"
        components: Component names along the path (without mode tags).
        source_pos: Input token position where the path terminates.
        source_token: The actual token string at source_pos.
        score: Signed contribution as % of total attribution.
        raw_score: Unnormalized signed contribution.
        modes: K/Q/V label at each hop (e.g. ["V", "K"]).
        depth: Number of hops (len(components) - 1).
    """
    chain: str
    components: List[str]
    source_pos: int
    source_token: str
    score: float
    raw_score: float
    modes: List[str]
    depth: int

    @classmethod
    def from_raw(cls, path_str: str, pos: int, pct: float,
                 raw: float, tokens: Sequence[str]) -> "Path":
        """Construct a Path from the raw (path_str, pos, score) triple."""
        parsed = parse_chain(path_str)
        # Strip mode tags and @pos from component names
        clean = []
        for elem in parsed:
            name = _ROLE_RE.sub("", elem)
            name = re.sub(r"@\d+$", "", name)
            clean.append(name.strip())
        modes = chain_modes(parsed)
        # Filter to just K/Q/V modes (skip 'emb', 'mlp', '?')
        attn_modes = [m for m in modes if m in ("K", "Q", "V")]

        tok = tokens[pos] if 0 <= pos < len(tokens) else "?"
        return cls(
            chain=path_str,
            components=clean,
            source_pos=pos,
            source_token=tok,
            score=pct,
            raw_score=raw,
            modes=attn_modes,
            depth=max(0, len(parsed) - 1),
        )


# ================================================================
#  TraceResult — full output of a single trace
# ================================================================

@dataclass
class TraceResult:
    """Complete output of a single trace call.

    Three levels of output:
      tokens/token_attribution — Level 1 (per-token credit)
      paths                   — Level 2 (named routes)
      component_flow          — Bridge (per-component aggregate)
    """

    # ── Metadata ──
    tokens: List[str]
    target_token: str
    target_prob: float
    target_logit_centered: float
    root: str = "target"

    # ── Level 1: Token attribution ──
    token_attribution: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── Level 2: Paths ──
    paths: List[Path] = field(default_factory=list)

    # ── Bridge: Component flow ──
    component_flow: Dict[str, float] = field(default_factory=dict)

    # ── Internals (for advanced use) ──
    importance: Dict[str, float] = field(default_factory=dict)
    suppress_ratio: Optional[float] = None

    # ── Display ──

    def print_tokens(self, top_k: Optional[int] = None):
        """Pretty-print token-level attribution with bar chart."""
        credit = self.token_attribution
        if credit.size == 0:
            print("  (no token attribution available)")
            return

        max_abs = max(np.abs(credit).max(), 1e-9)
        BAR_WIDTH = 10

        print(f"\nToken attribution for {self.target_token!r} "
              f"(p={self.target_prob:.4f}):")
        items = list(enumerate(zip(self.tokens, credit)))
        if top_k:
            items.sort(key=lambda x: abs(x[1][1]), reverse=True)
            items = items[:top_k]

        for i, (tok, c) in items:
            tok_repr = repr(tok)[:20]
            n_bar = int(round(abs(c) / max_abs * BAR_WIDTH))
            if c >= 0:
                bar = " " * BAR_WIDTH + "│" + "█" * n_bar
            else:
                bar = " " * (BAR_WIDTH - n_bar) + "█" * n_bar + "│"
            print(f"  {i:>3}  {tok_repr:<20}  {c:>+6.2f}%  {bar}")

    def print_paths(self, top_k: int = 20):
        """Pretty-print top paths ranked by |score|."""
        print(f"\nTop {min(top_k, len(self.paths))} paths "
              f"({len(self.paths)} total):")
        print(f"  {'#':>3}  {'score':>7}  {'src':>15}  path")
        print(f"  {'─'*3}  {'─'*7}  {'─'*15}  {'─'*50}")
        for i, p in enumerate(self.paths[:top_k], 1):
            src = f"{p.source_pos}({p.source_token!r})"
            chain = p.chain if len(p.chain) <= 50 else "…" + p.chain[-49:]
            print(f"  {i:>3}  {p.score:>+6.2f}%  {src:>15}  {chain}")

    def print_components(self, top_k: int = 30):
        """Pretty-print top components by |flow|."""
        if not self.component_flow:
            print("  (no component flow available)")
            return

        top = sorted(self.component_flow.items(),
                     key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        print(f"\nTop {len(top)} components by |flow|:")
        print(f"  {'#':>3}  {'flow':>9}  component")
        print(f"  {'─'*3}  {'─'*9}  {'─'*30}")
        for i, (name, val) in enumerate(top, 1):
            print(f"  {i:>3}  {val:>+9.4f}  {name}")

    def print(self, top_paths: int = 20, top_components: int = 30):
        """Pretty-print all three levels."""
        print("=" * 70)
        print(f"  target: {self.target_token!r}  "
              f"p={self.target_prob:.4f}  "
              f"logit={self.target_logit_centered:+.3f}")
        if self.root != "target":
            print(f"  root:   {self.root}")
        print("=" * 70)

        self.print_tokens()
        self.print_paths(top_k=top_paths)
        self.print_components(top_k=top_components)

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-safe with numpy conversion)."""
        return {
            "tokens": self.tokens,
            "target_token": self.target_token,
            "target_prob": float(self.target_prob),
            "target_logit_centered": float(self.target_logit_centered),
            "root": self.root,
            "token_attribution": self.token_attribution.tolist(),
            "paths": [
                {
                    "chain": p.chain,
                    "source_pos": p.source_pos,
                    "source_token": p.source_token,
                    "score": p.score,
                    "raw_score": p.raw_score,
                    "modes": p.modes,
                    "depth": p.depth,
                }
                for p in self.paths
            ],
            "component_flow": {
                k: float(v) for k, v in self.component_flow.items()
            },
        }
