"""
unpack.circuit - Circuit representation and I/O.

A Circuit is a set of model components (attention heads and MLPs)
discovered by the attribution algorithm or specified by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Set, Union


@dataclass
class Circuit:
    """A set of model components forming a discovered circuit.

    Attributes:
        components: Frozenset of component name strings, e.g.
            {"attn_9_head_9", "attn_7_head_3", "mlp_5"}.
        model_name: The model this circuit was discovered on.
        config_name: Which trace configuration produced this circuit.
        faith: Faith ratio (sufficiency score), if verified.
        knockout: Knockout score (necessity score), if verified.
    """
    components: FrozenSet[str] = field(default_factory=frozenset)
    model_name: Optional[str] = None
    config_name: Optional[str] = None
    faith: Optional[float] = None
    knockout: Optional[float] = None

    def __post_init__(self):
        if not isinstance(self.components, frozenset):
            self.components = frozenset(self.components)

    # ── Accessors ──

    @property
    def heads(self) -> FrozenSet[str]:
        """Just the attention heads in the circuit."""
        return frozenset(c for c in self.components if c.startswith("attn_"))

    @property
    def mlps(self) -> FrozenSet[str]:
        """Just the MLPs in the circuit."""
        return frozenset(c for c in self.components if c.startswith("mlp_"))

    def __len__(self) -> int:
        return len(self.components)

    def __contains__(self, item: str) -> bool:
        return item in self.components

    def __iter__(self):
        return iter(self.components)

    # ── Set operations ──

    def __and__(self, other: "Circuit") -> "Circuit":
        """Intersection of two circuits."""
        return Circuit(components=self.components & other.components)

    def __or__(self, other: "Circuit") -> "Circuit":
        """Union of two circuits."""
        return Circuit(components=self.components | other.components)

    def __sub__(self, other: "Circuit") -> "Circuit":
        """Difference: components in self but not in other."""
        return Circuit(components=self.components - other.components)

    def __xor__(self, other: "Circuit") -> "Circuit":
        """Symmetric difference."""
        return Circuit(components=self.components ^ other.components)

    # ── I/O ──

    def save(self, path: str):
        """Save circuit to a JSON file."""
        data = {
            "components": sorted(self.components),
            "model_name": self.model_name,
            "config_name": self.config_name,
            "faith": self.faith,
            "knockout": self.knockout,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Circuit":
        """Load circuit from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        # Support both the new format (dict with metadata) and the
        # old format (plain list of component names).
        if isinstance(data, list):
            return cls(components=frozenset(data))

        return cls(
            components=frozenset(data["components"]),
            model_name=data.get("model_name"),
            config_name=data.get("config_name"),
            faith=data.get("faith"),
            knockout=data.get("knockout"),
        )

    @classmethod
    def from_components(cls, components: Union[Set[str], list],
                        **kwargs) -> "Circuit":
        """Create a Circuit from a set or list of component names."""
        return cls(components=frozenset(components), **kwargs)

    def __repr__(self) -> str:
        parts = [f"Circuit({len(self)} components"]
        parts.append(f"{len(self.heads)} heads, {len(self.mlps)} MLPs")
        if self.faith is not None:
            parts.append(f"faith={self.faith:.3f}")
        if self.knockout is not None:
            parts.append(f"knockout={self.knockout:.3f}")
        return ", ".join(parts) + ")"
