"""
unpack.config - Trace configuration and named presets.

The five configurations from the paper span three independent axes:
  1. Attention key-side: K-only vs K+Q+V branches
  2. MLP dispatch: weighted (per-neuron) vs L2 (per-component norm)
  3. V-side dispatch: raw (activation-weighted) vs aligned (output-projected)

Named presets correspond to the configurations tested in the paper.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class TraceConfig:
    """Configuration for a single trace run.

    Attributes:
        beta: SafeDenom amplification floor. Default 0.8.
        branches: Which attention branches to trace. "k" for K-only,
            "kqv" for all three composition modes.
        branch_weights: Relative weights for K/Q/V branches when
            branches="kqv". Default (0.3, 0.3, 0.4).
        aligned: If True, use output-aligned V-side dispatch instead
            of raw activation weighting.
        mlp_rule: "weighted" (per-neuron value dispatch) or "l2"
            (per-component L2 norm).
        top_paths_k: Number of top paths to extract.
        path_min_frac: Minimum |score| fraction for path pruning.
    """
    beta: float = 0.8
    branches: str = "k"           # "k" or "kqv"
    branch_weights: Optional[Dict[str, float]] = None
    aligned: bool = False
    mlp_rule: str = "weighted"    # "weighted" or "l2"
    top_paths_k: int = 2000
    path_min_frac: float = 1e-4

    @property
    def enable_q_side(self) -> bool:
        return self.branches == "kqv"

    @property
    def enable_v_side(self) -> bool:
        return self.branches == "kqv"

    @property
    def _branch_weights_dict(self) -> Optional[Dict[str, float]]:
        if self.branches == "k":
            return None
        if self.branch_weights is not None:
            return self.branch_weights
        return {"K": 0.333, "Q": 0.333, "V": 0.333}

    def __post_init__(self):
        if self.branches not in ("k", "kqv"):
            raise ValueError(
                f"branches must be 'k' or 'kqv', got {self.branches!r}")
        if self.mlp_rule not in ("weighted", "l2"):
            raise ValueError(
                f"mlp_rule must be 'weighted' or 'l2', got {self.mlp_rule!r}")
        if self.aligned and self.mlp_rule == "l2":
            raise ValueError(
                "aligned=True requires mlp_rule='weighted' "
                "(L2 does not decompose at per-neuron level).")


# ── Named presets matching the paper's six configurations ──

PRESETS = {
    "default": TraceConfig(
        branches="k", mlp_rule="weighted", aligned=False,
    ),
    "k_only_l2": TraceConfig(
        branches="k", mlp_rule="l2", aligned=False,
    ),
    "k_only_aligned": TraceConfig(
        branches="k", mlp_rule="weighted", aligned=True,
    ),
    "kqv_weighted": TraceConfig(
        branches="kqv", mlp_rule="weighted", aligned=False,
    ),
    "kqv_l2": TraceConfig(
        branches="kqv", mlp_rule="l2", aligned=False,
    ),
    "kqv_aligned": TraceConfig(
        branches="kqv", mlp_rule="weighted", aligned=True,
    ),
}


def get_config(config) -> TraceConfig:
    """Resolve a config argument to a TraceConfig.

    Args:
        config: a TraceConfig instance, a preset name string, or None
            (returns the default).
    """
    if config is None:
        return PRESETS["default"]
    if isinstance(config, TraceConfig):
        return config
    if isinstance(config, str):
        if config not in PRESETS:
            raise ValueError(
                f"Unknown config preset {config!r}. "
                f"Available: {list(PRESETS.keys())}")
        return PRESETS[config]
    raise TypeError(f"config must be str, TraceConfig, or None; got {type(config)}")