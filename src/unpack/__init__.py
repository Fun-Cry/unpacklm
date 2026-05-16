"""
UNPACK: Unified Path Attribution through Component Keys.

Tokens, paths, and circuits from a single decomposition.

Quick start:
    import unpack

    tracer = unpack.Tracer("gpt2")
    result = tracer.trace(
        "Mary and John went to the store. John gave the bag to",
        target=" Mary",
        distractor=" John",
    )
    result.print()

Three levels of output:
    result.token_attribution    # Level 1: per-token signed credit
    result.paths                # Level 2: named end-to-end routes
    result.component_flow       # Bridge: per-component aggregate

See also:
    tracer.trace()     — single-prompt attribution
    unpack.discover()  — multi-prompt circuit discovery
    unpack.discover_one() — single-prompt circuit discovery
    tracer.verify()    — circuit verification via ablation (coming soon)
    unpack.validate()  — adapter contract validation
"""

__version__ = "0.1.0"

from unpack.tracer import Tracer
from unpack.result import TraceResult, Path
from unpack.circuit import Circuit
from unpack.discover import discover, discover_one
from unpack.config import TraceConfig, PRESETS
from unpack.validate import validate
from unpack.tasks.ioi import ioi_roles, WANG_CIRCUIT_HEADS

__all__ = [
    # Main entry point
    "Tracer",
    # Discovery
    "discover",
    "discover_one",
    # Output types
    "TraceResult",
    "Path",
    "Circuit",
    # Configuration
    "TraceConfig",
    "PRESETS",
    # Validation
    "validate",
    # Task helpers
    "ioi_roles",
    "WANG_CIRCUIT_HEADS",
]