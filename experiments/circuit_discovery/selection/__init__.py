"""Circuit selection: pick which components form the candidate circuit.

Selection is task-agnostic. Role-aware filtering uses a role mapping
supplied by the caller (or by --task on the CLI; see __main__.py).

Strategies live in selection/strategies/. Each module exposes
    select(prompts, *, terminal_role=None, role_keys=(),
           exclude=DEFAULT_EXCLUDE, **kwargs) -> SelectionResult

Add a strategy by dropping a file in strategies/ and registering it
in METHODS below.
"""

from .result import SelectionResult
from .strategies import partition_coverage, component_flow

METHODS = {
    "partition_coverage": partition_coverage.select,
    "component_flow":     component_flow.select,
}

__all__ = ["SelectionResult", "METHODS"]