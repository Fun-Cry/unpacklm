"""ablation_tracing.core — task-agnostic tracing pipeline.

    trace, build_intervention, compare           — primitives
    ExperimentConfig, run                         — generic runner
"""

from .trace      import TraceResult, Intervention, trace
from .ablation   import AblationConfig, build_intervention
from .compare    import (
    compare,
    DiffResult, ComponentDiff, PathDiff, EdgeDiff,
    SharedPathDiff, GainedOrLostPath,
)
from .runner     import (
    ExperimentConfig, ExperimentRun,
    run, load_runs,
)

__all__ = [
    "TraceResult", "Intervention", "trace",
    "AblationConfig", "build_intervention",
    "compare",
    "DiffResult", "ComponentDiff", "PathDiff", "EdgeDiff",
    "SharedPathDiff", "GainedOrLostPath",
    "ExperimentConfig", "ExperimentRun",
    "run", "load_runs",
]