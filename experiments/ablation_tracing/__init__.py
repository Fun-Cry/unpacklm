"""ablation_tracing — circuit tracing pipeline.

Top-level re-exports of the core API.  See `.core` for the runner and
`.ioi` for IOI-specific prompt + condition definitions.
"""

from .core import (
    # primitives
    TraceResult, Intervention, trace,
    AblationConfig, build_intervention,
    compare,
    DiffResult, ComponentDiff, PathDiff, EdgeDiff,
    SharedPathDiff, GainedOrLostPath,
    # runner
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