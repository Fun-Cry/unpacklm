"""
unpack.core - Algorithm internals.

For most users, the Tracer class is the right entry point.
This subpackage is for power users who need direct access to
the decomposition, recursion, and flow sweep primitives.
"""

from unpack.core.streamer import ComponentStreamer
from unpack.core.scorers import AttentionScorer, MLPScorer
from unpack.core.decomposition import (
    compute_target_direction,
    precompute_attn_shares,
    precompute_attn_shares_outproj,
    compute_mlp_decomp_principled,
    compute_mlp_decomp_geva,
    compute_mlp_decomp_outproj,
)
from unpack.core.recursion import (
    backward_recursive,
    backward_to_tokens,
    set_beta,
    get_beta,
    _safe_denom,
)
from unpack.core.prep import _prepare_trace_inputs
from unpack.core.flow import _run_flow_sweep
from unpack.core.path_views import (
    parse_chain,
    chain_source_position,
    chain_modes,
    dominant_attn_mode,
    group_paths_by_source,
    group_paths_by_mode,
    composition_summary,
    render_grouped,
)

__all__ = [
    "ComponentStreamer", "AttentionScorer", "MLPScorer",
    "compute_target_direction", "precompute_attn_shares",
    "precompute_attn_shares_outproj",
    "compute_mlp_decomp_principled", "compute_mlp_decomp_geva",
    "compute_mlp_decomp_outproj",
    "backward_recursive", "backward_to_tokens",
    "set_beta", "get_beta",
    "_prepare_trace_inputs", "_run_flow_sweep",
    "parse_chain", "chain_source_position", "chain_modes",
    "dominant_attn_mode",
    "group_paths_by_source", "group_paths_by_mode",
    "composition_summary", "render_grouped",
]
