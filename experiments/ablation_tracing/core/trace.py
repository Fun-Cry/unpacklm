"""
simple_trace.py — minimal interface for clean / ablated tracing.

This module is the new replacement for `experiments/ablation_tracing/`.
The old module is dead; everything we need is here:

    Step 1 (this file, clean only): TraceResult + trace()
    Step 2: AblationConfig + build_intervention() + trace(intervention=...)
    Step 3: compare(clean, ablated) -> DiffResult

Step 1: clean run.

The user passes a sentence and a target token. We run trace_flow,
reshape the output into a TraceResult, return. No ablation logic in
this step at all.

The TraceResult schema:

    sentence, tokens, target_token, target_prob, seq_len
        Bookkeeping. tokens is the list of decoded token strings,
        useful for printing / inspection.

    flow:    dict[component_name, signed_float]
        Path-credit through each component, summed across all positions
        the trace tree visited. The sum-over-positions is the natural
        compression because the same component at different positions
        contributes to different paths but rolls up to the same head /
        MLP identity.

    direct:  dict[component_name, signed_float]
        Direct logit attribution at the target position only. By
        construction this lives at one position — there's no per-position
        version because it's only meaningful at the target.

    paths:   list[(chain, src_pos, signed_score)]
        Top-K signed paths, ordered by |score|. Each path is a tuple
        of component names ordered target-ward (head of chain) ->
        source-ward (input). src_pos is the token index where the
        terminal source (embedding / pos_embedding) lives. Score is
        signed: positive = pushed target probability up, negative =
        down.

    edges_attn: dict[(src_component, tgt_component), signed_float]
    edges_mlp:  dict[(src_component, tgt_component), signed_float]
        Forward edges from the AttentionScorer / MLPScorer in core.py,
        evaluated at the target query position. Key is (source, target)
        where source is an upstream component name and target is a head
        / MLP name. Score is the std (attention) or L2 norm (MLP) of
        the source's contribution to the target's pre-softmax / pre-MLP
        signal at the target query position.

Component naming convention (inherited from the existing tracer):
    attention head:  "attn_{L}_head_{h}"     e.g. "attn_9_head_9"
    MLP:             "mlp_{L}"               e.g. "mlp_5"
    embedding:       "embedding"
    pos embedding:   "pos_embedding"   (only present for some models)
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

# trace_flow lives at the repo root (next to core.py, trace_recursive.py).
from unpack.compat import trace_flow as _trace_flow


# ============================================================
#   Public schema
# ============================================================

# Each path is a chain of component names, ordered target-ward
# (head of chain) -> source-ward (input). The source position is the
# token index where credit lands (the final embedding's position);
# internal components in the path don't carry per-step positions
# because the recursive walker only emits the terminal source pos.
PathChain  = Tuple[str, ...]            # e.g. ("attn_9_head_9", "mlp_0", "embedding")
SignedPath = Tuple[PathChain, int, float]   # (chain, src_pos, score)

# Edge keys are (source_component, target_component).
EdgeKey = Tuple[str, str]


@dataclass
class TraceResult:
    """Output of one trace, clean or ablated. Self-contained."""

    sentence:     str
    tokens:       List[str]
    target_token: str
    target_prob:  float
    seq_len:      int

    # Centered target logit at the trace's query position:
    #     target_logit - mean(logits over vocab)
    # This is the softmax-relevant quantity (softmax is shift-invariant)
    # and the canonical reference for "what is the model's effective
    # output for this target".  IMPORTANT: this is NOT generally equal
    # to ``sum(direct.values())`` — the per-component DLA dict drops
    # bias terms (attn output bias, layernorm bias, etc.).  The
    # difference is the bias contribution to the centered logit; see
    # ``bias_residual()``.
    target_logit_centered: float = 0.0

    # Per-component scalars at the target.
    flow:    Dict[str, float] = field(default_factory=dict)
    direct:  Dict[str, float] = field(default_factory=dict)

    # Ranked top-K signed paths.
    paths:   List[SignedPath] = field(default_factory=list)

    # Forward edges, target-query position only.
    edges_attn: Dict[EdgeKey, float] = field(default_factory=dict)
    edges_mlp:  Dict[EdgeKey, float] = field(default_factory=dict)

    def top_components(self, n: int = 10, by: str = "flow",
                       absolute: bool = True) -> List[Tuple[str, float]]:
        """Convenience: top-n components by flow or direct.

        Args:
            n:        how many to return.
            by:       "flow" or "direct".
            absolute: if True, rank by |score|; else by signed score.
        """
        src = self.flow if by == "flow" else self.direct
        items = list(src.items())
        items.sort(key=(lambda kv: abs(kv[1])) if absolute
                   else (lambda kv: kv[1]),
                   reverse=True)
        return items[:n]

    def bias_residual(self) -> float:
        """How much of the centered target logit is unaccounted for by
        the per-component direct dict.

        Equal to ``target_logit_centered - sum(direct.values())``.
        Comes from bias terms (W_O bias, layernorm bias, etc.) that the
        streamer drops during decomposition — they're real
        contributions to the model's logit, but not associated with
        any single named component.

        Useful as a sanity check: if ``abs(bias_residual()) >`` a few %
        of ``abs(target_logit_centered)``, the per-component scores
        are missing a non-trivial chunk of the actual signal.
        """
        return self.target_logit_centered - float(sum(self.direct.values()))


@dataclass
class Intervention:
    """A pre-built ablation, ready to install on a hook_manager and run.

    Produced by `build_intervention()` (in ablation.py).  Consumed by
    `trace(intervention=...)`.

    Fields
    ------
    interventions:
        List of ``(hook_name, fn)`` pairs.  ``fn`` is a callable
        ``(tensor, name) -> tensor`` suitable for
        ``HookManager.register_intervention``.  Multiple pairs sharing
        the same ``hook_name`` are registered sequentially and
        composed by the hook manager's list-based dispatch — this
        gives joint same-layer ablation (e.g. two heads at L=9 sharing
        the ``attn_9_pre_dense`` hook) for free.

    ablated_components:
        Set of component names whose contributions were swapped.
        Passed to the backward-attribution sweep so credit propagation
        terminates at these components rather than tracing through
        replaced contents.
    """
    interventions:      List[Tuple[str, Callable]]
    ablated_components: Set[str]


# ============================================================
#   Internal: reshape trace_flow's dict into TraceResult
# ============================================================

def _compress_flow(component_flow: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Sum each component's per-position credit array to a single scalar.

    Sum (rather than e.g. taking the target position) is correct because
    a component contributes to the target via the trace tree at any
    position the tree visits — for an MLP at layer 0 reading the IO
    token, that contribution lives at the IO position, not at the
    target position. The sum aggregates all such routes.
    """
    return {name: float(arr.sum()) for name, arr in component_flow.items()}


_PATH_SEP = "→"   # used by trace_recursive.py to glue components in a path string


def _build_paths(
    top_paths_raw: List[Tuple[str, int, float]],
) -> List[SignedPath]:
    """Parse trace_flow's path representation into (chain, src_pos, score).

    trace_flow.top_paths_raw is a list of (path_string, source_position,
    raw_credit). The path_string is built by trace_recursive as
    "name1→name2→...→nameK", target-ward to source-ward (the final
    name is the embedding / pos_embedding terminal).

    We split on "→" to recover the chain as a tuple of names. Score is
    raw signed credit in centered-logit units — directly comparable to
    `direct` scores.

    Bias-marker entries like "[ABLATED:attn_9_head_9]" are passed
    through as single-element chains; consumers that don't care about
    the ablation marker can filter them out.
    """
    out: List[SignedPath] = []
    for path_str, src_pos, credit in top_paths_raw:
        chain = tuple(path_str.split(_PATH_SEP))
        out.append((chain, int(src_pos), float(credit)))
    return out


def _build_edges_attn(
    forward_attn: Dict[Tuple[int, int, str], float],
) -> Dict[EdgeKey, float]:
    """Reshape trace_flow's (L, H, src_name) -> std into (src, tgt) -> std.

    Target component is the attention head at (L, H), named
    "attn_{L}_head_{H}". Source name is already a component name.
    """
    out: Dict[EdgeKey, float] = {}
    for (L, H, src_name), score in forward_attn.items():
        tgt_name = f"attn_{L}_head_{H}"
        out[(src_name, tgt_name)] = float(score)
    return out


def _build_edges_mlp(
    forward_mlp: Dict[Tuple[int, str], float],
) -> Dict[EdgeKey, float]:
    """Reshape trace_flow's (L, src_name) -> norm into (src, tgt) -> norm."""
    out: Dict[EdgeKey, float] = {}
    for (L, src_name), score in forward_mlp.items():
        tgt_name = f"mlp_{L}"
        out[(src_name, tgt_name)] = float(score)
    return out


def _filter_edges_topk(
    edges: Dict[EdgeKey, float],
    top_k_per_node: int,
) -> Dict[EdgeKey, float]:
    """Keep an edge if it ranks in the top-K outgoing edges of its source
    OR the top-K incoming edges of its target.

    Bounds storage at roughly 2·K·n_components regardless of total edge
    count — the L²·H² blowup on big models doesn't propagate downstream.
    Per-component top-K matches the granularity of the compare() diff,
    which only ever asks about edges adjacent to a specific component.

    The union (source OR target) means a low-magnitude edge between two
    busy components — which is in neither's local top-K — gets dropped.
    For the rewired-bucket analysis this is fine: rewiring shows up as
    large-magnitude changes that are easily within K=50.
    """
    if top_k_per_node is None or top_k_per_node <= 0:
        return dict(edges)

    # Group by source and by target.
    by_src: Dict[str, List[Tuple[EdgeKey, float]]] = {}
    by_tgt: Dict[str, List[Tuple[EdgeKey, float]]] = {}
    for key, score in edges.items():
        src, tgt = key
        by_src.setdefault(src, []).append((key, score))
        by_tgt.setdefault(tgt, []).append((key, score))

    keep: set = set()
    for group in (by_src.values(), by_tgt.values()):
        for entries in group:
            entries.sort(key=lambda kv: abs(kv[1]), reverse=True)
            for k, _ in entries[:top_k_per_node]:
                keep.add(k)

    return {k: edges[k] for k in keep}


def _result_from_trace_flow_dict(
    d: Dict, sentence: str, edges_top_k_per_node: int = 50,
) -> TraceResult:
    """Pure data reshape — no model interaction.

    edges_top_k_per_node bounds per-component-neighborhood edge storage.
    None or 0 to keep all edges.
    """
    edges_attn = _build_edges_attn(d.get("forward_attn_at_t", {}))
    edges_mlp  = _build_edges_mlp (d.get("forward_mlp_at_t",  {}))
    if edges_top_k_per_node:
        edges_attn = _filter_edges_topk(edges_attn, edges_top_k_per_node)
        edges_mlp  = _filter_edges_topk(edges_mlp,  edges_top_k_per_node)

    return TraceResult(
        sentence              = sentence,
        tokens                = list(d["tokens"]),
        target_token          = d["target_token"],
        target_prob           = float(d["target_prob"]),
        target_logit_centered = float(d["target_logit_centered"]),
        seq_len               = int(len(d["tokens"])),

        flow   = _compress_flow(d["component_flow"]),
        direct = {k: float(v) for k, v in d["importance"].items()},

        paths      = _build_paths(d["top_paths_raw"]),
        edges_attn = edges_attn,
        edges_mlp  = edges_mlp,
    )


# ============================================================
#   Public entry point: trace()
# ============================================================

def trace(
    model,
    tokenizer,
    sentence: str,
    target_token: str,
    *,
    distractor_token: str = None,
    target_position = "last",
    beta: float = 0.3,
    top_paths_k: int = 20,
    path_min_frac: float = 1e-3,
    edges_top_k_per_node: int = 50,
    hook_manager = None,
    intervention: Optional[Intervention] = None,
    enable_q_side: bool = True,
    enable_v_side: bool = True,
    branch_weights = None,
    geomean_min = None,
    mlp_geva_enabled: bool = False,
    mlp_outproj_enabled: bool = False,
    attn_outproj_enabled: bool = False,
) -> TraceResult:
    """Run a trace on `sentence` and return a TraceResult.

    Without `intervention`, this is a clean trace.  With `intervention`,
    its interventions are installed on `hook_manager` for the duration of the
    trace and `intervention.ablated_components` terminates the backward
    sweep at swapped components.

    Args:
        model, tokenizer: loaded HF model + tokenizer.
        sentence:         input prompt.
        target_token:     answer token (with leading space for GPT-2-style
                          BPE), e.g. " Mary".
        distractor_token: optional, used for IOI-style contrastive logit
                          differences. None for plain factual recall.
        target_position:  "last" or an int index.
        beta:             SafeDenom soft-floor for the recursive backward
                          pass (paper §3.3, eq. 5). Default 0.3 matches
                          the paper.
        top_paths_k:      number of top paths to return.
        path_min_frac:    pruning threshold for the recursive path walk
                          (smaller -> more paths, slower).
        edges_top_k_per_node:
                          bound on edge storage. Keep an edge if it's
                          among the top-K outgoing edges of its source
                          OR top-K incoming edges of its target. Set
                          None or 0 to keep all edges (small models
                          only — see the caveat in
                          _filter_edges_topk's docstring).
        hook_manager:     reuse one HookManager across calls to avoid
                          per-call hook setup. Required when
                          `intervention` is given.
        intervention:     pre-built ablation from `build_intervention()`.
                          Installed before the forward pass and cleared
                          after; the caller's interventions state is
                          reset on exit.

    Returns:
        TraceResult.
    """
    if intervention is not None:
        if hook_manager is None:
            raise ValueError(
                "trace(intervention=...) requires hook_manager. The "
                "intervention's callables are installed via the manager's "
                "register_intervention API."
            )
        hook_manager.clear_interventions()
        # Install pairs in order so that same-hook entries compose by
        # sequential application (chain of responsibility on the hook
        # manager side).
        for hook_name, fn in intervention.interventions:
            hook_manager.register_intervention(hook_name, fn)
        ablated_components = intervention.ablated_components
    else:
        ablated_components = None

    try:
        raw = _trace_flow(
            model, tokenizer, sentence,
            target_token       = target_token,
            distractor_token   = distractor_token,
            target_position    = target_position,
            beta               = beta,
            top_paths_k        = top_paths_k,
            path_min_frac      = path_min_frac,
            hook_manager       = hook_manager,
            ablated_components = ablated_components,
            enable_q_side      = enable_q_side,
            enable_v_side      = enable_v_side,
            branch_weights     = branch_weights,
            geomean_min        = geomean_min,
            mlp_geva_enabled   = mlp_geva_enabled,
            mlp_outproj_enabled = mlp_outproj_enabled,
            attn_outproj_enabled = attn_outproj_enabled,
        )
    finally:
        if intervention is not None:
            hook_manager.clear_interventions()

    return _result_from_trace_flow_dict(
        raw, sentence, edges_top_k_per_node=edges_top_k_per_node,
    )