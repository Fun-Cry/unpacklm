"""
ablation.py — AblationConfig + build_intervention.

Builds an Intervention object from a config + reference prompts.  The
caller installs the result via `trace(intervention=...)`:

    abc_refs = [
        "When Charlie and Alice went to the store. Bob gave a book to",
        "When Frank and Grace went to the store. Helen gave a book to",
        # ... 13 more length-matched ABC sentences ...
    ]

    cfg = AblationConfig(
        components = ["attn_9_head_9"],
        mode       = "mean",
        references = abc_refs,
        positions  = "target",
    )

    clean        = trace(model, tok, sentence, " Mary", hook_manager=hm)
    intervention = build_intervention(model, tok, hm, sentence, cfg)
    ablated      = trace(model, tok, sentence, " Mary",
                         hook_manager=hm, intervention=intervention)

Design notes
------------

Replace-based interventions.  Each component's contribution at the
chosen positions is overwritten with a value drawn from references —
no delta against the target's clean values.  This means:

- The intervention is well-defined regardless of upstream interventions
  modifying the residual: the slice is unconditionally overwritten.
- Heads are ablated at ``attn_{L}_pre_dense`` (the W_O input) by writing
  into the per-head z-slice.  By W_O linearity this produces the same
  residual-stream effect as replacing the head's post-W_O contribution,
  but the modification flows through the streamer's per-head
  reconstruction consistently (the sanity check verifies this).
- MLP / embedding ablations write at their own capture hooks, which are
  the natural intervention points for whole-component replacement.

Reference length must match the target's tokenized length exactly.

Positions:
    "target"  — only at the target query position (McGrath, default)
    "all"     — every position in the sequence (Wang's whole-sequence)
    list[int] — explicit position list, for surgical interventions
"""

import re
from dataclasses import dataclass
from typing import Callable, List, Literal, Tuple, Union

import numpy as np
import torch

from unpack.core import ComponentStreamer
from unpack._interventions import replace_at, replace_head_slice

from .trace import Intervention


# ============================================================
#   Public schema
# ============================================================

PositionSpec = Union[Literal["target", "all"], List[int]]


@dataclass
class AblationConfig:
    """Specification for one ablation: what, how, where.

    Validated eagerly in __post_init__ — bad configs raise at
    construction, not deep inside build_intervention.

    Fields
    ------
    components : list of component names to ablate together.  Names follow
        the convention "attn_{L}_head_{h}", "mlp_{L}", "embedding",
        "pos_embedding".  Multiple components ablated jointly are wired
        as separate intervention callables; same-hook callables (e.g.
        two heads of the same layer at attn_{L}_pre_dense) compose via
        the hook manager's list-based dispatch.

    mode : "mean" averages the component's contribution across all
        references; "resample" uses one specific reference selected by
        resample_index.

    references : list of reference sentences.  Each must tokenize to the
        same sequence length as the target.

    positions : "target" / "all" / list of int — where in the sequence
        the swap is applied.

    resample_index : only used when mode="resample".  Index into
        references.  Default 0.
    """
    components:     List[str]
    mode:           Literal["mean", "resample"]
    references:     List[str]
    positions:      PositionSpec        = "target"
    resample_index: int                 = 0

    def __post_init__(self):
        if not self.components:
            raise ValueError("AblationConfig.components is empty.")
        for c in self.components:
            if not isinstance(c, str) or not c:
                raise ValueError(
                    f"component name must be non-empty str, got {c!r}"
                )

        if self.mode not in ("mean", "resample"):
            raise ValueError(
                f"mode must be 'mean' or 'resample', got {self.mode!r}"
            )

        if not isinstance(self.references, list) or not self.references:
            raise ValueError(
                "AblationConfig.references must be a non-empty list of strings."
            )
        for i, r in enumerate(self.references):
            if not isinstance(r, str) or not r:
                raise ValueError(
                    f"references[{i}] must be a non-empty str, got {r!r}"
                )

        if self.mode == "resample":
            if not (0 <= self.resample_index < len(self.references)):
                raise ValueError(
                    f"resample_index={self.resample_index} out of range "
                    f"[0, {len(self.references)}) for mode='resample'."
                )

        if isinstance(self.positions, str):
            if self.positions not in ("target", "all"):
                raise ValueError(
                    f"positions string must be 'target' or 'all', "
                    f"got {self.positions!r}"
                )
        elif isinstance(self.positions, list):
            if not self.positions:
                raise ValueError("positions list is empty.")
            for p in self.positions:
                if not isinstance(p, int) or p < 0:
                    raise ValueError(
                        f"positions list must contain non-negative ints; "
                        f"got {p!r}"
                    )
        else:
            raise ValueError(
                f"positions must be 'target', 'all', or list[int]; "
                f"got {type(self.positions).__name__}"
            )


# ============================================================
#   Internal: component-name parsing
# ============================================================

# Heads ablate at attn_{L}_pre_dense (per-head slice of pre-W_O tensor).
# MLP/embedding ablate at their own capture hooks (full-tensor replace).
# These are the only three component types we support; anything else
# raises in _component_kind below.

_RE_HEAD = re.compile(r"^attn_(\d+)_head_(\d+)$")
_RE_MLP  = re.compile(r"^mlp_(\d+)$")


def _component_kind(name: str):
    """Parse a component name into (kind, layer, head_idx).

    kind is "head" / "mlp" / "embedding" / "pos_embedding".
    layer and head_idx are None where they don't apply.
    """
    if name == "embedding":
        return ("embedding", None, None)
    if name == "pos_embedding":
        return ("pos_embedding", None, None)
    m = _RE_HEAD.match(name)
    if m:
        return ("head", int(m.group(1)), int(m.group(2)))
    m = _RE_MLP.match(name)
    if m:
        return ("mlp", int(m.group(1)), None)
    raise ValueError(
        f"Unrecognized component name: {name!r}.  Supported: "
        f"'attn_{{L}}_head_{{h}}', 'mlp_{{L}}', 'embedding', 'pos_embedding'."
    )


# ============================================================
#   Internal: extract reference values per component
# ============================================================

def _extract_reference_values(
    model, tokenizer, hook_manager,
    ref_text: str,
    components: List[str],
    head_size: int,
) -> Tuple[dict, int]:
    """Run a clean forward on `ref_text` and pull the captured tensor
    each component needs as its replacement.

    Returns ({comp_name: tensor}, seq_len) where each tensor is on CPU.

    Per component-type:
      - head:          pre_dense_inputs[L] sliced at head's z range.
                       shape (1, S, head_size).
      - mlp:           mlp_outputs[L].          shape (1, S, d_model).
      - embedding:     embedding_outputs[0].    shape (1, S, d_model).
      - pos_embedding: position_embedding_outputs[0].  shape (1, S, d_model).

    Reads buffers directly — does not call iter_source_groups (which
    consumes pre_dense_inputs / mlp_outputs).
    """
    streamer = ComponentStreamer(model, tokenizer, hook_manager)
    streamer.set_context(ref_text)
    seq_len = int(streamer.seq_lens[0])

    out = {}
    for name in components:
        kind, layer, head_idx = _component_kind(name)

        if kind == "head":
            pre_dense = hook_manager.pre_dense_inputs[layer]      # (1, S, d_model)
            unmerged  = pre_dense.view(*pre_dense.shape[:-1],
                                       -1, head_size)             # (1, S, H, head_size)
            z         = unmerged[..., head_idx, :].detach().cpu().clone()
            out[name] = z                                         # (1, S, head_size)

        elif kind == "mlp":
            out[name] = hook_manager.mlp_outputs[layer].detach().cpu().clone()

        elif kind == "embedding":
            out[name] = hook_manager.embedding_outputs[0].detach().cpu().clone()

        elif kind == "pos_embedding":
            pos_outs = getattr(hook_manager, "position_embedding_outputs", None)
            if not pos_outs:
                raise ValueError(
                    f"Model has no position_embedding_outputs buffer; "
                    f"cannot ablate {name!r}."
                )
            out[name] = pos_outs[0].detach().cpu().clone()

    return out, seq_len


# ============================================================
#   Internal: positions
# ============================================================

def _resolve_positions(
    spec: PositionSpec, seq_len: int, target_pos: int,
) -> List[int]:
    if spec == "target":
        return [target_pos]
    if spec == "all":
        return list(range(seq_len))
    out = []
    for p in spec:
        if not (0 <= p < seq_len):
            raise ValueError(
                f"position {p} out of range [0, {seq_len}) for target sequence."
            )
        out.append(int(p))
    return out


# ============================================================
#   Internal: per-component-type intervention factory dispatch
# ============================================================

def _build_replacer(
    comp_name: str, replacement: torch.Tensor,
    positions: List[int], head_size: int,
) -> Tuple[str, Callable]:
    """Build (hook_name, callable) for one component's replacement.

    Each component type knows its own hook target and which factory
    from core.py to use.  Heads write into a pre_dense slice; MLP /
    embedding write the full tensor.
    """
    kind, layer, head_idx = _component_kind(comp_name)

    if kind == "head":
        hook = f"attn_{layer}_pre_dense"
        fn   = replace_head_slice(replacement, head_idx, head_size, positions)
        return hook, fn

    if kind == "mlp":
        hook = f"mlp_{layer}"
        fn   = replace_at(replacement, positions)
        return hook, fn

    if kind == "embedding":
        hook = "embedding"
        fn   = replace_at(replacement, positions)
        return hook, fn

    if kind == "pos_embedding":
        hook = "pos_embedding"
        fn   = replace_at(replacement, positions)
        return hook, fn

    raise AssertionError(f"unreachable: unknown kind {kind!r} for {comp_name!r}")


# ============================================================
#   Public: build_intervention
# ============================================================

def build_intervention(
    model,
    tokenizer,
    hook_manager,
    sentence:        str,
    ablation:        AblationConfig,
    *,
    target_position = "last",
) -> Intervention:
    """Build a ready-to-install Intervention from an AblationConfig.

    Runs reference forward passes, extracts replacement values from the
    captured component buffers (in pre-W_O space for heads; in module
    output space for MLP / embedding), aggregates per `ablation.mode`,
    masks to the requested positions, and packs into hook callables.

    Returned `Intervention.interventions` is a list of (hook_name, fn) pairs;
    multiple pairs sharing a hook (e.g. joint same-layer head ablation)
    are dispatched sequentially by the hook manager's list-based
    intervention dispatch.

    Raises ValueError if a reference's tokenized length doesn't match
    the target's, or if a position is out of range.
    """
    if hook_manager is None:
        raise ValueError(
            "build_intervention requires hook_manager.  Pass the right "
            "manager for your model (e.g. GPT2HookManager())."
        )

    # Register hooks if needed — idempotent.
    if not hook_manager.handles:
        hook_manager.register_hooks(model)

    # Per-head dimensionality comes from the manager — it knows the
    # model's architecture without us having to peek at config attrs
    # whose names vary between model families.
    head_size = hook_manager.get_head_size()

    # Resolve target sequence length to validate references against.
    streamer = ComponentStreamer(model, tokenizer, hook_manager)
    streamer.set_context(sentence)
    target_seq_len = int(streamer.seq_lens[0])
    t_pos = (target_seq_len - 1 if target_position == "last"
             else int(target_position))

    positions = _resolve_positions(ablation.positions, target_seq_len, t_pos)

    # Extract replacement values from references.  For mode='resample',
    # only one reference is consulted; for 'mean', all of them.
    refs = ([ablation.references[ablation.resample_index]]
            if ablation.mode == "resample" else ablation.references)

    per_ref: dict = {n: [] for n in ablation.components}
    for ref_text in refs:
        vecs, ref_len = _extract_reference_values(
            model, tokenizer, hook_manager, ref_text,
            ablation.components, head_size,
        )
        if ref_len != target_seq_len:
            raise ValueError(
                f"Reference length {ref_len} != target length "
                f"{target_seq_len} for reference {ref_text!r}.  Per-position "
                "ablation requires length-matched references; check that "
                "references share the target's template and use a "
                "single-token-verified name pool."
            )
        for name, vec in vecs.items():
            per_ref[name].append(vec)

    # Aggregate: mean across collected references, or pick the single
    # resampled one (which is already a length-1 list).
    aggregated = {
        name: torch.stack(arrs, dim=0).mean(dim=0)
        for name, arrs in per_ref.items()
    }

    # Build (hook, fn) pairs by dispatching per component type.
    pairs: List[Tuple[str, Callable]] = []
    for name in ablation.components:
        replacement = aggregated[name]
        hook, fn = _build_replacer(name, replacement, positions, head_size)
        pairs.append((hook, fn))

    return Intervention(
        interventions      = pairs,
        ablated_components = set(ablation.components),
    )