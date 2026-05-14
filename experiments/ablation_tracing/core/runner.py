"""runner.py — generic ablation experiment runner.

Loops over (prompt × condition) cells.  For each cell: trace clean,
trace with ablation, compare, save a JSON file.  Resume-friendly —
existing files are loaded instead of re-traced.

Task-agnostic.  Task-specific drivers (e.g. experiments/ablation_tracing/ioi)
just produce the prompt list and the condition list.

Prompt format
-------------

Each prompt is a dict with at least:
    prompt:           str        the input text
    target_token:     str        the answer token (with leading space if needed)
    references:       List[str]  reference prompts for the swap, length-matched

Optional, recognized by the runner:
    distractor_token: str        for contrastive logit-diff scoring

Any other keys are preserved verbatim into the saved run's `metadata`
field.  IOI prompts, for example, carry IO/S/template_type/abc_seed.

Condition format
----------------

Each condition is a (label, components_to_ablate_jointly) tuple.
Single-head ablation has length-1 components; joint ablation has more.
Labels become filename suffixes — keep them filesystem-safe.

Storage tiers
-------------

    "full":   diff + clean/ablated TraceResults (paths + edges).
              ~500 KB/run on GPT-2 small.  Default.  Lets you re-run
              compare() with new thresholds, or run new analyses
              without re-tracing.

    "medium": diff + clean/ablated direct & flow scalar dicts.
              ~80 KB/run.  Re-classifiable, but no path/edge data.

    "diff":   diff only.  ~50 KB/run.  Committed to all thresholds at
              run time.

Usage
-----

    from experiments.ablation_tracing.core import (
        ExperimentConfig, run,
    )

    cfg = ExperimentConfig(
        prompts    = my_prompts,             # list of dicts (see above)
        conditions = my_conditions,          # [(label, [comp, ...]), ...]
        out_dir    = "results/my_experiment",
    )
    run(model, tokenizer, hook_manager, cfg)
"""

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .ablation  import AblationConfig, build_intervention
from .compare   import (
    compare, DiffResult,
    ComponentDiff, EdgeDiff, GainedOrLostPath, PathDiff, SharedPathDiff,
)
from .trace     import trace, TraceResult


# Required + reserved keys in a prompt dict.  Anything else → metadata.
_REQUIRED_PROMPT_KEYS = {"prompt", "target_token", "references"}
_RESERVED_PROMPT_KEYS = _REQUIRED_PROMPT_KEYS | {"distractor_token"}


# ============================================================
#   Schema
# ============================================================

@dataclass
class ExperimentConfig:
    """Spec for an ablation experiment.

    Required:
        prompts:    list of prompt dicts (see module docstring).
        conditions: list of (label, components_to_ablate_jointly).
        out_dir:    where to write per-cell JSONs.

    Ablation knobs (apply to every condition):
        mode:           'mean' (McGrath) or 'resample' (Wang).
        positions:      'target' / 'all' / list[int].
        resample_index: only used when mode='resample'.

    Trace knobs (apply to both clean and ablated traces):
        beta, top_paths_k, edges_top_k_per_node, path_min_frac.

    Compare knobs:
        eps_path, eps_edge, abl_threshold, delta_threshold.

    Output:
        storage:    'full' / 'medium' / 'diff'.
        verbose:    print progress.
    """
    prompts:    List[Dict[str, Any]]
    conditions: List[Tuple[str, List[str]]]
    out_dir:    Optional[str] = None

    # ablation
    mode:           str                       = "mean"
    positions:      Union[str, List[int]]     = "target"
    resample_index: int                       = 0

    # trace
    beta:                  float = 0.3
    top_paths_k:           int   = 200
    edges_top_k_per_node:  int   = 50
    path_min_frac:         float = 1e-3

    # compare
    eps_path:        float = 0.001
    eps_edge:        float = 0.001
    abl_threshold:   float = 0.05
    delta_threshold: float = 0.02

    # output
    storage: str  = "full"
    verbose: bool = True


@dataclass
class ExperimentRun:
    """One (prompt, condition) cell.  Storage tier controls which fields
    are populated; unset fields stay None — check before reading."""
    # Identity
    prompt_idx:         int
    prompt:             str
    target_token:       str
    distractor_token:   Optional[str]
    label:              str               # condition label
    ablated_components: List[str]         # the joint ablation set
    metadata:           Dict[str, Any]    # task-specific, copied from the prompt dict
    storage:            str

    # Target-side scalars (always populated)
    clean_target_prob:    float
    clean_target_logit:   float
    ablated_target_prob:  float
    ablated_target_logit: float

    # Ablation-set direct-effect scalars (always populated). Mirrored
    # from ``diff`` so the top of each cell's JSON immediately shows
    # what the ablation cost in IO-axis credit, paired naturally with
    # the prob/logit before/after pair above.
    clean_direct_abl:     float
    abl_de_total:         float

    # Diff (always populated)
    diff: DiffResult = None

    # Per-component scalar dicts (medium and full)
    clean_direct:   Optional[Dict[str, float]] = None
    clean_flow:     Optional[Dict[str, float]] = None
    ablated_direct: Optional[Dict[str, float]] = None
    ablated_flow:   Optional[Dict[str, float]] = None

    # Full TraceResult fields (full only)
    clean_paths:        Optional[List[Any]] = None
    ablated_paths:      Optional[List[Any]] = None
    clean_edges_attn:   Optional[Dict[Tuple[str, str], float]] = None
    clean_edges_mlp:    Optional[Dict[Tuple[str, str], float]] = None
    ablated_edges_attn: Optional[Dict[Tuple[str, str], float]] = None
    ablated_edges_mlp:  Optional[Dict[Tuple[str, str], float]] = None


# ============================================================
#   Build / serialize a run
# ============================================================

def _split_metadata(p: Dict) -> Dict[str, Any]:
    return {k: v for k, v in p.items() if k not in _RESERVED_PROMPT_KEYS}


def _build_run(
    prompt_idx: int, p: Dict, label: str, comps: List[str],
    clean: TraceResult, ablated: TraceResult, diff: DiffResult,
    storage: str,
) -> ExperimentRun:
    run = ExperimentRun(
        prompt_idx         = prompt_idx,
        prompt             = p["prompt"],
        target_token       = p["target_token"],
        distractor_token   = p.get("distractor_token"),
        label              = label,
        ablated_components = list(comps),
        metadata           = _split_metadata(p),
        storage            = storage,

        clean_target_prob    = clean.target_prob,
        clean_target_logit   = clean.target_logit_centered,
        ablated_target_prob  = ablated.target_prob,
        ablated_target_logit = ablated.target_logit_centered,

        clean_direct_abl  = diff.clean_direct_abl,
        abl_de_total      = diff.abl_de_total,

        diff = diff,
    )
    if storage in ("medium", "full"):
        run.clean_direct   = dict(clean.direct)
        run.clean_flow     = dict(clean.flow)
        run.ablated_direct = dict(ablated.direct)
        run.ablated_flow   = dict(ablated.flow)
    if storage == "full":
        run.clean_paths        = list(clean.paths)
        run.ablated_paths      = list(ablated.paths)
        run.clean_edges_attn   = dict(clean.edges_attn)
        run.clean_edges_mlp    = dict(clean.edges_mlp)
        run.ablated_edges_attn = dict(ablated.edges_attn)
        run.ablated_edges_mlp  = dict(ablated.edges_mlp)
    return run


# ============================================================
#   JSON serialization (tuple keys & dataclasses)
# ============================================================

_EDGE_SEP = "||"


def _serialize_edges(edges):
    if edges is None:
        return None
    return {f"{src}{_EDGE_SEP}{tgt}": float(score)
            for (src, tgt), score in edges.items()}


def _deserialize_edges(d):
    if d is None:
        return None
    out = {}
    for k, v in d.items():
        src, tgt = k.split(_EDGE_SEP, 1)
        out[(src, tgt)] = float(v)
    return out


def _serialize_paths(paths):
    if paths is None:
        return None
    return [[list(chain), int(src_pos), float(score)]
            for (chain, src_pos, score) in paths]


def _deserialize_paths(lst):
    if lst is None:
        return None
    return [(tuple(chain), int(src_pos), float(score))
            for (chain, src_pos, score) in lst]


def _diff_to_jsonable(diff: DiffResult) -> Dict:
    return dataclasses.asdict(diff)


def _diff_from_jsonable(d: Dict) -> DiffResult:
    components = []
    for c in d["components"]:
        pd = c["paths"]
        paths = PathDiff(
            gained = [GainedOrLostPath(chain=tuple(g["chain"]),
                                       src_pos=g["src_pos"],
                                       score=g["score"])
                      for g in pd["gained"]],
            lost   = [GainedOrLostPath(chain=tuple(g["chain"]),
                                       src_pos=g["src_pos"],
                                       score=g["score"])
                      for g in pd["lost"]],
            shared = [SharedPathDiff(chain=tuple(s["chain"]),
                                     src_pos=s["src_pos"],
                                     clean_score=s["clean_score"],
                                     ablated_score=s["ablated_score"],
                                     delta=s["delta"])
                      for s in pd["shared"]],
        )
        components.append(ComponentDiff(
            name           = c["name"],
            role           = c["role"],
            clean_direct   = c["clean_direct"],
            ablated_direct = c["ablated_direct"],
            delta_direct   = c["delta_direct"],
            clean_flow     = c["clean_flow"],
            ablated_flow   = c["ablated_flow"],
            delta_flow     = c["delta_flow"],
            paths          = paths,
            edges_attn_in  = [EdgeDiff(**e) for e in c["edges_attn_in"]],
            edges_attn_out = [EdgeDiff(**e) for e in c["edges_attn_out"]],
            edges_mlp_in   = [EdgeDiff(**e) for e in c["edges_mlp_in"]],
            edges_mlp_out  = [EdgeDiff(**e) for e in c["edges_mlp_out"]],
        ))
    return DiffResult(
        components         = components,
        ablated_components = list(d["ablated_components"]),
        delta_p_target     = d["delta_p_target"],
        clean_direct_abl   = d["clean_direct_abl"],
        abl_de_total       = d["abl_de_total"],
        eps_path           = d["eps_path"],
        eps_edge           = d["eps_edge"],
    )


def _run_to_jsonable(run: ExperimentRun) -> Dict:
    d = {f.name: getattr(run, f.name) for f in dataclasses.fields(run)}
    d["diff"]               = _diff_to_jsonable(run.diff)
    d["clean_paths"]        = _serialize_paths(run.clean_paths)
    d["ablated_paths"]      = _serialize_paths(run.ablated_paths)
    d["clean_edges_attn"]   = _serialize_edges(run.clean_edges_attn)
    d["clean_edges_mlp"]    = _serialize_edges(run.clean_edges_mlp)
    d["ablated_edges_attn"] = _serialize_edges(run.ablated_edges_attn)
    d["ablated_edges_mlp"]  = _serialize_edges(run.ablated_edges_mlp)
    return d


def _run_from_jsonable(d: Dict) -> ExperimentRun:
    return ExperimentRun(
        prompt_idx         = d["prompt_idx"],
        prompt             = d["prompt"],
        target_token       = d["target_token"],
        distractor_token   = d.get("distractor_token"),
        label              = d["label"],
        ablated_components = list(d["ablated_components"]),
        metadata           = dict(d.get("metadata", {})),
        storage            = d["storage"],

        clean_target_prob    = d["clean_target_prob"],
        clean_target_logit   = d["clean_target_logit"],
        ablated_target_prob  = d["ablated_target_prob"],
        ablated_target_logit = d["ablated_target_logit"],

        # Backward-compat: pre-migration JSONs only have these in d["diff"].
        clean_direct_abl  = d.get("clean_direct_abl",
                                   d["diff"].get("clean_direct_abl", 0.0)),
        abl_de_total      = d.get("abl_de_total",
                                   d["diff"].get("abl_de_total", 0.0)),

        diff           = _diff_from_jsonable(d["diff"]),

        clean_direct   = d.get("clean_direct"),
        clean_flow     = d.get("clean_flow"),
        ablated_direct = d.get("ablated_direct"),
        ablated_flow   = d.get("ablated_flow"),

        clean_paths        = _deserialize_paths(d.get("clean_paths")),
        ablated_paths      = _deserialize_paths(d.get("ablated_paths")),
        clean_edges_attn   = _deserialize_edges(d.get("clean_edges_attn")),
        clean_edges_mlp    = _deserialize_edges(d.get("clean_edges_mlp")),
        ablated_edges_attn = _deserialize_edges(d.get("ablated_edges_attn")),
        ablated_edges_mlp  = _deserialize_edges(d.get("ablated_edges_mlp")),
    )


# ============================================================
#   Disk paths and resume logic
# ============================================================

def _run_json_path(out_dir: str, prompt_idx: int, label: str) -> str:
    safe = label.replace("/", "_").replace(":", "_")
    return os.path.join(out_dir, f"p{prompt_idx:04d}__{safe}.json")


def _save_run(run: ExperimentRun, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_run_to_jsonable(run), f)


def load_runs(out_dir: str) -> List[ExperimentRun]:
    """Load every saved run from a directory back into ExperimentRun objects."""
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(out_dir)
    runs = []
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue   # skip _spec.json and similar metadata files
        with open(os.path.join(out_dir, fn)) as f:
            runs.append(_run_from_jsonable(json.load(f)))
    return runs


# ============================================================
#   Validation
# ============================================================

def _validate_prompt(p: Dict, idx: int) -> None:
    missing = _REQUIRED_PROMPT_KEYS - p.keys()
    if missing:
        raise ValueError(
            f"prompts[{idx}] missing required keys: {sorted(missing)}"
        )


def _validate_config(cfg: ExperimentConfig) -> None:
    if cfg.storage not in ("full", "medium", "diff"):
        raise ValueError(
            f"storage must be 'full' / 'medium' / 'diff', got {cfg.storage!r}"
        )
    for i, p in enumerate(cfg.prompts):
        _validate_prompt(p, i)
    if not cfg.conditions:
        raise ValueError("conditions is empty.")
    for i, c in enumerate(cfg.conditions):
        if (not isinstance(c, (tuple, list))) or len(c) != 2:
            raise ValueError(
                f"conditions[{i}] must be (label, components_list), got {c!r}"
            )
        label, comps = c
        if not isinstance(label, str) or not label:
            raise ValueError(f"conditions[{i}] label must be non-empty str.")
        if not isinstance(comps, (list, tuple)) or not comps:
            raise ValueError(
                f"conditions[{i}] components must be a non-empty list."
            )


# ============================================================
#   Public entry point
# ============================================================

def run(
    model, tokenizer, hook_manager,
    cfg: ExperimentConfig,
) -> List[ExperimentRun]:
    """Trace + ablate + compare for every (prompt, condition) cell.

    Saves each cell to {cfg.out_dir}/pNNNN__{label}.json as it completes
    (if out_dir is set).  Re-running skips cells whose JSON already
    exists, so a crashed run resumes without re-tracing completed cells.
    """
    _validate_config(cfg)
    if cfg.out_dir:
        os.makedirs(cfg.out_dir, exist_ok=True)

    runs:    List[ExperimentRun] = []
    n_total = len(cfg.prompts) * len(cfg.conditions)
    n_done  = 0
    t0      = time.time()

    for pi, p in enumerate(cfg.prompts):
        # Resume: load existing cells, queue the rest.
        todo: List[Tuple[str, List[str]]] = []
        for label, comps in cfg.conditions:
            if cfg.out_dir:
                ckpt = _run_json_path(cfg.out_dir, pi, label)
                if os.path.exists(ckpt):
                    n_done += 1
                    if cfg.verbose:
                        print(f"[{n_done}/{n_total}] resume p{pi} {label}")
                    with open(ckpt) as f:
                        runs.append(_run_from_jsonable(json.load(f)))
                    continue
            todo.append((label, comps))
        if not todo:
            continue

        # Clean trace runs once per prompt and is reused across conditions.
        clean = trace(
            model, tokenizer, p["prompt"],
            target_token         = p["target_token"],
            distractor_token     = p.get("distractor_token"),
            beta                 = cfg.beta,
            top_paths_k          = cfg.top_paths_k,
            path_min_frac        = cfg.path_min_frac,
            edges_top_k_per_node = cfg.edges_top_k_per_node,
            hook_manager         = hook_manager,
        )

        for label, comps in todo:
            t_cell = time.time()

            abl_cfg = AblationConfig(
                components     = list(comps),
                mode           = cfg.mode,
                references     = list(p["references"]),
                positions      = cfg.positions,
                resample_index = cfg.resample_index,
            )
            intervention = build_intervention(
                model, tokenizer, hook_manager, p["prompt"], abl_cfg,
            )
            ablated = trace(
                model, tokenizer, p["prompt"],
                target_token         = p["target_token"],
                distractor_token     = p.get("distractor_token"),
                beta                 = cfg.beta,
                top_paths_k          = cfg.top_paths_k,
                path_min_frac        = cfg.path_min_frac,
                edges_top_k_per_node = cfg.edges_top_k_per_node,
                hook_manager         = hook_manager,
                intervention         = intervention,
            )
            diff = compare(
                clean, ablated, ablation=abl_cfg,
                eps_path        = cfg.eps_path,
                eps_edge        = cfg.eps_edge,
                abl_threshold   = cfg.abl_threshold,
                delta_threshold = cfg.delta_threshold,
            )
            run_obj = _build_run(
                pi, p, label, comps, clean, ablated, diff, cfg.storage,
            )
            runs.append(run_obj)
            if cfg.out_dir:
                _save_run(run_obj, _run_json_path(cfg.out_dir, pi, label))

            n_done += 1
            if cfg.verbose:
                dt = time.time() - t_cell
                elapsed = time.time() - t0
                eta = elapsed / n_done * (n_total - n_done) if n_done else 0
                print(f"[{n_done}/{n_total}] p{pi} {label}  "
                      f"Δp={diff.delta_p_target:+.3f}  "
                      f"({dt:.1f}s, ETA {eta/60:.1f}m)")

    return runs