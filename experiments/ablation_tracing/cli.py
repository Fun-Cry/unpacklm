"""Folder-based experiment runner for ablation tracing.

Each experiment is a folder containing exactly three Python modules:

    prompts.py     — defines build_prompts(tokenizer) -> list[dict]
    conditions.py  — defines CONDITIONS: list of (label, list[component_name])
    config.py      — defines CONFIG: dict with keys
                       model / ablation / trace / compare / output

Run with:

    python -m experiments.ablation_tracing <folder>

The folder must live under an 'experiments' tree so it can be imported
as a proper Python package (e.g. experiments/ablation_tracing/exp_foo/).
The runner snapshots the resolved spec to <out_dir>/_spec.json before
running, so the experiment's exact inputs are recoverable from disk.
"""

import argparse
import dataclasses
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from .core import ExperimentConfig, run


# ============================================================
#   Folder -> Python package name
# ============================================================

def _folder_to_module(folder: Path) -> Tuple[str, Path]:
    """Convert an absolute folder path into an importable module name and
    return the project root that should be on sys.path.

    Searches the path for an 'experiments' component and treats everything
    from there forward as the module path:

        /home/user/proj/experiments/ablation_tracing/exp_foo
            -> ('experiments.ablation_tracing.exp_foo',
                Path('/home/user/proj'))
    """
    folder = folder.resolve()
    parts = folder.parts
    try:
        i = parts.index("experiments")
    except ValueError:
        raise ValueError(
            f"Experiment folder must live under an 'experiments' tree; "
            f"got {folder}"
        )
    module_name  = ".".join(parts[i:])
    project_root = Path(*parts[:i]) if i > 0 else Path("/")
    return module_name, project_root


# ============================================================
#   Spec loading
# ============================================================

def load_experiment_spec(folder) -> Dict[str, Any]:
    """Import prompts/conditions/config from `folder` and return a spec dict."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Experiment folder does not exist: {folder}")

    module_name, project_root = _folder_to_module(folder)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    prompts_mod    = importlib.import_module(f"{module_name}.prompts")
    conditions_mod = importlib.import_module(f"{module_name}.conditions")
    config_mod     = importlib.import_module(f"{module_name}.config")

    if not hasattr(prompts_mod, "build_prompts"):
        raise AttributeError(
            f"{module_name}.prompts must define "
            f"build_prompts(tokenizer) -> list[dict]"
        )
    if not callable(prompts_mod.build_prompts):
        raise TypeError(f"{module_name}.prompts.build_prompts must be callable.")
    if not hasattr(conditions_mod, "CONDITIONS"):
        raise AttributeError(
            f"{module_name}.conditions must define CONDITIONS"
        )
    if not hasattr(config_mod, "CONFIG"):
        raise AttributeError(f"{module_name}.config must define CONFIG (dict).")

    return {
        "module":         module_name,
        "folder":         folder,
        "build_prompts":  prompts_mod.build_prompts,
        "conditions":     list(conditions_mod.CONDITIONS),
        "config":         dict(config_mod.CONFIG),
    }


# ============================================================
#   Model loading
# ============================================================

def load_model_from_config(model_cfg: Dict[str, Any]):
    """Dispatch on family and load (model, tokenizer, adapter)."""
    from unpack.models import load_model as _unpack_load_model, get_adapter

    family = model_cfg.get("family", "gpt2")
    size   = model_cfg.get("size",   "small")
    device = model_cfg.get("device", "cpu")
    cache  = model_cfg.get("cache_dir", None)

    if family == "gpt2":
        name_map = {"small": "gpt2", "medium": "gpt2-medium",
                    "large": "gpt2-large", "xl": "gpt2-xl"}
        model_name = name_map.get(size, size)
        load_kwargs = {}
    elif family == "pythia":
        step    = model_cfg.get("step", 143000)
        deduped = model_cfg.get("deduped", True)
        suffix  = "-deduped" if deduped else ""
        model_name = f"EleutherAI/pythia-{size}{suffix}"
        load_kwargs = {"step": step}
    else:
        raise ValueError(
            f"Unknown model family: {family!r}. Supported: 'gpt2', 'pythia'."
        )

    model, tokenizer = _unpack_load_model(
        model_name, device=device, cache_dir=cache, **load_kwargs)

    hook_manager = get_adapter(model)
    hook_manager.register_hooks(model)
    return model, tokenizer, hook_manager


# ============================================================
#   Build ExperimentConfig from a CONFIG dict
# ============================================================

def build_experiment_config(
    cfg_dict: Dict[str, Any],
    prompts: List[Dict],
    conditions: List[Tuple[str, List[str]]],
    default_out_dir: str,
) -> ExperimentConfig:
    """Map a dict-style CONFIG (plus resolved prompts/conditions) into an
    ExperimentConfig. Missing keys take the dataclass defaults.
    """
    abl_d  = cfg_dict.get("ablation", {})
    tr_d   = cfg_dict.get("trace",    {})
    cmp_d  = cfg_dict.get("compare",  {})
    out_d  = cfg_dict.get("output",   {})
    out_dir = out_d.get("dir") or default_out_dir

    return ExperimentConfig(
        prompts    = prompts,
        conditions = conditions,
        out_dir    = out_dir,

        mode           = abl_d.get("mode",           "mean"),
        positions      = abl_d.get("positions",      "target"),
        resample_index = abl_d.get("resample_index", 0),

        beta                 = tr_d.get("beta",                 0.3),
        top_paths_k          = tr_d.get("top_paths_k",          200),
        edges_top_k_per_node = tr_d.get("edges_top_k_per_node", 50),
        path_min_frac        = tr_d.get("path_min_frac",        1e-3),

        eps_path        = cmp_d.get("eps_path",        0.001),
        eps_edge        = cmp_d.get("eps_edge",        0.001),
        abl_threshold   = cmp_d.get("abl_threshold",   0.05),
        delta_threshold = cmp_d.get("delta_threshold", 0.02),

        storage = out_d.get("storage", "full"),
        verbose = out_d.get("verbose", True),
    )


# ============================================================
#   Spec snapshot
# ============================================================

def snapshot_spec(
    spec:        Dict[str, Any],
    prompts:     List[Dict],
    conditions:  List[Tuple[str, List[str]]],
    exp_config:  ExperimentConfig,
    out_dir:     str,
) -> str:
    """Write a self-contained _spec.json into out_dir.

    Captures everything needed to reproduce this experiment: the resolved
    prompts, conditions, the source folder, and the resolved
    ExperimentConfig. Writing this BEFORE the run starts means if the run
    crashes mid-way, the spec is still on disk for postmortem.
    """
    os.makedirs(out_dir, exist_ok=True)
    snap = {
        "module":     spec["module"],
        "folder":     str(spec["folder"]),
        "config":     spec["config"],
        "conditions": [list(c) for c in conditions],
        "prompts":    prompts,
        "experiment_config": {
            k: v for k, v in dataclasses.asdict(exp_config).items()
            # prompts/conditions duplicated above; drop from this section
            if k not in ("prompts", "conditions")
        },
    }
    path = os.path.join(out_dir, "_spec.json")
    with open(path, "w") as f:
        json.dump(snap, f, indent=2, default=str)
    return path


# ============================================================
#   Top-level entry point
# ============================================================

def run_experiment(folder, *, dry_run: bool = False, model_bundle=None):
    """Load + run the experiment in `folder`.

    `model_bundle` is an optional (model, tokenizer, hook_manager) triple
    used by tests to inject a synthetic model and skip the HF download
    path. In normal use leave it None and the runner loads what
    config['model'] says.
    """
    spec = load_experiment_spec(folder)
    folder = Path(folder).resolve()
    default_out_dir = str(folder / "results")

    if dry_run:
        print(f"[dry-run] module                : {spec['module']}")
        print(f"[dry-run] conditions ({len(spec['conditions'])}):")
        for label, comps in spec["conditions"]:
            print(f"           {label:<24} {comps}")
        cfg = spec["config"]
        print(f"[dry-run] model                 : {cfg.get('model')}")
        print(f"[dry-run] ablation              : {cfg.get('ablation')}")
        print(f"[dry-run] trace                 : {cfg.get('trace')}")
        print(f"[dry-run] compare               : {cfg.get('compare')}")
        out_dir = (cfg.get("output", {}).get("dir") or default_out_dir)
        print(f"[dry-run] would write to        : {out_dir}")
        return None

    if model_bundle is None:
        print(f"Loading model: {spec['config'].get('model')}")
        model, tokenizer, hook_manager = load_model_from_config(spec["config"]["model"])
    else:
        model, tokenizer, hook_manager = model_bundle
        print(f"Using injected model bundle (skipping config['model']).")

    print(f"Building prompts...")
    prompts = spec["build_prompts"](tokenizer)
    print(f"  resolved {len(prompts)} prompts")

    exp_config = build_experiment_config(
        spec["config"], prompts, spec["conditions"], default_out_dir,
    )
    print(f"Output dir: {exp_config.out_dir}")
    snapshot_path = snapshot_spec(
        spec, prompts, spec["conditions"], exp_config, exp_config.out_dir,
    )
    print(f"Spec snapshot: {snapshot_path}")

    print(f"Running {len(prompts)} prompts × {len(spec['conditions'])} conditions...")
    runs = run(model, tokenizer, hook_manager, exp_config)
    print(f"Done. {len(runs)} cells.")
    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Run an ablation_tracing experiment from a folder spec.",
    )
    parser.add_argument(
        "folder", type=str,
        help="Path to a folder containing prompts.py, conditions.py, config.py",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load and validate the spec without loading the model or running.",
    )
    args = parser.parse_args()

    run_experiment(args.folder, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
