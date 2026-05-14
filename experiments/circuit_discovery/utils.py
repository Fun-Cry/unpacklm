"""Shared utilities for circuit_discovery."""

import importlib.util
import os
import re
from typing import Any, Dict, Tuple

from unpack.models import load_model as _unpack_load_model, get_adapter


def _load_python_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_experiment_folder(folder: str) -> Tuple[Dict[str, Any], Any]:
    cfg_mod = _load_python_module(os.path.join(folder, "config.py"),
                                  "circ_discovery_cfg")
    pr_mod  = _load_python_module(os.path.join(folder, "prompts.py"),
                                  "circ_discovery_prompts")
    return cfg_mod.CONFIG, pr_mod.build_prompts


def load_model(cfg: Dict[str, Any]):
    """Instantiate model + tokenizer + adapter from CONFIG['model'].

    Returns (model, tokenizer, adapter, device).
    """
    family    = cfg["model"]["family"]
    size      = cfg["model"]["size"]
    device    = cfg["model"]["device"]
    cache_dir = cfg["model"].get("cache_dir")

    # Map family/size to HuggingFace model name
    if family == "gpt2":
        name_map = {"small": "gpt2", "medium": "gpt2-medium",
                    "large": "gpt2-large", "xl": "gpt2-xl"}
        model_name = name_map.get(size, size)
        load_kwargs = {}
    elif family == "pythia":
        deduped = cfg["model"].get("deduped", True)
        step    = cfg["model"].get("step", 143000)
        suffix  = "-deduped" if deduped else ""
        model_name = f"EleutherAI/pythia-{size}{suffix}"
        load_kwargs = {"step": step}
    else:
        raise ValueError(f"unsupported model family: {family}")

    model, tokenizer = _unpack_load_model(
        model_name, device=device, cache_dir=cache_dir, **load_kwargs)

    adapter = get_adapter(model)
    adapter.register_hooks(model)

    return model, tokenizer, adapter, device


# Path-step parsing

_STEP_RE = re.compile(r"^(.+?)@(-?\d+)$")

def parse_step(step: str) -> Tuple[str, int]:
    m = _STEP_RE.match(step)
    if m is None:
        return (step, -1)
    return (m.group(1), int(m.group(2)))

def chain_positions(chain) -> list:
    return [parse_step(s)[1] for s in chain]

def chain_components(chain) -> list:
    return [parse_step(s)[0] for s in chain]
