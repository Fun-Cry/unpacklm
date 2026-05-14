"""Task-agnostic helpers for circuit selection.

Nothing in this module knows about IOI specifically. The role mapping
is supplied as a parameter:

    role_keys: list of (metadata_field, label) pairs

Each prompt JSON's `metadata` dict may contain integer-valued fields
that name a token position; `role_keys` says how to map those fields
to short labels (IO, S1, S2, END, etc.) used by the selection strategy.

For IOI, the mapping lives at
    experiments.circuit_discovery.ioi.roles.ROLE_KEYS

For another task, define a sibling module with its own ROLE_KEYS and
pass --task <name> on the CLI.
"""

import glob
import json
import os
from typing import List, Optional, Sequence, Tuple


RoleKeys = Sequence[Tuple[str, str]]   # [(metadata_field, label), ...]

# Components excluded by default. embedding/pos_embedding are terminal
# and not meaningfully ablatable; mlp_0 in Pythia is part of the
# embedding pipeline (parallel-MLP), not a circuit component. None
# are IOI-specific. Override per-task if needed.
DEFAULT_EXCLUDE = frozenset({"embedding", "pos_embedding", "mlp_0"})


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────
def load_run(results_dir: str) -> Tuple[dict, List[dict]]:
    cfg_path = os.path.join(results_dir, "run_config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    prompts = []
    for path in sorted(glob.glob(os.path.join(results_dir, "prompt_*.json"))):
        with open(path) as f:
            prompts.append(json.load(f))
    return cfg, prompts


def filter_correct(prompts: List[dict], p_min: float) -> List[dict]:
    return [p for p in prompts if p["clean_target_prob"] >= p_min]


# ──────────────────────────────────────────────────────────────────────
# Role lookup
# ──────────────────────────────────────────────────────────────────────
def role_for_pos(prompt: dict, pos, role_keys: RoleKeys) -> Optional[str]:
    """Return role label of token position in this prompt, or None
    if no field in role_keys matches."""
    meta = prompt.get("metadata", {})
    for field, label in role_keys:
        v = meta.get(field)
        if v is not None and int(v) == int(pos):
            return label
    return None


# ──────────────────────────────────────────────────────────────────────
# Role-keys resolution for the CLI (--task or --positions-from-metadata)
# ──────────────────────────────────────────────────────────────────────
def parse_positions_arg(arg: str) -> RoleKeys:
    """Parse 'field=LABEL,field=LABEL' from --positions-from-metadata."""
    out = []
    for piece in arg.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(
                f"--positions-from-metadata entry without '=': {piece!r}"
            )
        field, label = piece.split("=", 1)
        out.append((field.strip(), label.strip()))
    return out


def resolve_role_keys(task: Optional[str],
                      positions: Optional[str]) -> RoleKeys:
    """Either --task <name> (imports task.roles.ROLE_KEYS) or
    --positions-from-metadata 'field=LABEL,...'. None of either ⇒ ()."""
    if positions and task:
        raise ValueError(
            "--task and --positions-from-metadata are mutually exclusive"
        )
    if positions:
        return parse_positions_arg(positions)
    if task:
        import importlib
        mod = importlib.import_module(
            f"experiments.circuit_discovery.{task}.roles"
        )
        return tuple(mod.ROLE_KEYS)
    return ()