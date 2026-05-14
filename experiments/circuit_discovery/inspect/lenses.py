"""Lens filters for path selection.

Each lens is a callable that returns True iff a path should be kept.
Lenses operate on integer positions only — they have no knowledge of
task-specific role labels (IO/S2/etc). Any task-specific interestingness
is encoded by the caller as a set of integer positions and passed as
a parameter.

Lenses receive (chain, src_pos) for every path. `src_pos` is the
terminal source position of the path — for chains that terminate at
an attention head without dispatching to source positions (recursion
truncated by min_frac), this is the head's read-from position which
otherwise never appears in `chain` annotations. Including it in the
position count fixes the "min_hops passes but min_positions doesn't"
artifact where truncated chains looked single-position despite the
attention head reading across positions.

Three lenses currently:

  length    — keep paths with at least N hops (chain length).
              Surfaces composition routes regardless of where they go.

  diversity — keep paths visiting at least N distinct token positions.
              Counts both `chain_positions(chain)` and `src_pos`.
              Surfaces cross-position routing (composition that moves
              information between tokens, the natural circuit signature).

  membership — keep paths visiting at least one token in a given set.
              Counts both `chain_positions(chain)` and `src_pos`.
              The targeted lens; the caller specifies which positions
              are interesting (e.g. {IO_pos, S2_pos} for IOI). Fully
              general — only the caller knows which positions matter.
"""

from typing import Iterable, Optional, Set
from ..utils import chain_positions


def _all_positions(chain, src_pos: Optional[int]) -> Set[int]:
    """All positions a path visits: every chain step's @position
    annotation plus the trailing source position recorded on the
    path tuple. Truncated attention chains look single-position
    in `chain` alone but have a meaningful `src_pos`.
    """
    positions = set(chain_positions(chain))
    if src_pos is not None:
        positions.add(int(src_pos))
    return positions


def lens_length(chain, src_pos, *, min_hops: int) -> bool:
    return len(chain) >= min_hops


def lens_diversity(chain, src_pos, *, min_positions: int) -> bool:
    return len(_all_positions(chain, src_pos)) >= min_positions


def lens_membership(chain, src_pos, *, target_positions: Set[int]) -> bool:
    if not target_positions:
        return False
    return any(p in target_positions for p in _all_positions(chain, src_pos))


# Map from config string to (callable, list of param names it expects).
LENSES = {
    "length":     (lens_length,     ("min_hops",)),
    "diversity":  (lens_diversity,  ("min_positions",)),
    "membership": (lens_membership, ("target_positions",)),
}


def make_lens_filter(lens_cfg: dict, prompt_dict: dict):
    """Build a 2-arg callable `lens(chain, src_pos) -> bool` from config.

    `lens_cfg` is the CONFIG['lens'] dict. `prompt_dict` is the per-
    prompt dict produced by build_prompts; for the membership lens we
    pull the target positions from a metadata key specified in config.

    The 'positions_from_metadata' key in lens_cfg tells the membership
    lens which prompt-metadata field to read positions from. This is
    the universal/task-agnostic way to wire task-specific position
    info into the lens.
    """
    typ = lens_cfg["type"]
    if typ not in LENSES:
        raise ValueError(f"unknown lens type: {typ}")
    fn, expected = LENSES[typ]

    if typ == "length":
        kw = {"min_hops": lens_cfg["min_hops"]}
    elif typ == "diversity":
        kw = {"min_positions": lens_cfg["min_positions"]}
    elif typ == "membership":
        key = lens_cfg.get("positions_from_metadata", "target_positions")
        positions = prompt_dict.get(key) or prompt_dict.get("metadata", {}).get(key)
        if positions is None:
            raise KeyError(
                f"membership lens needs prompt['{key}'] or "
                f"prompt['metadata']['{key}'] populated by prompts.py"
            )
        kw = {"target_positions": set(int(p) for p in positions)}
    else:
        raise AssertionError(typ)

    return lambda chain, src_pos: fn(chain, src_pos, **kw)