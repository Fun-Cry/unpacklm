"""IOI-specific utilities: position resolution, chain parsing, lenses.

Inlined from the old circuit_discovery package to remove that dependency.
"""

import re
from typing import Dict, List, Optional, Set, Tuple


# ── Chain parsing ──

_STEP_RE = re.compile(r"^(.+?)@(-?\d+)$")


def parse_step(step: str) -> Tuple[str, int]:
    """'attn_9_head_9[K]@14' → ('attn_9_head_9[K]', 14)"""
    m = _STEP_RE.match(step)
    if m is None:
        return (step, -1)
    return (m.group(1), int(m.group(2)))


def chain_positions(chain: List[str]) -> List[int]:
    return [parse_step(s)[1] for s in chain]


def chain_components(chain: List[str]) -> List[str]:
    return [parse_step(s)[0] for s in chain]


# ── IOI position resolution ──

def resolve_positions(prompt: str, io_token: str, s_token: str,
                      tokenizer) -> Optional[Dict[str, int]]:
    """Resolve IO/S1/S2/END token positions for one IOI prompt.

    Returns {"IO": 1, "S1": 2, "S2": 8, "END": 13} or None
    if names are multi-token.
    """
    ids = tokenizer.encode(prompt, add_special_tokens=False)

    def both_ids(s: str):
        s_stripped = s.lstrip()
        spaced = tokenizer.encode(" " + s_stripped, add_special_tokens=False)
        bare = tokenizer.encode(s_stripped, add_special_tokens=False)
        if len(spaced) != 1 or len(bare) != 1:
            return None
        return {spaced[0], bare[0]}

    io_set = both_ids(io_token)
    s_set = both_ids(s_token)
    if io_set is None or s_set is None:
        return None

    out: Dict[str, int] = {}
    s_count = 0
    for i, tid in enumerate(ids):
        if tid in io_set and "IO" not in out:
            out["IO"] = i
        elif tid in s_set:
            s_count += 1
            out[f"S{s_count}"] = i
    out["END"] = len(ids) - 1
    return out


# ── Lens filters ──

def _all_positions(chain: List[str], src_pos: Optional[int]) -> Set[int]:
    """All positions visited by a path: chain @-annotations + src_pos."""
    positions = set(chain_positions(chain))
    if src_pos is not None:
        positions.add(int(src_pos))
    return positions


def lens_membership(chain, src_pos, *, target_positions: Set[int]) -> bool:
    """Keep paths visiting at least one target position."""
    if not target_positions:
        return False
    return any(p in target_positions for p in _all_positions(chain, src_pos))


def lens_diversity(chain, src_pos, *, min_positions: int) -> bool:
    """Keep paths visiting at least N distinct positions."""
    return len(_all_positions(chain, src_pos)) >= min_positions


def lens_length(chain, src_pos, *, min_hops: int) -> bool:
    """Keep paths with at least N hops."""
    return len(chain) >= min_hops


def make_lens(lens_cfg: dict, prompt_metadata: dict):
    """Build a (chain, src_pos) → bool filter from config + prompt metadata.

    lens_cfg["type"]: "membership" | "diversity" | "length" | "none"
    """
    typ = lens_cfg.get("type", "membership")

    if typ == "none":
        return lambda chain, src_pos: True

    if typ == "membership":
        key = lens_cfg.get("positions_from_metadata", "target_positions")
        positions = prompt_metadata.get(key, [])
        target = set(int(p) for p in positions)
        return lambda chain, src_pos: lens_membership(
            chain, src_pos, target_positions=target)

    if typ == "diversity":
        min_pos = lens_cfg.get("min_positions", 2)
        return lambda chain, src_pos: lens_diversity(
            chain, src_pos, min_positions=min_pos)

    if typ == "length":
        min_h = lens_cfg.get("min_hops", 3)
        return lambda chain, src_pos: lens_length(
            chain, src_pos, min_hops=min_h)

    raise ValueError(f"Unknown lens type: {typ}")
