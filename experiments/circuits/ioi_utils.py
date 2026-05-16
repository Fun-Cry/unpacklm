"""IOI-specific utilities: position resolution, chain parsing, ABC references."""

import re
from typing import Dict, List, Optional, Set, Tuple

_STEP_RE = re.compile(r"^(.+?)@(-?\d+)$")
_BRANCH_RE = re.compile(r"\[([KQV])\]")

# IOI role mapping: metadata field → role label
ROLE_KEYS = [
    ("io_position",   "IO"),
    ("s1_position",   "S1"),
    ("s1p1_position", "S1p1"),
    ("s2_position",   "S2"),
    ("end_position",  "END"),
]

DEFAULT_EXCLUDE = {"embedding", "pos_embedding"}


# ── Chain parsing ──

def parse_step(step: str) -> Tuple[str, int]:
    """'attn_9_head_9[K]@14' → ('attn_9_head_9[K]', 14)"""
    m = _STEP_RE.match(step)
    if m is None:
        return (step, -1)
    return (m.group(1), int(m.group(2)))


def chain_positions(chain) -> List[int]:
    return [parse_step(s)[1] for s in chain]


def chain_components(chain) -> List[str]:
    return [parse_step(s)[0] for s in chain]


def strip_branch_tag(name: str) -> str:
    return _BRANCH_RE.sub("", name)


def step_branch_tag(step: str) -> Optional[str]:
    m = _BRANCH_RE.search(step)
    return m.group(1) if m else None


def chain_branch_profile(chain) -> Tuple[str, ...]:
    tags = []
    for step in chain[:-1]:
        b = step_branch_tag(step)
        if b is not None:
            tags.append(b)
    return tuple(tags)


# ── Position resolution ──

def resolve_positions(prompt: str, io_token: str, s_token: str,
                      tokenizer) -> Optional[Dict[str, int]]:
    """Resolve IO/S1/S2/END token positions for one IOI prompt."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)

    def both_ids(s):
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

    out = {}
    s_count = 0
    for i, tid in enumerate(ids):
        if tid in io_set and "IO" not in out:
            out["IO"] = i
        elif tid in s_set:
            s_count += 1
            out[f"S{s_count}"] = i
    out["END"] = len(ids) - 1
    return out


def role_for_pos(metadata: dict, pos: int, role_keys=ROLE_KEYS) -> Optional[str]:
    """Which role label (IO/S1/S2/END) does position `pos` have?"""
    for field, label in role_keys:
        v = metadata.get(field)
        if v is not None and int(v) == int(pos):
            return label
    return None


# ── Diversity lens ──

def diversity_filter(chain, src_pos, min_positions: int = 2) -> bool:
    """Keep paths visiting at least N distinct positions."""
    positions = set(chain_positions(chain))
    if src_pos is not None:
        positions.add(int(src_pos))
    return len(positions) >= min_positions


# ── Prompt loading ──

def load_ioi_prompts(tokenizer, n_prompts=100, seed=42):
    """Load IOI prompts with BOS prefix and position metadata including S1+1."""
    from utils.load_data import load_ioi_dataset

    ds = load_ioi_dataset(target=n_prompts * 3, seed=seed)
    raw = ds.metadata
    eos = tokenizer.eos_token or "<|endoftext|>"

    prompts = []
    for d in raw:
        roles = resolve_positions(d["prompt"], d["IO"], d["S"], tokenizer)
        if roles is None or "IO" not in roles or "S2" not in roles:
            continue
        # +1 for BOS prefix
        positions = {k: v + 1 for k, v in roles.items()}
        s1p1 = positions["S1"] + 1 if "S1" in positions else None
        prompts.append({
            "prompt": eos + d["prompt"],
            "target_token": d["IO"],
            "distractor_token": d["S"],
            "template_type": d.get("template_type", ""),
            "IO": d["IO"],
            "S": d["S"],
            "metadata": {
                "io_position": positions["IO"],
                "s1_position": positions.get("S1"),
                "s1p1_position": s1p1,
                "s2_position": positions.get("S2"),
                "end_position": positions["END"],
                "target_positions": [
                    positions["IO"],
                    positions.get("S1", positions["IO"]),
                    positions.get("S2", positions["IO"]),
                ],
            },
        })
        if len(prompts) >= n_prompts:
            break
    return prompts