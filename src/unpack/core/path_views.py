"""
core.path_views - Post-processing views over `top_paths` produced
by `trace_flow` and `backward_recursive`.

These are *views* on already-computed attribution paths. The functions
here do not run any model code; they only re-organize and pretty-print
path lists for human inspection or downstream aggregation.

Design notes
------------
- The canonical path representation throughout the project is the
  triple `(path_str, pos, score)` produced by `backward_recursive`:
      path_str:  "attn_8_head_5[V]@13->mlp_0@3->embedding@3"
      pos:       integer source position (the embedding's @N)
      score:     signed scalar contribution (raw or percent share,
                 depending on caller; this module treats it as
                 opaque scalar)

- Functions here accept that list directly so they can be called both
  from a CLI (trace_flow's --group-paths output) and from analysis
  scripts that loaded a saved trace JSON.

Public API
----------
    parse_chain(path_str)              -> list[str]
    chain_source_position(chain)       -> int
    chain_modes(chain)                 -> list[str]
    dominant_attn_mode(chain)          -> str | None
    group_paths_by_source(paths)       -> dict[int, list[entry]]
    group_paths_by_mode(paths)         -> dict[str, list[entry]]
    composition_summary(entries)       -> dict[str, dict]
    render_grouped(paths, tokens, ...) -> str
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Path entry as produced by backward_recursive / trace_flow.
# (path_str, source_position, score)
PathEntry = Tuple[str, int, float]


_POS_RE = re.compile(r"@(\d+)$")
_ROLE_RE = re.compile(r"\[([KQV])\]")
_PATH_SEP_RE = re.compile(r"\s*(?:→|->)\s*")


def parse_chain(path_str: str) -> List[str]:
    """Split a path string like 'A->B->C' or 'A→B→C' into ['A', 'B', 'C']."""
    return _PATH_SEP_RE.split(path_str)


def chain_source_position(chain: List[str]) -> int:
    """Position of the embedding (rightmost element) in the chain.

    Returns -1 if the rightmost element has no @N suffix.
    """
    if not chain:
        return -1
    m = _POS_RE.search(chain[-1])
    return int(m.group(1)) if m else -1


def chain_modes(chain: List[str]) -> List[str]:
    """Tag each chain element by kind.

    Attention elements get their role tag (K/Q/V) or '?' if untagged.
    MLP elements get 'mlp'. Embedding/pos_embedding get 'emb'.
    """
    modes: List[str] = []
    for elem in chain:
        if elem.startswith("attn_"):
            m = _ROLE_RE.search(elem)
            modes.append(m.group(1) if m else "?")
        elif elem.startswith("mlp_"):
            modes.append("mlp")
        elif elem.startswith("embedding") or elem.startswith("pos_embedding"):
            modes.append("emb")
        else:
            modes.append("?")
    return modes


def dominant_attn_mode(chain: List[str]) -> Optional[str]:
    """First K/Q/V attention role encountered scanning left-to-right.

    For a path like attn_X[V] -> mlp -> attn_Y[K] -> emb, returns 'V'
    (the role at the top-most attention step, closest to the prediction).
    Returns None if there is no attention element in the chain.
    """
    for m in chain_modes(chain):
        if m in ("K", "Q", "V"):
            return m
    return None


def group_paths_by_source(
    paths: Iterable[PathEntry],
) -> Dict[int, List[Tuple[float, str, List[str]]]]:
    """Group paths by their starting (embedding) position.

    Returns {position: [(score, path_str, parsed_chain), ...]}, each
    list sorted by |score| descending.
    """
    groups: Dict[int, List[Tuple[float, str, List[str]]]] = defaultdict(list)
    for path_str, _pos, score in paths:
        chain = parse_chain(path_str)
        src = chain_source_position(chain)
        groups[src].append((float(score), path_str, chain))
    for pos in groups:
        groups[pos].sort(key=lambda t: -abs(t[0]))
    return groups


def group_paths_by_mode(
    paths: Iterable[PathEntry],
) -> Dict[str, List[Tuple[float, str, List[str]]]]:
    """Group paths by dominant attention composition mode.

    Buckets: 'V', 'K', 'Q', 'MLP-only' (no attention element in chain).
    """
    groups: Dict[str, List[Tuple[float, str, List[str]]]] = defaultdict(list)
    for path_str, _pos, score in paths:
        chain = parse_chain(path_str)
        mode = dominant_attn_mode(chain) or "MLP-only"
        groups[mode].append((float(score), path_str, chain))
    for tag in groups:
        groups[tag].sort(key=lambda t: -abs(t[0]))
    return groups


def composition_summary(
    entries: Sequence[Tuple[float, str, List[str]]],
) -> Dict[str, Dict[str, float]]:
    """Per-mode count and absolute-score mass for a group of entries.

    Returns:
        {mode: {"count": int, "abs_mass": float, "signed_mass": float}}
    Where `mode` is 'V', 'K', 'Q', or 'MLP-only'.
    """
    summary: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"count": 0, "abs_mass": 0.0, "signed_mass": 0.0}
    )
    for score, _path, chain in entries:
        mode = dominant_attn_mode(chain) or "MLP-only"
        summary[mode]["count"] += 1
        summary[mode]["abs_mass"] += abs(score)
        summary[mode]["signed_mass"] += score
    return dict(summary)


def render_grouped(
    paths: Sequence[PathEntry],
    tokens: Sequence[str],
    *,
    by: str = "source",
    top: Optional[int] = None,
    truncate: int = 60,
    header: Optional[str] = None,
) -> str:
    """Pretty-print a path list grouped by source token (or by mode).

    Args:
        paths:    list of (path_str, pos, score) triples
        tokens:   list of token strings, indexed by position
        by:       'source' (group by starting token) or 'mode' (by K/Q/V)
        top:      consider only the first `top` paths from `paths`;
                  None for all
        truncate: max width to display per path string
        header:   optional one-line header line

    Returns:
        formatted multi-line string
    """
    if top is not None:
        paths = paths[:top]

    if by == "source":
        groups = group_paths_by_source(paths)
        # Order by total abs-score mass descending
        order = sorted(groups.keys(),
                       key=lambda k: -sum(abs(s) for s, _, _ in groups[k]))
    elif by == "mode":
        groups = group_paths_by_mode(paths)
        # Stable: V, K, Q, MLP-only
        priority = {"V": 0, "K": 1, "Q": 2, "MLP-only": 3}
        order = sorted(groups.keys(), key=lambda k: priority.get(k, 99))
    else:
        raise ValueError(f"unknown grouping {by!r}; expected 'source' or 'mode'")

    out: List[str] = []
    if header:
        out.append(header)

    for key in order:
        entries = groups[key]
        total_mass = sum(abs(s) for s, _, _ in entries)

        if by == "source":
            pos = key
            tok = tokens[pos] if 0 <= pos < len(tokens) else "?"
            heading = (f"\n── from [{pos}] {tok!r}  "
                       f"({len(entries)} paths, |score|={total_mass:.1f}) ──")
        else:
            heading = (f"\n── {key}-mode  "
                       f"({len(entries)} paths, |score|={total_mass:.1f}) ──")

        out.append(heading)

        # Composition breakdown line (only meaningful when grouping by source)
        if by == "source":
            comp = composition_summary(entries)
            parts = []
            for mode in sorted(comp, key=lambda m: -comp[m]["abs_mass"]):
                c = comp[mode]
                parts.append(f"{mode}:{c['count']}({c['abs_mass']:.1f})")
            out.append(f"   composition: {'  '.join(parts)}")

        out.append("")
        out.append(f"   {'#':>3}  {'score':>7}  path")
        out.append(f"   {'-'*3}  {'-'*7}  {'-' * truncate}")
        for i, (score, path_str, _chain) in enumerate(entries, 1):
            disp = path_str if len(path_str) <= truncate \
                   else "…" + path_str[-(truncate - 1):]
            out.append(f"   {i:>3}  {score:+7.2f}  {disp}")

    return "\n".join(out)
