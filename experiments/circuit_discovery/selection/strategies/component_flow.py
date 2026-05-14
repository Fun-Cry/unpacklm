"""Component-flow selection.

Aggregates each prompt's per-component cumulative path-score
(`ranked_components.cum_score` from the discover sweep) across all
prompts, then applies an elbow detector on the global ranking.

This is a "flat" alternative to partition_coverage — no role-set or
branch-profile partitioning, just a single ranking. Useful as:
  - a baseline to compare against partition_coverage
  - a fast first-pass pick when role labels aren't available
  - a sanity check that the deep-component story isn't an artifact
    of partitioning

Selection rule:
  1. For each prompt, sum |path_score| per component (already in the
     saved ranked_components, with branch tags stripped).
  2. Sum across prompts → global cum_score per component.
  3. Sort descending; run an elbow detector on the cumulative-share
     curve; keep components above the elbow rank.
  4. Optionally union with components touching at least one task role
     position (keeps role-active components even if they fall below
     the elbow).
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .._common import (
    DEFAULT_EXCLUDE, RoleKeys, role_for_pos,
)
from ...utils import chain_components, parse_step
from ..result import SelectionResult


_BRANCH_RE = re.compile(r"\[([KQV])\]")


def _strip_tag(name: str) -> str:
    return _BRANCH_RE.sub("", name)


def _aggregate_cum_score(prompts: List[dict],
                         exclude: Set[str]) -> Tuple[Dict[str, float],
                                                     Dict[str, int]]:
    """Aggregate per-component flow across all prompts.

    Reads the `component_flow` field saved by discover (the dense
    top-down flow sweep, no min_frac pruning, no path enumeration).
    Falls back to path-aggregated `ranked_components` for legacy data
    where `component_flow` wasn't saved.

    Returns (cum_score, n_prompts):
      cum_score[name] = Σ_prompts |flow|
      n_prompts[name] = how many prompts contained the component
    """
    cum_score: Dict[str, float] = defaultdict(float)
    n_prompts: Dict[str, int] = defaultdict(int)

    for prompt in prompts:
        seen_in_prompt: Set[str] = set()

        if "component_flow" in prompt:
            # Preferred: dense flow from the top-down sweep.
            for name, flow in prompt["component_flow"].items():
                name = _strip_tag(name)
                if name in exclude:
                    continue
                cum_score[name] += abs(float(flow))
                seen_in_prompt.add(name)
        else:
            # Fallback for legacy sweeps without component_flow field.
            for row in prompt.get("ranked_components", []):
                name = _strip_tag(row["name"])
                if name in exclude:
                    continue
                cum_score[name] += float(row["cum_score"])
                seen_in_prompt.add(name)

        for name in seen_in_prompt:
            n_prompts[name] += 1

    return dict(cum_score), dict(n_prompts)


def _elbow_index(shares: List[float]) -> int:
    """Maximum-perpendicular-distance elbow on a concave cumulative
    curve. Given a sorted-descending list of shares, returns the
    rank index k such that the cumulative curve's elbow is at point k.

    Lifted from partition_coverage's _elbow_index for consistency.
    """
    if not shares:
        return 0
    n = len(shares)
    # Cumulative curve: (0,0), (1,s[0]), (2,s[0]+s[1]), ..., (n,1).
    cum = [0.0]
    s = 0.0
    for v in shares:
        s += v
        cum.append(s)
    if cum[-1] <= 0:
        return 0

    # Line from (0,0) to (n, total). Perpendicular distance from each
    # (i, cum[i]) to that line, maximized.
    n_total = float(cum[-1])
    best_i = 0
    best_d = -1.0
    for i in range(1, n + 1):
        # Distance from point (i, cum[i]) to line from (0,0) to (n, n_total).
        # |n_total * i - n * cum[i]| / sqrt(n_total^2 + n^2)
        d = abs(n_total * i - n * cum[i])
        if d > best_d:
            best_d = d
            best_i = i
    return best_i


def select(
    prompts,
    *,
    terminal_role: Optional[str] = None,   # not used; kept for API uniformity
    role_keys: RoleKeys = (),
    exclude: Set[str] = DEFAULT_EXCLUDE,
    elbow_floor_k: int = 1,
    role_union: bool = False,
    **_,
) -> SelectionResult:
    """Aggregate cum_score across prompts; pick components above the
    global elbow.

    Args:
      role_keys:    Optional. If provided AND role_union=True, components
                    that appear in any chain touching a role position are
                    additionally retained, even if below the elbow.
      role_union:   Default False. If True, expand the elbow-picked set
                    by role-touching components.
      elbow_floor_k: Minimum number of components to keep regardless of
                    the elbow detector.
    """
    cum_score, n_prompts_map = _aggregate_cum_score(prompts, set(exclude))

    if not cum_score:
        return SelectionResult(
            method="component_flow",
            components=[],
            n_components=0,
            params={
                "role_keys":     list(role_keys),
                "exclude":       sorted(exclude),
                "elbow_floor_k": elbow_floor_k,
                "role_union":    role_union,
            },
        )

    # Sort by cum_score descending.
    ranked = sorted(cum_score.items(), key=lambda kv: -kv[1])
    total = sum(v for _, v in ranked)
    shares = [v / total for _, v in ranked] if total > 0 else []

    elbow_rank = _elbow_index(shares) if shares else 0
    elbow_rank = max(elbow_rank, min(elbow_floor_k, len(ranked)))

    elbow_picked = [name for name, _ in ranked[:elbow_rank]]
    union: Set[str] = set(elbow_picked)

    role_picked: Set[str] = set()
    if role_union and role_keys:
        for prompt in prompts:
            meta = prompt.get("metadata", {})
            role_positions = set()
            for field, _label in role_keys:
                v = meta.get(field)
                if v is not None:
                    role_positions.add(int(v))

            for path in prompt.get("ranked_paths", []):
                # If any step's @position lands on a role position,
                # keep all (tag-stripped) components in the chain.
                touched = False
                for step in path["chain"]:
                    _, pos = parse_step(step)
                    if int(pos) in role_positions:
                        touched = True
                        break
                if touched:
                    for c in chain_components(path["chain"]):
                        c = _strip_tag(c)
                        if c not in exclude:
                            role_picked.add(c)
        union |= role_picked

    # Diagnostics: full ranking with status.
    elbow_set = set(elbow_picked)
    full_ranking = []
    cum = 0.0
    for i, (name, score) in enumerate(ranked):
        cum += score
        if name in elbow_set:
            status = "kept"
        elif name in role_picked:
            status = "rescued_role"
        else:
            status = "below_elbow"
        full_ranking.append({
            "rank":           i + 1,
            "component":      name,
            "cum_score":      score,
            "share":          score / total if total > 0 else 0.0,
            "share_cum":      cum / total if total > 0 else 0.0,
            "n_prompts":      n_prompts_map.get(name, 0),
            "status":         status,
        })

    return SelectionResult(
        method="component_flow",
        components=sorted(union),
        n_components=len(union),
        params={
            "role_keys":     list(role_keys),
            "exclude":       sorted(exclude),
            "elbow_floor_k": elbow_floor_k,
            "role_union":    role_union,
        },
        diagnostics={
            "n_components_total":      len(ranked),
            "n_components_above_elbow": len(elbow_picked),
            "n_components_role_only":  len(role_picked - elbow_set),
            "elbow_rank":              elbow_rank,
            "ranking":                 full_ranking,
        },
    )