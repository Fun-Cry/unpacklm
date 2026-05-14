"""Partition-by-fingerprint selection.

A path's *fingerprint* is the set of task roles its chain touches
(union of role-labels over all chain steps; filler positions ignored).
Different fingerprints represent qualitatively different routes:
{IO, END} is direct name-mover transport, {IO, S2, END} is composition
through S2, {END} is self-loops with no information transport, etc.

Selection is a two-level filter:

  PARTITION LEVEL (which routes count?):
    1. Drop the {END}-only partition (self-loops, no real transport)
       and the empty partition (paths touching no task position).
    2. Keep partitions whose share of total kept mass is at least
       `partition_threshold`.
    3. Among partitions below threshold, run an elbow detector on
       their cumulative-share curve. Partitions up to the elbow are
       rescued. Disable with `elbow_rescue=False`.

  WITHIN-PARTITION LEVEL (which paths in a route count?):
    For each kept partition:
    1. Group paths by their *role-normalized* chain. Paths that share
       the same chain shape across prompts collapse to one bucket;
       each bucket carries a summed |score| and a count of distinct
       prompts it appears in.
    2. Drop normalized chains appearing in fewer than
       `min_prompt_fraction` of the kept prompts. A chain in 2 of 43
       prompts is per-prompt artifact; a chain in 30 of 43 is
       structural.
    3. Sort surviving chains by summed |score| descending; run elbow
       on the cumulative-score curve to find the natural cutoff.
       Falls back to keeping at least `elbow_floor_k` chains if the
       curve has no clear bend (e.g. a sparse partition with a few
       similar paths).
    4. Union the components in those chains.

Final circuit = union over per-partition component sets.

Diagnostics carry: full partition table with per-partition status
(kept/rescued/below_thresh/skipped), per-kept-partition number of
chains before/after occurrence filter / before/after elbow, and the
specific chains that were kept (with score and prompt-count). This
lets the writeup say "the IO-routing branch consists of paths
{a, b, c} appearing in {30, 25, 22} of 43 prompts; the S2 branch
consists of paths {d, e} ...".
"""

from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
import re

from .._common import (
    DEFAULT_EXCLUDE, RoleKeys, role_for_pos,
)
from ...utils import chain_components, parse_step
from ..result import SelectionResult


# Branch tag regex: matches "[K]", "[Q]", or "[V]" embedded in a step
# name like "attn_9_head_9[K]@13".
_BRANCH_RE = re.compile(r"\[([KQV])\]")


def step_branch_tag(step):
    """Return 'K', 'Q', 'V', or None for a step. None for MLP,
    embedding, or untagged attention heads (terminal hops with no
    outgoing branch decomposition)."""
    m = _BRANCH_RE.search(step)
    return m.group(1) if m else None


def chain_branch_profile(chain) -> Tuple[str, ...]:
    """Tuple of branch tags ('K'/'Q'/'V') along the chain, in order.

    A tag is recorded for each attention step that decomposed into
    its child via that branch. The final terminal step has no tag
    (no child to surface). MLP/embedding hops contribute nothing.
    Empty tuple = chain with no attention decomposition steps.
    """
    tags = []
    for step in chain[:-1]:  # last step is terminal
        b = step_branch_tag(step)
        if b is not None:
            tags.append(b)
    return tuple(tags)


def strip_branch_tag(component_name: str) -> str:
    """Strip [K]/[Q]/[V] from a component name. Leaves bare names
    untouched. Used so the picked-component set matches the bare
    names used by verification, ranked_components, etc.
    """
    return _BRANCH_RE.sub("", component_name)


# ──────────────────────────────────────────────────────────────────────
# Fingerprinting
# ──────────────────────────────────────────────────────────────────────
# A fingerprint pairs (role-set, branch-profile):
#   role-set:       frozenset of task-role labels touched by the chain
#                   (e.g. frozenset({"END","IO"})).
#   branch-profile: tuple of branch tags ('K'/'Q'/'V') consumed by the
#                   chain's attention decomposition steps in order
#                   (e.g. ("V","K") for a V-then-K composition).
#
# Cells (role_set × branch_profile) carve mechanism-distinct buckets:
#   ({END,IO},   (K,))         → name-mover-style direct read
#   ({END,IO},   (V,K))        → S-inh writes V at END, NM reads K at IO
#   ({END,S2},   (V,))         → S-inhibition direct reading from S2
#   ({END,S1,S2},(K,V))        → induction routing through S1 via V
# Selection picks top components within each cell, then unions.
Fingerprint = Tuple[FrozenSet[str], Tuple[str, ...]]


def chain_fingerprint(prompt: dict, chain, role_keys: RoleKeys
                      ) -> Fingerprint:
    """(role-set, branch-profile) tuple. Filler positions (no role
    match) and untagged attention hops contribute nothing to either
    component."""
    roles = set()
    for step in chain:
        _, pos = parse_step(step)
        role = role_for_pos(prompt, pos, role_keys)
        if role is not None:
            roles.add(role)
    return (frozenset(roles), chain_branch_profile(chain))


def fmt_fingerprint(fp: Fingerprint) -> str:
    """Stable string label, e.g. '{IO,END}|(V,K)' for a V-then-K
    chain touching IO and END."""
    roles, profile = fp
    role_str = "{}" if not roles else "{" + ",".join(sorted(roles)) + "}"
    prof_str = "()" if not profile else "(" + ",".join(profile) + ")"
    return f"{role_str}|{prof_str}"


# ──────────────────────────────────────────────────────────────────────
# Aggregation: collect signed paths grouped by fingerprint
# ──────────────────────────────────────────────────────────────────────
def aggregate_paths_by_fingerprint(
    prompts: List[dict],
    role_keys: RoleKeys,
    exclude: Set[str],
) -> Dict[Fingerprint, List[dict]]:
    """For each fingerprint, list path records (signed score, normalized
    chain, component set)."""
    by_fp: Dict[FrozenSet[str], List[dict]] = defaultdict(list)
    for prompt in prompts:
        # role_map for THIS prompt: int(token_pos) -> role_label
        role_map: Dict[int, str] = {}
        meta = prompt.get("metadata", {})
        for field, label in role_keys:
            v = meta.get(field)
            if v is not None and int(v) not in role_map:
                role_map[int(v)] = label

        for path in prompt["ranked_paths"]:
            fp = chain_fingerprint(prompt, path["chain"], role_keys)
            # Strip branch tags from component names so the picked-
            # component set matches the bare names downstream.
            comps = tuple(strip_branch_tag(c)
                          for c in chain_components(path["chain"])
                          if strip_branch_tag(c) not in exclude)
            norm = _normalized_chain(path["chain"], role_map)
            by_fp[fp].append({
                "prompt":            prompt["prompt"],
                "score":             float(path["score"]),
                "chain":             tuple(path["chain"]),
                "normalized_chain":  norm,
                "src_pos":           path["src_pos"],
                "components":        comps,
            })
    return dict(by_fp)


# ──────────────────────────────────────────────────────────────────────
# Per-partition path-coverage
# ──────────────────────────────────────────────────────────────────────
def _normalized_chain(chain, role_map: Dict[int, str]) -> Tuple[str, ...]:
    """Replace each step's @position with its role label (or raw int
    if no role assigned). Same chain shape across prompts becomes the
    same tuple."""
    out = []
    for step in chain:
        name, pos = parse_step(step)
        label = role_map.get(int(pos), str(pos))
        out.append(f"{name}@{label}")
    return tuple(out)


def select_within_partition(
    paths: List[dict],
    role_keys: RoleKeys,
    *,
    min_prompts_floor: int,
    elbow_floor_k: int = 1,
) -> Tuple[List[str], dict]:
    """Within-partition selection.

    1. Group paths by their role-normalized chain. The same chain shape
       across prompts collapses to one bucket. Score = sum of |path.score|;
       occurrence = number of distinct prompts the chain appears in.
    2. Drop normalized chains appearing in < min_prompts_floor prompts.
    3. Sort surviving chains by summed |score| descending, run elbow on
       the cumulative-score curve.
    4. Take chains up through the elbow rank (or `elbow_floor_k` if elbow
       returns less, so a sparse partition isn't reduced to zero).
    5. Return the union of components in those chains, plus diagnostic
       info: how many chains were dropped by the occurrence filter, the
       elbow's pick, the kept chains' summary.

    Why per-chain (not per-component) for the occurrence filter:
    a single component can appear in many different chains, some
    structural, some prompt-specific; filtering at chain level catches
    "this exact route only happens in 2 prompts" without losing the
    component if it shows up in another, recurring chain.
    """
    if not paths:
        return [], {"n_chains_total": 0, "n_chains_kept": 0,
                    "elbow_rank": 0, "kept_chains": []}

    # 1. Group by role-normalized chain across prompts.
    # Each prompt has its own role_map (different positions for the
    # same role across prompts), but normalize_chain is per-prompt
    # then we group across prompts.
    chain_buckets: Dict[Tuple[str, ...], dict] = {}
    for p in paths:
        # Re-derive the role_map for this path's prompt is awkward
        # without the prompt object — but each path record was built
        # with its prompt's role context. We re-normalize by inspecting
        # each step's position relative to the path's own positions.
        # However, we don't carry the prompt's metadata into the path
        # record. Instead, normalize by replacing positions matching
        # *this path's* role-bearing positions; the simpler approach
        # is to use the chain's `chain` field which carries the raw
        # @position labels, then compare each position's role via the
        # role information we stored earlier.
        # Simpler: each path record already has `chain` with raw positions.
        # We don't have the per-prompt role_map here; the caller must
        # have computed it. Pass it in via the path record.
        norm = p["normalized_chain"]
        bucket = chain_buckets.setdefault(norm, {
            "chain":     norm,
            "abs_score": 0.0,
            "n_paths":   0,
            "prompts":   set(),
            "components": tuple(),  # filled below from any path (same chain)
        })
        bucket["abs_score"] += abs(p["score"])
        bucket["n_paths"]   += 1
        bucket["prompts"].add(p["prompt"])
        if not bucket["components"]:
            bucket["components"] = p["components"]

    chain_list = list(chain_buckets.values())
    n_total = len(chain_list)

    # 2. Occurrence filter on normalized chains.
    after_occ = [c for c in chain_list
                 if len(c["prompts"]) >= min_prompts_floor]
    if not after_occ:
        return [], {
            "n_chains_total":     n_total,
            "n_chains_after_occ": 0,
            "n_chains_kept":      0,
            "elbow_rank":         0,
            "kept_chains":        [],
        }

    # 3. Sort by summed |score| descending; elbow on the *normalized*
    # cumulative-score curve.
    after_occ.sort(key=lambda c: -c["abs_score"])
    total_score = sum(c["abs_score"] for c in after_occ)
    shares = [c["abs_score"] / total_score for c in after_occ] if total_score > 0 else []

    elbow_rank = _elbow_index(shares) if shares else 0
    # elbow_floor_k acts as a minimum: even when elbow finds a clean
    # bend at rank 1, we keep at least `elbow_floor_k` chains. This
    # prevents the dominant partition from being collapsed to its top
    # chain alone (which loses, e.g., negative name movers that have
    # smaller magnitude than the primary positive mover but are still
    # circuit-relevant).
    elbow_rank = max(elbow_rank, min(elbow_floor_k, len(after_occ)))

    kept = after_occ[:elbow_rank]

    # 4. Union components.
    union: Set[str] = set()
    for c in kept:
        for comp in c["components"]:
            union.add(comp)

    # 5. Diagnostics
    kept_summary = [{
        "chain":        " → ".join(c["chain"]),
        "abs_score":    c["abs_score"],
        "share":        c["abs_score"] / total_score if total_score > 0 else 0.0,
        "n_paths":      c["n_paths"],
        "n_prompts":    len(c["prompts"]),
        "components":   sorted(c["components"]),
    } for c in kept]

    return sorted(union), {
        "n_chains_total":     n_total,
        "n_chains_after_occ": len(after_occ),
        "n_chains_kept":      len(kept),
        "elbow_rank":         elbow_rank,
        "kept_chains":        kept_summary,
    }


# ──────────────────────────────────────────────────────────────────────
# Strategy entry point
# ──────────────────────────────────────────────────────────────────────
# Fingerprints excluded from the "kept-mass" denominator AND the union.
# - frozenset()      — paths touching no task position (filler-only)
# - {END}            — self-loops at the prediction position
# Role-sets that should always be skipped, regardless of branch profile:
# - frozenset()      — paths touching no task position (filler-only)
# - {END}            — self-loops at the prediction position
SKIP_ROLE_SETS = frozenset([
    frozenset(),
    frozenset(["END"]),
])

# Legacy alias retained for any external code that imported it.
SKIP_FINGERPRINTS = SKIP_ROLE_SETS


def _elbow_index(shares):
    """Maximum-perpendicular-distance elbow on a concave cumulative
    curve. Given a sorted-descending list of shares, returns the
    rank index k such that the cumulative curve's elbow is at point k
    (1-indexed: keep the first k items). Returns 0 if the input is
    too small or shows no elbow.

    Algorithm: build cumulative curve (1..N, cum_share); the elbow
    is the point with greatest perpendicular distance to the line
    from (1, cum[0]) to (N, cum[-1]).
    """
    if len(shares) < 3:
        # 0 or 1 candidate is trivial; 2 has no curvature.
        return len(shares)

    # Cumulative curve over the candidate set
    cum = []
    s = 0.0
    for v in shares:
        s += v
        cum.append(s)

    n = len(cum)
    x0, y0 = 0.0, 0.0          # virtual origin so chord starts before rank 1
    xN, yN = float(n), cum[-1]

    # Perpendicular distance from each point (i+1, cum[i]) to chord (x0,y0)→(xN,yN)
    dx, dy = xN - x0, yN - y0
    denom = (dx * dx + dy * dy) ** 0.5
    if denom == 0:
        return 0

    best_d, best_k = -1.0, 0
    for i, c in enumerate(cum, start=1):
        # Distance from (i, c) to the chord
        d = abs(dy * i - dx * c + (dx * y0 - dy * x0)) / denom
        if d > best_d:
            best_d, best_k = d, i

    # Reject degenerate elbows: if the chosen distance is tiny,
    # the curve is essentially linear and there's no real elbow.
    # Threshold of 0.02 in the y-units (cumulative share) is a soft
    # sanity floor; tweak if it ever fires unexpectedly.
    if best_d < 0.02:
        return 0
    return best_k


def select(
    prompts,
    *,
    terminal_role: Optional[str] = None,   # not used; kept for API uniformity
    role_keys: RoleKeys = (),
    exclude: Set[str] = DEFAULT_EXCLUDE,
    partition_threshold: float = 0.05,
    min_prompt_fraction: float = 0.25,
    elbow_rescue: bool = True,
    elbow_floor_k: int = 1,
    **_,
) -> SelectionResult:
    if not role_keys:
        raise ValueError(
            "partition_coverage requires role_keys (use --task or "
            "--positions-from-metadata on the CLI)."
        )

    n_kept_prompts = len(prompts)
    min_prompts_floor = max(1, int(round(min_prompt_fraction * n_kept_prompts)))

    # Collect all paths, grouped by fingerprint.
    by_fp = aggregate_paths_by_fingerprint(prompts, role_keys, set(exclude))

    # Compute mass per partition; identify which are skipped.
    fp_total: Dict[Fingerprint, float] = {
        fp: sum(abs(r["score"]) for r in paths)
        for fp, paths in by_fp.items()
    }
    kept_fps = [fp for fp in by_fp if fp[0] not in SKIP_ROLE_SETS]
    kept_total_mass = sum(fp_total[fp] for fp in kept_fps)

    # Partition-share by fraction of *kept* mass.
    fp_share = {
        fp: (fp_total[fp] / kept_total_mass if kept_total_mass > 0 else 0.0)
        for fp in kept_fps
    }

    # Keep partitions above partition_threshold.
    above = [fp for fp in kept_fps if fp_share[fp] >= partition_threshold]
    above.sort(key=lambda fp: -fp_share[fp])

    # Elbow rescue: among partitions BELOW threshold, look for an elbow
    # on their cumulative-share curve and rescue partitions up to it.
    below = sorted(
        [fp for fp in kept_fps if fp_share[fp] < partition_threshold],
        key=lambda fp: -fp_share[fp],
    )
    rescued: List[Fingerprint] = []
    elbow_diag = {
        "enabled":      bool(elbow_rescue),
        "n_candidates": len(below),
    }
    if elbow_rescue and below:
        below_shares = [fp_share[fp] for fp in below]
        k = _elbow_index(below_shares)
        rescued = below[:k]
        elbow_diag.update({
            "elbow_rank":           k,
            "rescued_fingerprints": [fmt_fingerprint(fp) for fp in rescued],
            "candidate_shares":     below_shares,
        })

    kept_above = above + rescued

    # Per-partition selection: occurrence filter on normalized chains,
    # then elbow on cumulative within-partition score.
    union: Set[str] = set()
    per_partition = []
    for fp in kept_above:
        comps, info = select_within_partition(
            by_fp[fp], role_keys,
            min_prompts_floor=min_prompts_floor,
            elbow_floor_k=elbow_floor_k,
        )
        union.update(comps)
        per_partition.append({
            "fingerprint":         fmt_fingerprint(fp),
            "n_paths":             len(by_fp[fp]),
            "total_mass":          fp_total[fp],
            "share_of_kept":       fp_share[fp],
            "n_chains_total":      info["n_chains_total"],
            "n_chains_after_occ":  info["n_chains_after_occ"],
            "n_chains_kept":       info["n_chains_kept"],
            "elbow_rank":          info["elbow_rank"],
            "n_components":        len(comps),
            "components":          comps,
            "kept_chains":         info["kept_chains"],
        })

    # Full partition table for diagnostics.
    all_partitions = []
    total_all_mass = sum(fp_total.values())
    rescued_set = set(rescued)
    above_set = set(above)
    for fp in sorted(by_fp.keys(), key=lambda f: -fp_total[f]):
        if fp[0] in SKIP_ROLE_SETS:
            status = "skipped"
        elif fp in above_set:
            status = "kept"
        elif fp in rescued_set:
            status = "rescued"
        else:
            status = "below_thresh"
        all_partitions.append({
            "fingerprint":     fmt_fingerprint(fp),
            "n_paths":         len(by_fp[fp]),
            "total_mass":      fp_total[fp],
            "share_of_all":    (fp_total[fp] / total_all_mass
                                if total_all_mass > 0 else 0.0),
            "share_of_kept":   fp_share.get(fp, 0.0),
            "status":          status,
        })

    return SelectionResult(
        method="partition_coverage",
        components=sorted(union),
        n_components=len(union),
        params={
            "role_keys":           list(role_keys),
            "exclude":             sorted(exclude),
            "partition_threshold": partition_threshold,
            "min_prompt_fraction": min_prompt_fraction,
            "min_prompts_floor":   min_prompts_floor,
            "n_kept_prompts":      n_kept_prompts,
            "elbow_rescue":        elbow_rescue,
            "elbow_floor_k":       elbow_floor_k,
            "skip_fingerprints":   ["{}"  if not r else "{" + ",".join(sorted(r)) + "}"
                                    for r in SKIP_ROLE_SETS],
        },
        diagnostics={
            "kept_total_mass":  kept_total_mass,
            "all_partitions":   all_partitions,
            "kept_partitions":  per_partition,
            "elbow":            elbow_diag,
        },
    )