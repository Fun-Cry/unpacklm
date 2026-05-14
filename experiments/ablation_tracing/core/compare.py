"""
compare.py — diff two TraceResults.

Step 3. Given a clean trace and an ablated trace (same prompt, same
target, different ablation state), compute a component-centric diff
where each non-ablated component carries:

    role:         compensator / doubler / breakage / unclear
    direct, flow: clean / ablated / delta
    paths:        gained / lost / shared-with-delta — only those whose
                  chain passes through this component
    edges (×4):   attn-incoming, attn-outgoing, mlp-incoming,
                  mlp-outgoing — each filtered to edges adjacent to
                  this component, ranked by |delta|

Usage
-----

    from experiments.ablation_tracing import compare

    diff = compare(clean, ablated, ablation=cfg)

    for row in diff.components[:10]:
        print(row.name, row.role, row.delta_direct)

The DiffResult is fully self-contained — no model interaction. It's a
pure function of (clean, ablated, ablation).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .ablation import AblationConfig
from .trace import TraceResult


# ============================================================
#   Schema
# ============================================================

# Path identity: (chain_tuple, src_pos). Each chain element is a string
# like "attn_9_head_9@13" (component name + query position).
PathChain  = Tuple[str, ...]
PathKey    = Tuple[PathChain, int]


@dataclass
class SharedPathDiff:
    """A path present in both clean and ablated traces, with score delta."""
    chain:         PathChain
    src_pos:       int
    clean_score:   float
    ablated_score: float
    delta:         float


@dataclass
class GainedOrLostPath:
    """A path present in only one of {clean, ablated}."""
    chain:    PathChain
    src_pos:  int
    score:    float


@dataclass
class PathDiff:
    """Paths through one component, partitioned into three buckets."""
    gained: List[GainedOrLostPath] = field(default_factory=list)
    lost:   List[GainedOrLostPath] = field(default_factory=list)
    shared: List[SharedPathDiff]   = field(default_factory=list)


@dataclass
class EdgeDiff:
    src:           str
    tgt:           str
    clean_score:   float
    ablated_score: float
    delta:         float


@dataclass
class ComponentDiff:
    """One row of the per-component diff. Self-contained."""
    name:               str
    role:               str             # compensator / doubler / breakage / unclear

    clean_direct:       float
    ablated_direct:     float
    delta_direct:       float
    clean_flow:         float
    ablated_flow:       float
    delta_flow:         float

    paths:              PathDiff = field(default_factory=PathDiff)
    edges_attn_in:      List[EdgeDiff] = field(default_factory=list)
    edges_attn_out:     List[EdgeDiff] = field(default_factory=list)
    edges_mlp_in:       List[EdgeDiff] = field(default_factory=list)
    edges_mlp_out:      List[EdgeDiff] = field(default_factory=list)


@dataclass
class DiffResult:
    """Top-level result of compare()."""
    components:         List[ComponentDiff]
    ablated_components: List[str]
    delta_p_target:     float
    clean_direct_abl:   float        # sum of clean.direct[c] over ablated set — reference signal for classification
    abl_de_total:       float        # sum of (ablated.direct[c] - clean.direct[c]) over ablated set — diagnostic
    eps_path:           float
    eps_edge:           float

    def by_role(self, role: str) -> List[ComponentDiff]:
        """Convenience: filter rows by classification."""
        return [r for r in self.components if r.role == role]


# ============================================================
#   Internal helpers
# ============================================================

def _strip_pos(annotated: str) -> str:
    """Strip the @N suffix from a path-step name, returning the bare
    component name. 'attn_9_head_9@13' -> 'attn_9_head_9'."""
    return annotated.split("@", 1)[0]


def _classify(
    delta_direct:        float,
    clean_direct:        float,
    clean_direct_abl:    float,
    *,
    abl_threshold:       float,
    delta_threshold:     float,
) -> str:
    """Vote one component into compensator / doubler / breakage / unclear.

    Reference signal: ``clean_direct_abl`` — the total direct contribution
    of the ablated set in the clean run. This is "what was lost" when
    the ablation removed the head's IO-axis contribution. A compensator
    moved Δdirect with the same sign as ``clean_direct_abl`` (it picked
    up the kind of credit the ablated set was supplying); a doubler
    moved opposite (it removed credit in the same direction the
    ablation removed it).

    This matches the convention used in Wang et al. (IOI), McGrath et
    al. (Hydra effect), and Rushing & Nanda (self-repair): compensation
    is defined relative to the ablated head's *clean function*, not to
    its post-ablation Δdirect. Using Δdirect[ablated] as the reference
    would conflate a real compensator with the LayerNorm-rescaling
    artifact that produces small spurious changes in the ablated head's
    own projection (a known issue from Rushing & Nanda).

    A "breakage" is a component that was contributing in clean and now
    works against its own clean direction — independent of the
    ablation's sign.

    Below-threshold components are 'unclear' — too small to call.
    """
    if abs(clean_direct_abl) < abl_threshold:
        return "unclear"
    if abs(delta_direct) < delta_threshold:
        return "unclear"

    # Breakage: was contributing in clean, now flipped against itself
    # (independent of which direction the ablation pushed). Catch first.
    if (abs(clean_direct) >= delta_threshold
            and clean_direct * delta_direct < 0
            and abs(delta_direct) > abs(clean_direct) * 0.5):
        return "breakage"

    # Compensator: Δdirect along the same direction the ablated set was
    # supplying. Doubler: opposite direction (took away more of the
    # same kind of credit).
    if delta_direct * clean_direct_abl > 0:
        return "compensator"
    return "doubler"


def _path_passes_through(chain: PathChain, component: str) -> bool:
    """Is `component` (bare name) one of the steps in `chain`?"""
    return any(_strip_pos(step) == component for step in chain)


def _build_path_diff(
    clean_paths: List[Tuple[PathChain, int, float]],
    ablated_paths: List[Tuple[PathChain, int, float]],
    eps_path: float,
) -> Tuple[Dict[PathKey, GainedOrLostPath],
           Dict[PathKey, GainedOrLostPath],
           Dict[PathKey, SharedPathDiff]]:
    """Diff the global path lists into gained / lost / shared dicts
    keyed on (chain, src_pos). Shared filtered by |delta| > eps_path."""
    clean_map = {(chain, src): score for chain, src, score in clean_paths}
    abl_map   = {(chain, src): score for chain, src, score in ablated_paths}

    only_clean = clean_map.keys() - abl_map.keys()
    only_abl   = abl_map.keys()   - clean_map.keys()
    in_both    = clean_map.keys() & abl_map.keys()

    lost = {
        k: GainedOrLostPath(chain=k[0], src_pos=k[1], score=clean_map[k])
        for k in only_clean
    }
    gained = {
        k: GainedOrLostPath(chain=k[0], src_pos=k[1], score=abl_map[k])
        for k in only_abl
    }
    shared = {}
    for k in in_both:
        d = abl_map[k] - clean_map[k]
        if abs(d) >= eps_path:
            shared[k] = SharedPathDiff(
                chain         = k[0],
                src_pos       = k[1],
                clean_score   = clean_map[k],
                ablated_score = abl_map[k],
                delta         = d,
            )
    return gained, lost, shared


def _build_edge_diffs(
    clean_edges: Dict[Tuple[str, str], float],
    ablated_edges: Dict[Tuple[str, str], float],
    eps_edge: float,
) -> List[EdgeDiff]:
    """Return all (src, tgt) edges with |delta| > eps_edge.

    Edges in only one side are still shown — their other side has a
    score of 0.0 (the edge wasn't in the per-component-neighborhood
    top-K of that side, but the delta vs 0 may still be informative).
    """
    keys = clean_edges.keys() | ablated_edges.keys()
    out = []
    for k in keys:
        c = clean_edges.get(k, 0.0)
        a = ablated_edges.get(k, 0.0)
        d = a - c
        if abs(d) >= eps_edge:
            out.append(EdgeDiff(
                src=k[0], tgt=k[1],
                clean_score=c, ablated_score=a, delta=d,
            ))
    return out


# ============================================================
#   Public entry point
# ============================================================

def compare(
    clean:    TraceResult,
    ablated:  TraceResult,
    ablation: AblationConfig,
    *,
    eps_path:        float = 0.001,
    eps_edge:        float = 0.001,
    abl_threshold:   float = 0.05,
    delta_threshold: float = 0.02,
) -> DiffResult:
    """Compare clean vs ablated traces of the same prompt.

    Args:
        clean, ablated: TraceResults from trace() — typically one clean
            trace and one trace with intervention=... installed.
            on the same sentence and target.
        ablation: the AblationConfig that was used. Needed because role
            classification references the ablated set's clean direct
            attribution (what was lost).
        eps_path, eps_edge: filter shared/all entries by |delta|. Smaller
            eps -> more entries surfaced. Defaults match the old
            aggregator's noise floor.
        abl_threshold: ignore ablations where the ablated set's clean
            direct attribution is below this magnitude (no IO-axis
            signal to classify against — the ablated head wasn't
            doing anything in the first place).
        delta_threshold: a component's |Δdirect| must exceed this for
            its role to be classifiable; below, role='unclear'.

    Returns:
        DiffResult.
    """
    # --- target-side scalars ---
    delta_p_target = ablated.target_prob - clean.target_prob

    # Clean direct attribution of the ablated set. This is the
    # "what was lost" signal — used as the reference for role
    # classification. Wang/McGrath/Rushing-Nanda all use this
    # convention rather than Δdirect[ablated].
    clean_direct_abl = sum(
        clean.direct.get(c, 0.0) for c in ablation.components
    )
    # Δdirect on the ablated set, kept for diagnostic display.
    abl_de_total = sum(
        ablated.direct.get(c, 0.0) - clean.direct.get(c, 0.0)
        for c in ablation.components
    )

    # --- global diffs (computed once, attached per row below) ---
    gained_paths, lost_paths, shared_paths = _build_path_diff(
        clean.paths, ablated.paths, eps_path,
    )
    attn_edge_diffs = _build_edge_diffs(
        clean.edges_attn, ablated.edges_attn, eps_edge,
    )
    mlp_edge_diffs = _build_edge_diffs(
        clean.edges_mlp, ablated.edges_mlp, eps_edge,
    )

    # --- iterate components ---
    all_names = sorted(
        set(clean.direct.keys()) | set(ablated.direct.keys())
    )
    ablated_set = set(ablation.components)

    rows: List[ComponentDiff] = []
    for name in all_names:
        if name in ablated_set:
            continue                            # don't emit rows for the ablated heads themselves

        cd = clean.direct.get(name,   0.0)
        ad = ablated.direct.get(name, 0.0)
        dd = ad - cd
        cf = clean.flow.get(name,   0.0)
        af = ablated.flow.get(name, 0.0)
        df = af - cf

        role = _classify(
            delta_direct     = dd,
            clean_direct     = cd,
            clean_direct_abl = clean_direct_abl,
            abl_threshold    = abl_threshold,
            delta_threshold  = delta_threshold,
        )

        # Paths through this component, drawn from each global bucket.
        pd = PathDiff()
        for k, p in gained_paths.items():
            if _path_passes_through(p.chain, name):
                pd.gained.append(p)
        for k, p in lost_paths.items():
            if _path_passes_through(p.chain, name):
                pd.lost.append(p)
        for k, p in shared_paths.items():
            if _path_passes_through(p.chain, name):
                pd.shared.append(p)
        # Rank gained/lost by |score|, shared by |delta|.
        pd.gained.sort(key=lambda p: abs(p.score), reverse=True)
        pd.lost.sort  (key=lambda p: abs(p.score), reverse=True)
        pd.shared.sort(key=lambda p: abs(p.delta), reverse=True)

        # Edges adjacent to this component.
        attn_in  = [e for e in attn_edge_diffs if e.tgt == name]
        attn_out = [e for e in attn_edge_diffs if e.src == name]
        mlp_in   = [e for e in mlp_edge_diffs  if e.tgt == name]
        mlp_out  = [e for e in mlp_edge_diffs  if e.src == name]
        for ls in (attn_in, attn_out, mlp_in, mlp_out):
            ls.sort(key=lambda e: abs(e.delta), reverse=True)

        rows.append(ComponentDiff(
            name           = name,
            role           = role,
            clean_direct   = cd,
            ablated_direct = ad,
            delta_direct   = dd,
            clean_flow     = cf,
            ablated_flow   = af,
            delta_flow     = df,
            paths          = pd,
            edges_attn_in  = attn_in,
            edges_attn_out = attn_out,
            edges_mlp_in   = mlp_in,
            edges_mlp_out  = mlp_out,
        ))

    # Rank rows by |delta_direct| descending.
    rows.sort(key=lambda r: abs(r.delta_direct), reverse=True)

    return DiffResult(
        components         = rows,
        ablated_components = list(ablation.components),
        delta_p_target     = delta_p_target,
        clean_direct_abl   = clean_direct_abl,
        abl_de_total       = abl_de_total,
        eps_path           = eps_path,
        eps_edge           = eps_edge,
    )