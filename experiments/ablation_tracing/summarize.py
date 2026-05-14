"""Summarize ablation_tracing self-repair results across cells.

Run after the experiment has produced cells in <results_dir>:

    python -m experiments.ablation_tracing.summarize <results_dir>

Three sections are printed:

    1. HEADLINES (one row per condition)
       Δp, Δlogit, clean_direct_abl, abl_de_total — the
       "we replicate Wang/McGrath" summary.

    2. PER-COMPONENT ROLE DISTRIBUTION (one block per condition)
       Two-axis classification:
         primary axis: compensator vs doubler — sign(Δdirect *
                       clean_direct_abl). Compensator picked up the
                       direction the ablated set was supplying;
                       doubler took more away.
         breakage flag: did the component flip against its own clean
                       function?  Annotated as "of which N were
                       breakages" within each primary bucket.
       This recovers information the saved 4-way label hid: a
       previously-suppressing head whose suppression was disrupted
       and now helps IO recovery is _both_ a compensator and a
       breakage, and you want to know that.

    3. EDGE & PATH DRILLDOWN (top components only)
       For each top-K component in each condition, for each role with
       at least --min-role-count appearances: aggregated incoming /
       outgoing attn & mlp edges and gained / lost / shared paths
       through that component, with count, mean delta, std. The
       mechanism story.

By default the summary restricts to canonical-correct cells: prompts
where clean P(target) >= --p-min and clean (target - distractor) logit
> 0. This filters out cells where the model wasn't running the
canonical IOI circuit and the role labels would reflect non-canonical
behavior. Use --no-filter to summarize over every cell.
"""

import argparse
import contextlib
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# Make the project root importable when invoked via -m or directly.

from experiments.ablation_tracing import load_runs, ExperimentRun


# ============================================================
#   Filtering
# ============================================================

def is_canonical(run: ExperimentRun, p_min: float) -> bool:
    """Did the model run the canonical IOI circuit on this prompt in clean?

    Conservative: P(IO) >= p_min AND IO logit > S logit.
    target_logit_centered is (IO - S) when distractor_token is set, so
    > 0 means IO is preferred over S. Cells failing either check are
    dropped — their role labels reflect non-canonical behavior and
    would dilute the cross-cell signal.
    """
    return (run.clean_target_prob >= p_min
            and run.clean_target_logit > 0)


# ============================================================
#   Two-axis role assignment
# ============================================================

def primary_role(comp_diff, clean_direct_abl: float) -> str:
    """compensator vs doubler vs unclear, ignoring the breakage shadowing.

    Recomputes the primary axis directly from delta_direct *
    clean_direct_abl, so a component the saved classifier labelled
    "breakage" still gets a primary role here. "unclear" is preserved
    from the saved label (it depends on thresholds we don't store).
    """
    if comp_diff.role == "unclear":
        return "unclear"
    return ("compensator"
            if comp_diff.delta_direct * clean_direct_abl > 0
            else "doubler")


def is_breakage(comp_diff) -> bool:
    return comp_diff.role == "breakage"


# ============================================================
#   Stats helpers
# ============================================================

def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:           return (0.0, 0.0)
    if len(xs) == 1:     return (float(xs[0]), 0.0)
    return (float(np.mean(xs)), float(np.std(xs, ddof=1)))


def _median(xs: List[float]) -> float:
    return float(np.median(xs)) if xs else 0.0


def _strip_pos(annotated: str) -> str:
    """'attn_9_head_9@13' -> 'attn_9_head_9'."""
    return annotated.split("@", 1)[0]


# ============================================================
#   Visual annotation for the role-detail printer
# ============================================================

def _annotate_name(name: str, focus: str, ablated: set) -> str:
    """Wrap a bare component name with role markers.

    <<name>>   the section's focus compensator/doubler
    [ABL:name] in this condition's ablated set
    name       neither — printed as-is
    """
    if name == focus:
        return f"<<{name}>>"
    if name in ablated:
        return f"[ABL:{name}]"
    return name


def _annotate_step(step: str, focus: str, ablated: set) -> str:
    """Annotate one step in a path chain.

    Recognized formats:
      attn_9_head_9@13               -> wraps the bare name
      attn_9_head_9                  -> wraps the bare name
      [ABLATED:attn_9_head_9]@13     -> rewrites as [ABL:name]@pos
      [ABLATED:attn_9_head_9]        -> rewrites as [ABL:name]
    """
    # Split off @position suffix if any.
    if "@" in step:
        bare, pos = step.split("@", 1)
        suffix = "@" + pos
    else:
        bare, suffix = step, ""

    # Trace's existing [ABLATED:name] terminal marker — normalize to [ABL:name]
    # so the formatting matches the same-style markers we add elsewhere.
    if bare.startswith("[ABLATED:") and bare.endswith("]"):
        name = bare[len("[ABLATED:"):-1]
        return f"[ABL:{name}]{suffix}"

    return _annotate_name(bare, focus, ablated) + suffix


# ============================================================
#   Section 1: Headlines
# ============================================================

@dataclass
class Headline:
    label:               str
    n_total:             int
    n_kept:              int
    delta_p_mean:        float
    delta_p_median:      float
    delta_p_std:         float
    delta_logit_mean:    float
    delta_logit_median:  float
    delta_logit_std:     float
    clean_direct_abl:    float
    abl_de_total:        float


def build_headlines(
    runs_by_label: Dict[str, List[ExperimentRun]],
    p_min: float,
    no_filter: bool,
) -> List[Headline]:
    out = []
    for label, runs in runs_by_label.items():
        cells = runs if no_filter else [r for r in runs if is_canonical(r, p_min)]
        if not cells:
            out.append(Headline(label, len(runs), 0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue

        d_p     = [r.ablated_target_prob  - r.clean_target_prob  for r in cells]
        d_logit = [r.ablated_target_logit - r.clean_target_logit for r in cells]
        cda     = [r.clean_direct_abl for r in cells]
        adt     = [r.abl_de_total     for r in cells]

        d_p_m,   d_p_s   = _mean_std(d_p)
        d_l_m,   d_l_s   = _mean_std(d_logit)

        out.append(Headline(
            label              = label,
            n_total            = len(runs),
            n_kept             = len(cells),
            delta_p_mean       = d_p_m,
            delta_p_median     = _median(d_p),
            delta_p_std        = d_p_s,
            delta_logit_mean   = d_l_m,
            delta_logit_median = _median(d_logit),
            delta_logit_std    = d_l_s,
            clean_direct_abl   = _mean_std(cda)[0],
            abl_de_total       = _mean_std(adt)[0],
        ))
    return out


def print_headlines(rows: List[Headline]):
    print("=" * 122)
    print("  HEADLINES (per condition)")
    print("=" * 122)
    print(f"  {'condition':<16} {'n':>4} {'kept':>5}   "
          f"{'Δp mean':>9} {'Δp med':>9} {'Δp std':>8}   "
          f"{'Δlog mn':>9} {'Δlog md':>9} {'Δlog std':>9}   "
          f"{'cda':>7} {'adt':>7}")
    print("-" * 122)
    for r in rows:
        print(f"  {r.label:<16} {r.n_total:>4} {r.n_kept:>5}   "
              f"{r.delta_p_mean:>+9.4f} {r.delta_p_median:>+9.4f} {r.delta_p_std:>8.4f}   "
              f"{r.delta_logit_mean:>+9.3f} {r.delta_logit_median:>+9.3f} {r.delta_logit_std:>9.3f}   "
              f"{r.clean_direct_abl:>+7.2f} {r.abl_de_total:>+7.2f}")
    print("\n  cda = clean_direct_abl (sum of clean.direct over ablated set)")
    print("  adt = abl_de_total      (sum of Δdirect      over ablated set)")


# ============================================================
#   Section 2: Per-component role distribution (two-axis)
# ============================================================

@dataclass
class CompPrimaryStats:
    """Stats for one component in one primary role under one condition."""
    count:              int
    n_breakage:         int     # of count, how many were also breakages
    delta_direct_mean:  float
    delta_direct_std:   float
    delta_flow_mean:    float
    delta_flow_std:     float


def per_component_primary(
    cells: List[ExperimentRun],
) -> Dict[str, Dict[str, CompPrimaryStats]]:
    """For one (already-filtered) condition: {comp_name: {primary_role: stats}}.

    primary_role in {"compensator", "doubler"}. "unclear" is dropped.
    Within each primary bucket, we additionally count how many
    component-cells were saved as "breakage" (the shadowed sub-flag).
    """
    bucket: Dict[Tuple[str, str], List[Tuple[float, float, bool]]] = defaultdict(list)
    for run in cells:
        cda = run.clean_direct_abl
        for c in run.diff.components:
            pr = primary_role(c, cda)
            if pr == "unclear":
                continue
            bucket[(c.name, pr)].append(
                (c.delta_direct, c.delta_flow, is_breakage(c))
            )

    out: Dict[str, Dict[str, CompPrimaryStats]] = defaultdict(dict)
    for (name, pr), triples in bucket.items():
        dd = [t[0] for t in triples]
        df = [t[1] for t in triples]
        nb = sum(1 for t in triples if t[2])
        ddm, dds = _mean_std(dd)
        dfm, dfs = _mean_std(df)
        out[name][pr] = CompPrimaryStats(
            count             = len(triples),
            n_breakage        = nb,
            delta_direct_mean = ddm,
            delta_direct_std  = dds,
            delta_flow_mean   = dfm,
            delta_flow_std    = dfs,
        )
    return out


def print_role_distribution(label: str, n_kept: int, per_comp, top_k: int):
    scored = sorted(
        ((n, sum(s.count for s in roles.values())) for n, roles in per_comp.items()),
        key=lambda kv: kv[1], reverse=True,
    )

    print(f"\n{'─' * 122}")
    print(f"  ROLE DISTRIBUTION  —  condition '{label}'  (n_kept = {n_kept})")
    print(f"{'─' * 122}")
    print(f"  {'component':<22} "
          f"{'comp':>5} {'(brk)':>5}   {'doub':>5} {'(brk)':>5}   "
          f"{'Δd compen':>14}   {'Δd doubler':>14}   "
          f"{'Δflow compen':>15}")
    print("-" * 122)

    for name, _ in scored[:top_k]:
        roles = per_comp[name]
        c = roles.get("compensator")
        d = roles.get("doubler")

        def fmt_dd(s):
            if s is None or s.count == 0:
                return f"{'—':>14}"
            return f"{s.delta_direct_mean:>+5.2f} ± {s.delta_direct_std:.2f}".rjust(14)
        def fmt_df(s):
            if s is None or s.count == 0:
                return f"{'—':>15}"
            return f"{s.delta_flow_mean:>+5.2f} ± {s.delta_flow_std:.2f}".rjust(15)

        cn  = c.count      if c else 0
        cnb = c.n_breakage if c else 0
        dn  = d.count      if d else 0
        dnb = d.n_breakage if d else 0
        print(f"  {name:<22} "
              f"{cn:>5} {cnb:>5}   {dn:>5} {dnb:>5}   "
              f"{fmt_dd(c)}   {fmt_dd(d)}   {fmt_df(c)}")


# ============================================================
#   Section 3: Edge and path drilldown
# ============================================================

def _agg_edges(edge_lists, min_count=0, min_abs_mean=0.0):
    """[[EdgeDiff, ...], ...] -> [((src, tgt), count, mean, std), ...]

    Filters:
      min_count    — drop edges appearing in fewer than N cells
      min_abs_mean — drop edges with |mean delta| below this magnitude
    """
    bucket = defaultdict(list)
    for cell in edge_lists:
        for e in cell:
            bucket[(e.src, e.tgt)].append(e.delta)
    out = []
    for k, v in bucket.items():
        if len(v) < min_count:
            continue
        m, s = _mean_std(v)
        if abs(m) < min_abs_mean:
            continue
        out.append((k, len(v), m, s))
    out.sort(key=lambda x: (-x[1], -abs(x[2])))
    return out


def _agg_paths(path_lists, valfn, strip_positions=False,
               min_count=0, min_abs_mean=0.0):
    """[[Path, ...], ...] -> [(chain_tuple, count, mean, std), ...]

    By default keeps @position annotations on each step, so the same
    structural chain landing at different source tokens is counted as
    distinct paths. Set strip_positions=True to merge by structural
    chain only (loses the IO-vs-S2-vs-BOS distinction in IOI).

    Filters:
      min_count    — drop entries appearing in fewer than N cells
      min_abs_mean — drop entries with |mean| below this magnitude
                     (intended for cutting the noise-level ~1e-3 tail)
    """
    bucket = defaultdict(list)
    for cell in path_lists:
        for p in cell:
            if strip_positions:
                chain = tuple(_strip_pos(s) for s in p.chain)
            else:
                chain = tuple(p.chain)
            bucket[chain].append(valfn(p))
    out = []
    for k, v in bucket.items():
        if len(v) < min_count:
            continue
        m, s = _mean_std(v)
        if abs(m) < min_abs_mean:
            continue
        out.append((k, len(v), m, s))
    out.sort(key=lambda x: (-x[1], -abs(x[2])))
    return out


def component_role_detail(
    cells: List[ExperimentRun], comp_name: str, primary: str, top_k: int,
    strip_positions: bool = False,
    min_path_count: int = 0,
    min_path_mean:  float = 0.0,
    min_edge_count: int = 0,
    min_edge_mean:  float = 0.0,
) -> Dict:
    """Edge/path detail for one (component, primary_role) pair."""
    cdiffs = []
    for r in cells:
        cda = r.clean_direct_abl
        for c in r.diff.components:
            if c.name == comp_name and primary_role(c, cda) == primary:
                cdiffs.append(c)
                break
    if not cdiffs:
        return {"n_cells": 0}

    edge_kw = dict(min_count=min_edge_count, min_abs_mean=min_edge_mean)
    path_kw = dict(strip_positions=strip_positions,
                   min_count=min_path_count,
                   min_abs_mean=min_path_mean)

    return {
        "n_cells":        len(cdiffs),
        "edges_attn_in":  _agg_edges([c.edges_attn_in  for c in cdiffs], **edge_kw)[:top_k],
        "edges_attn_out": _agg_edges([c.edges_attn_out for c in cdiffs], **edge_kw)[:top_k],
        "edges_mlp_in":   _agg_edges([c.edges_mlp_in   for c in cdiffs], **edge_kw)[:top_k],
        "edges_mlp_out":  _agg_edges([c.edges_mlp_out  for c in cdiffs], **edge_kw)[:top_k],
        "paths_gained":   _agg_paths([c.paths.gained for c in cdiffs],
                                     valfn=lambda p: p.score, **path_kw)[:top_k],
        "paths_lost":     _agg_paths([c.paths.lost   for c in cdiffs],
                                     valfn=lambda p: p.score, **path_kw)[:top_k],
        "paths_shared":   _agg_paths([c.paths.shared for c in cdiffs],
                                     valfn=lambda p: p.delta, **path_kw)[:top_k],
    }


def print_role_detail(comp_name: str, primary: str,
                      n_role: int, n_total: int, detail: Dict,
                      ablated_set: set):
    pct = (n_role / n_total) * 100 if n_total else 0.0
    print(f"\n  ── {comp_name}  as  '{primary}'    "
          f"({n_role}/{n_total} cells, {pct:.0f}%) ──")
    print(f"     focus: <<{comp_name}>>     "
          f"ablated: {sorted(ablated_set)}")

    def annotate_edge(src, tgt):
        return (f"{_annotate_name(src, comp_name, ablated_set)} "
                f"→ {_annotate_name(tgt, comp_name, ablated_set)}")

    def annotate_chain(chain_tuple):
        return " ← ".join(_annotate_step(s, comp_name, ablated_set)
                          for s in chain_tuple)

    def section(title, entries, key_fmt, val_label):
        if not entries:
            return
        print(f"    {title}:")
        for k, count, m, s in entries:
            print(f"      {key_fmt(k):<60}  {count:>4}  "
                  f"{val_label} = {m:>+7.3f} ± {s:.3f}")

    section("incoming attn edges",  detail["edges_attn_in"],
            key_fmt=lambda k: annotate_edge(*k), val_label="Δ")
    section("outgoing attn edges",  detail["edges_attn_out"],
            key_fmt=lambda k: annotate_edge(*k), val_label="Δ")
    section("incoming mlp edges",   detail["edges_mlp_in"],
            key_fmt=lambda k: annotate_edge(*k), val_label="Δ")
    section("outgoing mlp edges",   detail["edges_mlp_out"],
            key_fmt=lambda k: annotate_edge(*k), val_label="Δ")
    section("paths GAINED  (only in ablated)", detail["paths_gained"],
            key_fmt=lambda c: annotate_chain(c), val_label="score")
    section("paths LOST    (only in clean)",   detail["paths_lost"],
            key_fmt=lambda c: annotate_chain(c), val_label="score")
    section("paths SHARED  (in both, by |delta|)", detail["paths_shared"],
            key_fmt=lambda c: annotate_chain(c), val_label="Δ")


# ============================================================
#   CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("results_dir")
    parser.add_argument("--top-k-components", type=int, default=15,
                        help="rank top-K components per condition (default: 15)")
    parser.add_argument("--top-k-detail",     type=int, default=8,
                        help="top-K edges/paths in role detail (default: 8)")
    parser.add_argument("--min-role-count",   type=int, default=5,
                        help="skip role detail with fewer than N cells "
                             "(default: 5)")
    parser.add_argument("--p-min",            type=float, default=0.1,
                        help="P(IO) threshold for canonical-correct filter "
                             "(default: 0.1)")
    parser.add_argument("--no-filter", action="store_true",
                        help="don't filter to canonical-correct cells")
    parser.add_argument("--no-detail", action="store_true",
                        help="skip section 3 (edge/path drilldown) — useful "
                             "for a quick top-level look")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="write summary to this file instead of stdout. "
                             "Status messages still go to stderr.")
    parser.add_argument("--strip-positions", action="store_true",
                        help="aggregate paths by structural chain only, "
                             "ignoring @position annotations. Loses the "
                             "IO-vs-S2-vs-BOS distinction in IOI but gives "
                             "tighter counts for cross-prompt patterns.")
    parser.add_argument("--min-path-count", type=int, default=0,
                        help="drop paths appearing in fewer than N cells "
                             "(default: 0 — keep all). Useful for cutting "
                             "single-prompt noise.")
    parser.add_argument("--min-path-mean", type=float, default=0.0,
                        help="drop paths with |mean score / Δ| below this "
                             "magnitude (default: 0.0). 0.005 cuts the "
                             "~1e-3 noise tail.")
    parser.add_argument("--min-edge-count", type=int, default=0,
                        help="drop edges appearing in fewer than N cells.")
    parser.add_argument("--min-edge-mean", type=float, default=0.0,
                        help="drop edges with |mean Δ| below this magnitude.")
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    print(f"loaded {len(runs)} cells from {args.results_dir}", file=sys.stderr)

    runs_by_label: Dict[str, List[ExperimentRun]] = defaultdict(list)
    for r in runs:
        runs_by_label[r.label].append(r)

    # Pick output stream. Status messages already go to stderr above
    # so the user sees progress even when --output is set.
    if args.output:
        out_stream = open(args.output, "w")
    else:
        out_stream = sys.stdout

    try:
        with contextlib.redirect_stdout(out_stream):
            # ── Section 1 ──
            rows = build_headlines(runs_by_label, args.p_min, args.no_filter)
            print_headlines(rows)

            # ── Sections 2 + 3 per condition ──
            for label, condition_runs in runs_by_label.items():
                cells = (condition_runs if args.no_filter
                         else [r for r in condition_runs
                               if is_canonical(r, args.p_min)])
                if not cells:
                    continue

                per_comp = per_component_primary(cells)
                print_role_distribution(label, len(cells), per_comp,
                                        top_k=args.top_k_components)

                if args.no_detail:
                    continue

                # rank top components by total compensator + doubler count
                ranked = sorted(
                    ((n, sum(s.count for s in roles.values()))
                     for n, roles in per_comp.items()),
                    key=lambda kv: kv[1], reverse=True,
                )
                top_names = [n for n, _ in ranked[:args.top_k_components]]

                for name in top_names:
                    for primary in ("compensator", "doubler"):
                        stats = per_comp[name].get(primary)
                        if stats is None or stats.count < args.min_role_count:
                            continue
                        detail = component_role_detail(
                            cells, name, primary, top_k=args.top_k_detail,
                            strip_positions=args.strip_positions,
                            min_path_count=args.min_path_count,
                            min_path_mean=args.min_path_mean,
                            min_edge_count=args.min_edge_count,
                            min_edge_mean=args.min_edge_mean,
                        )
                        print_role_detail(name, primary, stats.count,
                                          len(cells), detail,
                                          ablated_set=set(cells[0].ablated_components))
    finally:
        if out_stream is not sys.stdout:
            out_stream.close()
            print(f"summary written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()