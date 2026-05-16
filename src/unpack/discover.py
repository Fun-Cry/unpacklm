"""
unpack.discover - Hierarchical circuit discovery via DLA + recursive rerooting.

Algorithm:
  1. Seed: select late-layer attention heads with high direct logit
     attribution (|DLA|).
  2. Expand: BFS rerooting from each seed head, extracting top-k upstream
     attention heads per K/Q/V branch via backward recursion.
  3. Intersect: across prompts, keep components above a frequency threshold.

Usage:
    import unpack

    tracer = unpack.Tracer("gpt2", device="cuda:0")
    circuit = unpack.discover(
        tracer,
        prompts=[{"text": ..., "target": ..., "distractor": ...,
                  "positions": {"END": 13, "S2": 9, ...}}, ...],
        config="kqv_aligned",
    )
    print(circuit)  # Circuit(25 components, 15 heads, 10 MLPs)
"""

import re
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from unpack.circuit import Circuit

_BRANCH_RE = re.compile(r"\[([KQV])\]")


# ── Position strategies ──

def position_by_layer(positions: dict, layer: int, cutoff: int = 7) -> int:
    """Default: late layers (>= cutoff) reroot at END, earlier at S2.

    Works for IOI and similar tasks where late heads operate at the
    final position and early/mid heads at an earlier mention.
    """
    if layer >= cutoff:
        return positions["END"]
    return positions.get("S2", positions["END"])


def position_always_end(positions: dict, layer: int, **kw) -> int:
    """Always reroot at END."""
    return positions["END"]


# ── Core extraction ──

def _extract_attn_upstream(tracer, prep, cfg, head, pos,
                           top_k=3, n_paths_scan=200,
                           exclude_layers=None):
    """Reroot from head@pos, return top-k attn heads per K/Q/V mode.

    Returns: dict[mode] -> dict[head_name] -> (score, chain_str)
    """
    if exclude_layers is None:
        exclude_layers = set()
    root = f"{head}@{pos}"
    rr = tracer.trace_from_prep(prep, cfg, root=root)

    buckets = defaultdict(list)
    for path in rr.paths[:n_paths_scan]:
        parts = path.chain.split("\u2192")
        m = _BRANCH_RE.search(parts[0])
        mode = m.group(1) if m else None
        if mode is None:
            continue
        for step in parts[1:]:
            clean = _BRANCH_RE.sub("", step)
            clean = re.sub(r"@.*", "", clean).strip()
            if clean.startswith("attn_"):
                layer = int(clean.split("_")[1])
                if layer not in exclude_layers:
                    buckets[mode].append((abs(path.score), clean, path.chain))
                break

    result = {}
    for mode in ["K", "Q", "V"]:
        seen = {}
        for score, head_name, chain in sorted(
                buckets[mode], key=lambda x: -x[0]):
            if head_name not in seen:
                seen[head_name] = (score, chain)
            if len(seen) >= top_k:
                break
        result[mode] = seen
    return result


# ── Seed selection ──

def _select_seeds(attn_flow: Dict[str, float], min_layer: int = 10):
    """Pick seed heads from late layers using min(elbow, top10, 1/3 max).

    Returns: set of head names.
    """
    late = {k: v for k, v in attn_flow.items()
            if int(k.split("_")[1]) >= min_layer}
    ranked = sorted(late.items(), key=lambda x: -abs(x[1]))
    if not ranked:
        return set()

    scores = [abs(v) for _, v in ranked]
    cum = np.cumsum(scores)
    total = cum[-1]
    n = len(scores)

    # Elbow
    best_d, elbow_k = -1, n
    for i in range(1, n + 1):
        d = abs(total * i - n * cum[i - 1])
        if d > best_d:
            best_d = d
            elbow_k = i

    # Top 10
    top10_k = min(10, n)

    # 1/3 of max
    max_mag = scores[0]
    third_k = sum(1 for s in scores if s >= max_mag / 3)

    k = min(elbow_k, top10_k, third_k)
    return set(name for name, _ in ranked[:k])


# ── Single-prompt discovery ──

def discover_one(tracer, prompt: dict, config="kqv_aligned",
                 seed_min_layer: int = 10,
                 top_k_per_mode: int = 3,
                 max_depth: int = 3,
                 n_paths_scan: int = 200,
                 exclude_layers: Optional[Set[int]] = None,
                 mlp_flow_frac: Optional[float] = None,
                 position_fn: Callable = position_by_layer) -> dict:
    """Discover a circuit from a single prompt.

    Args:
        tracer: unpack.Tracer instance.
        prompt: dict with 'text', 'target', 'distractor', 'positions'.
            positions maps role names (e.g. "END", "S2") to int positions.
        config: trace config name or TraceConfig.
        seed_min_layer: only consider heads at this layer or later as seeds.
        top_k_per_mode: top-k upstream heads per K/Q/V branch per rerooting.
        max_depth: max BFS depth for rerooting.
        n_paths_scan: number of paths to scan per rerooting.
        exclude_layers: set of layer indices to exclude from final circuit
            (default: {0}, since layer-0 heads appear universally as
            embedding relays).
        mlp_flow_frac: include MLPs with |flow| >= this fraction of max
            MLP |flow|. Default: None (attention heads only).
        position_fn: callable(positions, layer) -> int position for rerooting.

    Returns: dict with 'circuit' (set), 'attn_circuit', 'mlp_circuit',
        'edges', 'seeds', 'depth_map', 'flow'.
    """
    if exclude_layers is None:
        exclude_layers = {0, 1}

    prep, cfg = tracer.prepare(
        prompt["text"], target=prompt["target"],
        distractor=prompt["distractor"], config=config)
    result = tracer.trace_from_prep(prep, cfg, root=None)

    all_flow = {k: v for k, v in result.component_flow.items()
                if "bias" not in k and k not in ("embedding", "pos_embedding")}
    dla = {k: v for k, v in result.importance.items()
           if k.startswith("attn_") and "bias" not in k}
    mlp_flow = {k: v for k, v in all_flow.items() if k.startswith("mlp_")}

    # Seeds from DLA
    seed_set = _select_seeds(dla, min_layer=seed_min_layer)

    # BFS rerooting
    attn_circuit = set(seed_set)
    edges = []
    queue = list(seed_set)
    visited = set()
    depth_map = {h: 0 for h in seed_set}

    while queue:
        head = queue.pop(0)
        if head in visited:
            continue
        visited.add(head)
        depth = depth_map[head]
        if depth >= max_depth:
            continue

        layer = int(head.split("_")[1])
        pos = position_fn(prompt["positions"], layer)

        upstream = _extract_attn_upstream(
            tracer, prep, cfg, head, pos, top_k_per_mode, n_paths_scan,
            exclude_layers=exclude_layers)

        for mode, heads in upstream.items():
            for up_head, (score, chain) in heads.items():
                edges.append((head, up_head, mode, score))
                if up_head not in attn_circuit:
                    up_layer = int(up_head.split("_")[1])
                    if up_layer not in exclude_layers:
                        attn_circuit.add(up_head)
                        depth_map[up_head] = depth + 1
                        queue.append(up_head)

    # Exclude layers
    attn_circuit = {h for h in attn_circuit
                    if int(h.split("_")[1]) not in exclude_layers}

    # MLPs via flow (optional)
    if mlp_flow_frac is not None:
        max_mlp = max((abs(v) for v in mlp_flow.values()), default=0)
        mlp_thresh = mlp_flow_frac * max_mlp
        mlp_circuit = {k for k, v in mlp_flow.items() if abs(v) >= mlp_thresh}
    else:
        mlp_circuit = set()

    circuit = attn_circuit | mlp_circuit

    return {
        "circuit": circuit,
        "attn_circuit": attn_circuit,
        "mlp_circuit": mlp_circuit,
        "edges": edges,
        "seeds": seed_set,
        "depth_map": depth_map,
        "dla": dla,
        "flow": all_flow,
    }


# ── Multi-prompt discovery ──

def discover(tracer, prompts: List[dict], config="kqv_aligned",
             freq_threshold: float = 0.5,
             seed_min_layer: int = 10,
             top_k_per_mode: int = 3,
             max_depth: int = 3,
             n_paths_scan: int = 200,
             exclude_layers: Optional[Set[int]] = None,
             mlp_flow_frac: Optional[float] = None,
             position_fn: Callable = position_by_layer,
             verbose: bool = True) -> Circuit:
    """Discover a circuit from multiple prompts by intersection.

    Runs discover_one on each prompt, then keeps components appearing
    in >= freq_threshold fraction of prompts.

    Args:
        tracer: unpack.Tracer instance.
        prompts: list of prompt dicts, each with 'text', 'target',
            'distractor', 'positions'.
        config: trace config name or TraceConfig.
        freq_threshold: keep components appearing in this fraction of prompts.
        (other args: see discover_one)

    Returns: Circuit object.
    """
    if exclude_layers is None:
        exclude_layers = {0, 1}

    n = len(prompts)
    all_results = []
    head_freq = Counter()

    for pi, p in enumerate(prompts):
        r = discover_one(
            tracer, p, config=config,
            seed_min_layer=seed_min_layer,
            top_k_per_mode=top_k_per_mode,
            max_depth=max_depth,
            n_paths_scan=n_paths_scan,
            exclude_layers=exclude_layers,
            mlp_flow_frac=mlp_flow_frac,
            position_fn=position_fn)
        all_results.append(r)

        for h in r["circuit"]:
            head_freq[h] += 1

        if verbose:
            print(f"  [{pi+1}/{n}] {len(r['seeds'])} seeds → "
                  f"{len(r['attn_circuit'])} attn + "
                  f"{len(r['mlp_circuit'])} mlp = "
                  f"{len(r['circuit'])} total")

    min_count = max(1, int(n * freq_threshold))
    components = frozenset(
        h for h, cnt in head_freq.items() if cnt >= min_count)

    if verbose:
        attn = [c for c in components if c.startswith("attn_")]
        mlps = [c for c in components if c.startswith("mlp_")]
        print(f"\n  Circuit (≥{freq_threshold*100:.0f}% of {n}): "
              f"{len(attn)} heads + {len(mlps)} MLPs = {len(components)}")

    return Circuit(
        components=components,
        model_name=getattr(tracer, '_model_name', None),
        config_name=config if isinstance(config, str) else None,
    )