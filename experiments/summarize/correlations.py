"""Edge correlation clustering via Leiden (CPM).

Pipeline:
  1. Build per-edge score vectors from raw HDF5
  2. (Optional) Filter low-variance edges automatically
  3. Compute pairwise Pearson correlations
  4. Cluster with Leiden CPM on the positive-correlation graph
  5. Store results in summary H5

Two modes:
  "topk"  — cluster only the pre-computed top-K edges (fast)
  "auto"  — enumerate ALL edges, variance-filter, then cluster

Results stored under {prefix}/correlations/{direction}/:
    attn_corr, mlp_corr, cross_corr, combined_corr  — correlation matrices
    attn_edges, mlp_edges                            — structured edge arrays
    attn_clusters, mlp_clusters, combined_clusters   — cluster labels
"""

import numpy as np


# ── Public entry point ──────────────────────────────────────────────

def compute_and_store(store, summary_path, direction="pos",
                      resolution=0.5, min_corr=0.1, mode="topk",
                      print_fn=print):
    """Compute correlations, cluster with Leiden, write to summary H5.

    Args:
        store:           DataStore with raw H5 open.
        summary_path:    Path to summary H5 (opened read-write).
        direction:       'pos' or 'neg'.
        resolution:      Leiden CPM resolution (higher = tighter clusters).
        min_corr:        Minimum correlation to keep as a graph edge.
        mode:            'topk' (use summary top-K) or 'auto' (all edges,
                         variance-filtered).
        print_fn:        Logging callback.
    """
    import h5py

    print_fn(f"\nCorrelation clustering ({direction}, mode={mode})...")

    if mode == "auto":
        result = _correlate_all_edges(store, direction, print_fn)
    else:
        from ..data.queries import compute_edge_correlations
        result = compute_edge_correlations(
            store, direction=direction, print_fn=print_fn)

    if not result:
        print_fn("  No data to correlate.")
        return

    # Close store's read handle so we can write
    if store._sf:
        store._sf.close()
        store._sf = None

    with h5py.File(summary_path, "a") as f:
        grp_name = f"{store.prefix}/correlations/{direction}"
        if grp_name in f:
            del f[grp_name]
        grp = f.create_group(grp_name)

        for graph in ("attn", "mlp"):
            if graph not in result:
                continue
            _write_graph(grp, graph, result[graph],
                         resolution, min_corr, print_fn)

        if "cross" in result:
            grp.create_dataset(
                "cross_corr",
                data=result["cross"]["corr"].astype(np.float16))
            print_fn(f"  Stored cross_corr {result['cross']['corr'].shape}")

        if "combined" in result:
            corr = result["combined"]["corr"]
            grp.create_dataset("combined_corr",
                               data=corr.astype(np.float16))
            if corr.shape[0] > 2:
                labels = cluster_leiden(corr, resolution, min_corr)
                grp.create_dataset("combined_clusters", data=labels)
                print_fn(f"  Combined clusters: {len(set(labels))}")

    print_fn("  Correlations saved.")


# ── Leiden clustering ───────────────────────────────────────────────

def cluster_leiden(corr, resolution=0.5, min_weight=0.1):
    """Cluster a correlation matrix with Leiden CPM.

    Falls back to hierarchical clustering if leidenalg is not installed.
    Returns int32 cluster labels.
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        return _cluster_hierarchical(corr, threshold=1.0 - resolution)

    G = corr_to_graph(corr, min_weight)
    partition = leidenalg.find_partition(
        G, leidenalg.CPMVertexPartition,
        weights="weight", resolution_parameter=resolution)
    return np.array(partition.membership, dtype=np.int32)


def corr_to_graph(corr, min_weight=0.1):
    """Build an igraph Graph from a correlation matrix.

    Only positive correlations above *min_weight* become edges.
    """
    import igraph as ig

    K = corr.shape[0]
    rows, cols = np.triu_indices(K, k=1)
    weights = corr[rows, cols].astype(np.float64)
    mask = weights > min_weight
    edges = list(zip(rows[mask].tolist(), cols[mask].tolist()))

    G = ig.Graph(n=K, edges=edges, directed=False)
    G.es["weight"] = weights[mask].tolist()
    return G


# ── Variance-based filtering ───────────────────────────────────────

def filter_by_variance(scores, edges, print_fn=print):
    """Drop edges whose score variance is below the auto-detected knee.

    Returns (filtered_scores, filtered_edges).
    """
    variances = np.var(scores, axis=1)
    threshold = _knee_point(np.sort(variances))
    mask = variances >= threshold
    print_fn(f"  Variance filter: kept {int(mask.sum())}/{len(mask)} "
             f"(threshold={threshold:.2e})")
    return scores[mask], [e for e, m in zip(edges, mask) if m]


def _knee_point(sorted_vals):
    """Knee/elbow via max perpendicular distance to the diagonal."""
    n = len(sorted_vals)
    if n < 3:
        return float(sorted_vals[0]) if n else 0.0
    xs = np.linspace(0, 1, n)
    span = sorted_vals[-1] - sorted_vals[0] + 1e-12
    ys = (sorted_vals - sorted_vals[0]) / span
    dists = np.abs(ys - xs) / np.sqrt(2)
    return float(sorted_vals[np.argmax(dists)])


# ── All-edges mode ──────────────────────────────────────────────────

def _correlate_all_edges(store, direction, print_fn):
    """Enumerate every edge, variance-filter, then correlate."""
    result = {}
    attn_scores, attn_edges = _all_edge_scores(store, "attn", print_fn)
    mlp_scores, mlp_edges = _all_edge_scores(store, "mlp", print_fn)

    if attn_scores is not None:
        attn_scores, attn_edges = filter_by_variance(
            attn_scores, attn_edges, print_fn)
        if len(attn_edges) > 1:
            corr = np.corrcoef(attn_scores)
            result["attn"] = {"edges": attn_edges, "corr": corr}
            print_fn(f"  Attn correlation matrix: {corr.shape}")

    if mlp_scores is not None:
        mlp_scores, mlp_edges = filter_by_variance(
            mlp_scores, mlp_edges, print_fn)
        if len(mlp_edges) > 1:
            corr = np.corrcoef(mlp_scores)
            result["mlp"] = {"edges": mlp_edges, "corr": corr}
            print_fn(f"  MLP correlation matrix: {corr.shape}")

    # Cross-graph
    if attn_scores is not None and mlp_scores is not None:
        attn_sids, attn_tids = store.get_attn_sids_tids()
        mlp_sids, mlp_tids = store.get_mlp_sids_tids()
        result = _add_cross_correlations(
            result, attn_scores, mlp_scores,
            attn_sids, attn_tids, mlp_sids, mlp_tids,
            attn_edges, mlp_edges, print_fn)

    return result


def _all_edge_scores(store, graph, print_fn):
    """Read score vectors for every (layer, head, comp) combination.

    Returns (scores [K, N], edge_dicts) or (None, None).
    """
    if graph == "attn":
        dset = store.get_attn_dataset()
        if dset is None:
            return None, None
        N, L, H, C = dset.shape
        print_fn(f"  Enumerating {L * H * C} attn edges ({N} tokens)...")
        scores = dset[:].reshape(N, -1).T.astype(np.float32)
        edges = [{"layer": l, "head": h, "comp": c, "value": 0.0}
                 for l in range(L) for h in range(H) for c in range(C)]
    elif graph == "mlp":
        dset = store.get_mlp_dataset()
        if dset is None:
            return None, None
        N, L, C = dset.shape
        print_fn(f"  Enumerating {L * C} mlp edges ({N} tokens)...")
        scores = dset[:].reshape(N, -1).T.astype(np.float32)
        edges = [{"layer": l, "comp": c, "value": 0.0}
                 for l in range(L) for c in range(C)]
    else:
        return None, None

    return scores, edges


def _add_cross_correlations(result, attn_scores, mlp_scores,
                            attn_sids, attn_tids, mlp_sids, mlp_tids,
                            attn_edges, mlp_edges, print_fn):
    """Compute cross-graph and combined correlation blocks."""
    attn_keys = {(int(attn_sids[i]), int(attn_tids[i])): i
                 for i in range(len(attn_sids))}
    mlp_keys = {(int(mlp_sids[i]), int(mlp_tids[i])): i
                for i in range(len(mlp_sids))}
    common = sorted(set(attn_keys) & set(mlp_keys))
    print_fn(f"  Cross-graph: {len(common)} common tokens")

    if len(common) > 100:
        ai = np.array([attn_keys[k] for k in common])
        mi = np.array([mlp_keys[k] for k in common])
        combined = np.vstack([attn_scores[:, ai], mlp_scores[:, mi]])
        full = np.corrcoef(combined)
        K_a = attn_scores.shape[0]
        result["cross"] = {"corr": full[:K_a, K_a:],
                           "attn_edges": attn_edges,
                           "mlp_edges": mlp_edges}
        result["combined"] = {"edges": attn_edges + mlp_edges,
                              "corr": full}
    return result


# ── Hierarchical fallback ──────────────────────────────────────────

def _cluster_hierarchical(corr, threshold=0.5):
    """Scipy agglomerative clustering (used when leidenalg missing)."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    dist = np.clip(1.0 - np.abs(corr), 0, 2)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    Z = linkage(squareform(dist, checks=False), method="average")
    return fcluster(Z, t=threshold, criterion="distance").astype(np.int32)


# ── H5 helpers ──────────────────────────────────────────────────────

def _write_graph(grp, graph, data, resolution, min_corr, print_fn):
    """Write one graph's correlation + clusters to an H5 group."""
    corr = data["corr"]
    grp.create_dataset(f"{graph}_corr", data=corr.astype(np.float16))
    _store_edge_list(grp, f"{graph}_edges", data["edges"], graph)
    print_fn(f"  Stored {graph}_corr {corr.shape}")

    if corr.shape[0] > 2:
        labels = cluster_leiden(corr, resolution, min_corr)
        grp.create_dataset(f"{graph}_clusters", data=labels)
        print_fn(f"  {graph.upper()} clusters: {len(set(labels))} "
                 f"(resolution={resolution})")


def _store_edge_list(grp, name, edges, graph):
    """Store edge list as structured numpy array."""
    if graph == "attn":
        dt = np.dtype([("layer", "i4"), ("head", "i4"),
                       ("comp", "i4"), ("value", "f4")])
        arr = np.array([(e["layer"], e["head"], e["comp"], e["value"])
                        for e in edges], dtype=dt)
    else:
        dt = np.dtype([("layer", "i4"), ("comp", "i4"), ("value", "f4")])
        arr = np.array([(e["layer"], e["comp"], e["value"])
                        for e in edges], dtype=dt)
    grp.create_dataset(name, data=arr)