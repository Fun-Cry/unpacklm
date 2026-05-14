"""Top-k edge extraction and per-edge example collection.

TopKTracker: extract top-k edges by mean from global stats.
collect_examples: second pass over raw data to find tokens that
    most strongly activate each top-k edge.
"""

import numpy as np
import heapq
import json
from tqdm import tqdm


class TopKTracker:
    """Extract top-k edges from a global mean array.

    Returns structured numpy arrays with fields (layer, head, comp, value)
    for attention, or (layer, comp, value) for MLP.
    """

    @staticmethod
    def extract_attn(global_mean, valid_mask, k):
        L, H, C = global_mean.shape
        flat_valid = valid_mask.ravel()
        flat_mean = global_mean.ravel()
        valid_idx = np.where(flat_valid)[0]
        valid_vals = flat_mean[valid_idx]

        dtype = np.dtype([('layer', 'i4'), ('head', 'i4'), ('comp', 'i4'), ('value', 'f4')])
        actual_k = min(k, len(valid_idx))

        # Positive
        pos_order = np.argsort(-valid_vals)[:actual_k]
        pos = np.zeros(actual_k, dtype=dtype)
        for i, oi in enumerate(pos_order):
            flat_i = valid_idx[oi]
            l, h, c = np.unravel_index(flat_i, (L, H, C))
            pos[i] = (l, h, c, valid_vals[oi])

        # Negative
        neg_order = np.argsort(valid_vals)[:actual_k]
        neg = np.zeros(actual_k, dtype=dtype)
        for i, oi in enumerate(neg_order):
            flat_i = valid_idx[oi]
            l, h, c = np.unravel_index(flat_i, (L, H, C))
            neg[i] = (l, h, c, valid_vals[oi])

        return pos, neg

    @staticmethod
    def extract_mlp(global_mean, valid_mask, k):
        L, C = global_mean.shape
        flat_valid = valid_mask.ravel()
        flat_mean = global_mean.ravel()
        valid_idx = np.where(flat_valid)[0]
        valid_vals = flat_mean[valid_idx]

        dtype = np.dtype([('layer', 'i4'), ('comp', 'i4'), ('value', 'f4')])
        actual_k = min(k, len(valid_idx))

        pos_order = np.argsort(-valid_vals)[:actual_k]
        pos = np.zeros(actual_k, dtype=dtype)
        for i, oi in enumerate(pos_order):
            flat_i = valid_idx[oi]
            l, c = np.unravel_index(flat_i, (L, C))
            pos[i] = (l, c, valid_vals[oi])

        neg_order = np.argsort(valid_vals)[:actual_k]
        neg = np.zeros(actual_k, dtype=dtype)
        for i, oi in enumerate(neg_order):
            flat_i = valid_idx[oi]
            l, c = np.unravel_index(flat_i, (L, C))
            neg[i] = (l, c, valid_vals[oi])

        return pos, neg


def collect_attn_examples(attn_dset, attn_sids, attn_tids, attn_pos, attn_neg,
                          examples_per_edge, chunk_size, min_tokens, print_fn):
    """Second pass: find top example tokens for each top-k attention edge.

    min_tokens filters by token position — examples at position < min_tokens
    are skipped since the model has too little context there.
    """
    N = attn_dset.shape[0]
    sids_flat = attn_sids[:].ravel()
    tids_flat = attn_tids[:].ravel()

    edges = []
    for arr in [attn_pos, attn_neg]:
        for row in arr:
            edges.append((int(row['layer']), int(row['head']), int(row['comp'])))
    edges = list(set(edges))

    edge_heaps = {e: [] for e in edges}
    edge_layers = np.array([e[0] for e in edges])
    edge_heads = np.array([e[1] for e in edges])
    edge_comps = np.array([e[2] for e in edges])
    n_edges = len(edges)

    print_fn(f"\nFinding top-{examples_per_edge} examples for {len(attn_pos)} attention edges…")
    if min_tokens > 0:
        print_fn(f"  Skipping token positions < {min_tokens}")

    pbar = tqdm(range(0, N, chunk_size), desc="Attn examples", unit="chunk",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for chunk_start in pbar:
        chunk_end = min(chunk_start + chunk_size, N)
        data_chunk = attn_dset[chunk_start:chunk_end]
        sids_chunk = sids_flat[chunk_start:chunk_end]
        tids_chunk = tids_flat[chunk_start:chunk_end]

        # Filter out early token positions (too little context)
        if min_tokens > 0:
            pos_mask = tids_chunk >= min_tokens
            if not pos_mask.any():
                continue
            data_chunk = data_chunk[pos_mask]
            sids_chunk = sids_chunk[pos_mask]
            tids_chunk = tids_chunk[pos_mask]

        chunk_len = data_chunk.shape[0]
        all_vals = data_chunk[:, edge_layers, edge_heads, edge_comps]
        all_abs = np.abs(all_vals)

        for ei in range(n_edges):
            vals = all_vals[:, ei]
            abs_v = all_abs[:, ei]
            heap = edge_heaps[edges[ei]]

            k = min(examples_per_edge, chunk_len)
            if chunk_len > k:
                top_in_chunk = np.argpartition(-abs_v, k)[:k]
            else:
                top_in_chunk = np.arange(chunk_len)

            for idx in top_in_chunk:
                v = float(vals[idx])
                av = float(abs_v[idx])
                entry = (av, v, int(sids_chunk[idx]), int(tids_chunk[idx]))
                if len(heap) < examples_per_edge:
                    heapq.heappush(heap, entry)
                elif av > heap[0][0]:
                    heapq.heapreplace(heap, entry)

    def to_json(structured_arr):
        result = []
        for row in structured_arr:
            l, h, c = int(row['layer']), int(row['head']), int(row['comp'])
            heap = edge_heaps.get((l, h, c), [])
            examples = sorted(heap, key=lambda x: -x[0])
            result.append({
                "layer": l, "head": h, "comp": c,
                "value": float(row['value']),
                "examples": [{"score": e[1], "sid": e[2], "tid": e[3]} for e in examples],
            })
        return result

    print_fn(f"  Stored examples for {len(edges)} unique attention edges")
    return json.dumps(to_json(attn_pos)), json.dumps(to_json(attn_neg))


def collect_mlp_examples(mlp_dset, mlp_sids, mlp_tids, mlp_pos, mlp_neg,
                         examples_per_edge, chunk_size, print_fn):
    """Second pass: find top example tokens for each top-k MLP edge.

    No position filter — MLP uses fixed-size weight matrices
    regardless of token position (unlike causal attention).
    """
    N = mlp_dset.shape[0]
    sids_flat = mlp_sids[:].ravel()
    tids_flat = mlp_tids[:].ravel()

    edges = []
    for arr in [mlp_pos, mlp_neg]:
        for row in arr:
            edges.append((int(row['layer']), int(row['comp'])))
    edges = list(set(edges))

    edge_heaps = {e: [] for e in edges}
    edge_layers = np.array([e[0] for e in edges])
    edge_comps = np.array([e[1] for e in edges])
    n_edges = len(edges)

    print_fn(f"\nFinding top-{examples_per_edge} examples for {len(mlp_pos)} MLP edges…")

    pbar = tqdm(range(0, N, chunk_size), desc="MLP examples", unit="chunk",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for chunk_start in pbar:
        chunk_end = min(chunk_start + chunk_size, N)
        data_chunk = mlp_dset[chunk_start:chunk_end]
        sids_chunk = sids_flat[chunk_start:chunk_end]
        tids_chunk = tids_flat[chunk_start:chunk_end]
        chunk_len = data_chunk.shape[0]

        all_vals = data_chunk[:, edge_layers, edge_comps]
        all_abs = np.abs(all_vals)

        for ei in range(n_edges):
            vals = all_vals[:, ei]
            abs_v = all_abs[:, ei]
            heap = edge_heaps[edges[ei]]

            k = min(examples_per_edge, chunk_len)
            if chunk_len > k:
                top_in_chunk = np.argpartition(-abs_v, k)[:k]
            else:
                top_in_chunk = np.arange(chunk_len)

            for idx in top_in_chunk:
                v = float(vals[idx])
                av = float(abs_v[idx])
                entry = (av, v, int(sids_chunk[idx]), int(tids_chunk[idx]))
                if len(heap) < examples_per_edge:
                    heapq.heappush(heap, entry)
                elif av > heap[0][0]:
                    heapq.heapreplace(heap, entry)

    def to_json(structured_arr):
        result = []
        for row in structured_arr:
            l, c = int(row['layer']), int(row['comp'])
            heap = edge_heaps.get((l, c), [])
            examples = sorted(heap, key=lambda x: -x[0])
            result.append({
                "layer": l, "comp": c,
                "value": float(row['value']),
                "examples": [{"score": e[1], "sid": e[2], "tid": e[3]} for e in examples],
            })
        return result

    print_fn(f"  Stored examples for {len(edges)} unique MLP edges")
    return json.dumps(to_json(mlp_pos)), json.dumps(to_json(mlp_neg))