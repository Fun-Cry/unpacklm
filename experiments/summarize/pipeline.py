"""Compute summary statistics from raw tracing data.

Reads raw HDF5 (head preferences + MLP norms) and produces a summary HDF5
with global mean/std, per-sentence mean/std, deviation scores, top-k edges,
and per-edge example tokens.

Usage:
    python -m analysis summarize --name exp1 --cache_dir /data --top_k 1000 --min_tokens 20
"""

import argparse
import numpy as np
import h5py
from tqdm import tqdm

from .masks import build_valid_mask_attn, build_valid_mask_mlp
from .accumulators import GlobalAccumulator, PerSentenceAccumulator
from .top_k import TopKTracker, collect_attn_examples, collect_mlp_examples


def load_component_names(h5f, model_id, dataset_path, attr_name="component_names_json"):
    prefix = f"model_{model_id}"
    dset = h5f.get(f"{prefix}/{dataset_path}")
    if dset is None:
        return []
    raw = dset.attrs.get(attr_name, "[]")
    if isinstance(raw, str):
        try:
            return eval(raw)
        except Exception:
            return []
    return list(raw)


def run(db_path, h5_path, output_path, model_id=1, top_k=200,
        chunk_size=1000, min_tokens=0, examples_per_edge=200):
    """Main pipeline entry point."""
    import builtins
    _print = builtins.print
    def print(*args, **kwargs):
        kwargs.setdefault('flush', True)
        _print(*args, **kwargs)

    print(f"Opening raw HDF5: {h5_path}")
    h5f = h5py.File(h5_path, 'r')
    prefix = f"model_{model_id}"

    # ── Load metadata ──
    attn_dset = h5f.get(f"{prefix}/head_preferences")
    has_attn = attn_dset is not None
    if has_attn:
        N_attn, L_attn, H_attn, C_attn = attn_dset.shape
        attn_comp_names = load_component_names(h5f, model_id, "head_preferences")
        attn_sids = h5f[f"{prefix}/head_preferences_sentence_ids"]
        attn_tids = h5f[f"{prefix}/head_preferences_token_indices"]
        print(f"Attention data: {N_attn} tokens, {L_attn} layers, {H_attn} heads, {C_attn} components")
    else:
        print("No attention data found, skipping.")

    mlp_dset = h5f.get(f"{prefix}/mlp_intermediate/norms")
    has_mlp = mlp_dset is not None
    if has_mlp:
        N_mlp, L_mlp, C_mlp = mlp_dset.shape
        mlp_comp_names = load_component_names(h5f, model_id, "mlp_intermediate/norms")
        mlp_sids = h5f[f"{prefix}/mlp_intermediate/sentence_ids"]
        mlp_tids = h5f[f"{prefix}/mlp_intermediate/token_indices"]
        print(f"MLP data: {N_mlp} tokens, {L_mlp} layers, {C_mlp} components")
    else:
        print("No MLP data found, skipping.")

    if not has_attn and not has_mlp:
        print("Nothing to process.")
        h5f.close()
        return

    # ── Architectural masks ──
    if has_attn:
        attn_valid = build_valid_mask_attn(L_attn, H_attn, attn_comp_names)
        print(f"Attention valid edges: {attn_valid.sum()} / {attn_valid.size}")
    if has_mlp:
        mlp_valid = build_valid_mask_mlp(L_mlp, mlp_comp_names)
        print(f"MLP valid edges: {mlp_valid.sum()} / {mlp_valid.size}")

    # ── Open output ──
    print(f"Writing summary to: {output_path}")
    out = h5py.File(output_path, 'w')
    out_prefix = f"model_{model_id}"

    meta_grp = out.require_group(f"{out_prefix}/meta")
    if has_attn:
        meta_grp.create_dataset("valid_mask_attn", data=attn_valid, compression="gzip")
        meta_grp.attrs["attn_component_names"] = str(attn_comp_names)
    if has_mlp:
        meta_grp.create_dataset("valid_mask_mlp", data=mlp_valid, compression="gzip")
        meta_grp.attrs["mlp_component_names"] = str(mlp_comp_names)

    per_sent_grp = out.require_group(f"{out_prefix}/per_sentence")

    # ── Initialize accumulators ──
    if has_attn:
        global_attn = GlobalAccumulator((L_attn, H_attn, C_attn), attn_valid)
        sent_attn = PerSentenceAccumulator(
            (L_attn, H_attn, C_attn), attn_valid, per_sent_grp, prefix_attn=True)
    if has_mlp:
        global_mlp = GlobalAccumulator((L_mlp, C_mlp), mlp_valid)
        sent_mlp = PerSentenceAccumulator(
            (L_mlp, C_mlp), mlp_valid, per_sent_grp, prefix_attn=False)

    # ── Pass 1: attention ──
    if has_attn:
        print("\nProcessing attention edges…")
        attn_sids_flat = attn_sids[:].ravel()

        pbar = tqdm(range(0, N_attn, chunk_size), desc="Attn", unit="chunk",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
        for chunk_start in pbar:
            chunk_end = min(chunk_start + chunk_size, N_attn)
            data_chunk = attn_dset[chunk_start:chunk_end]
            sids_chunk = attn_sids_flat[chunk_start:chunk_end]

            global_attn.update_chunk(data_chunk)
            sent_attn.update_chunk(data_chunk, sids_chunk)
            sent_attn.flush_stale()

            pbar.set_postfix_str(f"{chunk_end}/{N_attn} tok, {sent_attn.flushed} flushed, {len(sent_attn._data)} in mem")

        sent_attn.flush_all()
        print(f"  Attention done: {sent_attn.flushed} sentences")
        del sent_attn

    # ── Pass 1: MLP ──
    if has_mlp:
        print("\nProcessing MLP edges…")
        mlp_sids_flat = mlp_sids[:].ravel()

        pbar = tqdm(range(0, N_mlp, chunk_size), desc="MLP", unit="chunk",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
        for chunk_start in pbar:
            chunk_end = min(chunk_start + chunk_size, N_mlp)
            data_chunk = mlp_dset[chunk_start:chunk_end]
            sids_chunk = mlp_sids_flat[chunk_start:chunk_end]

            global_mlp.update_chunk(data_chunk)
            sent_mlp.update_chunk(data_chunk, sids_chunk)
            sent_mlp.flush_stale()

            pbar.set_postfix_str(f"{chunk_end}/{N_mlp} tok, {sent_mlp.flushed} flushed, {len(sent_mlp._data)} in mem")

        sent_mlp.flush_all()
        print(f"  MLP done: {sent_mlp.flushed} sentences")
        del sent_mlp

    # ── Finalize global stats + top-k ──
    global_grp = out.require_group(f"{out_prefix}/global")
    attn_pos = attn_neg = mlp_pos = mlp_neg = None

    if has_attn:
        print("\nFinalizing global attention stats...")
        g_mean, g_std, g_count = global_attn.finalize()
        global_grp.create_dataset("attn_edge_mean", data=g_mean, compression="gzip")
        global_grp.create_dataset("attn_edge_std", data=g_std, compression="gzip")
        global_grp.create_dataset("attn_edge_count", data=g_count, compression="gzip")

        print(f"  Extracting top-{top_k} attention edges...")
        attn_pos, attn_neg = TopKTracker.extract_attn(g_mean, attn_valid, top_k)
        global_grp.create_dataset("top_k_attn_pos", data=attn_pos)
        global_grp.create_dataset("top_k_attn_neg", data=attn_neg)
        global_grp.attrs["top_k"] = top_k

    if has_mlp:
        print("Finalizing global MLP stats...")
        g_mean, g_std, g_count = global_mlp.finalize()
        global_grp.create_dataset("mlp_edge_mean", data=g_mean, compression="gzip")
        global_grp.create_dataset("mlp_edge_std", data=g_std, compression="gzip")
        global_grp.create_dataset("mlp_edge_count", data=g_count, compression="gzip")

        print(f"  Extracting top-{top_k} MLP edges...")
        mlp_pos, mlp_neg = TopKTracker.extract_mlp(g_mean, mlp_valid, top_k)
        global_grp.create_dataset("top_k_mlp_pos", data=mlp_pos)
        global_grp.create_dataset("top_k_mlp_neg", data=mlp_neg)
        global_grp.attrs["top_k"] = top_k

    # ── Deviation scores ──
    if has_attn:
        attn_g_mean = np.array(global_grp["attn_edge_mean"])
        n_valid = attn_valid.sum()
        if n_valid > 0:
            print("\nComputing per-sentence deviation scores...")
            ps = out[f"{out_prefix}/per_sentence"]
            score_sids = []
            score_vals = []
            for sid_str in tqdm(sorted(ps.keys(), key=int), desc="Scores", unit="sent"):
                grp = ps[sid_str]
                if "attn_edge_mean" in grp:
                    sent_mean = np.array(grp["attn_edge_mean"])
                    score = float(np.abs(sent_mean - attn_g_mean)[attn_valid].sum() / n_valid)
                    score_sids.append(int(sid_str))
                    score_vals.append(score)
            global_grp.create_dataset("sentence_score_sids",
                                      data=np.array(score_sids, dtype=np.int64))
            global_grp.create_dataset("sentence_score_vals",
                                      data=np.array(score_vals, dtype=np.float32))
            print(f"  Scored {len(score_sids)} sentences")

    # ── Pass 2: examples ──
    if has_attn and attn_pos is not None and len(attn_pos) > 0:
        pos_json, neg_json = collect_attn_examples(
            attn_dset, attn_sids, attn_tids, attn_pos, attn_neg,
            examples_per_edge, chunk_size, min_tokens, print)
        global_grp.attrs["top_k_attn_pos_examples"] = pos_json
        global_grp.attrs["top_k_attn_neg_examples"] = neg_json

    if has_mlp and mlp_pos is not None and len(mlp_pos) > 0:
        pos_json, neg_json = collect_mlp_examples(
            mlp_dset, mlp_sids, mlp_tids, mlp_pos, mlp_neg,
            examples_per_edge, chunk_size, print)
        global_grp.attrs["top_k_mlp_pos_examples"] = pos_json
        global_grp.attrs["top_k_mlp_neg_examples"] = neg_json

    # ── Cleanup ──
    out.flush()
    out.close()
    h5f.close()
    print(f"\nDone. Summary written to {output_path}")


# ================================================================
#  CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute summary statistics from raw tracing data"
    )
    parser.add_argument("--name", type=str, required=True,
                        help="Base name for files ({name}.db, {name}.h5, {name}_summary.h5)")
    parser.add_argument("--cache_dir", type=str, default=".",
                        help="Directory containing data files")
    parser.add_argument("--model_id", type=int, default=1, help="Model ID in DB")
    parser.add_argument("--top_k", type=int, default=1000, help="Number of top edges to track")
    parser.add_argument("--min_tokens", type=int, default=20,
                        help="Min sentence length — filter short sentences from examples")
    parser.add_argument("--examples_per_edge", type=int, default=200,
                        help="Number of example tokens stored per ranked edge")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Rows to read at once from HDF5")

    args = parser.parse_args()
    import os
    base = os.path.join(args.cache_dir, args.name)

    run(db_path=base + ".db",
        h5_path=base + ".h5",
        output_path=base + "_summary.h5",
        model_id=args.model_id, top_k=args.top_k,
        chunk_size=args.chunk_size, min_tokens=args.min_tokens,
        examples_per_edge=args.examples_per_edge)


if __name__ == "__main__":
    main()