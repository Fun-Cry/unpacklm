"""
IOI scaling: per-prompt token attribution + model performance across Pythia scales.
Runs all six config variants per model by default.

Usage:
    python -m experiments.ioi_scaling.run --model EleutherAI/pythia-160m-deduped
    python -m experiments.ioi_scaling.run --model gpt2 --configs kqv_aligned default

Outputs results/ioi_scaling/<tag>_<config>.json with per-prompt records.
"""
import argparse, json, os
import numpy as np
from tqdm import tqdm

import unpack
from unpack.core.flow import _run_flow_sweep
from experiments.circuits.ioi_utils import load_ioi_prompts


CONFIGS = ["default", "k_only_l2", "k_only_aligned", "kqv_weighted", "kqv_l2", "kqv_aligned"]


def run(model, device, cache_dir, configs, n_prompts, seed, out_dir):
    tracer = unpack.Tracer(model, device=device, cache_dir=cache_dir)
    tok = tracer.tokenizer
    prompts = load_ioi_prompts(tok, n_prompts=n_prompts, seed=seed)
    tag = model.split("/")[-1].replace("-deduped", "")
    os.makedirs(out_dir, exist_ok=True)
    print(f"{len(prompts)} prompts, model={tag}")

    for config in configs:
        records = []
        for p in tqdm(prompts, desc=f"{tag}/{config}"):
            prep, cfg = tracer.prepare(
                p["prompt"], target=p["target_token"],
                distractor=p["distractor_token"], config=config,
            )
            meta = p["metadata"]
            io_pos = meta["io_position"]
            s1_pos = meta["s1_position"]
            s2_pos = meta["s2_position"]
            end_pos = meta["end_position"]

            credit_pct, suppress_ratio, component_flow = _run_flow_sweep(
                prep["importance"], prep["attn_shares"], prep["attention_weights"],
                prep["key_decomp"], prep["mlp_principled"], prep["mlp_l2"],
                prep["component_layer"], prep["component_order"],
                prep["num_layers"], prep["num_heads"], prep["seq_len"],
                prep["t_pos"],
                query_decomp=prep["query_decomp"] if cfg.enable_q_side else None,
                value_decomp=prep["value_decomp"] if cfg.enable_v_side else None,
                branch_weights=cfg._branch_weights_dict,
                attn_shares_outproj=prep.get("attn_shares_outproj"),
                mlp_geva=prep.get("mlp_geva"),
                mlp_outproj=prep.get("mlp_outproj"),
            )

            records.append({
                "prompt": p["prompt"],
                "io": p["IO"], "s": p["S"],
                "io_pos": io_pos, "s1_pos": s1_pos, "s2_pos": s2_pos, "end_pos": end_pos,
                "target_prob": float(prep["target_prob"]),
                "logit_diff": float(prep["target_logit_centered"]),
                "credit_io": float(credit_pct[io_pos]),
                "credit_s1": float(credit_pct[s1_pos]),
                "credit_s2": float(credit_pct[s2_pos]),
                "credit_end": float(credit_pct[end_pos]),
                "credit_bos": float(credit_pct[0]),
                "credit_all": credit_pct.tolist(),
                "importance": prep["importance"],
                "suppress_ratio": float(suppress_ratio),
            })

        path = os.path.join(out_dir, f"{tag}_{config}.json")
        with open(path, "w") as f:
            json.dump({"model": model, "config": config, "n_prompts": len(records),
                        "seed": seed, "records": records}, f)
        print(f"Saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m-deduped")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default="/local/s4283341")
    ap.add_argument("--configs", nargs="*", default=CONFIGS)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/ioi_scaling")
    args = ap.parse_args()
    run(**vars(args))
