"""
Three-way S2/C comparison: copying vs IOI suppression at duplicate position.

Processes one prompt at a time to keep memory bounded:
  1. Prep 3 conditions (3 forward passes)
  2. Sweep all configs (numpy only)
  3. Write record, release preps

Usage:
    python -m experiments.ioi_s2.run gpt2
    python -m experiments.ioi_s2.run pythia-160m
    python -m experiments.ioi_s2.run pythia-160m --configs kqv_weighted kqv_aligned
"""
import argparse, json, os
import numpy as np
from tqdm import tqdm

import unpack
from unpack.config import get_config
from unpack.core.flow import _run_flow_sweep
from experiments.ioi_s2.config import MODELS, ALL_CONFIGS, DEFAULT_N_PROMPTS, DEFAULT_SEED, DEFAULT_OUT_DIR
from experiments.ioi_s2.prep import generate_pairs


def sweep(prep, cfg):
    cr, supp, _ = _run_flow_sweep(
        prep["importance"], prep["attn_shares"], prep["attention_weights"],
        prep["key_decomp"], prep["mlp_principled"], prep["mlp_l2"],
        prep["component_layer"], prep["component_order"],
        prep["num_layers"], prep["num_heads"], prep["seq_len"],
        prep["t_pos"],
        query_decomp=prep["query_decomp"] if cfg.enable_q_side else None,
        value_decomp=prep["value_decomp"] if cfg.enable_v_side else None,
        branch_weights=cfg._branch_weights_dict,
        attn_shares_outproj=prep.get("attn_shares_outproj") if cfg.aligned else None,
        mlp_geva=prep.get("mlp_geva") if cfg.mlp_rule == "weighted" else None,
        mlp_outproj=prep.get("mlp_outproj") if cfg.aligned else None,
    )
    return cr, supp


def cond_fields(prep, cr, supp, prefix, io, s1, s2):
    """Extract all storable fields from one condition."""
    return {
        f"{prefix}_prob": float(prep["target_prob"]),
        f"{prefix}_logit_centered": float(prep["target_logit_centered"]),
        f"{prefix}_predictions": prep["predictions"][:5],
        f"{prefix}_suppress": float(supp),
        f"{prefix}_credit_all": [round(float(x), 2) for x in cr],
        f"{prefix}_cr_io": float(cr[io]),
        f"{prefix}_cr_s1": float(cr[s1]),
        f"{prefix}_cr_s2": float(cr[s2]),
        f"{prefix}_cr_end": float(cr[-1]),
        f"{prefix}_cr_bos": float(cr[0]),
    }


def run(model_name, device, cache_dir, configs, n_prompts, seed, out_dir):
    tag = model_name.split("/")[-1].replace("-deduped", "")
    os.makedirs(out_dir, exist_ok=True)

    tracer = unpack.Tracer(model_name, device=device, cache_dir=cache_dir)
    entries = generate_pairs(tracer.tokenizer, n_prompts, seed)
    cfgs = [(c, get_config(c)) for c in configs]

    # Open one file handle per config, count existing lines for resume
    handles = {}
    done_counts = {}
    for config, _ in cfgs:
        path = os.path.join(out_dir, f"{tag}_{config}.jsonl")
        done = 0
        if os.path.exists(path):
            with open(path) as f:
                done = sum(1 for _ in f)
        done_counts[config] = done
        if done >= len(entries):
            print(f"  {config}: already done ({done} lines)")
        else:
            handles[config] = open(path, "a")
            if done > 0:
                print(f"  {config}: resuming from {done}")

    if not handles:
        print("All configs done.")
        return

    # Find the minimum done count across active configs — skip that many prompts
    min_done = min(done_counts[c] for c in handles)

    # Process one prompt at a time
    for idx, entry in enumerate(tqdm(entries, desc=f"{tag}")):
        if idx < min_done:
            continue

        io = entry["io_pos"]
        s1 = entry["s1_pos"]
        s2 = entry["s2_pos"]

        # 4 forward passes (prep once with kqv_aligned superset)
        try:
            prep_c1, _ = tracer.prepare(entry["abc_text"], target=entry["c"], distractor=entry["a"], config="kqv_aligned")
            prep_c2, _ = tracer.prepare(entry["ioi_text"], target=entry["s"], distractor=entry["io"], config="kqv_aligned")
            prep_c3, _ = tracer.prepare(entry["abc_text"], target=entry["b"], distractor=entry["a"], config="kqv_aligned")
            prep_c4, _ = tracer.prepare(entry["ioi_text"], target=entry["io"], distractor=entry["s"], config="kqv_aligned")
        except Exception as e:
            print(f"  [{idx}] prep failed: {e}")
            # Write empty markers so line counts stay in sync
            for config, fh in handles.items():
                fh.write(json.dumps({"_skip": True, "error": str(e)}) + "\n")
                fh.flush()
            continue

        # Sweep all configs (numpy only, cheap)
        for config, cfg in cfgs:
            if config not in handles:
                continue
            if idx < done_counts[config]:
                continue

            cr_c1, supp_c1 = sweep(prep_c1, cfg)
            cr_c2, supp_c2 = sweep(prep_c2, cfg)
            cr_c3, supp_c3 = sweep(prep_c3, cfg)
            cr_c4, supp_c4 = sweep(prep_c4, cfg)

            rec = {
                "ioi_text": entry["ioi_text"],
                "abc_text": entry["abc_text"],
                "template_type": entry["template_type"],
                "io": entry["io"], "s": entry["s"],
                "a": entry["a"], "b": entry["b"], "c": entry["c"],
                "io_pos": io, "s1_pos": s1, "s2_pos": s2,
            }
            rec.update(cond_fields(prep_c1, cr_c1, supp_c1, "c1", io, s1, s2))
            rec.update(cond_fields(prep_c2, cr_c2, supp_c2, "c2", io, s1, s2))
            rec.update(cond_fields(prep_c3, cr_c3, supp_c3, "c3", io, s1, s2))
            rec.update(cond_fields(prep_c4, cr_c4, supp_c4, "c4", io, s1, s2))

            handles[config].write(json.dumps(rec) + "\n")
            handles[config].flush()

        # Release preps
        del prep_c1, prep_c2, prep_c3, prep_c4

    for fh in handles.values():
        fh.close()
    print("Done.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", nargs="?", help="Model preset: gpt2, pythia-160m, etc.")
    ap.add_argument("--model", help="Override: full model name")
    ap.add_argument("--device", help="Override: cuda device")
    ap.add_argument("--cache-dir", help="Override: HF cache dir")
    ap.add_argument("--configs", nargs="*", help="Override: config names")
    ap.add_argument("--n-prompts", type=int, default=DEFAULT_N_PROMPTS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    if args.preset and args.preset in MODELS:
        preset = MODELS[args.preset]
        model_name = args.model or preset["name"]
        device = args.device or preset["device"]
        cache_dir = args.cache_dir or preset["cache_dir"]
        configs = args.configs or preset["configs"]
    elif args.model:
        model_name = args.model
        device = args.device or "cuda:0"
        cache_dir = args.cache_dir or "/local/s4283341"
        configs = args.configs or ALL_CONFIGS
    else:
        ap.error("Provide a preset (gpt2, pythia-160m, ...) or --model")

    run(model_name, device, cache_dir, configs, args.n_prompts, args.seed, args.out_dir)


if __name__ == "__main__":
    main()