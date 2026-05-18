"""
Beta sensitivity analysis for token attribution + S2 three-way.

Preps 10 prompts once (4 forward passes each), then sweeps all beta values.
Usage: python -m experiments.ioi_s2.beta_sensitivity
"""
import numpy as np
import unpack
from unpack.config import get_config
from unpack.core.flow import _run_flow_sweep
from unpack.core.recursion import set_beta
from experiments.ioi_s2.prep import generate_pairs
from experiments.circuits.ioi_utils import resolve_positions

MODEL = "gpt2"
CONFIG = "kqv_aligned"
N = 10
BETAS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]

tracer = unpack.Tracer(MODEL, device="cuda:0", cache_dir=".")
cfg = get_config(CONFIG)
entries = generate_pairs(tracer.tokenizer, N, seed=42)


def sweep(prep, cfg):
    cr, supp, _ = _run_flow_sweep(
        prep["importance"], prep["attn_shares"], prep["attention_weights"],
        prep["key_decomp"], prep["mlp_principled"], prep["mlp_l2"],
        prep["component_layer"], prep["component_order"],
        prep["num_layers"], prep["num_heads"], prep["seq_len"],
        prep["t_pos"],
        query_decomp=prep["query_decomp"] if cfg.enable_q_side else None,
        value_decomp=prep["value_decomp"] if cfg.enable_v_side else None,
    )
    return cr, supp


# Prep all 4 conditions per prompt (GPU, done once)
print("Prepping...")
preps = []
for entry in entries:
    io = entry["io_pos"]
    s1 = entry["s1_pos"]
    s2 = entry["s2_pos"]
    try:
        p1, _ = tracer.prepare(entry["abc_text"], target=entry["c"], distractor=entry["a"], config=CONFIG)
        p2, _ = tracer.prepare(entry["ioi_text"], target=entry["s"], distractor=entry["io"], config=CONFIG)
        p3, _ = tracer.prepare(entry["abc_text"], target=entry["b"], distractor=entry["a"], config=CONFIG)
        p4, _ = tracer.prepare(entry["ioi_text"], target=entry["io"], distractor=entry["s"], config=CONFIG)
    except Exception as e:
        print(f"  skip: {e}")
        continue
    preps.append({"io": io, "s1": s1, "s2": s2, "p1": p1, "p2": p2, "p3": p3, "p4": p4})

print(f"{len(preps)} prompts prepped\n")

# Sweep all betas (numpy only)
print(f"{'beta':>6} | {'C→C':>6} {'S2→S':>6} {'C→B':>6} {'S1→S':>6} {'S1-S2':>6} | {'IO>S1':>6} {'IO>S2':>6} {'Top1':>5} {'MeanIO':>7}")
print("-" * 85)

for beta in BETAS:
    set_beta(beta)

    c1_vals, c2_vals, c3_vals, s1_vals = [], [], [], []
    io_gt_s1, io_gt_s2, top1_list, io_cr_list = [], [], [], []

    for entry in preps:
        io, s1, s2 = entry["io"], entry["s1"], entry["s2"]

        cr1, _ = sweep(entry["p1"], cfg)
        cr2, _ = sweep(entry["p2"], cfg)
        cr3, _ = sweep(entry["p3"], cfg)
        cr4, _ = sweep(entry["p4"], cfg)

        # S2 three-way
        c1_vals.append(cr1[s2])
        c2_vals.append(cr2[s2])
        c3_vals.append(cr3[s2])
        s1_vals.append(cr2[s1])

        # Token attribution (c4)
        io_gt_s1.append(cr4[io] > cr4[s1])
        io_gt_s2.append(cr4[io] > cr4[s2])
        top1_list.append(np.argmax(cr4) == io)
        io_cr_list.append(cr4[io])

    c1m = np.mean(c1_vals)
    c2m = np.mean(c2_vals)
    c3m = np.mean(c3_vals)
    s1m = np.mean(s1_vals)
    gap = s1m - c2m

    igs1 = np.mean(io_gt_s1)
    igs2 = np.mean(io_gt_s2)
    t1 = np.mean(top1_list)
    mio = np.mean(io_cr_list)

    print(f"{beta:>6.1f} | {c1m:>+6.1f} {c2m:>+6.1f} {c3m:>+6.1f} {s1m:>+6.1f} {gap:>+6.1f} | {igs1:>6.0%} {igs2:>6.0%} {t1:>5.0%} {mio:>+7.1f}%")

# Reset
set_beta(cfg.beta)
print(f"\n(default beta = {cfg.beta})")
