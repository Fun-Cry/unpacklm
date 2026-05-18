# analyze.py
# Usage: python -m experiments.ioi_s2.analyze [results_dir] [model_filter]
import json, glob, os, sys
import numpy as np

results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/ioi_s2"
model_filter = sys.argv[2] if len(sys.argv) > 2 else None

data = {}
for path in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
    name = os.path.basename(path).replace(".jsonl", "")
    tag, config = name.split("_", 1)
    if model_filter and model_filter not in tag:
        continue
    with open(path) as f:
        recs = [json.loads(line) for line in f]
    key = f"{tag}/{config}"
    data[key] = {"model": tag, "config": config, "n": len(recs), "records": recs}
    print(f"{key}: {len(recs)} prompts")

# %% Three-way summary per config
print(f"\n{'config':<25} {'C→C':>7} {'S2→S':>7} {'C→B':>7} {'S1→S':>7} {'S2>0%':>7} {'n':>4}")
for key, d in data.items():
    recs = d["records"]
    if not recs:
        continue
    c1 = np.mean([r["c1_cr_s2"] for r in recs])
    c2 = np.mean([r["c2_cr_s2"] for r in recs])
    c3 = np.mean([r["c3_cr_s2"] for r in recs])
    s1 = np.mean([r["c2_cr_s1"] for r in recs])
    s2_pos_rate = np.mean([r["c2_cr_s2"] > 0 for r in recs])
    print(f"{d['config']:<25} {c1:>+7.1f} {c2:>+7.1f} {c3:>+7.1f} {s1:>+7.1f} {s2_pos_rate:>7.0%} {d['n']:>4}")

# %% Detailed stats for each config
for key, d in data.items():
    recs = d["records"]
    if len(recs) < 5:
        continue
    c1 = np.array([r["c1_cr_s2"] for r in recs])
    c2 = np.array([r["c2_cr_s2"] for r in recs])
    c3 = np.array([r["c3_cr_s2"] for r in recs])
    s1 = np.array([r["c2_cr_s1"] for r in recs])
    io = np.array([r["c2_cr_io"] for r in recs])
    p1 = np.array([r["c1_prob"] for r in recs])
    p2 = np.array([r["c2_prob"] for r in recs])
    p3 = np.array([r["c3_prob"] for r in recs])

    print(f"\n=== {d['model']} / {d['config']} (n={len(recs)}) ===")
    print(f"  {'':>25} {'C→C':>10} {'S2→S':>10} {'C→B':>10}")
    print(f"  {'':>25} {'(copy)':>10} {'(copy+IOI)':>10} {'(no copy)':>10}")
    print(f"  {'mean credit at S2/C':<25} {c1.mean():>+10.1f} {c2.mean():>+10.1f} {c3.mean():>+10.1f}")
    print(f"  {'std':<25} {c1.std():>10.1f} {c2.std():>10.1f} {c3.std():>10.1f}")
    print(f"  {'> 0 rate':<25} {np.mean(c1>0):>10.0%} {np.mean(c2>0):>10.0%} {np.mean(c3>0):>10.0%}")
    print(f"  {'target prob':<25} {p1.mean():>10.3f} {p2.mean():>10.3f} {p3.mean():>10.3f}")

    print(f"\n  IOI (target=S) positions:")
    print(f"    cr(S1) = {s1.mean():+.1f} (std={s1.std():.1f})")
    print(f"    cr(S2) = {c2.mean():+.1f} (std={c2.std():.1f})")
    print(f"    cr(IO) = {io.mean():+.1f} (std={io.std():.1f})")
    print(f"    S1-S2 gap = {(s1-c2).mean():+.1f}")

    # Correlations
    r1 = np.corrcoef(p2, c2)[0, 1]
    r2 = np.corrcoef(p2, s1)[0, 1]
    print(f"\n  corr(P(S), cr(S2)) = {r1:+.3f}")
    print(f"  corr(P(S), cr(S1)) = {r2:+.3f}")

# %% Token attribution table (from c4: IOI target=IO)
has_c4 = any("c4_cr_io" in r for d in data.values() for r in d["records"][:1])
if has_c4:
    print(f"\n=== Token attribution table (IOI target=IO) ===")
    print(f"{'config':<25} {'IO>S1':>6} {'IO>S2':>6} {'Top-1':>6} {'Top-1*':>7} {'Mean IO':>8} {'IO prob':>8}")
    for key, d in data.items():
        recs = [r for r in d["records"] if "c4_cr_io" in r]
        if not recs:
            continue
        cr_io = np.array([r["c4_cr_io"] for r in recs])
        cr_s1 = np.array([r["c4_cr_s1"] for r in recs])
        cr_s2 = np.array([r["c4_cr_s2"] for r in recs])
        cr_all = [np.array(r["c4_credit_all"]) for r in recs]
        prob = np.array([r["c4_prob"] for r in recs])

        io_gt_s1 = np.mean(cr_io > cr_s1)
        io_gt_s2 = np.mean(cr_io > cr_s2)
        io_pos_arr = np.array([r["io_pos"] for r in recs])
        top1 = np.mean([np.argmax(c) == p for c, p in zip(cr_all, io_pos_arr)])
        # Top-1*: IO highest among name positions only
        top1_star = np.mean([
            cr_io[i] > cr_s1[i] and cr_io[i] > cr_s2[i]
            for i in range(len(recs))
        ])
        mean_io = cr_io.mean()
        mean_prob = prob.mean()
        print(f"{d['config']:<25} {io_gt_s1:>6.0%} {io_gt_s2:>6.0%} {top1:>6.0%} {top1_star:>7.0%} {mean_io:>+8.1f}% {mean_prob:>8.3f}")

# %% Cross-model comparison (best config per model)
models = {}
for key, d in data.items():
    tag = d["model"]
    recs = d["records"]
    if not recs:
        continue
    s2_mean = np.mean([r["c2_cr_s2"] for r in recs])
    if tag not in models or s2_mean < models[tag][1]:
        models[tag] = (key, s2_mean)

if len(models) > 1:
    print(f"\n=== Cross-model comparison (best config per model) ===")
    print(f"{'model':<20} {'config':<20} {'C→C':>7} {'S2→S':>7} {'C→B':>7} {'S1→S':>7} {'S1-S2':>7}")
    for tag, (key, _) in sorted(models.items()):
        recs = data[key]["records"]
        c1 = np.mean([r["c1_cr_s2"] for r in recs])
        c2 = np.mean([r["c2_cr_s2"] for r in recs])
        c3 = np.mean([r["c3_cr_s2"] for r in recs])
        s1 = np.mean([r["c2_cr_s1"] for r in recs])
        gap = s1 - c2
        print(f"{tag:<20} {data[key]['config']:<20} {c1:>+7.1f} {c2:>+7.1f} {c3:>+7.1f} {s1:>+7.1f} {gap:>+7.1f}")

# %% Scatter plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for tag, (key, _) in models.items():
        recs = data[key]["records"]
        if len(recs) < 5:
            continue
        c1 = [r["c1_cr_s2"] for r in recs]
        c2 = [r["c2_cr_s2"] for r in recs]
        c3 = [r["c3_cr_s2"] for r in recs]
        s1 = [r["c2_cr_s1"] for r in recs]
        p2 = [r["c2_prob"] for r in recs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        for j, (vals, label, color) in enumerate([
            (c1, "C→C (copy)", "#1D9E75"),
            (c2, "S2→S (copy+IOI)", "#D85A30"),
            (c3, "C→B (no copy)", "#534AB7"),
        ]):
            x = np.random.normal(j, 0.08, len(vals))
            ax1.scatter(x, vals, alpha=0.5, s=20, c=color, label=label)
            ax1.hlines(np.mean(vals), j-0.25, j+0.25, color=color, linewidth=2)
        ax1.axhline(0, color="gray", lw=0.5)
        ax1.set_xticks([0,1,2])
        ax1.set_xticklabels(["C→C", "S2→S", "C→B"])
        ax1.set_ylabel("Credit at S2/C (%)")
        ax1.legend(fontsize=7)

        ax2.scatter(p2, s1, alpha=0.5, s=20, c="#378ADD", label="cr(S1)")
        ax2.scatter(p2, c2, alpha=0.5, s=20, c="#D85A30", label="cr(S2)")
        ax2.axhline(0, color="gray", lw=0.5)
        ax2.set_xlabel("P(S)")
        ax2.set_ylabel("Credit (%)")
        ax2.legend(fontsize=8)

        fig.suptitle(f"{tag} ({data[key]['config']})")
        fig.tight_layout()
        out = os.path.join(results_dir, f"{tag}_s2.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")

except ImportError:
    print("matplotlib not available, skipping plots")