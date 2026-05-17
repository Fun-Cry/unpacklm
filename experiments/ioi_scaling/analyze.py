# %% Load results
import json, glob
import numpy as np

data = {}
for path in sorted(glob.glob("results/ioi_scaling/*.json")):
    with open(path) as f:
        d = json.load(f)
    tag = d["model"].split("/")[-1].replace("-deduped", "")
    key = f"{tag}/{d['config']}"
    data[key] = d
    print(f"{key}: {d['n_prompts']} prompts")

# %% Cross-scale summary
print(f"{'model/config':<35} {'logit_diff':>10} {'IO_prob':>8} {'IO_credit':>10} {'IO>S1':>6} {'IO>S2':>6} {'IO_top1':>8}")
for key, d in data.items():
    recs = d["records"]
    ld = np.mean([r["logit_diff"] for r in recs])
    prob = np.mean([r["target_prob"] for r in recs])
    cr_io = np.mean([r["credit_io"] for r in recs])
    io_gt_s1 = np.mean([r["credit_io"] > r["credit_s1"] for r in recs])
    io_gt_s2 = np.mean([r["credit_io"] > r["credit_s2"] for r in recs])
    io_top1 = np.mean([np.argmax(r["credit_all"]) == r["io_pos"] for r in recs])
    print(f"{key:<35} {ld:>+10.3f} {prob:>8.3f} {cr_io:>+10.1f}% {io_gt_s1:>6.0%} {io_gt_s2:>6.0%} {io_top1:>8.0%}")

# %% Within-model: credit_io vs logit_diff scatter (one per model, best config)
import matplotlib.pyplot as plt

# Group by model, pick best config by IO>S1
models = {}
for key, d in data.items():
    tag = d["model"].split("/")[-1].replace("-deduped", "")
    recs = d["records"]
    io_gt_s1 = np.mean([r["credit_io"] > r["credit_s1"] for r in recs])
    if tag not in models or io_gt_s1 > models[tag][1]:
        models[tag] = (key, io_gt_s1)

for tag, (key, _) in models.items():
    recs = data[key]["records"]
    ld = [r["logit_diff"] for r in recs]
    cr = [r["credit_io"] - r["credit_s1"] for r in recs]
    plt.figure(figsize=(5, 4))
    plt.scatter(ld, cr, alpha=0.5, s=20)
    plt.xlabel("Logit diff (IO - S)")
    plt.ylabel("Credit diff (IO% - S1%)")
    plt.title(f"{tag} ({data[key]['config']})")
    plt.axhline(0, color="gray", lw=0.5)
    plt.axvline(0, color="gray", lw=0.5)
    plt.tight_layout()
    plt.savefig(f"results/ioi_scaling/{tag}_scatter.png", dpi=150)
    plt.show()
    print(f"{tag}: corr = {np.corrcoef(ld, cr)[0,1]:.3f}")
