"""
Plot knockout results: strength vs Δppl scatter, colored by layer.
Usage: python plot_knockout.py /data/s4283341/160m_143k_knockout.json
"""

import json, sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "160m_143k_knockout.json"
with open(path) as f:
    data = json.load(f)

by_layer = defaultdict(list)
for k in data["knockouts"]:
    by_layer[k["source_layer"]].append(k)

layers = sorted(by_layer.keys())
cmap = plt.cm.viridis(np.linspace(0, 1, len(layers)))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

configs = [
    ("attn_strength", "delta_ppl_cut_attn", "Attention Strength", r"$\Delta$ppl (cut attn)", "Cut Attention Path"),
    ("mlp_strength",  "delta_ppl_cut_mlp",  "MLP Strength",       r"$\Delta$ppl (cut mlp)",  "Cut MLP Path"),
]

for ax, (x_key, y_key, xlabel, ylabel, title) in zip(axes, configs):
    for i, layer in enumerate(layers):
        pts = by_layer[layer]
        xs = [p[x_key] for p in pts]
        ys = [p[y_key] for p in pts]
        ax.scatter(xs, ys, color=cmap[i], s=30, zorder=3, label=f"Layer {layer}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = path.replace(".json", "_scatter.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")