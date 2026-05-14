"""Step 3: Verify — faithfulness + completeness via mean-ablation.

Loads circuit from select step, generates fresh IOI prompts at a
different seed, and measures:
  - faithfulness: ablate NOT(circuit) → logit diff preserved?
  - completeness: ablate circuit → logit diff destroyed?

Uses ABC corruptions as mean-ablation references (Wang-faithful).

Usage:
    python -m experiments.circuits.verify \
        --results-dir results/circuits/gpt2_default \
        --circuit-file results/circuits/gpt2_default/circuit.json \
        --device cuda:0 --verify-seed 7
"""

import argparse
import json
import os
from typing import List, Set

import numpy as np
import torch

from experiments.ablation_tracing.core.ablation import (
    AblationConfig, build_intervention,
)
from utils.load_data import load_ioi_with_abc


def all_components(num_layers, num_heads):
    """All patchable components: heads + MLPs."""
    heads = [f"attn_{l}_head_{h}"
             for l in range(num_layers) for h in range(num_heads)]
    mlps = [f"mlp_{l}" for l in range(num_layers)]
    return set(heads + mlps)


@torch.no_grad()
def logit_diff(model, tokenizer, sentence, io_id, s_id, device):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    logits = model(**inputs).logits[0, -1]
    return float(logits[io_id] - logits[s_id])


def verify(model, tokenizer, hook_manager, device,
           circuit: Set[str], prompts: List[dict],
           max_prompts=None, verbose=True):
    """Run faithfulness + completeness on prepared prompts.

    Each prompt dict needs: prompt, target_token, distractor_token, abc_refs.
    """
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    all_comps = all_components(num_layers, num_heads)
    circuit_only = circuit & all_comps
    not_circuit = all_comps - circuit_only

    if max_prompts:
        prompts = prompts[:max_prompts]

    rows = []
    for i, p in enumerate(prompts):
        io_id = tokenizer.encode(p["target_token"], add_special_tokens=False)
        s_id = tokenizer.encode(p["distractor_token"], add_special_tokens=False)
        if not io_id or not s_id:
            continue
        io_id, s_id = io_id[0], s_id[0]

        refs = p.get("references", [])
        if not refs:
            continue

        # 1. Clean
        ld_clean = logit_diff(model, tokenizer, p["prompt"],
                              io_id, s_id, device)

        # 2. Faithfulness: ablate NOT(circuit)
        try:
            cfg = AblationConfig(components=list(not_circuit), mode="mean",
                                 references=refs, positions="target")
            inter = build_intervention(model, tokenizer, hook_manager,
                                       p["prompt"], cfg)
            for hook_name, fn in inter.interventions:
                hook_manager.register_intervention(hook_name, fn)
            ld_faith = logit_diff(model, tokenizer, p["prompt"],
                                  io_id, s_id, device)
        finally:
            hook_manager.clear_interventions()

        # 3. Completeness: ablate circuit
        ld_comp = float("nan")
        if circuit_only:
            try:
                cfg = AblationConfig(components=list(circuit_only), mode="mean",
                                     references=refs, positions="target")
                inter = build_intervention(model, tokenizer, hook_manager,
                                           p["prompt"], cfg)
                for hook_name, fn in inter.interventions:
                    hook_manager.register_intervention(hook_name, fn)
                ld_comp = logit_diff(model, tokenizer, p["prompt"],
                                     io_id, s_id, device)
            finally:
                hook_manager.clear_interventions()

        rows.append({"ld_clean": ld_clean, "ld_faith": ld_faith,
                     "ld_comp": ld_comp})

        if verbose and (i < 3 or (i + 1) % 10 == 0):
            print(f"  [{i+1:3d}/{len(prompts)}] clean={ld_clean:+.3f}  "
                  f"faith={ld_faith:+.3f}  comp={ld_comp:+.3f}")

    if not rows:
        return {"n_prompts": 0}

    arr = np.array([(r["ld_clean"], r["ld_faith"], r["ld_comp"])
                    for r in rows])
    means = arr.mean(axis=0)
    sems = arr.std(axis=0, ddof=1) / np.sqrt(len(arr))

    faith_ratio = float(means[1] / means[0]) if means[0] != 0 else None
    comp_drop = float((means[0] - means[2]) / means[0]) if means[0] != 0 else None

    return {
        "n_prompts": len(rows),
        "circuit_size": len(circuit_only),
        "mean_ld": {"clean": float(means[0]), "faith": float(means[1]),
                    "comp": float(means[2])},
        "sem_ld": {"clean": float(sems[0]), "faith": float(sems[1]),
                   "comp": float(sems[2])},
        "faith_ratio": faith_ratio,
        "comp_drop": comp_drop,
        "rows": rows,
    }


def print_summary(s):
    if s["n_prompts"] == 0:
        print("  (no prompts verified)")
        return
    m, sem = s["mean_ld"], s["sem_ld"]
    print(f"\n  prompts: {s['n_prompts']}  |C|={s['circuit_size']}")
    print(f"            mean LD")
    print(f"  clean    {m['clean']:+.3f} ± {sem['clean']:.3f}")
    print(f"  faith    {m['faith']:+.3f} ± {sem['faith']:.3f}  "
          f"(ratio: {s['faith_ratio']:+.3f})")
    print(f"  comp     {m['comp']:+.3f} ± {sem['comp']:.3f}  "
          f"(drop:  {s['comp_drop']:+.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--circuit-file", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--verify-seed", type=int, default=7,
                    help="Seed for verification prompts (must differ from "
                         "discovery seed)")
    ap.add_argument("--n-verify-prompts", type=int, default=100)
    ap.add_argument("--n-abc-refs", type=int, default=10)
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Load circuit
    with open(args.circuit_file) as f:
        data = json.load(f)
    circuit = set(data["circuit"])

    # Load model info from run_config
    cfg_path = os.path.join(args.results_dir, "run_config.json")
    with open(cfg_path) as f:
        run_cfg = json.load(f)

    # Reconstruct model name
    family = run_cfg.get("family", "gpt2")
    size = run_cfg.get("size", "small")
    step = run_cfg.get("step", "final")
    load_kwargs = {}
    if family == "pythia":
        deduped = "-deduped" if run_cfg.get("deduped", True) else ""
        model_name = f"EleutherAI/pythia-{size}{deduped}"
        if step != "final":
            load_kwargs["step"] = step
    else:
        model_name = "gpt2"

    print(f"Loading {model_name}...")
    from unpack.models import load_model, get_adapter
    model, tokenizer = load_model(model_name, device=args.device,
                                  cache_dir=args.cache_dir, **load_kwargs)
    adapter = get_adapter(model)
    adapter.register_hooks(model)

    print(f"Generating {args.n_verify_prompts} verification prompts "
          f"(seed={args.verify_seed})")

    # Generate fresh IOI prompts with ABC references
    raw = load_ioi_with_abc(
        n_prompts=args.n_verify_prompts,
        n_abc_refs=args.n_abc_refs,
        tokenizer=tokenizer,
        seed=args.verify_seed,
    )

    # Build prepared prompts with references
    prepared = []
    for p in raw:
        refs = p.get("abc_refs", [])
        if not refs:
            # Length-matched fallback: use other prompts
            continue
        prepared.append({
            "prompt": p["prompt"],
            "target_token": p["target_token"],
            "distractor_token": p["distractor_token"],
            "references": refs,
        })

    # Filter by P(target) >= 0.3
    device = args.device
    filtered = []
    for p in prepared:
        io_id = tokenizer.encode(p["target_token"], add_special_tokens=False)
        if not io_id:
            continue
        inputs = tokenizer(p["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        prob = float(torch.softmax(logits, -1)[io_id[0]])
        if prob >= 0.3:
            filtered.append(p)

    print(f"  {len(filtered)}/{len(prepared)} prompts with P(target) >= 0.3")
    print(f"  circuit: {len(circuit)} components")

    summary = verify(model, tokenizer, adapter, device,
                     circuit, filtered,
                     max_prompts=args.max_prompts, verbose=True)
    print_summary(summary)

    out = args.out or os.path.join(args.results_dir, "verification.json")
    with open(out, "w") as f:
        json.dump({**summary, "circuit": sorted(circuit),
                   "verify_seed": args.verify_seed}, f, indent=2)
    print(f"\n  saved: {out}")


if __name__ == "__main__":
    main()
