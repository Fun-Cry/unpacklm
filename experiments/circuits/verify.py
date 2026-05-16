"""Circuit verification via mean-ablation.

Faithfulness: ablate components NOT in circuit → logit diff preserved?
Completeness: ablate components IN circuit → logit diff destroyed?

Only attention heads are ablated. MLPs recompute dynamically on the
patched residual stream (Wang-style).
"""

import numpy as np
import torch
from typing import List, Set

from experiments.ablation_tracing.core.ablation import (
    AblationConfig, build_intervention,
)
from utils.load_data import load_ioi_with_abc, build_abc_for_target


def all_heads(num_layers, num_heads):
    return set(f"attn_{l}_head_{h}"
               for l in range(num_layers) for h in range(num_heads))


@torch.no_grad()
def logit_diff(model, tokenizer, sentence, io_id, s_id, device):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    logits = model(**inputs).logits[0, -1]
    return float(logits[io_id] - logits[s_id])


def prepare_verify_prompts(tokenizer, model, device,
                           n_prompts=100, n_abc_refs=10,
                           seed=7, p_min=0.1):
    """Generate fresh IOI prompts with ABC refs, filter by P(target).
    
    Keeps generating with increasing pool sizes until n_prompts are collected.
    """
    prepared = []
    attempt = 0
    pool_size = n_prompts * 3

    while len(prepared) < n_prompts:
        current_seed = seed + attempt
        raw = load_ioi_with_abc(
            n_prompts=pool_size, n_abc_refs=n_abc_refs,
            tokenizer=tokenizer, seed=current_seed,
        )

        seen = set(p["prompt"] for p in prepared)
        for p in raw:
            if len(prepared) >= n_prompts:
                break
            refs = p.get("abc_refs", [])
            if not refs:
                continue
            if p["prompt"] in seen:
                continue

            io_id = tokenizer.encode(p["target_token"], add_special_tokens=False)
            if not io_id:
                continue

            inputs = tokenizer(p["prompt"], return_tensors="pt").to(device)
            with torch.no_grad():
                prob = float(torch.softmax(model(**inputs).logits[0, -1], -1)[io_id[0]])
            if prob >= p_min:
                prepared.append({
                    "prompt": p["prompt"],
                    "target_token": p["target_token"],
                    "distractor_token": p["distractor_token"],
                    "references": refs,
                })
                seen.add(p["prompt"])

        attempt += 1
        pool_size *= 2
        if attempt > 10:
            print(f"  Warning: only collected {len(prepared)}/{n_prompts} "
                  f"verify prompts after {attempt} attempts")
            break

    return prepared[:n_prompts]


def verify(model, tokenizer, adapter, device,
           circuit: Set[str], prompts: List[dict],
           verbose=True):
    """Run faith + knockout verification.

    Only ablates attention heads; MLPs recompute dynamically.
    """
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    every_head = all_heads(num_layers, num_heads)
    circuit_heads = circuit & every_head
    not_circuit_heads = every_head - circuit_heads

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

        ld_clean = logit_diff(model, tokenizer, p["prompt"],
                              io_id, s_id, device)

        # Faithfulness
        try:
            cfg = AblationConfig(components=list(not_circuit_heads),
                                 mode="mean", references=refs,
                                 positions="all")
            inter = build_intervention(model, tokenizer, adapter,
                                       p["prompt"], cfg)
            for hook_name, fn in inter.interventions:
                adapter.register_intervention(hook_name, fn)
            ld_faith = logit_diff(model, tokenizer, p["prompt"],
                                  io_id, s_id, device)
        finally:
            adapter.clear_interventions()

        # Knockout
        ld_comp = float("nan")
        if circuit_heads:
            try:
                cfg = AblationConfig(components=list(circuit_heads),
                                     mode="mean", references=refs,
                                     positions="all")
                inter = build_intervention(model, tokenizer, adapter,
                                           p["prompt"], cfg)
                for hook_name, fn in inter.interventions:
                    adapter.register_intervention(hook_name, fn)
                ld_comp = logit_diff(model, tokenizer, p["prompt"],
                                     io_id, s_id, device)
            finally:
                adapter.clear_interventions()

        rows.append({"ld_clean": ld_clean, "ld_faith": ld_faith,
                     "ld_comp": ld_comp})

        if verbose and (i < 3 or (i + 1) % 10 == 0):
            print(f"    [{i+1:3d}/{len(prompts)}] "
                  f"clean={ld_clean:+.3f} faith={ld_faith:+.3f} "
                  f"comp={ld_comp:+.3f}")

    if not rows:
        return {"n_prompts": 0}

    arr = np.array([(r["ld_clean"], r["ld_faith"], r["ld_comp"])
                    for r in rows])
    means = arr.mean(axis=0)
    sems = arr.std(axis=0, ddof=1) / np.sqrt(len(arr))

    faith = float(means[1] / means[0]) if means[0] != 0 else None
    knockout = float((means[0] - means[2]) / means[0]) if means[0] != 0 else None

    return {
        "n_prompts": len(rows),
        "circuit_size": len(circuit_heads),
        "mean_ld": {"clean": float(means[0]), "faith": float(means[1]),
                    "comp": float(means[2])},
        "sem_ld": {"clean": float(sems[0]), "faith": float(sems[1]),
                   "comp": float(sems[2])},
        "faith_ratio": faith,
        "comp_drop": knockout,
    }