"""Faithfulness + completeness verification — reference-source-agnostic.

Caller is responsible for preparing each prompt dict with:
    prompt:            str        target sentence
    target_token:      str        with leading space
    distractor_token:  str        with leading space
    references:        List[str]  same-length-tokenized prompts whose
                                  activations form the mean-ablation
                                  baseline. Caller chooses what these
                                  are: ABC corruptions (Wang-faithful),
                                  length-matched IOI prompts, etc.

For each prompt:
    1. Clean forward → logit_diff = logit(IO) - logit(S)
    2. Faithfulness: mean-ablate every component NOT in the circuit
       to the prompt's references → ablated logit_diff. If the circuit
       captures the task, this stays close to clean.
    3. Completeness: mean-ablate every component IN the circuit to the
       prompt's references → ablated logit_diff. If the circuit is
       necessary, this drops near zero.

Two entry points:

    verify(model, tokenizer, hook_manager, device, prepared_prompts,
           *, circuit, always_on, num_layers, num_heads, ...)
        Programmatic API. Returns a summary dict.

    main()
        CLI: loads saved trace JSONs from a results dir, prepares each
        prompt with either ABC refs (--abc) or length-matched IOI refs
        (default), then calls verify().
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch


from experiments.circuit_discovery.utils import load_model
from experiments.circuit_discovery.ioi.abc_prep import add_abc_references
from experiments.circuit_discovery.ioi.prompts import _resolve_positions
from experiments.ablation_tracing.core.ablation import (
    AblationConfig, build_intervention,
)
from experiments.circuit_discovery.selection._common import (
    filter_correct, load_run,
)
from utils.load_data import load_ioi_with_abc


def _regenerate_prompts(tokenizer, model, device, n_prompts, seed,
                        abc_seed_offset=10000):
    """Generate fresh IOI prompts and run forward passes to populate
    clean_target_prob / clean_target_logit_centered / position metadata,
    matching the structure that run_ioi_sweep saves to disk.

    Returns a list of dicts with the same keys as a per-prompt JSON
    (minus the trace fields like ranked_paths, which verification
    doesn't need).
    """
    raw = load_ioi_with_abc(
        n_prompts=n_prompts, n_abc_refs=0,
        tokenizer=tokenizer, seed=seed,
        abc_seed_offset=abc_seed_offset,
    )

    out = []
    n_failed = 0
    for p in raw:
        try:
            roles = _resolve_positions(p["prompt"], p["IO"], p["S"], tokenizer)
        except Exception as e:
            n_failed += 1
            continue

        # Forward pass to get target prob and centered logit.
        ids = tokenizer(p["prompt"], return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1]  # (vocab,)
        probs = torch.softmax(logits, dim=-1)
        tgt_id = tokenizer.encode(p["target_token"], add_special_tokens=False)[0]
        target_prob = float(probs[tgt_id])

        # Centered logit (target − mean over vocab).
        target_logit_centered = float(logits[tgt_id] - logits.mean())

        out.append({
            "prompt":                      p["prompt"],
            "target_token":                p["target_token"],
            "distractor_token":            p["distractor_token"],
            "clean_target_prob":           target_prob,
            "clean_target_logit_centered": target_logit_centered,
            "metadata": {
                "io_position":      roles["IO"],
                "s1_position":      roles.get("S1"),
                "s2_position":      roles.get("S2"),
                "end_position":     roles["END"],
                "target_positions": [roles["IO"], roles.get("S2", roles["IO"])],
                "template_type":    p["template_type"],
                "IO":               p["IO"],
                "S":                p["S"],
            },
        })
    if n_failed:
        print(f"  warning: dropped {n_failed} prompts on position resolution")
    return out


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def all_heads_and_mlps(num_layers: int, num_heads: int) -> List[str]:
    heads = [f"attn_{l}_head_{h}"
             for l in range(num_layers) for h in range(num_heads)]
    mlps = [f"mlp_{l}" for l in range(num_layers)]
    return heads + mlps


def single_token_id(tokenizer, token_str: str):
    if token_str is None:
        return None
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def install_intervention(hook_manager, intervention):
    hook_manager.clear_interventions()
    for hook_name, fn in intervention.interventions:
        hook_manager.register_intervention(hook_name, fn)


@torch.no_grad()
def logit_diff(model, tokenizer, sentence, io_id, s_id, device):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    out = model(**inputs)
    logits = out.logits[0, -1]
    return float(logits[io_id] - logits[s_id])


def _build_mean_intervention(model, tokenizer, hook_manager,
                             target_prompt, references, components):
    """Mean-ablation Intervention for one (target, components) pair.
    Reference-source-agnostic — the caller decides whether `references`
    are ABC corruptions, length-matched IOI prompts, or anything else."""
    cfg = AblationConfig(
        components=components, mode="mean",
        references=references, positions="target",
    )
    return build_intervention(model, tokenizer, hook_manager,
                              target_prompt, cfg)


def load_circuit_arg(args) -> Set[str]:
    if args.circuit_file:
        with open(args.circuit_file) as f:
            data = json.load(f)
        circuit = data["circuit"] if isinstance(data, dict) else data
    elif args.circuit:
        circuit = args.circuit
    else:
        sys.exit("must specify --circuit or --circuit-file")
    return set(circuit)


# ──────────────────────────────────────────────────────────────────────
# Length-matched IOI references (the previous default behavior)
# ──────────────────────────────────────────────────────────────────────
def add_ioi_length_matched_references(
    prompt_jsons: List[dict],
    tokenizer,
    keep_only: Optional[set] = None,
) -> List[dict]:
    """Build prepared-prompt dicts with references = other prompts of
    matching tokenized length. Mirrors the pre-refactor verifier
    behavior. Use this when ABC isn't available (non-IOI tasks)."""
    valid = []
    for p in prompt_jsons:
        if keep_only is not None and p["prompt"] not in keep_only:
            continue
        n = len(tokenizer.encode(p["prompt"], add_special_tokens=False))
        valid.append((n, p))

    by_len: Dict[int, List[dict]] = defaultdict(list)
    for n, p in valid:
        by_len[n].append(p)

    out = []
    for n, p in valid:
        bucket = by_len[n]
        refs = [q["prompt"] for q in bucket if q["prompt"] != p["prompt"]]
        if not refs:
            continue
        out.append({
            "prompt":           p["prompt"],
            "target_token":     p["target_token"],
            "distractor_token": p["distractor_token"],
            "references":       refs,
            "metadata":         dict(p.get("metadata", {})),
        })
    return out


# ──────────────────────────────────────────────────────────────────────
# Core: verify a circuit on prepared prompts
# ──────────────────────────────────────────────────────────────────────
def verify(
    model, tokenizer, hook_manager, device,
    prepared_prompts: List[dict],
    *,
    circuit: Set[str],
    always_on: Set[str],
    num_layers: int,
    num_heads: int,
    max_prompts: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run faithfulness + completeness over the prepared prompts.

    Each prompt dict must already include `references`. The caller is
    responsible for building those (ABC, length-matched IOI, etc).

    Returns:
        Dict with summary stats and per-prompt rows.
    """
    universe     = set(all_heads_and_mlps(num_layers, num_heads))
    not_circuit  = sorted(universe - set(circuit) - set(always_on))
    circuit_only = sorted(set(circuit) - set(always_on))

    targets = prepared_prompts[:max_prompts] if max_prompts else prepared_prompts
    rows = []
    for i, p in enumerate(targets):
        io_id = single_token_id(tokenizer, p["target_token"])
        s_id  = single_token_id(tokenizer, p["distractor_token"])
        if io_id is None or s_id is None:
            if verbose:
                print(f"  [skip] multi-token name: {p['prompt'][:60]!r}")
            continue
        if not p.get("references"):
            if verbose:
                print(f"  [skip] no references for {p['prompt'][:60]!r}")
            continue

        # 1. Clean
        hook_manager.clear_interventions()
        ld_clean = logit_diff(model, tokenizer, p["prompt"],
                              io_id, s_id, device)

        # 2. Faithfulness
        try:
            inter = _build_mean_intervention(
                model, tokenizer, hook_manager,
                target_prompt=p["prompt"],
                references=p["references"],
                components=not_circuit,
            )
            install_intervention(hook_manager, inter)
            ld_faith = logit_diff(model, tokenizer, p["prompt"],
                                  io_id, s_id, device)
        finally:
            hook_manager.clear_interventions()

        # 3. Completeness
        if circuit_only:
            try:
                inter = _build_mean_intervention(
                    model, tokenizer, hook_manager,
                    target_prompt=p["prompt"],
                    references=p["references"],
                    components=circuit_only,
                )
                install_intervention(hook_manager, inter)
                ld_comp = logit_diff(model, tokenizer, p["prompt"],
                                     io_id, s_id, device)
            finally:
                hook_manager.clear_interventions()
        else:
            ld_comp = float("nan")

        rows.append({
            "prompt":   p["prompt"],
            "ld_clean": ld_clean,
            "ld_faith": ld_faith,
            "ld_comp":  ld_comp,
            "n_refs":   len(p["references"]),
        })
        if verbose:
            print(f"  [{i+1:3d}/{len(targets)}] "
                  f"clean={ld_clean:+6.3f}  faith={ld_faith:+6.3f}  "
                  f"comp={ld_comp:+6.3f}  ({p['prompt'][:40]})")

    if not rows:
        return {"rows": [], "n_prompts": 0}

    arr = np.array([(r["ld_clean"], r["ld_faith"], r["ld_comp"])
                    for r in rows])
    means = arr.mean(axis=0)
    sems  = arr.std(axis=0, ddof=1) / np.sqrt(len(arr))
    correct = (arr > 0).mean(axis=0)

    return {
        "n_prompts":        len(rows),
        "circuit_size":     len(circuit_only),
        "ablated_in_faith": len(not_circuit),
        "mean_ld":     {"clean": float(means[0]),
                        "faith": float(means[1]),
                        "comp":  float(means[2])},
        "sem_ld":      {"clean": float(sems[0]),
                        "faith": float(sems[1]),
                        "comp":  float(sems[2])},
        "p_correct":   {"clean": float(correct[0]),
                        "faith": float(correct[1]),
                        "comp":  float(correct[2])},
        "faith_ratio": float(means[1] / means[0]) if means[0] != 0 else None,
        "comp_drop":   float((means[0] - means[2]) / means[0]) if means[0] != 0 else None,
        "rows":        rows,
    }


def print_summary(summary):
    if summary["n_prompts"] == 0:
        print("  (no prompts verified)")
        return
    m, s, c = summary["mean_ld"], summary["sem_ld"], summary["p_correct"]
    print()
    print(f"  prompts verified: {summary['n_prompts']}   "
          f"|C|={summary['circuit_size']}  "
          f"|NOT(C)|={summary['ablated_in_faith']}")
    print(f"            mean LD            P(IO > S)")
    print(f"  clean    {m['clean']:+6.3f} ± {s['clean']:.3f}    {c['clean']:.2f}")
    print(f"  faith    {m['faith']:+6.3f} ± {s['faith']:.3f}    {c['faith']:.2f}    "
          f"(ratio: {summary['faith_ratio']:+.2f})")
    print(f"  comp     {m['comp']:+6.3f} ± {s['comp']:.3f}    {c['comp']:.2f}    "
          f"(drop:  {summary['comp_drop']:+.2f})")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")

    cgrp = ap.add_mutually_exclusive_group(required=True)
    cgrp.add_argument("--circuit", nargs="+",
                      help="component names defining circuit C")
    cgrp.add_argument("--circuit-file",
                      help="JSON: {'circuit': [...]} or [...]")

    ap.add_argument("--always-on", nargs="*", default=[])
    ap.add_argument("--p-min", type=float, default=0.3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default="/data/s4283341")
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--out", default=None)

    # Reference source
    rgrp = ap.add_mutually_exclusive_group()
    rgrp.add_argument("--abc", action="store_true",
                      help="use ABC corruptions as mean-ablation refs "
                           "(Wang-faithful; default)")
    rgrp.add_argument("--ioi-length-matched", action="store_true",
                      help="use length-matched IOI prompts as refs "
                           "(pre-refactor behavior; available for "
                           "comparison)")
    ap.add_argument("--n-abc-refs", type=int, default=10)
    ap.add_argument("--abc-seed-offset", type=int, default=10000)
    ap.add_argument("--regen-prompts-seed", type=int, default=None,
                    help="Generate fresh IOI prompts at this seed and "
                         "verify the circuit on them, ignoring the "
                         "results dir's saved prompts. Used for out-of-"
                         "sample generalization checks.")
    ap.add_argument("--regen-prompts-n", type=int, default=100,
                    help="Number of fresh prompts when --regen-prompts-seed "
                         "is set (default 100).")

    args = ap.parse_args()

    # Default to ABC if neither flag was given.
    if not args.abc and not args.ioi_length_matched:
        args.abc = True

    circuit   = load_circuit_arg(args)
    always_on = set(args.always_on)

    run_cfg, prompts_all = load_run(args.results_dir)

    # Load model first; regen-prompts needs it for forward passes.
    model_cfg = {"model": {**run_cfg, "device": args.device,
                           "cache_dir": args.cache_dir}}
    model, tokenizer, hook_manager, device = load_model(model_cfg)
    num_layers = model.config.num_hidden_layers
    num_heads  = model.config.num_attention_heads

    # Optional: replace saved prompts with freshly-generated ones at a
    # different seed (out-of-sample generalization check).
    if args.regen_prompts_seed is not None:
        print(f"  regenerating {args.regen_prompts_n} prompts at "
              f"seed={args.regen_prompts_seed}")
        prompts_all = _regenerate_prompts(
            tokenizer, model, device,
            n_prompts=args.regen_prompts_n,
            seed=args.regen_prompts_seed,
            abc_seed_offset=args.abc_seed_offset,
        )

    prompts_kept = filter_correct(prompts_all, args.p_min)

    print(f"=== {args.results_dir} ===")
    print(f"  circuit:    {len(circuit)} components")
    print(f"  always-on:  {sorted(always_on) or '(none)'}")
    print(f"  prompts:    {len(prompts_kept)}/{len(prompts_all)} "
          f"(target_prob >= {args.p_min})")
    if args.regen_prompts_seed is not None:
        print(f"  prompt source: regenerated (seed={args.regen_prompts_seed})")
    print(f"  references: {'ABC corruptions' if args.abc else 'IOI length-matched'}")

    # Build prepared prompts
    if args.abc:
        prepared = add_abc_references(
            prompts_kept, tokenizer,
            n_abc_refs=args.n_abc_refs,
            abc_seed_offset=args.abc_seed_offset,
        )
    else:
        prepared = add_ioi_length_matched_references(
            prompts_kept, tokenizer,
        )
    print(f"  prepared: {len(prepared)} prompts with references")

    summary = verify(
        model, tokenizer, hook_manager, device, prepared,
        circuit=circuit, always_on=always_on,
        num_layers=num_layers, num_heads=num_heads,
        max_prompts=args.max_prompts, verbose=True,
    )
    print_summary(summary)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "results_dir":  args.results_dir,
                "circuit":      sorted(circuit),
                "always_on":    sorted(always_on),
                "reference_source": "abc" if args.abc else "ioi_length_matched",
                "n_abc_refs":   args.n_abc_refs if args.abc else None,
                **summary,
            }, f, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()