"""End-to-end pipeline test: discover -> patch -> report.

For one canonical-correct IOI prompt, we run the full pipeline:

  1. Run trace + lens filter to discover top paths.
  2. For each path, decompose into testable edges (skip terminals like
     embedding/pos_embedding).
  3. Run K-channel path patching on each edge, averaged across multiple
     ABC corruptions. Report the patching Δ alongside the discover
     score, so we can see which paths the method names AND validates.

This is the closed loop: method's own discoveries, validated by
its own patching machinery.

Run from project root:
    python -m experiments.circuit_discovery.patching.test_pipeline
"""

import os
import sys
from collections import defaultdict
from typing import List, Tuple, Optional


from experiments.circuit_discovery.utils import (
    load_experiment_folder, load_model, parse_step,
)
from experiments.circuit_discovery.discover import (
    discover_one,
)
from experiments.circuit_discovery.patching import (
    capture_path_patch_inputs, build_qk_score_patch,
    baseline_logit, run_with_intervention,
)
from experiments.circuit_discovery.ioi.prompts import (
    build_prompts as build_ioi_prompts, _resolve_positions,
)
from utils.load_data import load_ioi_with_abc


_TERMINALS = {"embedding", "pos_embedding"}


def chain_to_edges(chain: List[str]) -> List[Tuple[str, int, str, int]]:
    """Convert a discovered chain into a list of testable edges.

    The chain is in receiver→sender order, e.g.
        ["attn_9_head_9@12", "mlp_0@2", "embedding@2"]
    We want edges as (sender_name, sender_pos, recv_name, recv_pos):
        (mlp_0, 2, attn_9_head_9, 12)
    Skip edges where the sender is a terminal (embedding etc).
    Skip edges where the receiver is an MLP (we only patch into
    attention heads via the K-channel pipeline).
    """
    edges = []
    for i in range(len(chain) - 1, 0, -1):
        sender_name, sender_pos = parse_step(chain[i])
        recv_name,   recv_pos   = parse_step(chain[i - 1])
        if sender_name in _TERMINALS:
            continue
        if not recv_name.startswith("attn_"):
            continue
        edges.append((sender_name, sender_pos, recv_name, recv_pos))
    return edges


def parse_attn_head_name(name: str) -> Tuple[int, int]:
    """attn_9_head_6 -> (9, 6)"""
    parts = name.split("_")
    return int(parts[1]), int(parts[3])


def main():
    folder       = "experiments/circuit_discovery/ioi"
    n_abc_refs   = 10
    top_k_paths  = 8     # how many top paths to validate per prompt

    print(f"loading config + model from {folder}")
    cfg, _ = load_experiment_folder(folder)
    model, tokenizer, hook_manager, device = load_model(cfg)

    # Discover settings — same as a typical discover run
    beta = cfg.get("trace", {}).get("beta", 0.8)
    top_paths_k    = cfg.get("trace", {}).get("top_paths_k", 200)
    path_min_frac  = cfg.get("trace", {}).get("path_min_frac", 1e-4)
    lens_cfg = cfg.get("lens", {"type": "diversity", "min_diversity": 2})

    print(f"\nfinding canonical-correct IOI prompt (centered IO−S > 1.5)")
    raw = load_ioi_with_abc(
        n_prompts=20, n_abc_refs=n_abc_refs,
        tokenizer=tokenizer, seed=42,
    )

    p, base = None, None
    for cand in raw:
        b = baseline_logit(model, tokenizer, hook_manager,
                           cand["prompt"], cand["target_token"],
                           cand["distractor_token"], device=device)
        if b > 1.5:
            p, base = cand, b
            break
    if p is None:
        scored = sorted(
            [(baseline_logit(model, tokenizer, hook_manager,
                              c["prompt"], c["target_token"],
                              c["distractor_token"], device=device), c)
              for c in raw],
            reverse=True,
        )
        base, p = scored[0]

    clean_prompt     = p["prompt"]
    abc_refs         = p["abc_refs"]
    target_token     = p["target_token"]
    distractor_token = p["distractor_token"]

    roles = _resolve_positions(clean_prompt, p["IO"], p["S"], tokenizer)

    print(f"\nclean prompt:    {clean_prompt!r}")
    print(f"target / distractor: {target_token!r} / {distractor_token!r}")
    print(f"template: {p['template_type']}    "
          f"IO@{roles['IO']}  S2@{roles.get('S2','?')}  END@{roles['END']}")
    print(f"baseline (centered IO−S logit): {base:+.3f}")

    # ── 1. DISCOVER ───────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  Step 1: discover paths")
    print("=" * 78)

    # build the prompt dict that discover_one expects
    prompt_dict = {
        "prompt":           clean_prompt,
        "target_token":     target_token,
        "distractor_token": distractor_token,
        "metadata":         {
            "io_position": roles["IO"],
            "s2_position": roles.get("S2"),
            "end_position": roles["END"],
            "target_positions": [roles["IO"], roles.get("S2", roles["IO"])],
            "template_type": p["template_type"],
        },
    }

    discovery = discover_one(
        model, tokenizer, hook_manager,
        beta=beta, top_paths_k=top_paths_k,
        path_min_frac=path_min_frac, lens_cfg=lens_cfg,
        prompt_dict=prompt_dict,
    )

    paths = discovery["ranked_paths"][:top_k_paths]
    print(f"  found {len(discovery['ranked_paths'])} paths after lens filter; "
          f"showing top {len(paths)}:")
    for i, path in enumerate(paths):
        chain_str = " ← ".join(path["chain"])
        print(f"  [{i+1}] score={path['score']:+.4f}  {chain_str}")

    # ── 2. PATCH each non-trivial edge ───────────────────────────────
    print("\n" + "=" * 78)
    print(f"  Step 2: patch each edge ({n_abc_refs} ABC refs averaged)")
    print("=" * 78)

    # Diagnostic: state after discover
    print(f"  state after discover_one:")
    print(f"    PyTorch hook handles: {len(hook_manager.handles)}")
    print(f"    attention_input_cache keys: "
          f"{sorted(hook_manager.attention_input_cache.keys())}")
    print(f"    mlp_outputs len: {len(hook_manager.mlp_outputs)}")
    print(f"    pre_dense_inputs len: {len(hook_manager.pre_dense_inputs)}")

    # Collect all unique sender_specs and receiver_specs needed across the top paths
    all_edges_per_path = []
    sender_specs   = set()
    receiver_specs = set()
    for path in paths:
        edges = chain_to_edges(path["chain"])
        all_edges_per_path.append(edges)
        for s_n, s_p, r_n, r_p in edges:
            sender_specs.add((s_n, s_p))
            r_layer, r_head = parse_attn_head_name(r_n)
            receiver_specs.add((r_layer, r_head))

    sender_specs   = sorted(sender_specs)
    receiver_specs = sorted(receiver_specs)
    print(f"  {len(sender_specs)} unique sender specs, "
          f"{len(receiver_specs)} unique receiver heads")

    # Capture inputs once per ABC ref
    inputs_per_ref = []
    for ref_idx, ref in enumerate(abc_refs):
        try:
            inp = capture_path_patch_inputs(
                model, tokenizer, hook_manager,
                clean_prompt, ref,
                sender_specs=sender_specs,
                receiver_specs=receiver_specs,
                capture_corrupted_v=False,
                device=device,
            )
            inputs_per_ref.append(inp)
        except Exception as e:
            print(f"  capture failed for ref {ref_idx}: {e!r}")
            import traceback
            traceback.print_exc()
            print(f"    attention_input_cache keys: "
                  f"{sorted(hook_manager.attention_input_cache.keys())}")
            print(f"    pre_dense_inputs len: {len(hook_manager.pre_dense_inputs)}")
            print(f"    mlp_outputs len: {len(hook_manager.mlp_outputs)}")
            print(f"    embedding_outputs len: {len(hook_manager.embedding_outputs)}")
            print(f"    PyTorch hook handles count: {len(hook_manager.handles)}")
            print(f"    receiver_layers needed: "
                  f"{sorted({L for (L, _h) in receiver_specs})}")
            break   # stop after first failure for diagnosis

    print(f"  captured {len(inputs_per_ref)} ABC reference contexts")

    # ── 3. report ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  Step 3: results — discover score vs patching Δ per edge")
    print("=" * 78)

    for path_idx, (path, edges) in enumerate(zip(paths, all_edges_per_path)):
        chain_str = " ← ".join(path["chain"])
        print(f"\n  [{path_idx+1}] discover score = {path['score']:+.4f}")
        print(f"      chain: {chain_str}")
        if not edges:
            print(f"      (no testable edges — only terminals or MLP receivers)")
            continue

        for (s_n, s_p, r_n, r_p) in edges:
            r_layer, r_head = parse_attn_head_name(r_n)

            # Run patching across ABC refs, average
            deltas = []
            for inputs in inputs_per_ref:
                try:
                    iv = build_qk_score_patch(
                        inputs, hook_manager,
                        sender_name=s_n, sender_pos=s_p,
                        receiver_layer=r_layer, receiver_head=r_head,
                        query_pos=r_p,
                        use_corrupted_v=False, device=device,
                    )
                    d = run_with_intervention(
                        model, tokenizer, hook_manager, clean_prompt,
                        target_token, distractor_token, iv, base,
                        device=device,
                    )
                    deltas.append(float(d))
                except Exception as e:
                    pass

            if not deltas:
                marker = "!"
                line = "      ! patching failed"
            else:
                mean_d = sum(deltas) / len(deltas)
                if len(deltas) > 1:
                    var = sum((d - mean_d) ** 2 for d in deltas) / (len(deltas) - 1)
                    std_d = var ** 0.5
                else:
                    std_d = 0.0
                marker = "✓" if abs(mean_d) > 0.05 else " "
                line = (f"      {marker} {s_n}@{s_p} → {r_n}@{r_p}: "
                        f"Δ = {mean_d:+.4f} ± {std_d:.3f}  (n={len(deltas)})")
            print(line)


if __name__ == "__main__":
    main()