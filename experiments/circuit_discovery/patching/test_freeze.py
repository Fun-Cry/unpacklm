"""Test backup-head freezing during patching.

Self-repair hypothesis: when L9H9's K is disrupted, backup name movers
(L10H2, L10H10, L11H2 documented as compensators in the self-repair
experiments) step in to restore the IO direction, masking the patch's
output effect.

If self-repair is the explanation for L9H9's small patching Δ, then:
  - Patching L9H9 without freeze: small Δ (compensated)
  - Patching L9H9 with backups frozen: substantially larger Δ (uncompensated)

This script tests that hypothesis on canonical-correct IOI prompts.

Run from project root:
    python -m experiments.circuit_discovery.patching.test_freeze
"""

import os
import sys
from collections import defaultdict
from typing import List, Tuple


from experiments.circuit_discovery.utils import load_experiment_folder, load_model
from experiments.circuit_discovery.patching import (
    capture_path_patch_inputs, build_qk_score_patch,
    baseline_logit, run_with_intervention,
)
from experiments.circuit_discovery.ioi.prompts import _resolve_positions
from utils.load_data import load_ioi_with_abc


# Documented IOI backup name movers from self-repair experiments
BACKUP_HEADS: List[Tuple[int, int]] = [
    (10, 2), (10, 10), (11, 2),     # backup name movers
    (10, 6), (10, 7), (11, 1), (11, 3),   # observed compensators / suppression-flippers
]

# Edges to test: primary name movers (where self-repair should mask)
# vs negative name movers (where there's no backup, so patching should
# work with or without freeze)
EDGES_TO_TEST = [
    # (description, sender_name, sender_role, recv_layer, recv_head, query_role)
    ("NM L9H9 ← mlp_0@IO",     "mlp_0", "IO",  9, 9,   "END"),
    ("NM L9H6 ← mlp_0@IO",     "mlp_0", "IO",  9, 6,   "END"),
    ("NM L10H0 ← mlp_0@IO",    "mlp_0", "IO",  10, 0,  "END"),
    ("negNM L10H7 ← mlp_0@IO", "mlp_0", "IO",  10, 7,  "END"),
    ("negNM L11H10 ← mlp_0@IO","mlp_0", "IO",  11, 10, "END"),
    ("S-inh L8H6 ← mlp_0@S2",  "mlp_0", "S2",  8, 6,   "END"),
]


def main():
    folder = "experiments/circuit_discovery/ioi"
    n_prompts  = 20
    n_abc_refs = 10

    print(f"loading config + model from {folder}")
    cfg, _ = load_experiment_folder(folder)
    model, tokenizer, hook_manager, device = load_model(cfg)

    print(f"\nloading {n_prompts} IOI prompts × {n_abc_refs} ABC refs each")
    raw = load_ioi_with_abc(
        n_prompts=n_prompts, n_abc_refs=n_abc_refs,
        tokenizer=tokenizer, seed=42,
    )

    # Per-edge: list of prompt-level mean Δ (no-freeze and with-freeze)
    deltas_no_freeze:   dict = defaultdict(list)
    deltas_with_freeze: dict = defaultdict(list)
    n_prompts_used = 0

    for pi, p in enumerate(raw):
        clean_prompt     = p["prompt"]
        abc_refs         = p["abc_refs"]
        target_token     = p["target_token"]
        distractor_token = p["distractor_token"]

        try:
            base = baseline_logit(model, tokenizer, hook_manager,
                                   clean_prompt, target_token,
                                   distractor_token, device=device)
        except Exception:
            continue
        if base < 1.5:
            continue

        roles = _resolve_positions(clean_prompt, p["IO"], p["S"], tokenizer)
        if "IO" not in roles or "S2" not in roles:
            continue

        n_prompts_used += 1
        role_to_pos = {"IO": roles["IO"], "S2": roles["S2"], "END": roles["END"]}

        if n_prompts_used <= 3 or n_prompts_used % 5 == 0:
            print(f"  [{n_prompts_used}] prompt {pi} (base {base:+.2f})")

        # All sender + receiver specs needed
        sender_specs   = sorted({(s_n, role_to_pos[s_role])
                                  for _, s_n, s_role, _, _, _ in EDGES_TO_TEST})
        receiver_specs = sorted({(r_L, r_h)
                                  for _, _, _, r_L, r_h, _ in EDGES_TO_TEST})
        # Also need the backup heads in receiver_specs so we capture their data
        # Wait actually they go in freeze_specs, separately — see API.

        per_edge_no_freeze:   dict = defaultdict(list)
        per_edge_with_freeze: dict = defaultdict(list)

        for ref_idx, ref in enumerate(abc_refs):
            try:
                inputs = capture_path_patch_inputs(
                    model, tokenizer, hook_manager,
                    clean_prompt, ref,
                    sender_specs=sender_specs,
                    receiver_specs=receiver_specs,
                    freeze_specs=BACKUP_HEADS,
                    capture_corrupted_v=False,
                    device=device,
                )
            except Exception as e:
                continue

            for description, s_n, s_role, r_L, r_h, q_role in EDGES_TO_TEST:
                s_p = role_to_pos[s_role]
                q_p = role_to_pos[q_role]

                # NO FREEZE
                try:
                    iv_nf = build_qk_score_patch(
                        inputs, hook_manager,
                        sender_name=s_n, sender_pos=s_p,
                        receiver_layer=r_L, receiver_head=r_h, query_pos=q_p,
                        use_corrupted_v=False, device=device,
                    )
                    d_nf = run_with_intervention(
                        model, tokenizer, hook_manager, clean_prompt,
                        target_token, distractor_token, iv_nf, base,
                        device=device,
                    )
                    per_edge_no_freeze[description].append(float(d_nf))
                except Exception:
                    pass

                # WITH FREEZE — exclude the receiver itself from the freeze
                freeze_subset = [(fL, fh) for (fL, fh) in BACKUP_HEADS
                                  if not (fL == r_L and fh == r_h)]
                try:
                    iv_f = build_qk_score_patch(
                        inputs, hook_manager,
                        sender_name=s_n, sender_pos=s_p,
                        receiver_layer=r_L, receiver_head=r_h, query_pos=q_p,
                        use_corrupted_v=False,
                        freeze_heads=freeze_subset,
                        device=device,
                    )
                    d_f = run_with_intervention(
                        model, tokenizer, hook_manager, clean_prompt,
                        target_token, distractor_token, iv_f, base,
                        device=device,
                    )
                    per_edge_with_freeze[description].append(float(d_f))
                except Exception:
                    pass

        for description in per_edge_no_freeze:
            ds = per_edge_no_freeze[description]
            if ds:
                deltas_no_freeze[description].append(sum(ds) / len(ds))
        for description in per_edge_with_freeze:
            ds = per_edge_with_freeze[description]
            if ds:
                deltas_with_freeze[description].append(sum(ds) / len(ds))

    print(f"\nused {n_prompts_used} prompts × {n_abc_refs} corruptions each")
    print()

    print("=" * 90)
    print(f"  freeze heads (excluding receiver): {BACKUP_HEADS}")
    print()
    print(f"  {'edge':<28}  {'no freeze':>11}  {'with freeze':>13}  "
          f"{'ratio':>6}")
    print("=" * 90)
    for description, _, _, _, _, _ in EDGES_TO_TEST:
        ds_nf = deltas_no_freeze.get(description, [])
        ds_f  = deltas_with_freeze.get(description, [])
        if not ds_nf or not ds_f:
            print(f"  {description:<28}  no data")
            continue
        n = len(ds_nf)
        mean_nf = sum(ds_nf) / n
        mean_f  = sum(ds_f) / len(ds_f)
        ratio = (mean_f / mean_nf) if abs(mean_nf) > 1e-6 else float("inf")
        print(f"  {description:<28}  {mean_nf:>+11.4f}  {mean_f:>+13.4f}  "
              f"{ratio:>+6.2f}x")


if __name__ == "__main__":
    main()
