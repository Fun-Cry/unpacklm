"""Conditions for the IOI self-repair experiment.

Each condition is ``(label, components_to_ablate_jointly)``. The runner
produces one cell per ``(prompt, condition)`` pair, so a
100-prompt × 7-condition spec yields 700 cells. Labels become filename
suffixes — keep them filesystem-safe.

The conditions cover both directions of the IOI self-repair phenomenon:

  Positive circuit (Wang 2023, McGrath 2023). The primary name movers
  L9H9, L9H6, L10H0 attend from the final "to" position to the IO
  token and write a positive copy of it into the residual stream.
  Ablating any one of them is partially compensated by backup name
  movers and routing changes upstream; the joint ablation of all three
  is the cleanest probe of this self-repair behavior because it
  removes the dominant positive-direction source at once.

  Negative circuit (McDougall 2023). The negative name movers L10H7
  and L11H10 also attend to the IO but write _against_ its direction —
  copy suppression. Ablating them therefore _raises_ P(IO) above
  clean. This is the mirror self-repair story: where the positive
  circuit gets compensated when removed, the negative circuit gets
  unmasked. The joint ablation isolates that effect.

The single-head conditions exist for diagnostic breakdown — they let
the analysis ask "which heads compensate for L9H9 specifically vs L9H6
specifically" before pooling across the joint condition.

Role labels in the per-cell DiffResult use the Wang / McGrath / Rushing
& Nanda convention: a non-ablated component is a **compensator** if its
Δdirect points the same way as the ablated set's clean direct
attribution (it picked up the IOI-axis function the ablated set was
performing in clean), a **doubler** if it points the opposite way, and
**breakage** if it was contributing in clean and has now flipped
against itself. This convention stays well-defined when the model
fully recovers (Δp_target ≈ 0), unlike output-side classification.
"""

CONDITIONS = [
    # ──────────────────────────────────────────────────────────────────
    # Single-head positive name movers (Wang 2023).
    # Each writes the IO token's identity into the residual at the
    # final "to" position. Diagnostic breakdown: which compensators
    # are head-specific vs shared across the three.
    # ──────────────────────────────────────────────────────────────────
    ("nm_L9H9",       ["attn_9_head_9"]),
    ("nm_L9H6",       ["attn_9_head_6"]),
    ("nm_L10H0",      ["attn_10_head_0"]),

    # ──────────────────────────────────────────────────────────────────
    # Single-head negative name movers (McDougall 2023).
    # Each suppresses the IO token. Same diagnostic role for the
    # negative circuit.
    # ──────────────────────────────────────────────────────────────────
    ("negnm_L10H7",   ["attn_10_head_7"]),
    ("negnm_L11H10",  ["attn_11_head_10"]),

    # ──────────────────────────────────────────────────────────────────
    # Joint positive-circuit ablation: the clean Wang / McGrath
    # self-repair probe. Removes the dominant IO-promoting source;
    # P(IO) drops, downstream backup name movers and S-inhibition
    # routing changes pick up the slack.
    # ──────────────────────────────────────────────────────────────────
    ("nm_joint",      ["attn_9_head_9", "attn_9_head_6", "attn_10_head_0"]),

    # ──────────────────────────────────────────────────────────────────
    # Joint negative-circuit ablation: the McDougall copy-suppression
    # mirror probe. Removes the suppressors; P(IO) rises above clean.
    # The non-ablated negative heads (or downstream attention shifts)
    # may partially re-establish the suppression.
    # ──────────────────────────────────────────────────────────────────
    ("negnm_joint",   ["attn_10_head_7", "attn_11_head_10"]),
]