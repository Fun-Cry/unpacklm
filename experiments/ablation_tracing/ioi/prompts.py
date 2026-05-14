"""Prompts for the IOI self-repair experiment.

Generates IOI prompts plus per-target length-matched ABC references via
``utils.load_data.load_ioi_with_abc``. That helper returns dicts with:

    prompt            : the IOI sentence
                        (e.g. "When Mary and John went to the store,
                         John gave a drink to")
    target_token      : " <IO>" — the answer
    distractor_token  : " <S>"  — used for logit-difference target
                        direction (the standard IOI metric)
    abc_refs          : list of ABC sentences. Each shares the target's
                        template, place, and object; only the names
                        change. The three names are sampled to exclude
                        the target's own IO and S, so the duplication
                        signal the IOI circuit reads is fully broken.
    template_type     : "ABBA" or "BABA"
    IO, S, abc_seed   : carried through as metadata

The single rename below is ``abc_refs -> references`` so the runner
recognizes them; everything else (template_type, IO, S, abc_seed) flows
through as metadata on each saved cell so cross-prompt analysis can
slice by template type, IO/S identity, etc.

ABC references make mean ablation a "without this head doing IO"
counterfactual rather than a "without this head at all" counterfactual:
the head still receives a structurally identical input (a sentence in
the same template with three random names), so non-IOI behavior of the
head is preserved while its IOI-specific behavior is ablated out.

Single-token name verification (inside load_ioi_with_abc) guarantees
that every ABC reference tokenizes to exactly the target's length, which
is required by the position-aware mean-ablation machinery.
"""

from utils.load_data import load_ioi_with_abc


# ──────────────────────────────────────────────────────────────────────
# Knobs.  Tweak here to scale the experiment up or down.
# ──────────────────────────────────────────────────────────────────────

# Number of IOI target prompts. Symmetric ABBA/BABA split, so 100 gives
# roughly 50 of each template type for downstream stratification. 100
# prompts × 7 conditions = 700 cells; well under 20 min on GPU for
# GPT-2 small.
N_PROMPTS = 100

# Pool size for mean ablation per target. Larger = lower-variance
# reference, at proportional cost in build_intervention's reference
# forwards (one forward per ref per condition × prompt). 15 is a
# comfortable middle ground for GPT-2 small.
N_ABC_REFS = 15

# Seed for IOI prompt sampling. ABC references use seed + offset per
# target (see load_ioi_with_abc) so re-running with the same SEED gives
# identical prompts AND identical ABC samples.
SEED = 42


def build_prompts(tokenizer):
    raw = load_ioi_with_abc(
        n_prompts  = N_PROMPTS,
        n_abc_refs = N_ABC_REFS,
        tokenizer  = tokenizer,
        seed       = SEED,
    )

    prompts = []
    for d in raw:
        prompts.append({
            # ── Required / recognized by the runner ──
            "prompt":           d["prompt"],
            "target_token":     d["target_token"],
            "distractor_token": d["distractor_token"],
            "references":       d["abc_refs"],
            # ── Metadata: copied verbatim into each saved cell ──
            "template_type":    d["template_type"],
            "IO":               d["IO"],
            "S":                d["S"],
            "abc_seed":         d["abc_seed"],
        })
    return prompts