"""Config for the IOI circuit-discovery experiment.

Defaults match the path-recovery analysis that produced the headline
result: 50 IOI prompts on GPT-2 small, β=0.8, large top_paths_k and a
permissive path_min_frac to surface composition routes that fragment
into smaller individual scores.

The lens defaults to 'diversity ≥ 2' — paths visiting at least two
distinct token positions, which is the cross-position-routing
signature of compositional computation. For the targeted IO/S2
follow-up, swap to 'membership' with positions_from_metadata
pointing at the prompts.py-populated target_positions list.

Model is configured here. Swap the `model` block to run on a different
architecture; everything else (prompts, trace settings, lens) is
model-agnostic. Examples below; uncomment one.
"""

CONFIG = {
    # ── GPT-2 small (default) ─────────────────────────────────────
    # "model": {
    #     "family":    "gpt2",
    #     "size":      "small",
    #     "device":    "cuda:0",
    #     "cache_dir": None,
    # },

    # ── Pythia 410M deduped, final step ───────────────────────────
    # "model": {
    #     "family":    "pythia",
    #     "size":      "410m",
    #     "deduped":   True,
    #     "step":      143000,
    #     "device":    "cuda:0",
    #     "cache_dir": ".",
    # },

    # ── Pythia 1.4b deduped, final step ───────────────────────────
    "model": {
        "family":    "pythia",
        "size":      "1.4b",
        "deduped":   True,
        "step":      143000,
        "device":    "cuda:1",
        "cache_dir": ".",
    },

    "trace": {
        "beta":          0.8,
        "top_paths_k":   200,
        # 1e-4 is one order tighter than the runner default; permissive
        # enough to surface low-magnitude composition routes (S-inhibition
        # paths) without producing too much noise. Lower (1e-5) for
        # extra-sensitive recovery; higher (1e-3) for a cleaner top-K.
        "path_min_frac": 1e-4,
    },

    "lens": {
        # type: "length" | "diversity" | "membership"
        "type": "diversity",

        # length-lens param
        "min_hops": 3,

        # diversity-lens param
        "min_positions": 2,

        # membership-lens param: which prompt-metadata field carries the
        # set of integer token positions that the path must touch at
        # least one of. prompts.py populates 'target_positions' with
        # the list [io_position, s2_position] per prompt.
        "positions_from_metadata": "target_positions",
    },
}