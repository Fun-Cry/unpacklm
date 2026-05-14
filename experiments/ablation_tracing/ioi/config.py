"""Config for the IOI self-repair experiment.

Replicates the Wang (2023) / McGrath (2023) positive-circuit
self-repair phenomenon and the McDougall (2023) negative-circuit
copy-suppression unmasking phenomenon on GPT-2 small. The flow,
direct, edge, and path outputs from each cell let the analysis go
beyond direct-attribution descriptions of "which head took over" to
also specify _which upstream edges shifted_ to drive the takeover.

    model    — GPT-2 small. Wang / McGrath / McDougall all reference
               this model so the comparison is direct.

    ablation — mean across ABC references at the target query position
               only (McGrath 2023 do(A^l_t = ã^l_t) convention). Each
               ABC ref shares the target's template, place, and object
               and uses three random names that exclude the target's
               own IO and S — so the duplication signal that drives
               the IOI circuit is fully broken without changing the
               sentence's structural type. Mean over N_ABC_REFS refs
               (see prompts.py) gives the swap a low-variance "without
               this head doing IO" interpretation.

    trace    — beta = 0.8 matches the paper's IOI evaluation (Section
               5; Appendix B sensitivity). Higher β means less
               amplification of cancelling depth-0 splits, which is
               appropriate for IOI where the dominant heads have
               similar magnitudes; small denominator cancellations
               would otherwise amplify noise into the path scores.

    compare  — abl_threshold = 0.05 filters out ablations of heads
               whose clean direct attribution to the IO axis was
               below noise (no role to classify against). The role
               classifier (compensator / doubler / breakage / unclear)
               uses sum_{c in ablated_set} clean.direct[c] as its
               reference signal — the Wang / McGrath / Rushing & Nanda
               convention. A component's role describes how it moved
               relative to the function of the ablated set, not
               relative to delta_p_target. This stays well-defined
               when the model fully recovers and |Δp| ≈ 0, where an
               output-side classifier would have no signal to use.

    output   — full storage tier so post-hoc analyses can re-run
               compare() with new thresholds, and so path / edge data
               is retained for the upstream-routing-change phase of
               the analysis.
"""

CONFIG = {
    "model": {
        "family":    "gpt2",
        "size":      "small",
        "device":    "cuda:1",
        "cache_dir": None,
    },

    "ablation": {
        "mode":      "mean",
        "positions": "target",
    },

    "trace": {
        "beta":                 0.8,
        "top_paths_k":          200,
        "edges_top_k_per_node": 50,
        "path_min_frac":        0.001,
    },

    "compare": {
        "eps_path":        0.001,
        "eps_edge":        0.001,
        "abl_threshold":   0.05,
        "delta_threshold": 0.02,
    },

    "output": {
        "storage": "full",
        "verbose": True,
        "dir":     None,   # None -> <folder>/results/
    },
}