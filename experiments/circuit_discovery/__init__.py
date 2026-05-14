"""circuit_discovery — single-forward-pass circuit-finding pipeline.

Mirrors the structure of `experiments.ablation_tracing`:

    discover.py — universal: trace + lens filter + per-prompt JSON output
    validate.py — universal: read discover output, ablate top-K, measure
                  Δ-logit (TODO)
    summarize.py — universal: aggregate per-prompt JSONs into cross-
                   prompt rankings (TODO)
    <task>/      — task-specific config, prompts, and optional translator
                   (e.g. ioi/ adds an IOI-aware translator that maps raw
                   @position to @ROLE for human-readable tables)

Stage outputs feed into one another: discover -> validate -> summarize,
each independently re-runnable.
"""
