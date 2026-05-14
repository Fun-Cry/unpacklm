#!/bin/bash
# Run the Pythia IOI sweeps: cross-scale and cross-checkpoint.
#
# Both sweeps save per-prompt JSONs (every prompt — failures included)
# under results/ioi_sweep/<run_name>/. Filtering by target probability
# happens at analysis time, not capture time.
#
# Run from project root:
#     bash experiments/circuit_discovery/run_pythia_sweeps.sh
#
# Skip-if-exists logic: if a results dir already has run_config.json,
# the sweep skips it. Safe to interrupt and resume.

set -e
set -o pipefail

CACHE=.
N_PROMPTS=100
RESULTS_BASE=results/ioi_sweep
mkdir -p "$RESULTS_BASE"

# ── 1. CROSS-SCALE SWEEP (final step, deduped, varying size) ──────────

cross_scale() {
    SIZE=$1
    RUN_NAME="pythia_${SIZE//./_}_step143000"
    OUT="$RESULTS_BASE/$RUN_NAME"
    if [ -d "$OUT" ] && [ -f "$OUT/run_config.json" ]; then
        echo "=== SKIP $RUN_NAME (already exists) ==="
        return 0
    fi
    echo "=== $RUN_NAME ==="
    python -m experiments.circuit_discovery.run_ioi_sweep \
        --family pythia --size "$SIZE" --step 143000 \
        --deduped 1 --cache-dir "$CACHE" \
        --n-prompts "$N_PROMPTS" \
        --results-dir "$OUT"
}

cross_scale 70m
cross_scale 160m
cross_scale 410m
cross_scale 1b
cross_scale 1.4b
cross_scale 2.8b
cross_scale 6.9b

# ── 2. CROSS-CHECKPOINT SWEEP (one scale, many steps) ────────────────
# Pick the scale where the circuit is clearly present at final step
# but small enough to run many checkpoints quickly. 1.4B is the
# sweet spot.

CKPT_SIZE=1.4b
STEPS="512 1000 4000 16000 32000 64000 143000"

for STEP in $STEPS; do
    RUN_NAME="pythia_${CKPT_SIZE//./_}_step${STEP}"
    OUT="$RESULTS_BASE/$RUN_NAME"
    if [ -d "$OUT" ] && [ -f "$OUT/run_config.json" ]; then
        echo "=== SKIP $RUN_NAME (already exists) ==="
        continue
    fi
    echo "=== $RUN_NAME ==="
    python -m experiments.circuit_discovery.run_ioi_sweep \
        --family pythia --size "$CKPT_SIZE" --step "$STEP" \
        --deduped 1 --cache-dir "$CACHE" \
        --n-prompts "$N_PROMPTS" \
        --results-dir "$OUT"
done

echo
echo "all sweeps done. results under $RESULTS_BASE/"
ls -1 "$RESULTS_BASE/"