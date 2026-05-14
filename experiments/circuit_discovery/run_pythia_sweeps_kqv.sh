#!/bin/bash
# K+Q+V Pythia IOI sweeps across scales.
#
# Each sweep saves per-prompt JSONs under
#     results/ioi_sweep/<run_name>/
# Filtering by target probability happens at analysis time, not capture
# time.
#
# Run from project root:
#     bash experiments/circuit_discovery/run_pythia_sweeps_kqv.sh
#
# Skip-if-exists logic: if a results dir already has run_config.json,
# the sweep skips it. Safe to interrupt and resume.
#
# Defaults: K+Q+V (no-split, weights 1/1/1). To run an ablation variant,
# add --no-q-side / --no-v-side / --branch-weights to the call below.

set -e
set -o pipefail

CACHE=${CACHE:-/data/s4283341}
N_PROMPTS=${N_PROMPTS:-100}
RESULTS_BASE=${RESULTS_BASE:-/data/s4283341/results/ioi_sweep}
DEVICE=${DEVICE:-cuda:0}

# Sizes to sweep. Override via CLI:
#     bash run_pythia_sweeps_kqv.sh 410m 1.4b
SIZES=( "$@" )
if [ ${#SIZES[@]} -eq 0 ]; then
    SIZES=( 70m 160m 410m 1b 1.4b 2.8b )
fi

mkdir -p "$RESULTS_BASE"

run_one() {
    local SIZE=$1
    local STEP=${2:-143000}
    local TAG="kqv"
    local RUN_NAME="pythia_${SIZE//./_}_step${STEP}_${TAG}"
    local OUT="$RESULTS_BASE/$RUN_NAME"

    if [ -d "$OUT" ] && [ -f "$OUT/run_config.json" ]; then
        echo "=== SKIP $RUN_NAME (already exists) ==="
        return 0
    fi
    echo "=== $RUN_NAME ==="
    python -m experiments.circuit_discovery.run_ioi_sweep \
        --family pythia --size "$SIZE" --step "$STEP" \
        --deduped 1 --cache-dir "$CACHE" \
        --device "$DEVICE" \
        --n-prompts "$N_PROMPTS" \
        --results-dir "$OUT"
}

# ── Cross-scale sweep at final step ───────────────────────────────────
for SIZE in "${SIZES[@]}"; do
    run_one "$SIZE" 143000
done

echo
echo "all sweeps done. results under $RESULTS_BASE/"
ls -1 "$RESULTS_BASE/"
