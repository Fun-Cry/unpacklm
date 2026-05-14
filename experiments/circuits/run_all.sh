#!/bin/bash
# Full pipeline: discover → select → verify for all configs × models.
#
# Usage:
#   bash experiments/circuits/run_all.sh --device cuda:0 --cache-dir /local/user
#
#   # Single model:
#   MODEL=gpt2 bash experiments/circuits/run_all.sh --device cuda:0

set -e

DEVICE="${DEVICE:-cuda:0}"
N_PROMPTS="${N_PROMPTS:-100}"
CACHE_DIR="${CACHE_DIR:-.}"
BASE_DIR="${BASE_DIR:-results/circuits}"
DISCOVER_SEED=42
VERIFY_SEED=7

while [[ $# -gt 0 ]]; do
    case $1 in
        --device) DEVICE="$2"; shift 2 ;;
        --n-prompts) N_PROMPTS="$2"; shift 2 ;;
        --cache-dir) CACHE_DIR="$2"; shift 2 ;;
        --base-dir) BASE_DIR="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ -n "$MODEL" ]; then
    MODELS=("$MODEL")
else
    MODELS=("gpt2" "EleutherAI/pythia-160m-deduped")
fi

CONFIGS=(default k_only_l2 k_only_aligned kqv_weighted kqv_l2 kqv_aligned)

echo "============================================================"
echo "  Circuit Pipeline: ${#CONFIGS[@]} configs × ${#MODELS[@]} models"
echo "  discover_seed=$DISCOVER_SEED  verify_seed=$VERIFY_SEED"
echo "  device=$DEVICE  n_prompts=$N_PROMPTS"
echo "============================================================"

for MODEL_NAME in "${MODELS[@]}"; do
    # Short tag for directory names
    if [[ "$MODEL_NAME" == "gpt2" ]]; then
        TAG="gpt2_small"
    else
        TAG=$(echo "$MODEL_NAME" | sed 's|.*/||; s|-deduped||; s|-|_|g')
    fi

    for CONFIG in "${CONFIGS[@]}"; do
        DIR="$BASE_DIR/${TAG}_${CONFIG}"
        echo ""
        echo "── $TAG / $CONFIG ──"

        echo "  [1/3] discover..."
        python -u -m experiments.circuits.discover \
            --model "$MODEL_NAME" --device "$DEVICE" \
            --cache-dir "$CACHE_DIR" --config "$CONFIG" \
            --n-prompts "$N_PROMPTS" --seed "$DISCOVER_SEED" \
            --results-dir "$DIR" --force

        echo "  [2/3] select..."
        python -u -m experiments.circuits.select "$DIR"

        echo "  [3/3] verify..."
        python -u -m experiments.circuits.verify \
            --results-dir "$DIR" \
            --circuit-file "$DIR/circuit.json" \
            --device "$DEVICE" --cache-dir "$CACHE_DIR" \
            --verify-seed "$VERIFY_SEED" \
            --n-verify-prompts 100

        echo "  ✓ $TAG / $CONFIG"
    done
done

# Summary
echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
printf "  %-16s %-18s %4s %8s %8s\n" "Model" "Config" "|C|" "Faith" "KO"
echo "  ---------------------------------------------------------------"
for MODEL_NAME in "${MODELS[@]}"; do
    if [[ "$MODEL_NAME" == "gpt2" ]]; then TAG="gpt2_small"
    else TAG=$(echo "$MODEL_NAME" | sed 's|.*/||; s|-deduped||; s|-|_|g'); fi

    for CONFIG in "${CONFIGS[@]}"; do
        V="$BASE_DIR/${TAG}_${CONFIG}/verification.json"
        if [ -f "$V" ]; then
            python -c "
import json
with open('$V') as f: d = json.load(f)
fr = d.get('faith_ratio') or 0
cd = d.get('comp_drop') or 0
print(f'  $TAG  $CONFIG  {d[\"circuit_size\"]:>3}  {fr:>+.3f}   {cd:>+.3f}')
" 2>/dev/null || echo "  $TAG  $CONFIG  — error"
        fi
    done
done
