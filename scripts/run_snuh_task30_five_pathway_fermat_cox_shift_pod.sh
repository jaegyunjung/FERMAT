#!/usr/bin/env bash

# Run with `exec bash <this-file>` so the terminal stays open until a key is
# pressed, then closes with the analysis exit status.

set -uo pipefail

TASK_DIR=/home/khdp-user/workspace/fermat-data/task30
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_DIR=$(dirname "$SCRIPT_DIR")
CONFIG="$CODE_DIR/config/snuh_task30_five_pathway_fermat_cox_shift.json"
RANGE_DIR="$TASK_DIR/outputs/continuous_shift_ranges_20260718_075041"
OUT="$TASK_DIR/outputs/five_pathway_fermat_cox_shift_20260718"
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG="$TASK_DIR/logs/five_pathway_fermat_cox_shift_${RUN_ID}.log"
RUNNER="$SCRIPT_DIR/run_snuh_task30_five_pathway_fermat_cox_shift.py"

mkdir -p "$TASK_DIR/logs"

(
    set -euo pipefail
    test -f "$CONFIG"
    test -f "$RANGE_DIR/gpu_continuous_pilot_samples.csv"
    test -f "$RANGE_DIR/raw/patient_source_stable_ranges.parquet"
    python3 -m py_compile "$RUNNER"
    python3 "$RUNNER" --self-test
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python3 "$RUNNER" \
        --config-file "$CONFIG" \
        --range-dir "$RANGE_DIR" \
        --output-dir "$OUT" \
        --resume
) 2>&1 | tee "$LOG"

STATUS=${PIPESTATUS[0]}
printf '\n## EXIT_STATUS\n%s\n' "$STATUS"
printf '## LOG_FILE\n%s\n' "$LOG"
printf '## OUTPUT_DIR\n%s\n' "$OUT"

if [[ "$STATUS" -ne 0 ]]; then
    printf '\n## LOG_TAIL\n'
    tail -n 150 "$LOG"
elif [[ -f "$OUT/RETURN_THIS.txt" ]]; then
    printf '\n## RETURN_THIS\n'
    cat "$OUT/RETURN_THIS.txt"
fi

printf '\n아무 키나 누르면 이 셸이 닫힙니다...'
IFS= read -r -n 1 _
printf '\n'
exit "$STATUS"
