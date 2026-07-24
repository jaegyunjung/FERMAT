#!/usr/bin/env bash

set -uo pipefail

TASK_DIR=/home/khdp-user/workspace/fermat-data/task30
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INPUT_DIR="$TASK_DIR/outputs/five_pathway_fermat_cox_shift_20260718"
OUTPUT_DIR="$TASK_DIR/outputs/five_pathway_shift_cpu_analysis_20260718"
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG="$TASK_DIR/logs/five_pathway_shift_cpu_analysis_${RUN_ID}.log"
RUNNER="$SCRIPT_DIR/analyze_snuh_task30_five_pathway_shift_results.py"

mkdir -p "$TASK_DIR/logs"

(
    set -euo pipefail
    test -f "$INPUT_DIR/patient_shift_embeddings_and_risks.parquet"
    python3 -m py_compile "$RUNNER"
    python3 "$RUNNER" --self-test
    python3 "$RUNNER" \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --bootstrap-samples 2000 \
        --overwrite
) 2>&1 | tee "$LOG"

STATUS=${PIPESTATUS[0]}
printf '\n## EXIT_STATUS\n%s\n' "$STATUS"
printf '## LOG_FILE\n%s\n' "$LOG"
printf '## OUTPUT_DIR\n%s\n' "$OUTPUT_DIR"

if [[ "$STATUS" -ne 0 ]]; then
    printf '\n## LOG_TAIL\n'
    tail -n 150 "$LOG"
elif [[ -f "$OUTPUT_DIR/RETURN_THIS.txt" ]]; then
    printf '\n## RETURN_THIS\n'
    cat "$OUTPUT_DIR/RETURN_THIS.txt"
fi

printf '\n아무 키나 누르면 이 셸이 닫힙니다...'
IFS= read -r -n 1 _
printf '\n'
exit "$STATUS"
