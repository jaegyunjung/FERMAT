#!/usr/bin/env bash

# Run with `exec bash <this-file>` so the terminal stays open until a key is
# pressed, then closes with the analysis exit status.
#
# CPU-only. No GPU, no checkpoint, no rollout.
# First run: BOOTSTRAP=0 (patient / event counts, ESS, balance).
# Final run: BOOTSTRAP=500 (adds 95% CIs).  e.g.  BOOTSTRAP=500 exec bash <this>

set -uo pipefail

TASK_DIR=/home/khdp-user/workspace/fermat-data/task30
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_DIR=$(dirname "$SCRIPT_DIR")
CONFIG="$CODE_DIR/config/snuh_task30_recency_group_htn_cad.json"
RUNNER="$SCRIPT_DIR/audit_snuh_task30_recency_group_curves.py"

PATHWAY_PATIENTS="$TASK_DIR/outputs/pathway_feasibility_20260717_175059/raw/pathway_patient_level.parquet"
FEATURES=/home/khdp-user/workspace/fermat-data/task19/outputs/baseline_features/baseline_features_20180101.parquet

BOOTSTRAP=${BOOTSTRAP:-0}
RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT="$TASK_DIR/outputs/recency_group_htn_cad_${RUN_ID}"
LOG="$TASK_DIR/logs/recency_group_htn_cad_${RUN_ID}.log"

mkdir -p "$TASK_DIR/logs" "$TASK_DIR/outputs"

(
    set -euo pipefail
    test -f "$CONFIG"
    test -f "$PATHWAY_PATIENTS"
    test -f "$FEATURES"
    python3 -m py_compile "$RUNNER"
    python3 "$RUNNER" --self-test
    python3 "$RUNNER" \
        --config-file "$CONFIG" \
        --pathway-patient-file "$PATHWAY_PATIENTS" \
        --feature-file "$FEATURES" \
        --output-dir "$OUT" \
        --bootstrap-samples "$BOOTSTRAP"
) 2>&1 | tee "$LOG"

STATUS=${PIPESTATUS[0]}
printf '\n## EXIT_STATUS\n%s\n' "$STATUS"
printf '## LOG_FILE\n%s\n' "$LOG"
printf '## OUTPUT_DIR\n%s\n' "$OUT"

if [[ "$STATUS" -ne 0 ]]; then
    printf '\n## LOG_TAIL\n'
    tail -n 150 "$LOG"
elif [[ -f "$OUT/return_summary.txt" ]]; then
    printf '\n## RETURN_SUMMARY\n'
    cat "$OUT/return_summary.txt"
fi

printf '\n아무 키나 누르면 이 셸이 닫힙니다...'
IFS= read -r -n 1 _
printf '\n'
exit "$STATUS"
