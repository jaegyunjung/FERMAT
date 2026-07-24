#!/usr/bin/env bash
set -euo pipefail

TASK_DIR=/home/khdp-user/workspace/fermat-data/task30
CODE_DIR=${1:?usage: run_snuh_task30_hba1c_adm_ccw_v2_pod.sh CODE_DIR}
REUSE_INPUT_DIR=${REUSE_INPUT_DIR:-$TASK_DIR/outputs/hba1c_adm_ccw_inputs_20260719_v1}
INPUT_DIR_OVERRIDE=${INPUT_DIR_OVERRIDE:-}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-50}
BALANCE_ONLY=${BALANCE_ONLY:-0}
RESUME=${RESUME:-0}
if [[ -n "$INPUT_DIR_OVERRIDE" ]]; then
  INPUT_OUT=$INPUT_DIR_OVERRIDE
else
  INPUT_OUT=$TASK_DIR/outputs/hba1c_adm_ccw_inputs_v3_$RUN_TAG
fi
CCW_OUT=$TASK_DIR/outputs/hba1c_adm_ccw_v2r2_$RUN_TAG
AUDIT_OUT=$TASK_DIR/outputs/hba1c_adm_ccw_v2r2_audit_$RUN_TAG
LOG=$TASK_DIR/logs/hba1c_adm_ccw_v2r2_$RUN_TAG.log

mkdir -p "$TASK_DIR/logs" "$TASK_DIR/outputs"
if [[ -z "$INPUT_DIR_OVERRIDE" && ! -f "$REUSE_INPUT_DIR/raw_private/eligible_person_ids.parquet" ]]; then
  echo "MISSING_REUSABLE_INPUT=$REUSE_INPUT_DIR" >&2
  exit 2
fi

exec > >(tee -a "$LOG") 2>&1

echo "## PREFLIGHT"
python3 "$CODE_DIR/scripts/snuh_task30_adm_ccw_core.py" --self-test
python3 "$CODE_DIR/scripts/extract_snuh_task30_hba1c_adm_ccw_inputs.py" --self-test
python3 "$CODE_DIR/scripts/run_snuh_task30_hba1c_adm_ccw.py" --self-test
python3 "$CODE_DIR/scripts/audit_snuh_task30_hba1c_adm_ccw_results.py" --self-test

echo "## EXTRACT_V3_INPUTS"
if [[ -n "$INPUT_DIR_OVERRIDE" ]]; then
  for required in ccw_patients.parquet ccw_monthly_covariates.parquet covariate_spec.json; do
    if [[ ! -f "$INPUT_OUT/$required" ]]; then
      echo "MISSING_EXISTING_INPUT=$INPUT_OUT/$required" >&2
      exit 3
    fi
  done
  echo "[REUSE] existing completed input without database extraction: $INPUT_OUT"
elif [[ "$RESUME" == 1 && -f "$INPUT_OUT/raw_private/eligible_person_ids.parquet" ]]; then
  python3 "$CODE_DIR/scripts/extract_snuh_task30_hba1c_adm_ccw_inputs.py" \
    --resume \
    --output-dir "$INPUT_OUT"
else
  python3 "$CODE_DIR/scripts/extract_snuh_task30_hba1c_adm_ccw_inputs.py" \
    --checkpoint-reuse-from "$REUSE_INPUT_DIR" \
    --output-dir "$INPUT_OUT"
fi

echo "## FIT_MONTHLY_CCW"
if [[ "$RESUME" == 1 && -f "$CCW_OUT/manifest.json" ]]; then
  echo "[RESUME] completed CCW point estimate reused: $CCW_OUT"
else
  CCW_OVERWRITE=()
  if [[ "$RESUME" == 1 && -d "$CCW_OUT" ]]; then
    CCW_OVERWRITE=(--overwrite)
  fi
  if python3 "$CODE_DIR/scripts/run_snuh_task30_hba1c_adm_ccw.py" \
      --input-dir "$INPUT_OUT" \
      --output-dir "$CCW_OUT" \
      "${CCW_OVERWRITE[@]}"; then
    :
  else
    status=$?
    echo "## STATUS"
    echo "FAILED_CENSOR_MODEL_FIT"
    echo "STATUS=$status"
    echo "INPUT_OUT=$INPUT_OUT"
    echo "CCW_OUT=$CCW_OUT"
    echo "LOG=$LOG"
    if [[ -f "$CCW_OUT/censor_model_diagnostics.csv" ]]; then
      echo "## CENSOR_MODEL_DIAGNOSTICS"
      cat "$CCW_OUT/censor_model_diagnostics.csv"
    fi
    exit "$status"
  fi
fi

echo "## AUDIT"
AUDIT_RESUME=()
if [[ "$RESUME" == 1 && -f "$AUDIT_OUT/bootstrap_full_refit_draws.csv" ]]; then
  AUDIT_RESUME=(--resume-bootstrap)
fi
AUDIT_MODE=()
if [[ "$BALANCE_ONLY" == 1 ]]; then
  AUDIT_MODE=(--balance-only)
  echo "[INFO] corrected balance audit only; completed bootstrap is not rerun"
fi
if python3 "$CODE_DIR/scripts/audit_snuh_task30_hba1c_adm_ccw_results.py" \
    --input-dir "$INPUT_OUT" \
    --ccw-dir "$CCW_OUT" \
    --output-dir "$AUDIT_OUT" \
    --bootstrap-mode full_refit \
    --bootstrap-scope val \
    --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
    --checkpoint-every 10 \
    "${AUDIT_MODE[@]}" \
    "${AUDIT_RESUME[@]}"; then
  :
else
  status=$?
  echo "## STATUS"
  echo "FAILED_AUDIT_OR_BOOTSTRAP"
  echo "STATUS=$status"
  echo "INPUT_OUT=$INPUT_OUT"
  echo "CCW_OUT=$CCW_OUT"
  echo "AUDIT_OUT=$AUDIT_OUT"
  echo "LOG=$LOG"
  if [[ -f "$AUDIT_OUT/bootstrap_full_refit_draws.csv" ]]; then
    echo "BOOTSTRAP_CHECKPOINT=$AUDIT_OUT/bootstrap_full_refit_draws.csv"
  fi
  exit "$status"
fi

echo "## COMPLETE"
echo "INPUT_OUT=$INPUT_OUT"
echo "CCW_OUT=$CCW_OUT"
echo "AUDIT_OUT=$AUDIT_OUT"
echo "LOG=$LOG"
cat "$AUDIT_OUT/return_summary.txt"
