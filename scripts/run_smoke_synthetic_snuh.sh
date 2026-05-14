#!/usr/bin/env bash
# End-to-end Synthetic SNUH smoke harness for FERMAT.
#
# Behavior:
#   1. If $SYNTHETIC_SNUH_DUCKDB is set and the file exists, run
#      inspect + preprocess on the real Synthetic SNUH DuckDB file.
#   2. Else, fall back to scripts/generate_self_synthetic_4col.py to
#      produce a 4-column synthetic dataset that exercises every
#      TokenType.
#
# Always then:
#   3. Validate bin files (check_fermat_bin.py)
#   4. Compute summary statistics (summarize_fermat_dataset.py)
#   5. Run mapping audit if scripts/audit_fermat_mapping.py exists
#   6. Run smoke training (train.py + config/train_fermat_synthetic_snuh.py)
#
# The old 3-column Delphi regression is not part of the SNUH 4-column sprint.
# It must never make this harness fail.
#
# Usage:
#     SYNTHETIC_SNUH_DUCKDB=data/synthetic_snuh_raw.duckdb \
#       bash scripts/run_smoke_synthetic_snuh.sh
#
#     # Or, fallback path:
#     bash scripts/run_smoke_synthetic_snuh.sh

set -euo pipefail

mkdir -p logs data/synthetic_snuh

DUCKDB="${SYNTHETIC_SNUH_DUCKDB:-data/synthetic_snuh_raw.duckdb}"

if [[ -f "$DUCKDB" ]]; then
  echo "[harness] using real Synthetic SNUH at $DUCKDB"
  echo "[harness] step 1: inspect schema"
  python3 scripts/inspect_synthetic_snuh_schema.py \
    --duckdb "$DUCKDB" \
    --out logs/synthetic_snuh_schema_summary.txt \
    2>&1 | tee logs/01_inspect.log

  echo "[harness] step 2: preprocess to 4-column"
  python3 scripts/preprocess_synthetic_snuh_to_fermat.py \
    --duckdb "$DUCKDB" \
    --out_dir data/synthetic_snuh \
    --vocab_cap 2000 \
    --val_frac 0.1 \
    --seed 42 \
    2>&1 | tee logs/02_preprocess.log
else
  echo "[harness] SYNTHETIC_SNUH_DUCKDB not set or file missing — fallback"
  echo "[harness] step 1+2: generate self-synthetic 4-column dataset"
  python3 scripts/generate_self_synthetic_4col.py \
    --out_dir data/synthetic_snuh \
    --n_patients 2000 \
    --seed 42 \
    2>&1 | tee logs/02_preprocess.log
fi

echo "[harness] step 3: validate bin"
python3 scripts/check_fermat_bin.py \
  --data_dir data/synthetic_snuh \
  --out logs/fermat_bin_check.log \
  2>&1 | tee logs/03_check_bin.log

echo "[harness] step 4: summarize dataset"
python3 scripts/summarize_fermat_dataset.py \
  --data_dir data/synthetic_snuh \
  --out_txt logs/fermat_dataset_summary.txt \
  --out_md logs/fermat_dataset_summary.md \
  2>&1 | tee logs/04_summary.log

if [[ -f scripts/audit_fermat_mapping.py ]]; then
  echo "[harness] step 5: mapping audit"
  python3 scripts/audit_fermat_mapping.py \
    --duckdb "$DUCKDB" \
    --data-dir data/synthetic_snuh \
    --out-md logs/fermat_mapping_audit.md \
    --out-txt logs/fermat_mapping_audit.txt \
    2>&1 | tee logs/05_mapping_audit.log
else
  echo "[harness] step 5 skipped: scripts/audit_fermat_mapping.py not found" \
    | tee logs/05_mapping_audit.log
fi

echo "[harness] step 6: smoke training (4-column Synthetic SNUH)"
python3 train.py config/train_fermat_synthetic_snuh.py \
  --device=cpu \
  2>&1 | tee logs/fermat_synthetic_snuh_smoke_test.log

echo "[harness] optional Delphi regression skipped: not part of the SNUH 4-column sprint" \
  | tee logs/fermat_3col_regression_test.log

echo "[harness] done. SNUH 4-column pipeline succeeded."
