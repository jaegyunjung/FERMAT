# SNUH Foundation Pretraining Runbook

## Task 10: smoke comparison

On an L4 GPU pod:

```bash
python scripts/run_snuh_task10.py \
  --data-dir outputs/snuh_tokenization_etl/patient_001pct_seed_42 \
  --output-dir out/snuh-task10
```

The command runs the window audit, baseline arm, LAB-context arm, deterministic
checkpoint evaluation, and comparison report.

Primary output:

```text
out/snuh-task10/reports/comparison.md
```

Use clinical-only metrics for arm comparison. Objective CE uses different
target sets and is not directly comparable.

## Task 11: measurement staging gate

On the CPU/ETL pod:

```bash
python scripts/audit_snuh_measurement_staging.py \
  --patient-buckets 1 \
  --create-indexes \
  --output outputs/snuh_measurement_staging_audit.json
```

Review:

- stage creation and index runtime
- table and index size
- numeric/categorical overlap
- linear full-scale size projection

The projection is only a first estimate. Increase to 5% before adopting a
single-scan staging design for the full ETL.

## Task 12: scaling sweep

Run this only after choosing the baseline or LAB-context target policy:

```bash
python scripts/run_snuh_scaling_sweep.py \
  --data-dir outputs/snuh_tokenization_etl/patient_001pct_seed_42 \
  --policy lab-context \
  --output-dir out/snuh-scaling-lab-context
```

Primary output:

```text
out/snuh-scaling-lab-context/scaling_summary.md
```

Choose a production candidate using validation trend, effective targets per
second, and peak VRAM. The 1% pilot is for relative comparison, not final
perplexity.
