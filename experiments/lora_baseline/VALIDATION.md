# Refactor validation — 2026-10-22

Base commit: `b1dd40c753c5aa9aea1d1a67e309dead531c8d9d`.
Protocol: `2026-10-22.3`. No patient records, released weights or GPU training
were used for these checks. Actual v1 reproduction remains pending.

## Reproducible checks

Environment: WSL Ubuntu-24.04, PyTorch `2.12.1+cu132`, CPU,
`OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`.

```bash
python -m unittest discover -s experiments/lora_baseline/tests -v
python experiments/lora_baseline/audit_v1.py \
  --reference-dir /path/to/original/scripts --out /tmp/v1_audit.json
```

The suite contains 33 tests, including both task pipelines, gate issuance and
rejection, checkpoint reload in separate processes, unchanged frozen weights,
partial/unequal gradient accumulation, historical seed/order and preprocessing,
eligibility validation, changed data, modified target files, output protection,
wrong-base checkpoint rejection, and import-safe command entry points.

The original review's weighted gradient (-0.4) and trailing accumulation (-2)
regressions remain covered. The formerly accepted alternate-target gate is now
rejected by both the CLI and the validator. Probability reload checks compare
numeric arrays with absolute tolerance 1e-6 across batch sizes, with exact
patient IDs, labels and weights.

## Independent local source parity

The audit extracts only the original risk preprocessing, fitting and prediction
functions. It uses a synthetic cohort with eligible censored patients, missing
and infinite values, and small standard deviations. For all four C candidates:

- FP32 training features: exact equality.
- Fitted coefficients and intercept: exact equality; maximum error 0.
- Validation probabilities: maximum absolute error `4.440892098500626e-16`.
- Selected C: identical (`0.01` for this synthetic fixture; not a clinical result).

Audited source SHA-256:

| Local source | SHA-256 |
|---|---|
| task30_d3_analysis_core.py | 4a09a88093a59d100876dc0e0cf7a3325dcd13be88e6cfcd08292c7b25f6bc96 |
| train_task30_d3_comparators.py | afccc67876b0820faef0bd48552e4b9d1f60e2ca6a6ceb5122192883b882e46f |

## Before real runs

Export explicit disease-specific eligibility masks; preserve original clinical
and observation-weight precision. Verify cohort IDs, prefixes, vocabulary,
frozen embeddings and outcome definitions against saved v1 artifacts. Run the
historical head-only recipe and gate before any LoRA experiment, then measure
GPU time/memory on a small run. No speed or clinical performance claim follows
from this synthetic validation.

Old gates/checkpoints must be regenerated. The linear record baseline does not
repeat the original linear/MLP architecture search. Stable record ranking ties
are an explicit difference whose effect on the actual cohort is still unknown.
Interrupted-training resume is not implemented; use a fresh output folder.
