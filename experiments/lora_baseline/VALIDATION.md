# Refactor validation — 2026-10-22

Base commit: `b1dd40c753c5aa9aea1d1a67e309dead531c8d9d`.
Protocol: `2026-10-22.5`. The automated tests below use synthetic inputs and
random weights only. Actual CUDA record reproduction is documented separately
in RUN_RESULTS_KO.md; do not confuse it with the synthetic suite. Disease
reproduction remains pending. The full record suite was launched after review;
its start evidence is in RECORD_MAIN_PLAN_KO.md, not proof of completed results.
Bounded LoRA timing is reported separately in LORA_BENCHMARK_KO.md.

## Reproducible checks

Environment: WSL Ubuntu-24.04, PyTorch `2.12.1+cu132`, CPU,
`OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`.

```bash
python -m unittest discover -s experiments/lora_baseline/tests -v
python experiments/lora_baseline/audit_v1.py \
  --reference-dir /path/to/original/scripts --out /tmp/v1_audit.json
```

The suite contains 55 tests, including both task pipelines, gate issuance and
rejection, checkpoint reload in separate processes, unchanged frozen weights,
partial/unequal gradient accumulation, historical seed/order and preprocessing,
eligibility validation, changed data, modified target files, output protection,
wrong-base checkpoint rejection, and import-safe command entry points.
The added regressions cover fixed C=10, rejection of a matching-metric gate
from grid selection/wrong C, saved prefix mapping and cutoff checks, and
refusal to export patient data into Git or overwrite an existing export.
Additional checks cover diagnostic-only gates, cache patient order, aggregate
tracking restrictions, GPU limits, normalization over unlabeled training patients,
original inference-pool mapping/content integrity, and Windows-worktree revision
lookup from WSL. The bounded benchmark integration test verifies actual adapter
updates, unchanged frozen weights, and absence of test outputs/reproduction gates.
Runtime projection tests include the final partial batch and evaluation costs.
The suite passed in 72.174 seconds. New checks include atomic-write/publish
failures, checksum/contract rejection, RNG roundtrip, and uninterrupted versus
resumed head/LoRA training including partial gradient accumulation. Parameters,
optimizer, RNG, epoch metrics and validation predictions agree exactly; a
separate prediction process checks the saved validation reference. Epoch-cap
resume does not execute additional epochs. Tracking's final optimizer update
count agrees with the committed state after both uninterrupted and resumed runs.
The release model configuration is hashed separately from the weights; changing
dropout with unchanged weights changes the resume contract.
These results are from torch 2.12.1, not the other agent's torch 2.14.1 environment.

The reviewed full-suite additions verify experimental optimizer batch override,
rejection of this override for v1/LoRA, resolved update counts, the result integrity
checker (including rejection of a corrupted summary), fixed A/C/LoRA queue order,
matched C/LoRA schedules, and patient-ID pairing for secondary contrasts.

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

This audit checks the earlier grid's primitive fit and prediction functions.
It does not validate grid selection as the published v1 recipe. The final
manuscript's fixed-C10 revision is documented separately in PREPARATION_KO.md.

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
remain an explicit implementation difference. The saved-head audit on all 1,862
test patients found no boundary ties, changed top-10 sets or Recall@10 change.
Record epoch-boundary resume is implemented; use a fresh output folder. Real CUDA
resume/reload evidence is recorded separately in CHECKPOINT_RESUME_KO.md. Risk
resume and arbitrary BF16 batch-context replay remain outside this validation.
