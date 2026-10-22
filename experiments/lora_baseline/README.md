# LoRA fine-tuning baseline

Code for the fine-tuning baseline requested in the NeurIPS GenAI4Health review
(reviewer RRnf, meta-review item 4). [PROTOCOL.md](PROTOCOL.md) fixes the design;
read it before running anything on real data.

The local preparation status and ordered execution checklist are in
[PREPARATION_KO.md](PREPARATION_KO.md). Optional saved-artifact exporters require
`pip install -r experiments/lora_baseline/requirements-preparation.txt`.

The scripts import the frozen model code from `releases/fermat_snuh_20261021`
and never modify the release. Weights (`artifacts/model.pt`) and patient data are
not in git; point `--release` at an unpacked collaborator archive.

```
lora.py                    LoRA wrapper, injection, state export, frozen-base audit
check_gate.py              reproduction gate: head_only summary vs protocol_targets.json -> gate.json with run identity
protocol_targets.json      v1 reference values and tolerances per task and disease
predict.py                 rebuild a trained arm from trainable_state.pt and reproduce its outputs
data.py                    dataset layout (documented in the module docstring) and validation
metrics.py                 Recall@K / F1 / precision / hit, weighted AUROC and Brier, paired bootstrap
common.py                  model, accumulation, checkpoint and provenance plumbing
gates.py                   target-content and experiment-identity verification
recipes.py                 audited v1 preprocessing, seed/order and selection
tracking.py                aggregate-only W&B and durable local JSONL (optional)
runtime_guard.py           optional current-boot WSL GPU profile and new host-event checks
benchmarking.py            bounded LoRA step/validation timing, CUDA peaks and runtime projections
audit_v1.py                synthetic parity audit against local original source
audit_record_embeddings.py guarded original-batch/exported-batch CUDA BF16 parity
audit_record_ties.py        saved-head comparison of seeded and stable tie rules
prepare_record_v1.py       verify and export saved record prefixes, labels and 50-path baseline outside Git
prepare_risk_baselines.py  verify published fixed-C10 artifacts and export comparison probabilities outside Git
train_record.py            task R: items on the first clinical date after the history
train_risk.py              task D: five-year disease risk for one disease
make_synthetic_dataset.py  random fixture for pipeline tests (not patient data)
tests/                     unit tests and an end-to-end CPU smoke test, no artifacts needed
```

## Check the pipeline without data or weights

```bash
pip install -r releases/fermat_snuh_20261021/requirements.txt
python -m unittest discover -s experiments/lora_baseline/tests -v
```

## Run on the exported cohorts

```bash
cd experiments/lora_baseline
R=/path/to/unpacked/FERMAT_SNUH_20261021_compared   # holds artifacts/model.pt
D=/path/to/exported/record_dataset                   # data.py layout

# Task R: reproduction arm, gate, then LoRA arm (seed 1)
python train_record.py --release $R --data $D --out runs/record/head_only_v1 --arm head_only --recipe v1 --seed 20260909 --batch-size 32 \
    --train-patient-ids /path/to/v1_record_train_ids.npy --device cuda --bf16
python check_gate.py --summary runs/record/head_only_v1/summary.json --out runs/record/gate.json
python train_record.py --release $R --data $D --out runs/record/lora_s1 --arm lora --seed 1 --gate runs/record/gate.json \
    --train-patient-ids /path/to/v1_record_train_ids.npy --device cuda --bf16 \
    --baseline-recall /path/to/generation50_recall.csv --baseline-label generation50
python predict.py --release $R --data $D --state runs/record/lora_s1/trainable_state.pt --out runs/record/lora_s1_reload --device cuda --bf16

# Task D, one disease at a time: full-pool reproduction, gate, then both arms on the same 100k list
E=/path/to/exported/risk_dataset
python train_risk.py --release $R --data $E --out runs/risk/ckd/head_only_full --arm head_only --disease ckd --device cuda --bf16
python check_gate.py --summary runs/risk/ckd/head_only_full/summary.json --out runs/risk/ckd/gate.json
python train_risk.py --release $R --data $E --out runs/risk/ckd/head_only_100k --arm head_only --recipe experimental --disease ckd \
    --train-patient-ids /path/to/risk_train_100k_ids.npy --device cuda --bf16
python train_risk.py --release $R --data $E --out runs/risk/ckd/lora_s1 --arm lora --disease ckd --seed 1 --gate runs/risk/ckd/gate.json \
    --train-patient-ids /path/to/risk_train_100k_ids.npy --device cuda --bf16 \
    --baseline-probabilities runs/risk/ckd/head_only_100k/test_patient_probabilities.csv --baseline-label head_only
```

Local source comparison is recorded in PROTOCOL.md Section 12. Actual record
CUDA BF16 reproduction passed; see RUN_RESULTS_KO.md and results/record_v1_20261022/.
A 128-step record LoRA timing pilot is complete (LORA_BENCHMARK_KO.md). Disease
reproduction and full LoRA performance experiments remain pending. Synthetic tests alone
do not establish actual v1 reproduction.

Risk exports require boolean `eligible_<disease>.npy` per split. Preprocessing
uses all eligible training patients, including censored patients; optimization
uses selected observed patients. Preserve float64 clinical values/weights from
v1 when available. See data.py for the complete contract.

For new matched record comparisons, run head-only with `--recipe experimental
--seed 1` (also 2 and 3) on the same patient list as LoRA. Only historical
`--recipe v1` runs can issue a real gate. Nonhistorical optimizer overrides need
the experimental label.

Optional original-source audit (requires pandas):

```bash
python audit_v1.py --reference-dir /path/to/original/scripts --out /tmp/v1_audit.json
```

Version .5 restores the original record scaler cohort: all 19,992 training
patients, including patients without a next-record target. Real record exports
must contain `preprocessing_train/`; only 18,474 observed-target patients enter
head optimization. Also supply `embedding_pool/` containing all 23,991 original
inference rows: it preserves the original batch-32 composition, including the
final partial batch. This pool is only used for inference, not for fitting the
scaler. Both pools are bound to the dataset identity. It invalidates .4 gates.
Version .4 fixes risk v1 at C=10 with 200 updates and checks that policy in the
gate. Grid selection requires `--recipe experimental`; it cannot issue a real
gate. Version .3 introduced the corrected preprocessing. Regenerate old
gates/checkpoints. Use an empty output directory per run. Prediction reads
precision/smoke seed from its checkpoint and verifies the base model.
Interrupted-run resume is not implemented. No actual patient-data training
has been performed by these tests.

Every run writes `summary.json`, a per-patient test CSV and `trainable_state.pt`
(adapters, head and preprocessing constants). Synthetic `predict.py` reload is
tested; real BF16 replay across arbitrary batch contexts is not guaranteed.
See RUN_RESULTS_KO.md for the original-pool requirement. Commit only
`summary.json` and `gate.json` files, under
`results/`.

For a source-faithful head diagnostic, use `--recipe diagnostic_v1
--embeddings-cache /path/to/export`; this keeps historical seed/order and
hyperparameters but cannot authorize LoRA. Then run fresh `--recipe v1
--reference-embeddings /path/to/export --batch-size 32`: the reference check
compares every extracted embedding before fitting. Head training remains batch
256. See the preparation report for the expanded-cohort gate restriction.

Optional tracking: install `requirements-tracking.txt`, then pass `--tracking
online --wandb-entity YOUR_ENTITY --run-name YOUR_NAME`. Only explicitly selected
numeric aggregates and non-patient run configuration are sent. Code, console,
automatic metadata/statistics and model/data artifacts are disabled. Local
`tracking/aggregate_history.jsonl` survives connection failures. The local
`tracking.json` distinguishes connection/finish failures; do not equate a local
write with confirmed remote delivery. `--tracking offline` never syncs online.

Before a full record LoRA run, add `--benchmark-steps 128 --benchmark-warmup 10`
to a gated LoRA command. This performs 128 actual optimizer updates, excluding
the first 10 duration samples from steady-speed estimation (the optimizer's
100-step LR warmup is unchanged). It times full validation but does not score
test patients, select a best epoch or issue a gate. Outputs are private
`benchmark.json` and a checkpoint marked `benchmark_only`; full experiments
start afresh. This mode currently requires gradient accumulation 1 and no more
than one epoch of training steps. CUDA peak allocated/reserved bytes are reported
separately from sampled whole-device memory. ETA includes per-epoch validation;
checkpoint/network/bootstrap costs and early stopping are not predicted.
