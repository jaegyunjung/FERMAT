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
checkpointing.py           atomic epoch transactions, checksum/contract validation and RNG restoration
audit_record_resume.py     bounded real CUDA resume/reload audit using validation only
audit_risk_resume.py       risk-specific CUDA resume/reload audit, 128 patients and validation only
run_risk_preflight.py      C10 gates -> resume audit -> bounded timing pilots; never main LoRA training
risk_cache.py              original-batch CUDA parity and verified shared risk embeddings
run_record_suite.py        fixed A/C/LoRA seed queue with artifact checks and fail-stop behavior
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
prepare_risk_outcomes.py   original-model replay and private clinical/outcome/weight preparation
prepare_risk_prefixes.py   source-hash verified prefixes and original 64-patient extraction context
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

# Task D: bounded preparation only, using the full original cohort for each disease.
E=/path/to/exported/risk_dataset
K=/private/new_risk_cache
P=/path/to/current_boot_gpu_profile.json
M=/private/original_fixed_c10/models
python risk_cache.py --release $R --data $E --out $K --arm head_only --batch-size 64 \
    --device cuda --bf16 --gpu-profile $P --tracking online --wandb-entity YOUR_ENTITY
python run_risk_preflight.py --release $R --data $E --embedding-cache $K --fixed-models $M \
    --gpu-profile $P --out /private/new_risk_preflight --wandb-entity YOUR_ENTITY
# Review the preflight report and measured ETA before starting the main disease experiment.
```

Local source comparison is recorded in PROTOCOL.md Section 12. Actual record
CUDA BF16 reproduction passed; see RUN_RESULTS_KO.md and results/record_v1_20261022/.
A 128-step record LoRA timing pilot and all nine A/C/LoRA record experiments are
complete. See RECORD_MAIN_PLAN_KO.md and results/record_main_20261022/ for final
results and an independent bootstrap audit. All three disease C10 reproduction
gates passed, and all 329,583 CUDA BF16 embeddings exactly match the originals.
Disease main C/LoRA runs started on 2026-10-22 at 21:31 KST; the 100k expansion remains pending. Risk runtime
additions pass 59 CPU tests; measured preparation results are documented in
PREPARATION_KO.md Section 6. Synthetic tests
alone do not establish v1 reproduction.

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
Both record and risk training support epoch-boundary resume into a new output
directory. Actual CUDA risk uninterrupted/resumed states and independently
reloaded validation probabilities agree exactly. See CHECKPOINT_RESUME_KO.md.
Synthetic tests alone do not establish real-data training or replay correctness.

Completed full runs write `summary.json`, a per-patient test CSV and `trainable_state.pt`
(adapters, head and preprocessing constants). Paused, benchmark and validation-only
runs have their own outputs and do not score test patients. Real CUDA BF16 LoRA
resume and separate-process reload matched exactly on all 1,867 validation patients
at batch 32; see CHECKPOINT_RESUME_KO.md. Replay across arbitrary batch contexts is
not guaranteed.
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

Record resume uses `--resume /prior/run/last_training.json`, with unchanged data,
recipe, seed, epoch cap, precision and training settings, and a new `--out`.
`--stop-after-epoch N` commits an epoch and exits before test evaluation. Each
epoch writes an immutable `checkpoints/epoch_NNNN.pt`; the atomic manifest with
SHA-256 is authoritative, and `best_model.pt` is a convenience inference copy.
An interrupted epoch is replayed from the last committed epoch. Old final
checkpoints and benchmark-only checkpoints do not contain resumable optimizer
state. Training source hashes and numerical environment must match; documentation
changes alone do not invalidate resume.

Use `--validation-only` with LoRA or experimental head-only for bounded diagnostics.
It never evaluates test or issues a gate, and cannot be resumed as a full run.
`predict.py --split validation --reference-scores RUN/validation_scores.pt`
compares independent reload with the live-model scores at the same batch size.

For the reviewed full record suite, see RECORD_MAIN_PLAN_KO.md. Experimental
head-only supports `--optimizer-batch-size` independently of extraction/scoring
batch size. Historical v1 batch 256 is unchanged. Freeze the execution checkout
before starting; do not modify it during the run or resume from changed sources.

Disease queue: `run_risk_suite.py` supervises a separate frozen training checkout,
C nine runs then LoRA seeds 1/2/3 across three diseases. See PREPARATION_KO.md
Section 7 and results/risk_main_20261022/launch.json. A launch record is not a
completed result; live status stays in the private output directory.
