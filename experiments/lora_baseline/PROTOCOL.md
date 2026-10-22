# LoRA fine-tuning baseline — protocol (drafted 2026-10-22, before any run)

Protocol version `2026-10-22.2` (`common.PROTOCOL_VERSION`); gates record it and
a run refuses a gate from another version.

**Status: preregistered, not executed.** No LoRA model has been trained on SNUH
data as of this date. Everything below was fixed before the first GPU run; any
later deviation must be recorded in the results document as a protocol change,
and a result obtained after such a change is exploratory.

## 1. Why this experiment exists

The medRxiv v1 manuscript (posted 2026-10-13) compares two uses of one frozen
foundation model: representation-based prediction (frozen last-event hidden
state plus a task head) and generation-based prediction (sampled trajectories).
Reviewer RRnf (comment 3) and the meta-review (weakness 4) asked for supervised
fine-tuning, for example LoRA, as the natural third option. v1 answers this only
as Limitation 4 ("could change the relative performance in either task").

The concern is substantive. In the record task the representation arm is the
weakest form of the method (a linear head on a fixed vector), and strengthening
it already moved the result: the same head trained on 100,000 instead of 18,474
patients rose from 55.68% to 63.42% Recall@10, halving the gap to generation
(71.00%) from 15.31 to 7.58 percentage points. Fine-tuning may close more of it,
or widen the representation advantage in disease risk. Either outcome changes
how the task-dependence claim is worded, so the experiment is run under rules
fixed in advance.

## 2. Arms

| Arm | Model weights | Trained parameters | Role |
|---|---|---|---|
| `head_only` | frozen 93,000-update release | task head only | reproduces the v1 representation arm inside this pipeline; must land close to the v1 numbers before the LoRA arm is interpreted |
| `lora` | frozen base + LoRA adapters | adapters + task head, jointly | the new baseline |
| generation (50 paths) | frozen | none | reused from the saved v1 outputs, not rerun |
| 100k head | frozen | head | reused from the head-scaling evidence, not rerun |
| direct next-token selection | frozen | none | reused from the Task 31 evidence; only after its target definition is confirmed equal to v1's |

The base model is `releases/fermat_snuh_20261021` (stage-2 update 93,000; see
MODEL_CARD.md). The same-day and waiting-time heads are not trained and the
adapted model is **not** used for generation; this protocol compares predictors,
not generators.

## 3. Adaptation specification (fixed)

| Item | Value |
|---|---|
| Method | LoRA (Hu et al., 2021): `y = Wx + (α/r)·BAx`, `B` zero-initialised, `W` frozen |
| Target modules | `attn.c_attn` (fused q/k/v, 640→1920) and `attn.c_proj` (640→640) in all 10 blocks |
| Rank / alpha / dropout | r = 8, α = 16, LoRA dropout 0.05 |
| Adapter parameters | 10 × (8·(640+1920) + 8·(640+640)) = **307,200**, 0.47% of the 65,221,122 frozen base |
| Head parameters, trained in both arms | record task: 17,460 × 641 = **11,191,860**; risk task: 645 |
| Total trained parameters (LoRA arm) | record task **11,499,060**; risk task 307,845. The 0.47% figure describes the adapters only and is never quoted as the total |
| Base dropout | the release value (0.1) stays active in `model.train()` |
| Precision | BF16 autocast for the transformer body on CUDA; heads and losses in FP32 |
| Optimiser | AdamW with PyTorch default betas (0.9, 0.999). LoRA arm: gradient clip 1.0, 100 warm-up steps then constant. Head-only arm: no clipping, no warm-up (v1) |
| Gradient accumulation | gradients of the per-patient *sum* are accumulated and divided by the number of patients actually accumulated, before clipping; the trailing partial accumulation of every epoch is applied with that same normalisation, and an epoch with zero optimizer updates aborts the run. Not applicable to the full-batch head-only risk fit (rejected) |
| Head initialisation | record head: weights N(0, 0.01), bias = logit((positives + 0.5)/(n + 1)) per item; risk head: zero weights, bias = logit of the weighted training prevalence. Both follow the local v1 code (Section 11) and are identical in both arms |
| Loss definitions | record: BCE summed over the 17,460 output items, averaged over patients (v1). Risk: the full-cohort objective Σ w·BCE / Σ w; mini-batches use the fixed training-set mean weight as normaliser so that accumulation over patients recovers exactly that objective, never a per-batch weight sum |
| Learning rates | adapters 2e-4; head 1e-3 (record), 1e-3 (risk, LoRA arm) |
| Weight decay | 0.01 (record both arms, risk LoRA arm); 1e-5 (risk head-only, as v1) |
| Head input scale | hidden state standardised with the **frozen** model's training-set mean/sd (sd floored at 1e-4 as in v1), computed before injection and held fixed, so both arms feed the head on the same scale |
| Audit | after training every pre-injection tensor is checked bit-identical (`assert_base_unchanged`) |
| Saved state | `trainable_state.pt` holds adapters, head, every standardisation constant (embedding mean/sd; clinical medians, mean/sd), the prediction vocabulary and the run configuration. `predict.py` rebuilds the predictor from this file alone in a separate process and must reproduce the training run's test outputs exactly; the test suite checks this |

Sensitivity settings, run only after the primary analysis and reported as such:
r = 32 with α = 64; targets `attention_mlp` (adds `mlp.c_fc`, `mlp.c_proj`).

## 4. Task R — items recorded on the first clinical date after the history

Follows manuscript Section 2.3 exactly: the first DX/RX/PX date after the
cutoff, all distinct DX/RX/PX tokens on that date as the target set, Recall@10
averaged over patients as the primary metric, stable outcome-independent tie
order.

| Item | Value |
|---|---|
| Patients | the v1 cohorts: 18,474 training, 1,867 validation, 1,862 test, supplied as explicit patient-ID lists (`--train-patient-ids`); a listed patient missing from the export aborts the run, and the SHA-256 of the list used is written to `summary.json`. Splits are checked for overlapping patient IDs before training |
| Prediction vocabulary | the 17,460 DX/RX/PX tokens of the released head (`artifacts/head_classes.npy`) |
| Head | linear 640→17,460, random initialisation, unweighted multilabel BCE (v1 Appendix A) |
| `head_only` recipe | cached frozen embeddings; batch 256; ≤60 epochs; stop after 10 epochs without validation improvement (v1) |
| `lora` recipe | batch 32 prefixes (gradient accumulation allowed); ≤20 epochs; stop after 5 epochs without improvement |
| Selection | validation Recall@10, validation BCE as tie-breaker; the test split is scored once per run |
| Seeds | 1, 2, 3 for both arms; seed 1 is the primary run, all three are reported |
| Secondary | both arms repeated with the **same 100,000 patient IDs** used in the head-scaling evidence, loaded from that experiment's saved list via `--train-patient-ids` (a random re-draw is not the same cohort; `--train-subset` refuses a size larger than the pool instead of silently using all patients) |

Primary contrast: LoRA (seed 1) minus saved generation (50 paths), paired
patient bootstrap with 5,000 resamples on the 1,862 test patients, conditional
on fitted predictors and saved paths (v1 Section 2.4). Secondary contrasts:
LoRA minus `head_only` from this pipeline; LoRA minus the 100k head.

Reproduction gate (enforced by code): `head_only` seed 1 must give test
Recall@10 within 1.0 percentage point of the v1 linear heads (55.68–55.98%;
target 55.68 in `protocol_targets.json`). `check_gate.py` writes `gate.json`
carrying the run identity (Section 10); `train_record.py --arm lora` refuses to
start unless a gate passed **and** matches the run. If the gate fails, the cohort export, target definition or
head recipe differs from v1 and no LoRA run is made until the discrepancy is
explained. Before the gate run, the head recipe in this pipeline is diffed
against the local v1 head-training code (bias initialisation, loss reduction,
optimizer settings, early-stopping bookkeeping): Appendix A of v1 fixes the
learning rate, weight decay, batch size, epoch cap and patience, but not those
details, and a failed gate must be attributable to data rather than to a recipe
difference.

## 5. Task D — five-year chronic kidney disease, fatty liver, diabetes

Follows the v1 detailed probability comparison (Section 2.2): weighted logistic
predictor on age, sex, log prior rows, log active days, with or without the
patient representation; observation-weighted AUROC and Brier; disease before
death is a case, death before disease an observed non-case.

| Item | Value |
|---|---|
| Patients | v1 eligible training pool (per disease; 235,670–256,536 eligible), v1 validation pool, the v1 10,000-patient test cohort with its disease-specific eligibility |
| Weights | the v1 clipped inverse observation probabilities, exported as they were used; **not refitted** here |
| Features | standardised hidden state (640) concatenated with the four standardised clinical variables (training-set median imputation, mean/sd scaling) |
| `head_only` recipe | cached frozen embeddings, full-batch AdamW, 200 steps, learning rate 0.03, weight decay 1e-5, no model selection (v1 Appendix A) |
| `lora` recipe | batch 32; ≤3 epochs; the epoch with the lowest validation weighted BCE is selected; stop after 1 epoch without improvement |
| Training-set size | primary: the same 100,000 observed-outcome training patients for both arms, supplied as a patient-ID list. The reproduction gate is the `head_only` fit on the **full** pool (the v1 predictor); a gate fitted on the full pool is accepted for a LoRA run on any list drawn from that pool, and a gate fitted on a subset is accepted only for the identical list |
| Seeds | 1, 2, 3 for the LoRA arm of every disease, fixed in advance and all reported; seed 1 is the primary run. The head-only full-batch fit is deterministic given its initialisation and is run once per training set |

Primary contrast per disease: LoRA minus representation-based logistic
(`head_only`, same training subset): AUROC difference and Brier difference,
paired patient bootstrap with 2,000 resamples, weights held fixed. This differs
from v1 Section 2.4, which refitted the observation model inside each resample;
the fixed-weight interval is the 2026-10-21 analysis standard. Both can be
reported if the saved per-patient probabilities are run through the v1 tooling.

Reproduction gate (enforced by code, per disease): full-pool `head_only` AUROC
within 0.01 of v1 (0.871 CKD, 0.752 fatty liver, 0.784 diabetes) **and** weighted
Brier within 0.002 (0.0204, 0.0273, 0.0452); both checks are required
(`protocol_targets.json`). `train_risk.py --arm lora` refuses a gate for another
task, disease, model, dataset, patient list or label definition, a gate missing
either metric, and a gate built with custom targets. Otherwise, as in Section 4.

## 6. Interpretation rules (fixed)

1. A difference is called only when its paired 95% interval excludes zero.
2. Task R: if LoRA ≥ generation with an interval excluding zero, the v1 claim
   "generation performed better for same-date items" is restated as a
   comparison against the frozen-representation arm only. If generation remains
   better, the advantage is reported against the strongest predictor tried.
3. Task D: if LoRA improves on the frozen representation, the representation
   advantage over generation is reported with the stronger arm; if it does not,
   the frozen head remains the reference and the LoRA result is reported as a
   negative finding with its interval.
4. No metric from the test split influences any training, selection or
   replication choice: the number of seeds is fixed above and does not depend on
   any observed interval. The sensitivity settings (Section 3) are never promoted
   to primary.
5. Cost is reported next to accuracy: GPU-hours per arm, and inference cost per
   patient (one forward pass for both predictor arms versus 50 trajectories).

## 7. Compute and feasibility (estimates, not measured)

One forward-and-backward pass of the 65M model over a 512-event prefix is small;
with batch 32 and BF16 on one RTX 5090, the record task (18,474 patients × ≤20
epochs) is expected to take on the order of one to a few hours per seed, and the
risk task (100,000 patients × ≤3 epochs) a similar order per disease. These are
estimates from model size only and must be replaced by measured wall-clock times
in the results document. The frozen embeddings for `head_only` are cached once
per split.

## 8. Inputs to prepare locally (not in this repository)

1. Export the three splits per task in the `data.py` layout (prefixes exactly as
   fed to the v1 predictors; model IDs = stored ID + 1; padding 0 / −10000).
2. Record task: the CSR target sets and `classes.npy` (or use the release head
   classes). Risk task: `clinical.npy`, `outcome_<disease>.npy`,
   `weight_<disease>.npy` from the v1 observation model.
3. Per-patient Recall@10 of the saved generation arm and of the 100k head as
   `patient_id,recall_at_10` CSVs, and per-patient probabilities of the v1
   representation arm as `patient_id,probability` CSVs, for the paired bootstraps.
   The patient-ID lists of the v1 cohorts and of the 100k head-scaling training
   set as `.npy` files for `--train-patient-ids`.
4. The local v1 head-training code, to diff its recipe against `train_record.py`
   / `train_risk.py` before the gate runs (Section 4).
5. Confirm whether the Task 31 "direct selection" targets (first ≤8 clinical
   events among the first 64 post-index events) equal the v1 target definition;
   if not, do not place the two numbers in one table.

## 9. Gate identity (what a gate must agree on with the run)

`gate.json` records: protocol version; task and disease; whether the model is a
smoke model; the release `model.pt` SHA-256; the SHA-256 of `dataset.json`, of
the validation and test patient-ID arrays, and of the label definition (record:
prediction vocabulary plus validation/test target sets; risk: validation/test
outcome and weight arrays); the training selection mode, size and patient-ID
hash; each checked metric with target, tolerance and value. `validate_gate`
compares all of these with the run before the LoRA arm starts. Targets come from
`protocol_targets.json` only; a gate built with command-line targets is marked
`custom_targets` and is accepted only by smoke runs.

## 10. Outputs and what enters the repository

Each run writes `summary.json` (arguments, git commit, release model SHA-256,
dataset provenance, training-patient SHA-256 and selection mode, the gate it ran
under, every epoch's validation metrics, optimizer update count, selected epoch,
test metrics, adapter/head/total parameter counts, frozen-base audit, bootstrap
contrasts), per-patient test CSVs, `test_scores.pt` (record task) and
`trainable_state.pt` (adapters, head and preprocessing constants). Only `summary.json` files with
aggregate numbers are committed, under `experiments/lora_baseline/results/`;
per-patient files stay internal, as for every other evidence folder.

## 11. Reporting plan for the next manuscript version

- New results subsection "Supervised fine-tuning" with one table: for each task,
  frozen representation (v1), 100k head (record only), LoRA, generation, with
  paired intervals against generation (record) and against the frozen
  representation (risk).
- Limitation 4 is rewritten from "not evaluated" to the measured outcome.
- The cost paragraph in the Discussion gains the fine-tuning training cost and
  its one-pass inference cost.
- The v1 numbers remain labelled as v1; no number from this protocol replaces a
  v1 number silently.

## 12. v1 recipe comparison (to complete locally before the gate runs)

Reference files (local, not in git): `scripts/run_next_record_bundle.py`
(next-record head), `scripts/weekend_head_scaling.py` (head scaling),
`scripts/train_task30_d3_comparators.py` and `scripts/task30_d3_analysis_core.py`
(disease predictors, preprocessing, metrics). Status values: **match**,
**intentional change** (applies to both arms, reported as such), **fix needed**,
**unverified** (must be diffed against the file named; cannot be settled from
the manuscript).

| Item | v1 local code | This branch | Status |
|---|---|---|---|
| Record head weight init | N(0, 0.01) | N(0, 0.01) | match (fixed 2026-10-22) |
| Record head bias init | logit((positives + 0.5)/(n + 1)) | same | match (fixed 2026-10-22) |
| Record training loss | BCE summed over items, mean over patients | same | match (fixed 2026-10-22) |
| Embedding sd floor | max(sd, 1e-4) | max(sd, 1e-4) | match (fixed 2026-10-22) |
| Risk head weight init | zeros | zeros | match (fixed 2026-10-22) |
| Risk objective | weighted BCE, weights clipped IPCW | Σ w·BCE / Σ w over the training set | match in definition; **unverified** whether v1 normalised by Σ w or by n |
| Record head optimizer | AdamW lr 1e-3, wd 0.01, batch 256 (Appendix A) | same; betas (0.9, 0.999) | betas **unverified** |
| Risk head optimizer | AdamW lr 0.03, wd 1e-5, 200 full-batch steps (Appendix A) | same; betas (0.9, 0.999) | betas and whether the fit was full-batch **unverified** |
| Gradient clipping / warm-up (head-only) | — | none / none | **unverified** |
| Embedding extraction precision | CUDA BF16 autocast, FP32 head (MODEL_CARD.md) | `--bf16` autocast on CUDA, FP32 head | match when `--bf16` is set on CUDA |
| Patient order and seeds | — | numpy `default_rng(seed)` permutation per epoch; three seeds | **unverified** (ordering scheme) |
| Tie handling at ranking | patient-seeded tie break (MODEL_CARD.md) | stable class-order tie break (as the released head) | intentional change; expected effect ≤ 0.04 pp (v1 Appendix A: 70.96 vs 71.00) |
| Early stopping / selection | validation Recall@10, BCE tie-break, patience 10, ≤ 60 epochs (Appendix A) | same | match in definition; bookkeeping (strict vs non-strict improvement, evaluation every epoch) **unverified** |
| LoRA-arm settings (clip 1.0, warm-up 100, lr 2e-4, dropout 0.05) | n/a | LoRA arm only | intentional, new |

Rules: every **unverified** row is settled by diffing the named file, and the
result is written into this table with the commit that settles it. A gate run is
made only after no row reads **fix needed**. Settings are never adjusted to make
the gate pass; a failed gate is explained from code and data differences.
