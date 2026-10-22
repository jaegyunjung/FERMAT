# LoRA fine-tuning baseline — protocol (drafted 2026-10-22; amended during reproduction)

Protocol version `2026-10-22.5` (`common.PROTOCOL_VERSION`); gates record it and
a run refuses a gate from another version.

A bounded 128-update record LoRA timing pilot and the reviewed A/C/LoRA record suite
have completed. See RECORD_MAIN_PLAN_KO.md for final results and LORA_BENCHMARK_KO.md
for pilot timings and limits. Version .5 restores
the original record normalization cohort and inference batch context, based on
source-code and saved-artifact comparisons during reproduction diagnostics.
These amendments occurred after the initial diagnostics, not prospectively.
See RUN_RESULTS_KO.md for execution evidence and amendment history. Changes to
the experimental design after observing LoRA results must be labeled exploratory.

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
| Head initialisation | record head: weights N(0, 0.01), bias = logit((positives + 0.5)/(n + 1)) per item; risk head: zero weights, bias = logit of the weighted training prevalence. Both follow the local v1 code (Section 12) and are identical in both arms |
| Loss definitions | record: BCE summed over the 17,460 output items, averaged over patients (v1). Risk: the full-cohort objective Σ w·BCE / Σ w; mini-batches use the fixed training-set mean weight as normaliser so that accumulation over patients recovers exactly that objective, never a per-batch weight sum |
| Learning rates | adapters 2e-4; head 1e-3 (record), 1e-3 (risk, LoRA arm) |
| Weight decay | 0.01 (record both arms, risk LoRA arm); 1e-5 (risk v1 head-only: fixed C=10 under 1e-4/C) |
| Head input scale | fixed frozen-model statistics. Record: all 19,992 original training patients, including those without targets, sd lower bound 1e-4. Risk: all eligible training patients including censored patients, sd>1e-8 else1, finite-value median imputation. Both arms share the task-specific transform |
| Audit | after training every pre-injection tensor is checked bit-identical (`assert_base_unchanged`) |
| Saved state | `trainable_state.pt` holds adapters, head, every standardisation constant, vocabulary and run configuration. Synthetic separate-process reload tests check probabilities at absolute tolerance 1e-6 and record logits at rtol/atol 1e-5, with exact IDs/labels/weights. A bounded real CUDA BF16 LoRA audit matched all 1,867 validation logits and top-10 sets exactly at batch 32 (CHECKPOINT_RESUME_KO.md). Replay must preserve the inference batch context; arbitrary batch-size or split-only replay is not established |

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
| Schedule control C (added before full experiments) | `head_only --recipe experimental --optimizer-batch-size 32`; ≤20 epochs, patience 5, warmup 100, clip 1.0, seeds 1/2/3. A retains the original head schedule at seeds 1/2/3. See RECORD_MAIN_PLAN_KO.md for fixed order, inference boundaries and limitations |
| Selection | validation Recall@10, validation BCE as tie-breaker; the test split is scored once per run |
| Seeds | historical candidates: 20260908, 20260909, 20260910; the saved selected v1 head uses 20260909 (pass explicitly). New comparisons: seeds 1, 2, 3; use `--recipe experimental` for new head-only seeds and `--recipe lora` for LoRA. Only historical recipe runs issue real gates |
| Secondary | both arms repeated with the **same 100,000 patient IDs** used in the head-scaling evidence, loaded from that experiment's saved list via `--train-patient-ids` (a random re-draw is not the same cohort; `--train-subset` refuses a size larger than the pool instead of silently using all patients) |

Primary contrast: LoRA (seed 1) minus saved generation (50 paths), paired
patient bootstrap with 5,000 resamples on the 1,862 test patients, conditional
on fitted predictors and saved paths (v1 Section 2.4). Secondary contrasts:
LoRA minus `head_only` from this pipeline; LoRA minus the 100k head.

Reproduction gate (enforced by code): `head_only --recipe v1 --seed 20260909` must give test
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
| Weights | v1 clipped inverse observation probabilities. Historical arrays were not archived with the supplied manifests; preparation reconstructs them once using the hash-verified original observation-model function and freezes the arrays for all arms. Test AUROC/Brier replay must match the published C10 results within 1e-8. This is not a bitwise comparison to historical weight arrays; full-train C10 refitting remains required. No refitting during training or the fixed-weight bootstrap. |
| Features | four standardised clinical variables followed by 640 hidden components, matching v1 column order; statistics from all eligible train, optimization only on observed selected train |
| `head_only` recipe | cached embeddings, full-batch AdamW, 200 steps, lr .03, fixed C=10 and wd=1e-5; no C selection or early stopping. Earlier grid selection is available only with `--recipe experimental` |
| `lora` recipe | batch 32; ≤3 epochs; the epoch with the lowest validation weighted BCE is selected; stop after 1 epoch without improvement |
| Training-set size | revised before any disease LoRA outcomes on 2026-10-22: use the full original observed training pool in both arms (CKD 148,048; fatty liver 145,782; diabetes 133,947). This replaces the earlier 100,000-patient proposal so the main comparison directly answers improvement over the published predictor. Preprocessing still uses every eligible training patient, including censored patients. Separate disease-specific full-pool gates are required. |
| Seeds | 1, 2, 3 for the LoRA arm of every disease, fixed in advance and all reported; seed 1 is the primary run. The head-only full-batch fit is deterministic given its initialisation and is run once per training set |

Primary contrast per disease: LoRA minus representation-based logistic
(`head_only`, same full training pool): AUROC difference and Brier difference,
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

The record suite is complete; measured runtime is in RECORD_MAIN_PLAN_KO.md.
The risk task now uses 133,947–148,048 observed patients × ≤3 epochs per disease.
The record-pilot-based estimate is 10.6 hours for all nine risk LoRA fits,
or a provisional 12–15 hours including preparation compute and overhead;
see PREPARATION_KO.md Section 5. A disease-specific CUDA pilot must replace this
estimate. Shared, identity-checked frozen embedding caching is planned but is
not yet implemented for the risk trainer.

## 8. Inputs to prepare locally (not in this repository)

1. Export the three splits per task in the `data.py` layout (prefixes exactly as
   fed to the v1 predictors; model IDs = stored ID + 1; padding 0 / −10000).
2. Record task: the CSR target sets and `classes.npy` (or use the release head
   classes). Risk task: `clinical.npy`, `outcome_<disease>.npy`,
   `weight_<disease>.npy` from the v1 observation model and explicit boolean
   `eligible_<disease>.npy`. Eligible censored patients contribute to preprocessing
   with outcome -1 and weight0. Preserve float64 clinical values and weights from
   v1 when available; float32 exports are also accepted.
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

## 9. Gate identity and validation

Version .5 binds task/disease, model hash, precision, protocol version, metadata,
prediction vocabulary/labels, validation/test IDs and selected training IDs.
It also streams a hash of every consumed array across all three splits,
including the complete training pool and eligibility. No patient rows are
copied into the gate. Keep input files immutable during a run.
This reads all training/validation/test input bytes on every invocation, without
an mtime cache. Include this I/O in wall-clock estimates; its measured duration
is recorded as `gate_identity.dataset_hash_seconds` (not an identity constraint).

Official thresholds come from `protocol_targets.json`. Different contents
supplied by either `--targets` or per-metric overrides are custom. On consumption,
the validator verifies the target hash, compares thresholds to the official
table and recomputes each check from finite values. It also requires positive
training updates/selected epoch and `recipe=v1` for real runs. Custom thresholds
are accepted only by synthetic smoke runs. A passed flag alone is insufficient.

A full-pool gate covers subsets only on the identical complete dataset. A
subset gate covers only the same ordered training list. Changed data, precision
or protocol invalidates the gate; old gate files must be regenerated.

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

## 12. Local v1 source audit

Audited 2026-10-22 before any real LoRA run. Code alignment and synthetic parity
do not establish actual cohort reproduction. Version .3 corrects the earlier
implementation and recipe descriptions; it does not alter official gate targets.

| Item | Verified local v1 recipe | Implementation/status |
|---|---|---|
| Record initialization | reset seed before head; N(0,.01); FP32 torch.logit((positive+.5)/(n+1)) | restored |
| Record objective/optimizer | item-summed BCE/patient count; AdamW.001, wd.01, default betas, batch256 | matched for recipe=v1 |
| Record scale | all 19,992 original training patients, including those without a next-record target; sd floor 1e-4 | version .5 requires a separate preprocessing_train pool; optimization remains on 18,474 patients |
| Record randomness | historical seeds20260908/+1/+2; zero-based epoch reseeds torch/CUDA and numpy with seed+epoch*101 | recipes.record_epoch_order |
| Record selection | score>best+1e-10 or abs(delta)<=1e-10 and smaller BCE; patience10, max60 | recipes.record_improved |
| Risk preprocessing cohort | every eligible train patient, including censored outcomes | explicit eligibility mask, independent of observed optimization subset |
| Risk preprocessing | float64 median, nonfinite median->0; nonfinite input->median; sd>1e-8 else1 | shared recipes.fit_risk_scaler/risk_features |
| Risk initialization | zero weights, numpy weighted prevalence bias clipped to[1e-6,1-1e-6] | matched |
| Risk fitting | sum(w*BCE)/sum(w); full batch 200 steps; AdamW.03, default betas, wd=1e-4/C | recipes.fit_risk_candidate |
| Risk regularization | published revision fixed C=10 for all six predictors; diabetes was refitted after the earlier grid | v1 has one C=10 fit; grid selection remains experimental and cannot issue a real gate |
| Risk prediction | FP32 coefficients evaluated in float64, logits clipped[-40,40] | shared recipes.risk_probabilities |
| Head-only clip/warm-up | absent in both originals | absent; incompatible real v1 overrides rejected |
| Record architecture search | original included linear and mlp256, three seeds each | this protocol reproduces the linear baseline, not the full architecture search |
| Record ties | original validation torch.topk, final metrics patient-seeded ties | stable class-order ranking; saved-head audit found 0 changed top-10 sets in 1,862 test patients; no general bound inferred |
| Embedding extraction | original saved embeddings, token mapping, prefixes and precision | record: all 23,991 CUDA BF16 vectors exactly match saved values using original batch context; risk remains pending |
| LoRA settings | no historical counterpart | new protocol |

Source files in the original local checkout: `scripts/run_next_record_bundle.py`,
`scripts/next_record_bundle_common.py`, `scripts/weekend_head_scaling.py`,
`scripts/train_task30_d3_comparators.py`, `scripts/task30_d3_analysis_core.py`.

Run `audit_v1.py --reference-dir /path/to/original/scripts --out /tmp/audit.json`
to compare the original risk preprocessing/fit/predict functions using synthetic
data. It extracts only those functions, records source hashes, and never runs
original main functions or queries patient data. The initial CPU audit found
identical FP32 features and fitted coefficients, maximum probability error
4.44e-16, and identical selected C for the synthetic exploratory grid. This
checks the fitting functions, not the published regularization policy.

The local `FIXED_C10_REVISION_KO.md` and `build/recompute_fixed_c.py` in
`reports/genai4health_paper_20260908/` establish the final recipe: CKD and fatty
liver reused C=10 heads; both diabetes heads were refitted at C=10. The earlier
grid had selected C=.01 for diabetes. The revision explicitly says C=10 was
chosen after examining existing results; do not call it prospectively specified.
Version .4 gates record fixed selection, candidate list [10], selected C=10,
and 200 updates. Matching AUROC/Brier from another C or a grid is insufficient.

Disease cohort export, frozen embedding parity and outcome/weight definitions
remain to be checked. Record execution evidence is in RUN_RESULTS_KO.md. If a real gate fails, explain those differences; never
tune the recipe or thresholds to the test result.

## 13. Migration and operational boundaries

- Export explicit `eligible_*.npy`; missing masks fail. Eligibility must not be
  reconstructed as weight>0.
- Version .2/.3/.4 gates/checkpoints are incompatible. Regenerate them; editing a
  version string is not migration.
- `gates.py` owns evidence checks; `recipes.py` owns historical task recipes;
  `common.py` owns model, accumulation and artifact plumbing. CLI imports have
  no training or argument-parsing side effects.
- Training/prediction refuse nonempty output directories. Use one directory
  per run. Record training can resume from a committed epoch manifest into a new
  directory; optimizer, best state, patience and all RNG streams are restored.
  Data/settings/source/numerical-environment changes are rejected. Risk training
  does not yet support resume. The resume format is versioned independently of
  this statistical protocol; the model recipe and gate thresholds are unchanged.
- Checkpoints identify the frozen model and precision; prediction verifies
  both. Smoke model initialization has a separate seed so changing an experiment
  seed does not silently change the base model.
- Deliberate head-only changes use `--recipe experimental`; they cannot issue
  real reproduction gates. New LoRA runs use `--recipe lora`.

## 14. Diagnostic execution and expanded cohorts

`--recipe diagnostic_v1 --embeddings-cache PATH` uses the historical record
optimizer, seed and epoch order on saved embeddings, but cannot issue a real
gate. Fresh v1 runs can require `--reference-embeddings PATH`: all consumed
embeddings must match at rtol=1e-5, atol=1e-6 before head fitting starts.
Extraction/scoring uses `--batch-size 32`; historical head optimization remains
batch 256. The preprocessing pool is hashed as part of the dataset identity.
The separate `embedding_pool/` preserves all 23,991 original inference rows in
their original order. Extract it once in batches of 32, then map embeddings to
the labeled splits and the 19,992-patient training-only scaler pool. Its inputs
must match all mapped task prefixes exactly and enter the dataset hash. Held-out
rows in this inference pool never enter scaler fitting or optimization.

The 18,474-patient gate does not authorize a new 100k export. Even unchanged
validation/test IDs cannot compensate for a changed training-content hash.
Expanded-cohort runs are deferred until a protocol amendment defines and tests
historical-reference versus new-pool compatibility. Do not bypass the check or
relabel an experimental 100k head as v1. Shared risk embedding caching is also
deferred; its future key must bind model, prefix content, precision and extraction
semantics, independently of disease-specific outcomes and eligibility.

Before a risk gate, require exact float64 agreement of the clinical median,
mean and standard deviation with the saved fixed-C10 transform; FP32 transformed
training features must agree exactly in an independent source comparison, and
saved-model probability replay must have maximum absolute error <=1e-6.
Treat any failure as a preprocessing/identity diagnostic, not a reason to tune
test thresholds. Cold-cache hashing duration must be reported separately from
the existing record warm-cache measurement.
