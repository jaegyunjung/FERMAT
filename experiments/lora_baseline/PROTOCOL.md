# LoRA fine-tuning baseline — protocol (drafted 2026-10-22, before any run)

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
| Trainable adapter parameters | 10 × (8·(640+1920) + 8·(640+640)) = **307,200**, 0.47% of 65,221,122 |
| Base dropout | the release value (0.1) stays active in `model.train()` |
| Precision | BF16 autocast for the transformer body on CUDA; heads and losses in FP32 |
| Optimiser | AdamW (β = 0.9, 0.99), gradient clip 1.0, 100 warm-up steps, constant learning rate afterwards |
| Learning rates | adapters 2e-4; head 1e-3 (record), 1e-3 (risk, LoRA arm) |
| Weight decay | 0.01 (record both arms, risk LoRA arm); 1e-5 (risk head-only, as v1) |
| Head input scale | hidden state standardised with the **frozen** model's training-set mean/sd, computed before injection and held fixed, so both arms feed the head on the same scale |
| Audit | after training every pre-injection tensor is checked bit-identical (`assert_base_unchanged`); only adapters and head are saved |

Sensitivity settings, run only after the primary analysis and reported as such:
r = 32 with α = 64; targets `attention_mlp` (adds `mlp.c_fc`, `mlp.c_proj`).

## 4. Task R — items recorded on the first clinical date after the history

Follows manuscript Section 2.3 exactly: the first DX/RX/PX date after the
cutoff, all distinct DX/RX/PX tokens on that date as the target set, Recall@10
averaged over patients as the primary metric, stable outcome-independent tie
order.

| Item | Value |
|---|---|
| Patients | the v1 cohorts: 18,474 training, 1,867 validation, 1,862 test (same patient IDs; their SHA-256 is written to `summary.json`) |
| Prediction vocabulary | the 17,460 DX/RX/PX tokens of the released head (`artifacts/head_classes.npy`) |
| Head | linear 640→17,460, random initialisation, unweighted multilabel BCE (v1 Appendix A) |
| `head_only` recipe | cached frozen embeddings; batch 256; ≤60 epochs; stop after 10 epochs without validation improvement (v1) |
| `lora` recipe | batch 32 prefixes (gradient accumulation allowed); ≤20 epochs; stop after 5 epochs without improvement |
| Selection | validation Recall@10, validation BCE as tie-breaker; the test split is scored once per run |
| Seeds | 1, 2, 3 for both arms; seed 1 is the primary run, all three are reported |
| Secondary | both arms repeated with the fixed 100,000-patient training subset used in the head-scaling evidence (`--train-subset 100000 --subset-seed 20260910`, same subset for both arms) |

Primary contrast: LoRA (seed 1) minus saved generation (50 paths), paired
patient bootstrap with 5,000 resamples on the 1,862 test patients, conditional
on fitted predictors and saved paths (v1 Section 2.4). Secondary contrasts:
LoRA minus `head_only` from this pipeline; LoRA minus the 100k head.

Reproduction gate: `head_only` seed 1 must give test Recall@10 within 1.0
percentage point of the v1 linear heads (55.68–55.98%). If it does not, the
cohort export or target definition differs from v1 and the LoRA result is not
reported until the discrepancy is explained.

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
| Training-set size | primary: a fixed 100,000-patient subset of observed-outcome training patients, **identical for both arms** (`--train-subset 100000`). `head_only` is additionally fitted on the full pool to confirm agreement with v1 |
| Seeds | 1 only (one fit per disease per arm); seeds 2–3 if the primary contrast interval is within 0.01 AUROC of zero |

Primary contrast per disease: LoRA minus representation-based logistic
(`head_only`, same training subset): AUROC difference and Brier difference,
paired patient bootstrap with 2,000 resamples, weights held fixed. This differs
from v1 Section 2.4, which refitted the observation model inside each resample;
the fixed-weight interval is the 2026-10-21 analysis standard. Both can be
reported if the saved per-patient probabilities are run through the v1 tooling.

Reproduction gate: full-pool `head_only` AUROC within 0.01 of v1 (0.871 CKD,
0.752 fatty liver, 0.784 diabetes) and Brier within 0.002. Otherwise, as above.

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
4. No metric from the test split influences any training or selection choice.
   The sensitivity settings (Section 3) are never promoted to primary.
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
4. Confirm whether the Task 31 "direct selection" targets (first ≤8 clinical
   events among the first 64 post-index events) equal the v1 target definition;
   if not, do not place the two numbers in one table.

## 9. Outputs and what enters the repository

Each run writes `summary.json` (arguments, git commit, release model SHA-256,
dataset provenance, training-patient SHA-256, every epoch's validation metrics,
selected epoch, test metrics, parameter counts, frozen-base audit, bootstrap
contrasts), per-patient test CSVs, `test_top10.pt` (record task) and
`trainable_state.pt` (adapters and head only). Only `summary.json` files with
aggregate numbers are committed, under `experiments/lora_baseline/results/`;
per-patient files stay internal, as for every other evidence folder.

## 10. Reporting plan for the next manuscript version

- New results subsection "Supervised fine-tuning" with one table: for each task,
  frozen representation (v1), 100k head (record only), LoRA, generation, with
  paired intervals against generation (record) and against the frozen
  representation (risk).
- Limitation 4 is rewritten from "not evaluated" to the measured outcome.
- The cost paragraph in the Discussion gains the fine-tuning training cost and
  its one-pass inference cost.
- The v1 numbers remain labelled as v1; no number from this protocol replaces a
  v1 number silently.
