# SNUH Task19-25 Current Progress

This note records the current analysis state after the FERMAT-2048 rerun,
Task19 LightGBM evaluation, Task20 marker-backed Cox evaluation, and Task23
31-phenotype Cox evaluation, Task22 bag-of-codes stress test, and Task25
generation proof-of-concept. It is an internal checkpoint for continuing the
next tasks, not a final manuscript section.

## Fixed Result Paths

| analysis | output |
|---|---|
| FERMAT-2048 checkpoint | `/home/khdp-user/workspace/fermat-data/task21/outputs/block2048_full_10l640_20260629/block_2048` |
| FERMAT-2048 Task19 embeddings | `/home/khdp-user/workspace/fermat-data/task21/outputs/fermat_embeddings_2018_5y_block2048_best/fermat_embeddings_20180101_5y_last.parquet` |
| Task20 Cox block512, 1000 bootstrap | `/home/khdp-user/workspace/fermat-data/task20/outputs/cox_survival_2018_5y_block512_1000ci_20260703` |
| Task20 Cox block2048, 1000 bootstrap | `/home/khdp-user/workspace/fermat-data/task20/outputs/cox_survival_2018_5y_block2048_1000ci_20260703` |
| Task23 Cox 31 phenotypes, block2048, 1000 bootstrap | `/home/khdp-user/workspace/fermat-data/task23/outputs/cox_survival_2018_5y_31phenotypes_block2048_1000ci_20260704` |
| Task25 generation final test, FERMAT-2048 | `/home/khdp-user/workspace/fermat-data/task25/outputs/generation_test_final_500p_20260704_v7` |

## 31 Phenotypes

The downstream disease-risk benchmark uses 31 incident phenotype groups at the
2018-01-01 index date and 5-year horizon. The rationale was to select diseases
with enough incident positives for stable testing while spanning chronic
metabolic, cardiovascular, respiratory, liver, cancer, neurologic, urologic,
ophthalmologic, and common procedure-detection phenotypes.

The current 31 phenotype list is:

`asthma`, `atrial_fibrillation`, `benign_prostatic_hyperplasia`,
`breast_cancer`, `cataract`, `chronic_hepatitis_b`,
`chronic_kidney_disease`, `colon_polyp`, `coronary_artery_disease`,
`depression_or_mood_disorder`, `diabetes`, `dyslipidemia`, `epilepsy`,
`fatty_liver`, `gallbladder_polyp`, `gallstone`, `gastritis`,
`gastroesophageal_reflux`, `hearing_loss`, `hepatocellular_carcinoma`,
`hypertension`, `intracranial_aneurysm`, `ischemic_stroke`, `lung_cancer`,
`obesity`, `osteoarthritis`, `osteoporosis_or_osteopenia`,
`overactive_bladder`, `pneumonia`, `retinal_disorder`,
`spinal_stenosis_or_disc`.

## Task19 LightGBM 5-Year Binary Prediction

Primary comparison:

`lgbm_age_sex_counts` versus `lgbm_fermat_embedding_counts`, using
FERMAT-2048 embeddings.

| metric | result |
|---|---:|
| phenotypes | 31 |
| AUROC point estimate improved | 31/31 |
| AUROC significant improvement | 27/31 |
| AUROC significant worsening | 0/31 |
| AUPRC point estimate improved | 31/31 |
| AUPRC significant improvement | 27/31 |
| AUPRC significant worsening | 0/31 |
| Brier point estimate improved | 31/31 |
| Brier significant improvement | 28/31 |
| Brier significant worsening | 0/31 |

The four AUROC non-significant phenotypes were `gallbladder_polyp`,
`intracranial_aneurysm`, `ischemic_stroke`, and `lung_cancer`. All four had
non-negative point-estimate deltas except the Cox sensitivity result for
`lung_cancer` and `ischemic_stroke`, discussed below.

## Task23 Cox 31-Phenotype Time-to-Event Sensitivity

Primary comparison:

`cox_baseline` versus `cox_fermat_baseline`, using FERMAT-2048 embeddings.

| metric | point estimate improved | significant improvement | significant worsening |
|---|---:|---:|---:|
| C-index | 29/31 | 27/31 | 0/31 |
| 5-year AUROC | 29/31 | 27/31 | 0/31 |

Largest C-index gains:

| phenotype | baseline | FERMAT-2048 | delta [95% CI] |
|---|---:|---:|---|
| chronic_hepatitis_b | 0.693 | 0.807 | +0.114 [+0.086, +0.141] |
| chronic_kidney_disease | 0.765 | 0.861 | +0.096 [+0.078, +0.113] |
| epilepsy | 0.666 | 0.760 | +0.094 [+0.063, +0.124] |
| fatty_liver | 0.659 | 0.744 | +0.085 [+0.065, +0.104] |
| hepatocellular_carcinoma | 0.744 | 0.828 | +0.084 [+0.058, +0.108] |
| gallbladder_polyp | 0.627 | 0.710 | +0.082 [+0.054, +0.112] |

The non-significant Cox phenotypes were `intracranial_aneurysm`,
`overactive_bladder`, `lung_cancer`, and `ischemic_stroke`. There was no
significant worsening.

## Task20 Clinical-Marker Cox

Task20 is a separate marker-backed comparison for five phenotypes:
`chronic_kidney_disease`, `diabetes`, `fatty_liver`, `chronic_hepatitis_b`,
and `hepatocellular_carcinoma`.

For the fixed comparison `cox_clinical_baseline` versus
`cox_fermat_clinical_baseline`, FERMAT-2048 significantly improved all five
phenotypes by both C-index and 5-year AUROC with 1000 bootstrap samples.

This is distinct from Task23:

| task | baseline |
|---|---|
| Task20 | clinical marker features plus age/sex/utilization counts |
| Task23 | age/sex/utilization counts only |

## Task22 Bag-of-Codes Baseline

Task22 tests whether the FERMAT downstream gain can be explained by simple
pre-index code frequency. The sparse feature set uses train-only DX/RX/PX token
vocabulary with `min_train_patients=200`, token-type top-K filtering, and a
maximum feature cap of 10,000.

Observed feature profile:

| item | value |
|---|---:|
| rows | 291,901 |
| phenotypes | 31 |
| sparse bag features | 3,600 |
| non-zero entries | 13,491,373 |
| density | 1.28% |
| DX features | 1,154 |
| RX features | 1,801 |
| PX features | 645 |

Model ladder:

| model | features |
|---|---|
| B1 | age + sex + utilization counts |
| F1 | B1 + FERMAT-2048 embedding |
| B2 | B1 + sparse DX/RX/PX bag-of-codes |
| F2 | B2 + FERMAT-2048 embedding |

Bootstrap summary:

| comparison | metric | point improved | significant improved | significant worse | median delta |
|---|---|---:|---:|---:|---:|
| F1 vs B1 | AUROC | 31/31 | 28/31 | 0/31 | +0.0344 |
| F1 vs B1 | AUPRC | 31/31 | 28/31 | 0/31 | +0.0283 |
| B2 vs B1 | AUROC | 31/31 | 30/31 | 0/31 | +0.0465 |
| B2 vs B1 | AUPRC | 31/31 | 30/31 | 0/31 | +0.0444 |
| F2 vs B2 | AUROC | 15/31 | 2/31 | 2/31 | -0.0002 |
| F2 vs B2 | AUPRC | 13/31 | 1/31 | 1/31 | -0.0014 |
| F2 vs B1 | AUROC | 31/31 | 29/31 | 0/31 | +0.0443 |
| F2 vs B1 | AUPRC | 31/31 | 30/31 | 0/31 | +0.0464 |

Primary interpretation:

- FERMAT clearly improves over the original B1 baseline.
- Sparse bag-of-codes is itself a stronger baseline than B1.
- After adding sparse code frequency, FERMAT's incremental gain is small and
  phenotype-specific.
- The main Task19/23 claim remains valid against age/sex/utilization baselines,
  but Task22 shows that much of the broad predictive gain is captured by
  explicit pre-index DX/RX/PX frequency.

For F2 versus B2, AUROC significantly improved only for `dyslipidemia` and
`osteoarthritis`, and significantly worsened for `chronic_hepatitis_b` and
`epilepsy`. AUPRC significantly improved only for
`depression_or_mood_disorder`, and significantly worsened for
`chronic_hepatitis_b`.

## Context Length Sensitivity

FERMAT-2048 was trained to reduce long-history truncation. In the Task19/20
eligible cohort, 42.8% of patients exceed 512 pre-index tokens, while 8.9%
exceed 2048 tokens. Token coverage improves from 37.8% at block512 to 74.6% at
block2048.

The block2048 versus block512 downstream delta is small and phenotype-specific.
It should be reported as context-length sensitivity rather than the main claim.
The main claim is that FERMAT embeddings add value over baseline features.

## Genomics Tokens

Task17 genomics tokens are included in the training data as context-only
information. Coverage is small relative to the full cohort, so current
downstream performance should not be interpreted as evidence for an independent
genomics effect.

## Task25 Generation Proof-of-Concept

Task25 evaluates whether the FERMAT-2048 checkpoint can generate future
clinical tokens beyond a simple train-frequency generator. The final test was
run once on the test split after choosing generation-time calibration parameters
on the validation split.

Final test configuration:

| item | value |
|---|---:|
| split | test |
| seed patients | 500 |
| max generated events per patient | 20 |
| minimum future clinical events | 5 |
| prefix tokens | 512 |
| stratified by clinical rows band | yes |
| repeat policy | same-day repeat penalty 1.0 |
| same-day calibration | temperature 1.5, probability cap 0.90 |
| bootstrap samples | 500 |

The comparison has three sources:

| source | definition |
|---|---|
| real future | observed future DX/RX/PX/DTH events after the prefix |
| FERMAT generation | model-generated DX/RX/PX/DTH tokens and event times from `model.generate()` |
| frequency generation | train-frequency clinical token and gap sampler |

Main distribution results:

| metric | FERMAT vs real | frequency vs real | frequency - FERMAT [95% CI] | winner |
|---|---:|---:|---|---|
| token_id JSD | 0.187 | 0.272 | +0.085 [+0.079, +0.115] | FERMAT |
| token_id total variation | 0.343 | 0.437 | +0.093 [+0.074, +0.121] | FERMAT |
| top-10 token fraction absolute error | 0.0075 | 0.0411 | +0.0336 [+0.0237, +0.0472] | FERMAT |
| consecutive-repeat absolute error | 0.0160 | 0.1116 | +0.0956 [+0.0859, +0.1063] | FERMAT |
| same-day absolute error | 0.0130 | 0.0337 | +0.0207 [-0.0031, +0.0410] | tie |
| gap-bucket JSD | 0.0177 | 0.0047 | -0.0130 [-0.0169, -0.0100] | frequency |
| gap-bucket total variation | 0.0572 | 0.0429 | -0.0142 [-0.0338, +0.0014] | tie |
| token-type JSD | 0.0087 | 0.0033 | -0.0054 [-0.0100, -0.0010] | frequency |
| token-type total variation | 0.1077 | 0.0591 | -0.0485 [-0.0753, -0.0132] | frequency |

Observed source-level summaries:

| source | same-day fraction | repeat rate | top-10 token fraction | age reversal |
|---|---:|---:|---:|---:|
| FERMAT generation | 0.793 | 0.097 | 0.107 | 0.000 |
| frequency generation | 0.840 | 0.0018 | 0.073 | 0.000 |
| real future | 0.806 | 0.113 | 0.114 | 0.000 |

Seed sampling was balanced across history-length bands:

| clinical rows band | seed patients | median patient length | p99 patient length | max patient length |
|---|---:|---:|---:|---:|
| 513-1024 | 125 | 739 | 1,012 | 1,018 |
| 1025-2048 | 125 | 1,386 | 2,026 | 2,032 |
| 2049-4096 | 125 | 2,711 | 4,016 | 4,066 |
| >4096 | 125 | 6,005 | 63,887 | 88,192 |

Task25 conclusion:

- FERMAT shows a statistically supported token-level generative signal:
  generated future clinical token identities are significantly closer to
  observed future tokens than a train-frequency baseline.
- The repeat-rate pathology from the earlier smoke run was corrected by
  replacing hard same-day no-repeat with a soft same-day repeat penalty.
- The same-day fraction is close to observed future records, but the CI for
  same-day absolute error still overlaps zero.
- Full temporal realism is not established. The gap-bucket distribution remains
  weaker than the frequency baseline on JSD, and token-type distribution also
  favors the frequency baseline.

Permissible wording:

> FERMAT generated future clinical token identities that were significantly
> closer to observed future tokens than a train-frequency baseline.

Non-permissible wording at this stage:

> FERMAT is a clinically realistic full temporal patient-trajectory simulator.

## Next Tasks

1. Task24: first-record and utilization sensitivity.
2. Decide whether to keep Task25 as a generation PoC or extend it into a
   separate temporal-calibration study.
