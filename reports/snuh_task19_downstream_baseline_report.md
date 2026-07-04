# SNUH Downstream Disease-Risk Baseline Report

This internal report fixes the first downstream disease-risk benchmark for the
SNUH-CDM FERMAT foundation model. The benchmark asks whether frozen FERMAT
patient embeddings add predictive information for incident disease prediction
beyond basic demographic and pre-index utilization features.

Primary setting:

| item | value |
|---|---|
| index date | 2018-01-01 |
| prediction horizon | 5 years |
| phenotypes | 31 incident disease groups |
| cohort rule | pre-index observation, full follow-up, and no prior target disease |
| split | existing FERMAT train/val/test patient split |
| foundation checkpoint | `ckpt.pt` from the SNUH full-cohort training run |
| embedding | 640-dimensional FERMAT hidden state, last pre-index token |
| context limit | most recent 512 pre-index tokens |

All embeddings are extracted using events strictly before the index date. The
benchmark therefore evaluates prospective disease onset prediction rather than
reconstructing already-observed post-index events.

## Executive Summary

FERMAT embeddings add clinically useful predictive information for broad
five-year disease onset prediction. The strongest and most defensible model is
not embedding-only, but embedding combined with basic patient features
(`embedding+counts`).

Supported claims:

| Claim | Evidence |
|---|---|
| FERMAT embeddings add information beyond age, sex, and pre-index utilization | Against LightGBM age/sex/count baseline, FERMAT+counts improves AUROC in 30/31 and AUPRC in 31/31 phenotypes by point estimate |
| The improvement is statistically stable for most phenotypes | Bootstrap 95% CI supports significant AUROC improvement in 26/31 and AUPRC improvement in 26/31 phenotypes |
| The result is not just a weak-linear-baseline artifact | The main comparison uses LightGBM age/sex/counts as the baseline |
| Probability error improves on average | Brier improves in 27/31 logistic comparisons and 31/31 LightGBM comparisons |
| Raw calibration remains incomplete | ECE improves in only 11/31 phenotypes for both logistic and LightGBM comparisons |

Explicit limits:

| Limitation | Evidence / note |
|---|---|
| Embedding-only is not a replacement for age/sex/count features | LightGBM embedding-only has significant AUROC degradation in 5/31 phenotypes |
| Raw probabilities are not ready as clinical probabilities | Decile ECE is mixed despite improved discrimination and Brier |
| Clinical risk-score and biomarker comparisons are not yet done | HbA1c, eGFR/creatinine, AFP/liver markers, and cardiovascular scores require separate raw-value extraction |
| Modality attribution is not established | Current results do not prove which input source, such as LAB or RX, drives each phenotype gain |
| Long sequences are truncated | Embeddings use the most recent 512 pre-index tokens, which may underuse distant history |

## Benchmark Design

The task predicts whether a patient will develop each disease within 5 years of
2018-01-01. For each phenotype, patients with prior evidence of that disease
before the index date are excluded from that phenotype's eligible set.

The evaluated models are:

| model set | learner | input features | role |
|---|---|---|---|
| `age_sex_counts` | logistic regression | age, sex token, DX/RX/PX counts, unique concepts, active days | interpretable baseline |
| `fermat_embedding` | logistic regression | FERMAT embedding only | embedding-only linear probe |
| `fermat_embedding_counts` | logistic regression | FERMAT embedding + age/sex/counts | main linear FERMAT model |
| `lgbm_age_sex_counts` | LightGBM | age, sex token, DX/RX/PX counts, unique concepts, active days | strong non-linear baseline |
| `lgbm_fermat_embedding` | LightGBM | FERMAT embedding only | embedding-only non-linear probe |
| `lgbm_fermat_embedding_counts` | LightGBM | FERMAT embedding + age/sex/counts | main strong FERMAT model |

The main downstream claim uses `lgbm_fermat_embedding_counts` against
`lgbm_age_sex_counts`.

## Logistic Baseline Results

Comparison: `fermat_embedding_counts` versus logistic `age_sex_counts`.

| metric | result |
|---|---:|
| phenotypes | 31 |
| AUROC point estimate improved | 28/31 |
| AUPRC point estimate improved | 30/31 |
| AUROC bootstrap-CI significant improvement | 25/31 |
| AUPRC bootstrap-CI significant improvement | 27/31 |
| significant degradation | 0 phenotypes |
| median delta AUROC | +0.0422 |
| median delta AUPRC | +0.0377 |

This establishes that FERMAT embeddings add predictive information beyond the
first simple baseline. It is not the final evidence standard because logistic
regression can be criticized as a weak baseline for tabular features.

## LightGBM Baseline Results

Comparison: `lgbm_fermat_embedding_counts` versus `lgbm_age_sex_counts`.

| metric | result |
|---|---:|
| phenotypes | 31 |
| AUROC point estimate improved | 30/31 |
| AUPRC point estimate improved | 31/31 |
| AUROC bootstrap-CI significant improvement | 26/31 |
| AUPRC bootstrap-CI significant improvement | 26/31 |
| AUROC bootstrap-CI significant degradation | 0/31 |
| AUPRC bootstrap-CI significant degradation | 0/31 |
| median delta AUROC | +0.0317 |
| median delta AUPRC | +0.0270 |

This is the primary result. It directly addresses the concern that FERMAT only
beats a weak linear model. Even when age/sex/count features are modeled with a
non-linear LightGBM baseline, adding FERMAT embeddings improves discrimination
for most phenotypes and does not significantly degrade any phenotype.

### Strongest LightGBM Gains

Top phenotypes by AUROC gain for `lgbm_fermat_embedding_counts` versus
`lgbm_age_sex_counts`:

| phenotype | test positives | baseline AUROC | FERMAT+counts AUROC | delta AUROC | baseline AUPRC | FERMAT+counts AUPRC | delta AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| chronic_kidney_disease | 677 | 0.7838 | 0.8681 | +0.0844 | 0.0639 | 0.1618 | +0.0980 |
| chronic_hepatitis_b | 367 | 0.7299 | 0.7970 | +0.0671 | 0.0481 | 0.1730 | +0.1249 |
| epilepsy | 306 | 0.6827 | 0.7495 | +0.0668 | 0.0438 | 0.1232 | +0.0794 |
| hepatocellular_carcinoma | 310 | 0.7697 | 0.8340 | +0.0643 | 0.0556 | 0.1267 | +0.0711 |
| osteoporosis_or_osteopenia | 1,605 | 0.7872 | 0.8421 | +0.0549 | 0.1282 | 0.2207 | +0.0926 |
| depression_or_mood_disorder | 1,156 | 0.7317 | 0.7845 | +0.0528 | 0.0787 | 0.1539 | +0.0752 |
| dyslipidemia | 3,286 | 0.7086 | 0.7568 | +0.0482 | 0.1793 | 0.2291 | +0.0498 |
| diabetes | 1,399 | 0.7189 | 0.7638 | +0.0449 | 0.0743 | 0.1134 | +0.0390 |
| fatty_liver | 868 | 0.7056 | 0.7476 | +0.0420 | 0.0558 | 0.0781 | +0.0223 |
| atrial_fibrillation | 448 | 0.7867 | 0.8285 | +0.0417 | 0.0374 | 0.0751 | +0.0377 |

These gains are concentrated in chronic, progressive, or clinically patterned
conditions. That interpretation is biologically plausible, but modality-level
ablation is required before attributing the gains to specific inputs such as LAB
values, medication exposure, or diagnosis trajectories.

### Weak or Uncertain LightGBM Gains

Lowest phenotypes by AUROC gain:

| phenotype | test positives | baseline AUROC | FERMAT+counts AUROC | delta AUROC | baseline AUPRC | FERMAT+counts AUPRC | delta AUPRC | CI interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| lung_cancer | 268 | 0.7620 | 0.7613 | -0.0006 | 0.0179 | 0.0228 | +0.0049 | AUROC and AUPRC uncertain |
| benign_prostatic_hyperplasia | 882 | 0.9219 | 0.9247 | +0.0028 | 0.1511 | 0.1781 | +0.0270 | AUROC uncertain, AUPRC improved |
| breast_cancer | 430 | 0.8544 | 0.8582 | +0.0038 | 0.0485 | 0.0553 | +0.0068 | AUROC and AUPRC uncertain |
| ischemic_stroke | 289 | 0.7591 | 0.7681 | +0.0091 | 0.0179 | 0.0270 | +0.0091 | AUROC uncertain, AUPRC marginally improved |
| intracranial_aneurysm | 335 | 0.7080 | 0.7193 | +0.0113 | 0.0144 | 0.0174 | +0.0030 | AUROC uncertain, AUPRC improved |
| overactive_bladder | 650 | 0.7873 | 0.8016 | +0.0143 | 0.0624 | 0.0762 | +0.0138 | AUROC and AUPRC improved |

No phenotype shows significant degradation for `lgbm_fermat_embedding_counts`.
For some phenotypes, especially lung cancer and breast cancer, the stronger
LightGBM baseline leaves little or uncertain additional gain.

## Embedding-Only Probe

Comparison: `lgbm_fermat_embedding` versus `lgbm_age_sex_counts`.

| metric | result |
|---|---:|
| phenotypes | 31 |
| AUROC point estimate improved | 22/31 |
| AUPRC point estimate improved | 30/31 |
| AUROC bootstrap-CI significant improvement | 16/31 |
| AUPRC bootstrap-CI significant improvement | 19/31 |
| AUROC bootstrap-CI significant degradation | 5/31 |
| AUPRC bootstrap-CI significant degradation | 1/31 |
| median delta AUROC | +0.0134 |
| median delta AUPRC | +0.0158 |

Embedding-only carries useful signal, especially by AUPRC, but it does not
uniformly replace demographic and utilization features. The final model claim
should therefore center on `embedding+counts`, not embedding-only.

Lowest embedding-only AUROC deltas:

| phenotype | test positives | baseline AUROC | embedding-only AUROC | delta AUROC | baseline AUPRC | embedding-only AUPRC | delta AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| benign_prostatic_hyperplasia | 882 | 0.9219 | 0.8509 | -0.0711 | 0.1511 | 0.1170 | -0.0341 |
| breast_cancer | 430 | 0.8544 | 0.8200 | -0.0344 | 0.0485 | 0.0536 | +0.0051 |
| lung_cancer | 268 | 0.7620 | 0.7277 | -0.0343 | 0.0179 | 0.0225 | +0.0046 |
| intracranial_aneurysm | 335 | 0.7080 | 0.6822 | -0.0257 | 0.0144 | 0.0180 | +0.0036 |
| overactive_bladder | 650 | 0.7873 | 0.7703 | -0.0170 | 0.0624 | 0.0702 | +0.0077 |

This pattern suggests that age, sex, and utilization intensity remain important
explicit covariates for several diseases. FERMAT should be framed as
complementary to those features.

## Calibration

Calibration was evaluated from saved patient-level test predictions. The main
calibration quantities are:

| metric | interpretation |
|---|---|
| Brier score | mean squared error of predicted probability; lower is better |
| calibration-in-large | mean predicted risk minus observed prevalence; closer to 0 is better |
| decile ECE | weighted mean absolute calibration gap across risk deciles; lower is better |
| decile MCE | maximum decile calibration gap; lower is better |

### Aggregate Calibration

| source | model set | phenotypes | median Brier | median ECE | median abs calibration-in-large | max ECE |
|---|---|---:|---:|---:|---:|---:|
| logistic | age_sex_counts | 31 | 0.016741 | 0.001864 | 0.000424 | 0.008592 |
| logistic | fermat_embedding | 31 | 0.016789 | 0.002544 | 0.000485 | 0.005255 |
| logistic | fermat_embedding_counts | 31 | 0.016778 | 0.002439 | 0.000516 | 0.005603 |
| LightGBM | lgbm_age_sex_counts | 31 | 0.016748 | 0.001898 | 0.000524 | 0.005716 |
| LightGBM | lgbm_fermat_embedding | 31 | 0.016745 | 0.002363 | 0.001184 | 0.005208 |
| LightGBM | lgbm_fermat_embedding_counts | 31 | 0.016730 | 0.002016 | 0.001085 | 0.005435 |

### Calibration Change Versus Baseline

| comparison | Brier improved | ECE improved |
|---|---:|---:|
| logistic `fermat_embedding_counts` vs logistic `age_sex_counts` | 27/31 | 11/31 |
| LightGBM `lgbm_fermat_embedding_counts` vs LightGBM `lgbm_age_sex_counts` | 31/31 | 11/31 |

FERMAT improves Brier score across nearly all comparisons, including all
LightGBM phenotypes. Decile calibration is mixed: ECE improves in only 11/31
phenotypes in both logistic and LightGBM settings. Calibration-in-large is
generally small, so average risk is usually close to observed prevalence, but
risk-decile calibration is not uniformly improved.

Calibration claim boundary:

| Supported | Not supported |
|---|---|
| FERMAT improves discrimination and average probability error | FERMAT raw probabilities are fully calibrated |
| Calibration-in-large is generally close to zero | FERMAT improves ECE for most phenotypes |
| Additional probability calibration may be useful | Raw FERMAT probabilities should be used directly as clinical risk without calibration |

## Main Claim Boundary

Recommended claim:

> FERMAT embeddings provide predictive information complementary to age, sex,
> and pre-index utilization features for five-year incident disease prediction.
> Against a strong LightGBM age/sex/count baseline, adding FERMAT embeddings
> improves AUROC in 30/31 and AUPRC in 31/31 phenotypes by point estimate, with
> bootstrap confidence intervals supporting significant improvement in 26/31
> phenotypes for both metrics and no phenotype showing significant degradation.

Recommended calibration qualifier:

> FERMAT improves risk ranking and Brier error, but raw probability calibration
> remains mixed. Additional calibration should be evaluated before using raw
> probabilities as clinical risk estimates.

Recommended embedding-only qualifier:

> FERMAT embeddings alone carry signal, but they do not uniformly replace basic
> demographic and utilization features. The strongest model is the combined
> embedding+counts model.

## Next Clinical Comparator Work

The next benchmark should compare FERMAT against disease-specific clinical
markers or risk scores, in the spirit of Delphi-2M's comparison to existing
clinical risk tools. This is a separate step from the current age/sex/count
benchmark because it requires extracting raw clinical marker values.

Priority comparator candidates:

| priority | phenotype | comparator features | reason |
|---|---|---|---|
| 1 | chronic kidney disease | latest and historical creatinine/eGFR | large FERMAT gain and clinically central biomarker |
| 1 | hepatocellular carcinoma / hepatitis B / fatty liver | AST, ALT, bilirubin, platelet, AFP, HBV-related markers if available | strong FERMAT gains and lab-heavy disease biology |
| 1 | diabetes | HbA1c, fasting glucose, glucose | direct comparison to Delphi-2M's diabetes limitation |
| 2 | cardiovascular disease / stroke | BP, lipids, smoking if available, risk-score components | important but more complex to define for Korean cohorts |

For each clinical comparator, use the same index date, split, eligibility,
washout, and horizon. Compare:

| model | purpose |
|---|---|
| marker-only | disease-specific clinical standard |
| counts-only | existing broad tabular baseline |
| FERMAT-only | whether embedding already carries marker-like information |
| FERMAT+marker | whether FERMAT adds information beyond the marker |
| FERMAT+marker+counts | best combined model |

The diabetes comparator is especially important because FERMAT includes LAB
tokens as pre-index context. LAB tokens are not prediction targets in the SNUH
pretraining setup, but they are present as input context, so the patient
embedding may contain laboratory signal that Delphi-2M-style diagnosis-only
models lack.

## Reproducibility Notes

Pod task directory:

```text
/home/khdp-user/workspace/fermat-data/task19
```

Key generated outputs:

```text
/home/khdp-user/workspace/fermat-data/task19/outputs/patient_phenotype_labels_wide
/home/khdp-user/workspace/fermat-data/task19/outputs/baseline_features
/home/khdp-user/workspace/fermat-data/task19/outputs/fermat_embeddings_2018_5y_all/fermat_embeddings_20180101_5y_last.parquet
/home/khdp-user/workspace/fermat-data/task19/outputs/prediction_ci_2018_5y_all
/home/khdp-user/workspace/fermat-data/task19/outputs/lightgbm_ci_2018_5y_all
/home/khdp-user/workspace/fermat-data/task19/outputs/calibration_summary_2018_5y
```

Main result files:

| output | purpose |
|---|---|
| `prediction_metrics.csv` | logistic model metrics |
| `bootstrap_delta_ci.csv` | logistic delta bootstrap intervals |
| `test_predictions.parquet` | logistic patient-level test predictions |
| `lightgbm_metrics.csv` | LightGBM model metrics |
| `lightgbm_bootstrap_delta_ci.csv` | LightGBM delta bootstrap intervals |
| `lightgbm_test_predictions.parquet` | LightGBM patient-level test predictions |
| `calibration_summary.csv` | Brier, calibration-in-large, ECE, MCE |
| `calibration_bins.csv` | decile calibration bins |

Code artifacts:

| script | role |
|---|---|
| `scripts/build_snuh_task19_phenotype_group_counts.py` | phenotype candidate counting |
| `scripts/build_snuh_task19_patient_labels.py` | incident disease labels |
| `scripts/build_snuh_task19_baseline_features.py` | pre-index count features |
| `scripts/extract_snuh_task19_fermat_embeddings.py` | FERMAT embedding extraction |
| `scripts/run_snuh_task19_baseline_models.py` | logistic count baselines |
| `scripts/run_snuh_task19_embedding_models.py` | logistic FERMAT models |
| `scripts/run_snuh_task19_prediction_ci.py` | patient-level predictions and logistic CIs |
| `scripts/run_snuh_task19_lightgbm_ci.py` | LightGBM predictions and CIs |

