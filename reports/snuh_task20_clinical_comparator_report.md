# SNUH Clinical Marker Comparator Report

This internal report fixes the first clinical-marker comparator results for the
SNUH-CDM FERMAT downstream disease-risk benchmark. The benchmark asks whether
FERMAT embeddings add predictive information beyond disease-relevant raw LAB
markers and basic pre-index patient features.

Primary setting:

| item | value |
|---|---|
| index date | 2018-01-01 |
| prediction horizon | 5 years |
| phenotypes | chronic kidney disease, diabetes, fatty liver, chronic hepatitis B, hepatocellular carcinoma |
| cohort rule | same incident disease-risk labels as the downstream benchmark |
| split | existing FERMAT train/val/test patient split |
| foundation checkpoint | `ckpt.pt` from the SNUH full-cohort training run |
| embedding | 640-dimensional FERMAT hidden state, last pre-index token |
| clinical markers | raw numeric LAB markers measured before the index date |
| learner | LightGBM 4.6.0 |

The main comparison is `lgbm_fermat_marker_counts` versus
`lgbm_marker_counts`. This asks whether adding FERMAT embeddings improves a
model that already has clinical marker values plus age, sex, and pre-index
utilization counts.

## Executive Summary

FERMAT adds information beyond clinical marker models for several, but not all,
marker-backed disease-risk tasks.

Supported claims:

| Claim | Evidence |
|---|---|
| CKD has the cleanest marker-level result | FERMAT+marker+counts significantly improves AUROC over eGFR/creatinine+counts in all cohorts |
| Fatty liver has broad marker-level improvement | FERMAT+marker+counts significantly improves AUROC, AUPRC, and Brier in all cohorts |
| Diabetes shows marker-level added value in the full eligible and marker-available cohorts | AUROC improves significantly in both cohorts, and all-eligible AUPRC/Brier also improve |
| Chronic hepatitis B benefits most clearly by AUPRC and Brier | AUPRC and Brier improve significantly in all cohorts; AUROC improves significantly only in the full eligible cohort |
| HCC remains unresolved against marker+counts | AUROC and AUPRC improvements are not significant versus marker+counts |

Explicit limits:

| Limitation | Evidence / note |
|---|---|
| HCC should not be claimed as beating the marker model | `marker+counts` versus FERMAT+marker+counts CI crosses 0 for AUROC and AUPRC |
| CKD improvement is primarily AUROC | AUPRC and Brier deltas versus marker+counts are not significant |
| Diabetes recent-marker subgroup is not significant | Recent 2-year HbA1c/glucose cohort has CI crossing 0 |
| Marker availability is not random | Marker-available and recent-marker cohorts are selected clinical subgroups |
| Raw clinical comparator is still limited to selected LAB markers | Cardiovascular clinical scores and AFP-based HCC models are not included yet |

Additional survival-analysis result:

| Claim | Evidence |
|---|---|
| APOLLO-style Cox evaluation supports FERMAT's added value | In full-cohort censored Cox models, the best FERMAT model significantly improves C-index and 5-year AUROC over both count-only and clinical-marker baselines in all five marker-backed phenotypes |
| Cox is strongest for CKD, HCC, and chronic hepatitis B | Best FERMAT Cox C-index is 0.8779 for CKD, 0.8375 for HCC, and 0.8043 for chronic hepatitis B |
| Cox changes the HCC interpretation | Unlike the binary LightGBM marker comparator, Cox survival ranking shows significant FERMAT improvement over liver-marker clinical baseline for HCC |
| The Cox result is a survival sensitivity analysis, not a replacement | Cox includes censored follow-up and ranks time-to-event risk; binary LightGBM remains the fully observed 5-year classification comparator |

## Clinical Markers

The raw LAB marker features were extracted from source numeric measurements
before the index date and summarized as latest value, days since latest value,
min/max/mean/std, one-year and two-year means, and measurement counts.

| phenotype | marker set used for model features | marker used for marker-available cohort |
|---|---|---|
| chronic kidney disease | serum creatinine, eGFR MDRD | eGFR MDRD |
| diabetes | HbA1c, serum glucose | HbA1c |
| fatty liver | AST, ALT, total bilirubin, platelet count | any liver marker |
| chronic hepatitis B | AST, ALT, total bilirubin, platelet count | any liver marker |
| hepatocellular carcinoma | AST, ALT, total bilirubin, platelet count | any liver marker |

The eGFR CKD-EPI marker was not used in the first comparator because its
coverage was too low in the extracted feature table.

## Analysis Cohorts

Each phenotype is evaluated in three cohorts:

| cohort | definition | purpose |
|---|---|---|
| `all_eligible` | all patients eligible for that phenotype's incident disease label | practical setting where FERMAT applies even when a marker is missing |
| `marker_available` | eligible patients with the phenotype-specific marker available | direct marker-versus-FERMAT comparison |
| `marker_recent_2y` | marker-available patients with at least one marker measured within 730 days before index | stricter comparison where marker values are more current |

## Main Results

Comparison: `lgbm_fermat_marker_counts` versus `lgbm_marker_counts`.

### Chronic Kidney Disease

| cohort | metric | marker+counts | FERMAT+marker+counts | delta | 95% CI |
|---|---|---:|---:|---:|---:|
| all eligible | AUROC | 0.893144 | 0.909255 | +0.016112 | +0.009032 to +0.023539 |
| all eligible | AUPRC | 0.239618 | 0.243375 | +0.003757 | -0.017229 to +0.025266 |
| all eligible | Brier | 0.013202 | 0.013189 | -0.000013 | -0.000218 to +0.000198 |
| marker available | AUROC | 0.895241 | 0.909565 | +0.014325 | +0.007524 to +0.021463 |
| marker available | AUPRC | 0.274014 | 0.264281 | -0.009733 | -0.032657 to +0.014311 |
| marker available | Brier | 0.016924 | 0.016991 | +0.000067 | -0.000206 to +0.000348 |
| recent marker | AUROC | 0.898921 | 0.918652 | +0.019731 | +0.012357 to +0.027739 |
| recent marker | AUPRC | 0.298751 | 0.295690 | -0.003061 | -0.032288 to +0.023995 |
| recent marker | Brier | 0.020154 | 0.020107 | -0.000046 | -0.000457 to +0.000393 |

Interpretation: FERMAT significantly improves CKD risk ranking by AUROC beyond
eGFR/creatinine+counts. AUPRC and Brier should be treated as broadly comparable,
not improved.

### Diabetes

| cohort | metric | marker+counts | FERMAT+marker+counts | delta | 95% CI |
|---|---|---:|---:|---:|---:|
| all eligible | AUROC | 0.826964 | 0.834773 | +0.007808 | +0.003106 to +0.012538 |
| all eligible | AUPRC | 0.199304 | 0.212505 | +0.013200 | +0.002526 to +0.023371 |
| all eligible | Brier | 0.030866 | 0.030513 | -0.000353 | -0.000559 to -0.000146 |
| marker available | AUROC | 0.836043 | 0.843309 | +0.007266 | +0.001874 to +0.012653 |
| marker available | AUPRC | 0.273741 | 0.286035 | +0.012294 | -0.009756 to +0.032331 |
| marker available | Brier | 0.048817 | 0.048259 | -0.000558 | -0.001165 to +0.000015 |
| recent marker | AUROC | 0.858440 | 0.861574 | +0.003134 | -0.001976 to +0.008556 |
| recent marker | AUPRC | 0.316233 | 0.316779 | +0.000546 | -0.026947 to +0.027848 |
| recent marker | Brier | 0.052531 | 0.052370 | -0.000161 | -0.001071 to +0.000781 |

Interpretation: FERMAT adds information beyond HbA1c/glucose+counts in the full
eligible and marker-available cohorts by AUROC. In the recent-marker subgroup,
FERMAT is comparable but not significantly better.

### Fatty Liver

| cohort | metric | marker+counts | FERMAT+marker+counts | delta | 95% CI |
|---|---|---:|---:|---:|---:|
| all eligible | AUROC | 0.767505 | 0.784128 | +0.016623 | +0.008244 to +0.025672 |
| all eligible | AUPRC | 0.081948 | 0.099924 | +0.017976 | +0.008082 to +0.028581 |
| all eligible | Brier | 0.019510 | 0.019334 | -0.000176 | -0.000310 to -0.000044 |
| marker available | AUROC | 0.770569 | 0.790978 | +0.020410 | +0.009885 to +0.030649 |
| marker available | AUPRC | 0.089139 | 0.109114 | +0.019975 | +0.009735 to +0.030782 |
| marker available | Brier | 0.020585 | 0.020349 | -0.000236 | -0.000395 to -0.000074 |
| recent marker | AUROC | 0.768068 | 0.796876 | +0.028807 | +0.016769 to +0.041761 |
| recent marker | AUPRC | 0.099387 | 0.125231 | +0.025844 | +0.010486 to +0.041222 |
| recent marker | Brier | 0.024674 | 0.024309 | -0.000365 | -0.000622 to -0.000111 |

Interpretation: Fatty liver is the strongest clinical-comparator result. FERMAT
significantly improves discrimination, positive-case ranking, and Brier error
over liver-marker+counts models in every cohort.

### Chronic Hepatitis B

| cohort | metric | marker+counts | FERMAT+marker+counts | delta | 95% CI |
|---|---|---:|---:|---:|---:|
| all eligible | AUROC | 0.808785 | 0.824673 | +0.015887 | +0.001908 to +0.029453 |
| all eligible | AUPRC | 0.112268 | 0.170038 | +0.057770 | +0.031898 to +0.084838 |
| all eligible | Brier | 0.008092 | 0.007792 | -0.000300 | -0.000466 to -0.000126 |
| marker available | AUROC | 0.829275 | 0.839676 | +0.010402 | -0.003145 to +0.024328 |
| marker available | AUPRC | 0.127673 | 0.192274 | +0.064601 | +0.032172 to +0.097151 |
| marker available | Brier | 0.008500 | 0.008159 | -0.000341 | -0.000574 to -0.000115 |
| recent marker | AUROC | 0.836766 | 0.850542 | +0.013776 | -0.004713 to +0.031086 |
| recent marker | AUPRC | 0.144992 | 0.219666 | +0.074674 | +0.043071 to +0.113434 |
| recent marker | Brier | 0.010847 | 0.010301 | -0.000546 | -0.000860 to -0.000236 |

Interpretation: FERMAT strongly improves AUPRC and Brier beyond liver
marker+counts for chronic hepatitis B. AUROC improvement is significant only in
the full eligible cohort.

### Hepatocellular Carcinoma

| cohort | metric | marker+counts | FERMAT+marker+counts | delta | 95% CI |
|---|---|---:|---:|---:|---:|
| all eligible | AUROC | 0.859968 | 0.847696 | -0.012272 | -0.026494 to +0.000396 |
| all eligible | AUPRC | 0.146133 | 0.158175 | +0.012042 | -0.016591 to +0.039513 |
| all eligible | Brier | 0.006551 | 0.006517 | -0.000034 | -0.000164 to +0.000099 |
| marker available | AUROC | 0.874598 | 0.873111 | -0.001487 | -0.015987 to +0.011373 |
| marker available | AUPRC | 0.160711 | 0.189944 | +0.029233 | -0.006309 to +0.063612 |
| marker available | Brier | 0.007029 | 0.006913 | -0.000117 | -0.000306 to +0.000062 |
| recent marker | AUROC | 0.895736 | 0.898386 | +0.002651 | -0.012333 to +0.017588 |
| recent marker | AUPRC | 0.177315 | 0.198727 | +0.021412 | -0.014281 to +0.056644 |
| recent marker | Brier | 0.009051 | 0.008951 | -0.000100 | -0.000339 to +0.000138 |

Interpretation: HCC should be reported as comparable or unresolved versus
liver-marker+counts, not improved. FERMAT+marker+counts remains much better than
counts-only, but it does not significantly exceed the marker+counts model.

## Counts-Only Reference

Although the main clinical comparator is marker+counts, FERMAT+marker+counts is
also compared against counts-only to show the overall value of marker plus
trajectory information.

| phenotype | cohort | AUROC delta vs counts-only | AUPRC delta vs counts-only | Brier delta vs counts-only |
|---|---|---:|---:|---:|
| chronic hepatitis B | all eligible | +0.094762 | +0.121954 | -0.000627 |
| chronic hepatitis B | marker available | +0.094796 | +0.135338 | -0.000749 |
| chronic hepatitis B | recent marker | +0.080983 | +0.140836 | -0.001012 |
| chronic kidney disease | all eligible | +0.125505 | +0.179521 | -0.001757 |
| chronic kidney disease | marker available | +0.148306 | +0.192283 | -0.002655 |
| chronic kidney disease | recent marker | +0.159045 | +0.217257 | -0.003541 |
| diabetes | all eligible | +0.115858 | +0.138160 | -0.002784 |
| diabetes | marker available | +0.160279 | +0.179059 | -0.006414 |
| diabetes | recent marker | +0.145617 | +0.169476 | -0.007695 |
| fatty liver | all eligible | +0.078543 | +0.044173 | -0.000487 |
| fatty liver | marker available | +0.094404 | +0.050404 | -0.000621 |
| fatty liver | recent marker | +0.101260 | +0.058327 | -0.000866 |
| hepatocellular carcinoma | all eligible | +0.077950 | +0.102609 | -0.000444 |
| hepatocellular carcinoma | marker available | +0.100423 | +0.126428 | -0.000575 |
| hepatocellular carcinoma | recent marker | +0.098765 | +0.109955 | -0.000659 |

All values in this counts-only reference have confidence intervals excluding 0
in the favorable direction. This is not the primary clinical-marker claim, but
it confirms that marker plus trajectory information is much stronger than
demographic and utilization count features alone.

## Best AUROC Model

| phenotype | cohort | best AUROC model | test rows | positives | AUROC | AUPRC | Brier |
|---|---|---|---:|---:|---:|---:|---:|
| chronic hepatitis B | all eligible | FERMAT+marker | 42,439 | 367 | 0.825289 | 0.173053 | 0.007757 |
| chronic hepatitis B | marker available | FERMAT+marker | 35,158 | 323 | 0.844818 | 0.193029 | 0.008156 |
| chronic hepatitis B | recent marker | FERMAT+marker+counts | 21,907 | 259 | 0.850542 | 0.219666 | 0.010301 |
| chronic kidney disease | all eligible | FERMAT+marker+counts | 43,482 | 677 | 0.909255 | 0.243375 | 0.013189 |
| chronic kidney disease | marker available | FERMAT+marker | 28,344 | 583 | 0.913630 | 0.270098 | 0.016949 |
| chronic kidney disease | recent marker | FERMAT+marker+counts | 20,263 | 505 | 0.918652 | 0.295690 | 0.020107 |
| diabetes | all eligible | FERMAT+marker+counts | 39,712 | 1,399 | 0.834773 | 0.212505 | 0.030513 |
| diabetes | marker available | FERMAT+marker+counts | 12,361 | 735 | 0.843309 | 0.286035 | 0.048259 |
| diabetes | recent marker | FERMAT+marker+counts | 6,344 | 424 | 0.861574 | 0.316779 | 0.052370 |
| fatty liver | all eligible | FERMAT+marker+counts | 42,264 | 868 | 0.784128 | 0.099924 | 0.019334 |
| fatty liver | marker available | FERMAT+marker+counts | 35,032 | 762 | 0.790978 | 0.109114 | 0.020349 |
| fatty liver | recent marker | FERMAT+marker+counts | 21,951 | 577 | 0.796876 | 0.125231 | 0.024309 |
| hepatocellular carcinoma | all eligible | marker+counts | 43,201 | 310 | 0.859968 | 0.146133 | 0.006551 |
| hepatocellular carcinoma | marker available | FERMAT+marker | 35,917 | 279 | 0.885622 | 0.167909 | 0.007004 |
| hepatocellular carcinoma | recent marker | FERMAT+marker | 22,604 | 229 | 0.900256 | 0.198238 | 0.008946 |

This table is descriptive. The claim table above should be used for statistical
interpretation because it fixes the baseline and reports bootstrap confidence
intervals.

## Cox Survival Analysis

An APOLLO-style Cox analysis was added as a survival sensitivity analysis. It
uses the same 2018-01-01 index date and five-year horizon, but keeps patients
with censored follow-up rather than requiring complete five-year observation.
The Cox learner is a ridge-penalized linear model fit on fixed feature sets.

Model sets:

| model | features |
|---|---|
| `cox_baseline` | age, sex, pre-index utilization and concept counts |
| `cox_clinical_baseline` | baseline plus phenotype-specific raw LAB marker summaries |
| `cox_fermat_embedding` | FERMAT embedding only |
| `cox_fermat_baseline` | FERMAT embedding plus baseline features |
| `cox_fermat_clinical_baseline` | FERMAT embedding plus baseline and raw LAB marker features |

Primary survival metrics are C-index and five-year AUROC. The bootstrap CIs
below use 200 paired bootstrap samples and should be refreshed to 1000 samples
for final reporting.

### Cox Full-Cohort Results

| phenotype | best Cox model | test rows | events | C-index | 5-year AUROC |
|---|---|---:|---:|---:|---:|
| chronic hepatitis B | FERMAT+clinical baseline | 42,439 | 367 | 0.804290 | 0.805768 |
| chronic kidney disease | FERMAT+clinical baseline | 43,481 | 677 | 0.877910 | 0.880481 |
| diabetes | FERMAT+clinical baseline | 39,712 | 1,402 | 0.790131 | 0.794432 |
| fatty liver | FERMAT+baseline | 42,264 | 868 | 0.753177 | 0.755335 |
| hepatocellular carcinoma | FERMAT embedding | 43,201 | 311 | 0.837534 | 0.837331 |

### Cox Improvement Over Clinical Baseline

Comparison: best FERMAT Cox model versus `cox_clinical_baseline`.

| phenotype | metric | clinical baseline | best FERMAT model | delta | 95% CI |
|---|---|---:|---:|---:|---:|
| chronic hepatitis B | C-index | 0.672060 | 0.804290 | +0.132230 | +0.104782 to +0.163524 |
| chronic hepatitis B | 5-year AUROC | 0.672811 | 0.805768 | +0.132957 | +0.105833 to +0.162731 |
| chronic kidney disease | C-index | 0.843820 | 0.877910 | +0.034090 | +0.020393 to +0.049104 |
| chronic kidney disease | 5-year AUROC | 0.845900 | 0.880481 | +0.034580 | +0.023365 to +0.045570 |
| diabetes | C-index | 0.775278 | 0.790131 | +0.014853 | +0.008603 to +0.021174 |
| diabetes | 5-year AUROC | 0.779444 | 0.794432 | +0.014989 | +0.008011 to +0.023237 |
| fatty liver | C-index | 0.576510 | 0.753177 | +0.176667 | +0.155523 to +0.198930 |
| fatty liver | 5-year AUROC | 0.577117 | 0.755335 | +0.178219 | +0.158971 to +0.196481 |
| hepatocellular carcinoma | C-index | 0.734672 | 0.837534 | +0.102863 | +0.069170 to +0.136746 |
| hepatocellular carcinoma | 5-year AUROC | 0.734329 | 0.837331 | +0.103002 | +0.067787 to +0.132790 |

Interpretation: In censored survival ranking, FERMAT embeddings add signal
beyond clinical marker baselines in all five marker-backed phenotypes. This is
stronger than the binary HCC marker-comparator result and suggests that FERMAT
captures time-to-event risk structure not fully represented by the selected
liver markers.

### Cox Versus Binary Comparator

The Cox and binary analyses answer related but distinct questions.

| analysis | population | target | main use |
|---|---|---|---|
| binary LightGBM | patients with complete five-year label eligibility | five-year incident disease yes/no | direct classification comparator against clinical marker features |
| Cox survival | at-risk patients with censored follow-up retained | time to incident disease or censoring | APOLLO-style survival ranking and censoring-aware sensitivity analysis |

Therefore C-index and binary AUROC should not be compared as identical numbers
across the two analyses. The key question is whether FERMAT improves over the
matched baseline within each analysis. The answer is mixed but favorable in the
binary marker comparator, and uniformly favorable in the Cox survival analysis.

## Claim Boundary

Recommended claim:

> In marker-backed five-year disease-risk tasks, FERMAT embeddings provide
> information complementary to raw clinical LAB markers and basic utilization
> features. The strongest evidence is in CKD and fatty liver, with additional
> support in diabetes and chronic hepatitis B. In binary classification, HCC is
> not yet improved over the marker+counts model; in Cox survival analysis,
> FERMAT significantly improves censored time-to-event ranking for HCC.

More specific supported claims:

| phenotype | supported statement |
|---|---|
| CKD | FERMAT improves AUROC beyond eGFR/creatinine+counts in all cohorts |
| fatty liver | FERMAT improves AUROC, AUPRC, and Brier beyond liver-marker+counts in all cohorts |
| diabetes | FERMAT improves AUROC beyond HbA1c/glucose+counts in all eligible and marker-available cohorts |
| chronic hepatitis B | FERMAT improves AUPRC and Brier beyond liver-marker+counts in all cohorts |
| HCC | FERMAT is comparable to marker+counts; improvement is not established |
| Cox survival analysis | FERMAT improves censored C-index and five-year AUROC over clinical-marker Cox baselines in all five phenotypes |

HCC is not yet improved over the binary marker+counts classifier and should be
treated as unresolved in that setting. The Cox survival result supports a
separate, censoring-aware HCC ranking claim.

Statements to avoid:

| Avoid | Reason |
|---|---|
| FERMAT beats clinical markers across all tasks | HCC is not improved over marker+counts |
| FERMAT beats HbA1c in diabetes | The tested model uses HbA1c/glucose plus FERMAT; HbA1c-only versus FERMAT-only is not isolated here |
| FERMAT improves CKD AUPRC beyond marker models | CKD AUPRC CI crosses 0 |
| FERMAT probabilities are clinically calibrated | This report evaluates discrimination and Brier; probability calibration needs separate review |
| Cox proves calibrated absolute risk | Cox results here focus on C-index and five-year AUROC; calibration bins were generated but not yet reviewed |

## Reproducibility

Pod output directory:

```text
/home/khdp-user/workspace/fermat-data/task20/outputs/clinical_comparator_core5_2018_5y
```

Input artifacts:

| artifact | path |
|---|---|
| labels | `/home/khdp-user/workspace/fermat-data/task19/outputs/patient_phenotype_labels_wide/patient_phenotype_labels_wide_20180101.parquet` |
| age/sex/count features | `/home/khdp-user/workspace/fermat-data/task19/outputs/baseline_features/baseline_features_20180101.parquet` |
| FERMAT embeddings | `/home/khdp-user/workspace/fermat-data/task19/outputs/fermat_embeddings_2018_5y_all/fermat_embeddings_20180101_5y_last.parquet` |
| raw LAB marker features | `/home/khdp-user/workspace/fermat-data/task20/outputs/lab_marker_features/lab_marker_features_wide_20180101.parquet` |

Generated outputs:

| file | purpose |
|---|---|
| `clinical_comparator_metrics.csv` | model-level test metrics |
| `clinical_comparator_bootstrap_delta_ci.csv` | paired bootstrap CI for model deltas |
| `clinical_comparator_test_predictions.parquet` | patient-level test predictions |
| `clinical_comparator_marker_config.csv` | marker mapping used by phenotype |
| `manifest.json` | run configuration and artifact paths |

Cox survival output directory:

```text
/home/khdp-user/workspace/fermat-data/task20/outputs/cox_survival_2018_5y_full_20260629
```

Cox outputs:

| file | purpose |
|---|---|
| `cox_survival_metrics.csv` | Cox model-level C-index and five-year AUROC |
| `cox_survival_bootstrap_delta_ci.csv` | paired bootstrap CI for Cox model deltas |
| `cox_survival_test_predictions.parquet` | patient-level test risk scores and five-year risks |
| `cox_survival_calibration_bins.csv` | decile calibration bins for survival risk review |
| `cox_survival_training_history.csv` | Cox optimization history |
| `manifest.json` | Cox run configuration and artifact paths |

Run configuration:

| item | value |
|---|---|
| LightGBM version | 4.6.0 |
| bootstrap samples | 1000 |
| bootstrap seed | 42 |
| number of estimators | 800 |
| early stopping rounds | 50 |
| recent marker window | 730 days |
| Cox ridge | 0.1 |
| Cox bootstrap samples | 200 |

## Next Step

Before expanding to more clinical scores, the next focused analysis should be a
diabetes-specific decomposition:

| comparison | purpose |
|---|---|
| HbA1c-only versus FERMAT-only | tests whether FERMAT embedding already carries HbA1c-like signal |
| HbA1c/glucose+counts versus FERMAT+HbA1c/glucose+counts | tests incremental value over measured markers |
| recent HbA1c subgroup | tests a fairer comparison against current biomarker values |

This will support a cleaner comparison to Delphi-2M's diabetes discussion,
where biomarker availability was a central limitation.
