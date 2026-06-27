# SNUH Full-Cohort Baseline Report

This internal report fixes the current full-cohort SNUH-CDM baseline result for
FERMAT. It uses the capped 20,000-patient test evaluation produced from the
SNUH full-cohort training run with GENOMICS context tokens.

The primary checkpoint is `objective_saved_best`, the validation-selected
`ckpt.pt` at step 83,000. All main claims below use this single checkpoint
unless explicitly stated otherwise.

## Executive Summary

The full-cohort SNUH-CDM trajectory transformer baseline is established. The
model provides meaningful frequent-token clinical candidate generation,
same-day event grouping, and short- to medium-horizon timing signal. The result
should not be read as rare-code prediction, mortality modeling, or long-horizon
point-time prediction.

Supported claims:

| Claim | Evidence |
|---|---|
| Full-cohort trajectory baseline is established | Primary checkpoint clinical top-1 `0.3353`, top-5 `0.6509`, top-10 `0.7400` |
| Frequent clinical event candidate generation is meaningful | Head-token top-5 `0.6820`; DX top-1 `0.4107`; RX/PX top-10 `0.7366`/`0.7590` |
| Same-day event grouping is strong | Same-day AUROC `0.8033`, AUPRC `0.9121`, Brier `0.1137` |
| Timing signal is meaningful within 1 year | Median absolute error: `3.35` days for 1-7 days, `13.47` for 8-30 days, `27.74` for 31-90 days, `68.94` for 91-365 days |

Explicit limits:

| Limitation | Evidence / note |
|---|---|
| Tail and rare exact-token prediction remain weak | Tail top-1 `0.0054`; rare top-1/top-5 `0.0000` |
| DTH modeling is not supported as a claim | DTH target count is only `508` |
| Long-gap timing is not reliable as point prediction | Over-365-day median AE `3475.67` days; likely long-gap/censoring-like targets |
| Complex patients are harder | Performance falls with age, visit density, and 1024+ sequence length |
| Calendar-year stratification is unavailable | Current age-only binary events require an event-date sidecar |

## Checkpoint Robustness

Checkpoint selection is not a major driver of the observed conclusions. The
post-hoc metric-specific checkpoints differ only modestly, so reporting one
primary checkpoint avoids metric shopping. The selected primary checkpoint is
the saved validation-selected best checkpoint, `ckpt.pt` at step 83,000.

| label | step | checkpoint | selection criterion |
|---|---:|---|---|
| objective_best | 70,000 | `ckpt_70000.pt` | lowest stored validation objective among periodic checkpoints |
| ce_best | 70,000 | `ckpt_70000.pt` | lowest stored validation CE among periodic checkpoints |
| time_nll_best | 60,000 | `ckpt_60000.pt` | lowest stored validation different-day loss among periodic checkpoints |
| same_day_best | 80,000 | `ckpt_80000.pt` | lowest stored validation same-day loss among periodic checkpoints |
| objective_saved_best | 83,000 | `ckpt.pt` | saved validation-selected best checkpoint |
| latest | 86,000 | `ckpt_latest.pt` | latest checkpoint in the original run directory |

| label | clinical CE | top1 | top5 | time NLL | time MAE days | same-day AUROC | same-day Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| objective_best | 3.2320 | 0.3323 | 0.6457 | 5.0467 | 525.59 | 0.7920 | 0.1186 |
| ce_best | 3.2320 | 0.3323 | 0.6457 | 5.0467 | 525.59 | 0.7920 | 0.1186 |
| time_nll_best | 3.3168 | 0.3175 | 0.6304 | 5.0479 | 515.82 | 0.7926 | 0.1180 |
| same_day_best | 3.1967 | 0.3373 | 0.6520 | 5.0122 | 526.20 | 0.7952 | 0.1154 |
| objective_saved_best | 3.2038 | 0.3353 | 0.6509 | 5.0346 | 540.68 | 0.8033 | 0.1137 |
| latest | 3.1892 | 0.3390 | 0.6535 | 5.0250 | 540.38 | 0.7953 | 0.1156 |

## Primary Checkpoint Metrics

Primary checkpoint: `objective_saved_best`, step 83,000, `ckpt.pt`.

| metric | value |
|---|---:|
| Clinical CE | 3.2038 |
| Clinical top-1 | 0.3353 |
| Clinical top-5 | 0.6509 |
| Clinical top-10 | 0.7400 |
| Different-day time NLL | 5.0346 |
| Different-day mean absolute error | 540.68 days |
| Different-day median absolute error | 21.06 days |
| Different-day p95 absolute error | 1032.47 days |
| Same-day AUROC | 0.8033 |
| Same-day AUPRC | 0.9121 |
| Same-day Brier | 0.1137 |

Overall time MAE is retained as a tail-sensitive diagnostic, not as the primary
timing metric. Timing claims should use NLL, median absolute error, and
horizon-stratified errors.

## Type-Specific Clinical Prediction

| token type | targets | CE | top1 | top5 | top10 |
|---|---:|---:|---:|---:|---:|
| DX | 1,668,359 | 3.2489 | 0.4107 | 0.6687 | 0.7359 |
| RX | 2,407,960 | 3.2026 | 0.2927 | 0.6381 | 0.7366 |
| PX | 803,613 | 3.1121 | 0.3065 | 0.6524 | 0.7590 |
| DTH | 508 | 5.7751 | 0.0748 | 0.1831 | 0.2776 |

DX exact-token prediction is strongest. RX and PX have lower top-1 accuracy but
meaningful top-k candidate recall. DTH should not be claimed because the target
count is very small.

## Token Frequency Buckets

| bucket | targets | CE | top1 | top5 |
|---|---:|---:|---:|---:|
| head | 4,526,264 | 2.9515 | 0.3536 | 0.6820 |
| medium | 307,734 | 5.7950 | 0.1159 | 0.2881 |
| tail | 43,355 | 10.4883 | 0.0054 | 0.0174 |
| rare | 3,087 | 12.5347 | 0.0000 | 0.0000 |
| unseen_near_rare | 0 | NA | NA | NA |

The aggregate clinical top-k result is driven mostly by head tokens. Tail and
rare exact-code prediction remain clear limitations.

## Patient and Sequence Stratification

### Age Group

| age group | targets | CE | top1 | top5 |
|---|---:|---:|---:|---:|
| 0-17 | 12 | 3.4368 | 0.2500 | 0.2500 |
| 18-39 | 225,282 | 3.0736 | 0.3858 | 0.6780 |
| 40-64 | 1,981,698 | 3.1256 | 0.3504 | 0.6634 |
| 65-79 | 2,087,089 | 3.2312 | 0.3271 | 0.6456 |
| 80+ | 586,814 | 3.4203 | 0.2941 | 0.6170 |

The 0-17 group is too small to interpret. Performance decreases with age.

### Sex Token

| sex token | targets | CE | top1 | top5 |
|---|---:|---:|---:|---:|
| sex_token_7 | 2,022,903 | 3.1885 | 0.3343 | 0.6543 |
| sex_token_8 | 2,857,537 | 3.2146 | 0.3360 | 0.6485 |

Sex-token differences are small. The token registry should be used before
mapping these token IDs to human-readable sex labels in external material.

### Visit Density

| visit density | targets | CE | top1 | top5 |
|---|---:|---:|---:|---:|
| 0-4/year | 894,870 | 3.0044 | 0.4064 | 0.6751 |
| 5-19/year | 2,080,762 | 3.0963 | 0.3391 | 0.6705 |
| 20-49/year | 1,063,440 | 3.4415 | 0.2872 | 0.6156 |
| 50+/year | 493,268 | 3.6414 | 0.2740 | 0.5815 |

Higher visit density corresponds to lower prediction performance, likely
reflecting more complex or higher-entropy trajectories.

### Sequence Length

| sequence length | targets | CE | top1 | top5 |
|---|---:|---:|---:|---:|
| 0-127 | 417,510 | 3.1170 | 0.4122 | 0.6545 |
| 128-511 | 1,288,536 | 2.9038 | 0.3879 | 0.6952 |
| 512-1023 | 1,095,348 | 3.1163 | 0.3333 | 0.6685 |
| 1024+ | 2,079,046 | 3.4533 | 0.2884 | 0.6134 |

Very long trajectories are harder. This may reflect both patient complexity and
context truncation.

### Calendar Year

Calendar-year stratification is not available from the current age-only binary
event format. An event-date sidecar is required to evaluate calendar drift,
entry gaps, in-care gaps, and long observation gaps.

## Same-Day Classification

The same-day head predicts whether the next clinical target belongs to the same
event day or to a different future day. It should not be interpreted as a
disease-outcome AUROC.

Primary checkpoint:

| metric | value |
|---|---:|
| targets | 4,880,440 |
| same-day prevalence | 0.7939 |
| AUROC | 0.8033 |
| AUPRC | 0.9121 |
| Brier score | 0.1137 |
| Brier reliability | 0.0036 |
| Brier resolution | 0.0536 |
| Brier uncertainty | 0.1636 |

Calibration bins:

| probability bin | targets | mean probability | observed same-day rate |
|---|---:|---:|---:|
| 0.0-0.1 | 86,320 | 0.0294 | 0.0035 |
| 0.1-0.2 | 59,277 | 0.1506 | 0.0397 |
| 0.2-0.3 | 66,113 | 0.2514 | 0.0824 |
| 0.3-0.4 | 83,016 | 0.3524 | 0.1693 |
| 0.4-0.5 | 111,849 | 0.4524 | 0.2848 |
| 0.5-0.6 | 163,614 | 0.5537 | 0.4400 |
| 0.6-0.7 | 265,278 | 0.6542 | 0.5969 |
| 0.7-0.8 | 472,244 | 0.7548 | 0.7479 |
| 0.8-0.9 | 893,126 | 0.8559 | 0.8599 |
| 0.9-1.0 | 2,679,603 | 0.9690 | 0.9214 |

The same-day head has strong discrimination and favorable Brier reliability,
but calibration bins show mild overconfidence, especially outside the central
well-calibrated bins and in the highest-probability bin.

## Different-Day Timing

Primary checkpoint timing summary:

| metric | value |
|---|---:|
| targets | 1,005,888 |
| model NLL | 5.0346 |
| model MAE | 540.68 days |
| model median AE | 21.06 days |
| model p95 AE | 1032.47 days |
| constant-rate baseline NLL | 7.9001 |
| median-gap baseline MAE | 953.33 days |
| NLL improvement over constant-rate | 2.8655 |
| MAE improvement over median baseline | 412.66 days |
| beats baseline | true |

Horizon-stratified timing:

| horizon | targets | actual median days | MAE | median AE | p95 AE | NLL |
|---|---:|---:|---:|---:|---:|---:|
| 0 days | 0 | NA | NA | NA | NA | NA |
| 1-7 days | 354,876 | 3.00 | 23.01 | 3.35 | 119.09 | 2.9465 |
| 8-30 days | 234,262 | 14.00 | 37.89 | 13.47 | 156.39 | 4.3175 |
| 31-90 days | 168,224 | 56.00 | 42.32 | 27.74 | 138.55 | 5.4065 |
| 91-365 days | 169,821 | 161.00 | 86.59 | 68.94 | 223.35 | 6.5888 |
| over 365 days | 78,705 | 11,220.00 | 6416.30 | 3475.67 | 17860.74 | 12.4361 |

The model learns useful short- and medium-horizon timing signal. The aggregate
mean absolute error is dominated by the over-365-day long-gap bucket and should
not be used as the primary time metric.

## Patient-Level Bootstrap Confidence Intervals

Bootstrap configuration: 18,407 patients, 1,000 samples, seed 42.

| checkpoint | clinical CE | top1 | top5 | top10 | time NLL | time MAE days | same-day Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| objective_saved_best | 3.2035 [3.1895, 3.2190] | 0.3354 [0.3332, 0.3378] | 0.6509 [0.6485, 0.6532] | 0.7400 [0.7378, 0.7420] | 5.0344 [5.0044, 5.0691] | 540.78 [531.64, 550.61] | 0.1137 [0.1129, 0.1144] |
| same_day_best | 3.1964 [3.1822, 3.2119] | 0.3373 [0.3351, 0.3397] | 0.6520 [0.6495, 0.6543] | 0.7410 [0.7388, 0.7430] | 5.0120 [4.9869, 5.0378] | 526.30 [517.49, 535.75] | 0.1155 [0.1147, 0.1162] |
| latest | 3.1888 [3.1747, 3.2043] | 0.3391 [0.3368, 0.3415] | 0.6535 [0.6510, 0.6558] | 0.7419 [0.7397, 0.7440] | 5.0248 [4.9966, 5.0574] | 540.48 [531.41, 550.29] | 0.1156 [0.1148, 0.1164] |

These are per-metric bootstrap intervals, not paired checkpoint-difference
intervals. They support the qualitative conclusion that checkpoint differences
are small, while the primary checkpoint has the best same-day Brier interval
among the compared alternatives.

## Reproducibility Notes

Evidence source: capped 20,000-patient held-out test evaluation.

Pod output directory:

```text
/home/khdp-user/workspace/fermat-data/out/task16_full_10l640_genomics_d1_20260623T051230Z/paper_eval_test_capped20k
```

Generated JSON files used:

```text
objective_best.evaluation.json
ce_best.evaluation.json
time_nll_best.evaluation.json
same_day_best.evaluation.json
objective_saved_best.evaluation.json
latest.evaluation.json
suite_manifest.json
checkpoint_comparison.md
```

This report intentionally avoids public README exposure of internal run
workflow, Pod commands, or task-numbered development details.
