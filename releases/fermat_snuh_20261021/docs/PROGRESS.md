# Project progress through 2026-10-21

FERMAT's project scope is a clinical foundation model usable through both patient
representations and generated trajectories. ETHOS is one comparator in the
long-term risk evaluation, not the sole purpose of the model.

## Completed work

| Work | Result and provenance |
|---|---|
| SNUH model development | Two-stage 65.22M model, 3.77M patients, 1.238B events; validation-best stage-2 update 93,000 |
| Original manuscript evaluations | Embedding-based long-term disease prediction and historical next-record-day generation; these remain the original experiments |
| Frozen-embedding head scaling | 5k, 10k, 18,474, 50k, 100k training patients; selected 100k head Recall@10 63.42% on 1,862 test patients |
| Historical next-record-day comparison | Saved 50-path generation Recall@10 71.00%; original head 55.68%; larger head reduces the gap to 7.58 percentage points |
| Corrected joint generation | Same/different-day branches aligned with the training mask, without changing foundation weights |
| Corrected disease generation extension | Completed 1,999 patients × 50 paths = 99,950 paths; reused 19,990 from the first 10 paths and added 79,960; completed 2026-10-20 |
| ETHOS disease comparison | Existing outputs evaluated with explicit END/token-cap handling; termination semantics substantially affect conclusions |
| Task 31 ablation | Separate timing-head ablation finished 100k updates; not promoted as the default joint model |
| Collaborator handoff | Model-only weights, exact architecture, corrected inference, frozen registry/lab bins, selected next-record head, synthetic examples and verification |

## Five-year disease risk comparison

The analysis standard fixed on 2026-10-21 treats generated record END and token
caps as right censoring, disease as the event of interest, and death as a competing
event. It retains events observed before censoring and does not extrapolate a
flat risk curve to five years after the risk set disappears with remaining survival.
Each model uses all 50 paths in its Aalen–Johansen estimate where five-year risk
is supported. Clinical follow-up uses five-fold cross-fitted reverse-KM IPCW;
Brier divides by the eligible patient count, not the sum of weights. The paired
bootstrap uses 2,000 patient resamples with the fitted weights held fixed.

This standard was set after prior results had been inspected, not prospectively
preregistered. Generated END censoring requires an unverified noninformative
censoring assumption. The comparison below is conditional on common evaluable
patients, not an unconditional population-level result.

| Disease | Common evaluable patients | FERMAT Brier | ETHOS Brier | FERMAT AUROC | ETHOS AUROC |
|---|---:|---:|---:|---:|---:|
| Chronic kidney disease | 1,825 | 0.04828 | 0.01456 | 0.81491 | 0.80017 |
| Fatty liver | 1,800 | 0.03490 | 0.02959 | 0.69844 | 0.73874 |
| Diabetes | 1,634 | 0.06479 | 0.04044 | 0.67225 | 0.80296 |

Lower Brier is better. ETHOS was better on all three Brier comparisons, with paired
intervals excluding zero. FERMAT's kidney-disease AUROC point estimate alone is
not a superiority claim. These comparisons do not replace the original manuscript's
embedding-versus-generation experiments. No fitted risk calibration is built into
the weights or inference package.

## What this enables next

The receiver can run embeddings, generated records, the optional next-record
head, or start additional training. For Samsung Life, the immediate next task is
to map available codes and units to the frozen input specification, report missing
concept coverage, then validate against actual outcomes on that dataset. No
Samsung Life performance result exists in this release.

For the manuscript, keep a versioned separation between historical next-record-day
results, corrected generation, head scaling, and the later ETHOS comparison.
Next-record-day 71.00% must not be labeled as newly measured performance of the
corrected generator. This handoff performs no new model training or long generation.

## Aggregate evidence

`evidence/head_scaling.json` preserves the verified aggregate learning-curve report.
`evidence/five_year_*.csv` preserves aggregate conditional metrics and paired
intervals. Patient-level prefixes, labels, IDs, trajectories, and database settings
are not part of the public repository or the transfer archive.
