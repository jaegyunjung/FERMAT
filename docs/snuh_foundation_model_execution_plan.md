# SNUH FERMAT Foundation Model Execution Plan

## Objective

Pretrain a general-purpose autoregressive FERMAT model on longitudinal SNUH
OMOP CDM trajectories, freeze the resulting checkpoint, and initially evaluate
time-to-event prediction for primary gastric cancer (`C16.*`). The same
checkpoint and downstream protocol should later support other cancer types.

## Confirmed Data Facts

- PostgreSQL database: `cdm`
- OMOP schema: `cdm2024_official`
- Patients: 3,773,568
- Condition events: 98,574,969
- Valid observed database end: 2025-02-05
- Invalid `observation_period` rows outside 1900-2100: 107
- Core OMOP and vocabulary tables are present and readable.
- Source condition vocabulary is predominantly ICD10 and maps to standard
  SNOMED concepts through the included OMOP vocabulary tables.

No Athena download is required for the official SNUH CDM. FERMAT still needs a
model token registry that maps selected OMOP concepts to dense embedding row
IDs.

## Input Domains

The first structured-data foundation model retains the Synthetic SNUH token
types:

- `DX`: standard condition concept
- `RX`: standard drug concept
- `PX`: standard procedure concept
- `LAB`: measurement concept and result representation
- `DTH`: death
- `SEX`: static sex token
- `LIFESTYLE`: retained as a token type but empty unless a reliable SNUH source
  is identified

Each event is represented in the FERMAT four-column format:

```text
patient_id, age_in_days, token_id, token_type_id
```

`patient_id` separates trajectories and is not embedded as a model feature.
The model receives token embedding, continuous age encoding, and token-type
embedding.

## Patient Split

1. Create one deterministic patient-level 70/15/15 train/validation/test split.
2. Persist the split seed and patient assignment.
3. Require zero patient overlap across splits.
4. Create all vocabulary selections and numeric LAB cutpoints from train
   patients only.
5. Apply the frozen train mapping to validation and test patients.

## LAB Policy

Audit the actual train distribution before choosing frequency thresholds.

- Frequent numeric `(measurement_concept_id, unit_concept_id)`:
  train-only decile cutpoints and one conditional `(test, unit, bin)` token.
- Rare numeric measurement:
  retain a test-performed token without value binning.
- Categorical measurement:
  retain `(measurement_concept_id, value_as_concept_id)`.
- Missing result:
  retain a test-performed token when clinically and technically valid.
- Repeated identical patient/date/test/unit numeric rows:
  collapse using a documented representative statistic such as the median.
- Never merge numeric values across incompatible units.

The frequent/rare threshold must be selected from coverage and stability
statistics, not an imported fixed top-N rule.

## Foundation Pretraining

The initial baseline retains the existing decoder-only FERMAT architecture.
It learns:

- next clinical event token
- time until the next event

Primary pretraining outputs:

- validation cross-entropy and perplexity
- top-1 and top-k token accuracy
- new-onset token accuracy separated from repeated-event accuracy
- next-event-time likelihood and error summaries

Architecture modernization and continuous-time versus discrete-time ablations
are later experiments, not prerequisites for the first baseline.

## Gastric Cancer Downstream Task

Primary gastric cancer codes:

```text
C160, C161, C162, C163, C164, C165,
C166, C168, C169, C1690, C1691, C1699
```

The downstream model uses the frozen foundation representation and predicts
time to first primary gastric cancer. Patients with prior qualifying gastric
cancer at the prediction origin are excluded. Death and observation end are
treated as censoring according to the selected survival estimand.

Five- and ten-year risk are evaluation horizons derived from the predicted
survival distribution, not separate model targets.

Evaluation:

- concordance index
- IPCW time-dependent AUROC
- AUPRC at selected horizons
- integrated Brier score
- calibration at 1, 3, 5, and 10 years

Sensitivity definitions such as `D37.1` or secondary digestive malignancy
codes remain separate from the primary `C16.*` analysis.

## Execution Order

### Phase 1: ETL Pilot

1. Audit table, date, concept, value, and unit completeness.
2. Create the deterministic patient split.
3. Audit numeric and categorical measurement distributions.
4. Select the LAB frequent/rare threshold from measured coverage.
5. Build train-only token vocabulary and LAB cutpoints in PostgreSQL.
6. Stream compact train/validation/test event shards without copying source
   CDM tables.
7. Verify split overlap, unknown-token rate, sequence lengths, domain balance,
   output size, runtime, and peak memory.

Recommended first pod:

```text
High-Memory, 16 vCPU, 128 GB RAM, 100 GB block storage
```

This is a pilot choice, not a proven sufficient production size. Scale to
32 vCPU / 256 GB only if measured memory, runtime, or parallelism requires it.

### Phase 2: Training Smoke Test

Run a small FERMAT model and verify end-to-end loading, decreasing loss,
throughput, context length, and GPU memory.

Recommended first GPU pod:

```text
L4, 1 GPU (24 GB VRAM), 8 vCPU, 48 GB RAM
```

### Phase 3: Foundation Pretraining

Choose model size after a small scaling sweep. A likely single-GPU production
candidate is:

```text
L40S, 1 GPU (48 GB VRAM), 16 vCPU, 96 GB RAM
```

The current training code does not implement distributed training, so
multi-GPU pods should not be selected until that support is added and tested.

### Phase 4: Downstream Evaluation

Freeze the selected foundation checkpoint, create cancer-specific survival
cohorts, and train consistent downstream heads. Start with gastric cancer and
then reuse the protocol for additional cancer types.

## Storage Rule

Every listed pod has only 100 GB block storage. Do not copy the 300+ GB
measurement table or other source CDM tables. Perform split joins, LAB
aggregation, cutpoint application, and tokenization inside PostgreSQL, and
write only compact event shards and reproducibility artifacts.

## Required Reproducibility Artifacts

- database snapshot identifier and effective end date
- SQL and preprocessing code version
- patient split assignments and seed
- token vocabulary
- LAB cutpoints and unit policy
- duplicate-collapse rules
- model configuration and package versions
- training seed and checkpoint
- downstream cohort definitions
- prediction files and evaluation outputs
