# FERMAT

**Foundation model for Exploring Real-world Multimodal health data using Autoregressive Trajectory modeling**

FERMAT is a generative transformer for longitudinal clinical trajectories. It
serializes each patient's history as an autoregressive sequence of clinical
events and learns both what event is likely to occur next and when it is likely
to occur.

The current development line supports two execution paths:

- a public Synthetic SNUH 4-column pipeline used for reproducible code
  verification; and
- a private SNUH-CDM full-cohort pretraining pipeline run on research Pods,
  including Task 17 GENOMICS conditioning tokens.

## Motivation

A patient's trajectory is not a list of diagnoses. Long before an outcome such
as diabetic nephropathy appears, there may be years of prescriptions,
procedures, screening values, hospital measurements, and other context. FERMAT
models these heterogeneous events in a single age-ordered sequence rather than
restricting the history to diagnosis codes.

## Data Sources

FERMAT is designed for linked Korean healthcare data, including national
claims, screening, death registry, cancer registry, cohort resources, and
hospital CDMs. The public repository keeps data-specific credentials and raw
private SNUH artifacts out of Git; Pod-side outputs are produced under
`/home/khdp-user/workspace/fermat-data`.

| Source | Data | What it captures |
|--------|------|-----------------|
| NHIS | eligibility, health screening, death and cancer screening records | demographics, biennial biomarkers, lifestyle, mortality |
| HIRA | claims, diagnoses, procedures, prescriptions | nationwide healthcare utilization |
| Statistics Korea | cause-of-death records | out-of-hospital deaths with cause codes |
| KDCA | KNHANES, KoGES, registries | cohort and public-health context |
| Cancer registry | cancer diagnoses | registry-confirmed cancers |
| Hospital CDMs | OMOP tables and local source values | hospital-level clinical detail and external validation |

## Multimodal Token Vocabulary

Each event is stored in the 4-column FERMAT binary format:

```text
patient_id | age_in_days | token_id | token_type
```

`patient_id` is a dense integer, `age_in_days` is the patient's age at the
event, `token_id` is a global vocabulary ID, and `token_type` identifies the
modality.

Example:

```text
001 | 9131  | 42   | DX        # hypertension
001 | 9496  | 815  | RX        # amlodipine
001 | 9861  | 1102 | LAB       # fasting glucose bucket
001 | 10592 | 55   | DX        # type 2 diabetes
001 | 10957 | 830  | RX        # metformin
```

### Token Types

| Type | Role in the trajectory |
|------|------------------------|
| DX | diagnosis |
| RX | drug prescription |
| PX | procedure |
| LAB | lab / measurement context |
| GENOMICS | genomics / tumor biomarker context |
| LIFESTYLE | screening questionnaire context |
| DTH | death event |
| SEX | static demographic conditioning token |
| NO_EVENT | structural no-event token |
| PAD | sequence padding |

In the SNUH full-cohort run, LAB and GENOMICS were used as context rather than
as prediction targets.

### GENOMICS Tokens

Task 17 adds a unified `GENOMICS:*` namespace for tumor biomarker and genomics
context. Examples:

```text
GENOMICS:EGFR:MUTATION:EXON19DEL
GENOMICS:ALK:STATUS:NEGATIVE_OR_WT
GENOMICS:PDL1:TPS:70
```

The integrated SNUH run created 1,170 GENOMICS events for 615 patients, with
278 unique token keys. Source provenance is retained in
`genomics_token_events.csv`:

- `canonical_report`: 2004-2020 molecular pathology / NGS note parser outputs.
- `weak_clinical_summary`: 2021+ `observation_source_value=기타`,
  `observation_concept_id=1340204` biomarker summary text.

GENOMICS tokens are conditioning-only by design: somatic biomarker facts are
treated as known context for downstream clinical prediction, not as events that
the model should learn to predict from prior clinical history.

## Architecture

Each input event is embedded as:

```text
TokenEmb(token_id) + AgeEncoding(age_in_days) + TypeEmb(token_type)
```

- `TokenEmb` is a learned vocabulary embedding tied to the output projection.
- `AgeEncoding` is a continuous sinusoidal age representation followed by a
  learned projection.
- `TypeEmb` lets the transformer distinguish modalities.

The sequence is passed through causal transformer blocks. Same-day events can
be masked from attending to each other with `mask_ties=True`.

The current SNUH pretraining configuration uses:

- a next-token softmax head;
- a two-stage time objective with a same-day classifier;
- a decoupled waiting-time head that predicts log event rate separately from
  the token logits; and
- objective-based checkpoint selection.

The SNUH Task 16 production candidate used 10 layers, 10 heads, 640 hidden
dimensions, context length 512, bfloat16, and `loss_dt_weight=0.3`.

## Training And Evaluation

FERMAT supports:

- 3-column Delphi-compatible input;
- 4-column typed FERMAT input;
- full checkpoint save/resume;
- best-checkpoint and latest-checkpoint outputs;
- deterministic capped checkpoint evaluation; and
- clinical next-token, waiting-time, and same-day metrics.

The SNUH Task 16 full-cohort run completed 100,000 iterations. The
validation-selected best checkpoint was at iteration 83,000. Capped validation
evaluation on 20,000 patients reported:

| Metric | Value |
|--------|------:|
| Clinical CE | 3.1940 |
| Clinical top-1 | 33.91% |
| Clinical top-5 | 65.24% |
| Clinical top-10 | 74.09% |
| Waiting-time NLL | 5.0487 |
| Waiting-time MAE | 539.5 days |
| Waiting-time median absolute error | 21.4 days |
| Beats waiting-time baseline | true |
| Same-day AUROC | 0.8035 |
| Same-day AUPRC | 0.9123 |
| Same-day Brier | 0.1140 |

The final/latest checkpoint at 100,000 iterations is retained, but validation
selection uses the 83,000-iteration best checkpoint.

## Ablation Design

The original ablation plan progressively adds modalities to quantify which
contexts improve trajectory modeling. That design remains important, but the
current README does not treat it as completed evidence. The ablation section
should be expanded after the corresponding experiments are run and reported
with shared metrics.

Planned comparisons include:

- DX-only baseline;
- DX + RX;
- DX + RX + PX;
- adding LAB context;
- adding lifestyle context; and
- adding GENOMICS conditioning for oncology-focused analyses.

## Current Implementation Status

| Component | Status |
|-----------|--------|
| Core transformer (`model.py`) | Implemented with token, age, and type embeddings |
| Same-day masking | Implemented |
| Decoupled time head | Implemented and used for SNUH Task 16 |
| Two-stage time objective | Implemented |
| 3-column Delphi compatibility | Implemented |
| 4-column FERMAT data loading | Implemented |
| Synthetic SNUH preprocessing | Implemented |
| Synthetic SNUH training/evaluation | Implemented |
| SNUH full-cohort Task 15 ETL runner | Implemented in Pod workflow |
| SNUH Task 16 full-cohort training | Implemented and run to 100k iterations |
| SNUH Task 17 biomarker form discovery | Implemented |
| SNUH Task 17 GENOMICS token integration | Implemented as conditioning-only context |
| NHIS/HIRA production preprocessing | Not implemented in this repo |
| Ablation result table | Planned after experiments |

## Reproducing The Public Synthetic SNUH Pipeline

If `data/synthetic_snuh_raw.duckdb` is available:

```bash
SYNTHETIC_SNUH_DUCKDB=data/synthetic_snuh_raw.duckdb \
  bash scripts/run_smoke_synthetic_snuh.sh
```

This runs schema inspection, preprocessing, bin validation, dataset summary,
mapping audit, and CPU smoke training. If the DuckDB file is missing, the
harness falls back to a self-synthetic 4-column dataset for code verification.

Train the public synthetic next-token model:

```bash
python train.py config/train_fermat_synthetic_snuh_token_prediction_longer.py --device=cpu
```

Evaluate:

```bash
python scripts/evaluate_token_prediction.py \
  --ckpt FERMAT-synthetic-snuh-token-prediction-longer/ckpt_top1_best.pt \
  --data-dir data/synthetic_snuh \
  --device cpu
```

Generate examples:

```bash
python scripts/demo_next_token_prediction.py \
  --ckpt FERMAT-synthetic-snuh-token-prediction-longer/ckpt_top1_best.pt \
  --data-dir data/synthetic_snuh \
  --device cpu
```

## SNUH Pod Workflow

The private SNUH workflow is documented in:

- `docs/snuh_pretraining_runbook.md`
- `docs/snuh_task17_genomic_variant_audit.md`

Task 16 bundles are created with:

```bash
python scripts/build_snuh_task16_bundle.py
```

Task 17 / GENOMICS tooling includes:

```text
scripts/discover_snuh_task17_biomarker_forms.py
scripts/add_snuh_task17_genomics_tokens.py
```

## License

MIT
