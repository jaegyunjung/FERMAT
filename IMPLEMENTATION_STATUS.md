# FERMAT Implementation Status Report

**Foundation model for Exploring Real-world Multimodal health data using Autoregressive Trajectory modeling**

Date: 2026-05-19

---

## A. Current Status Summary

| Category | Status |
|----------|--------|
| Core architecture (`model.py`) | **Implemented** |
| 3-column Delphi compatibility | **Implemented** |
| 4-column FERMAT data loading/training | **Implemented** |
| Synthetic SNUH DuckDB → 4-column preprocessing | **Implemented** |
| Synthetic SNUH mapping audit | **Implemented** |
| Synthetic SNUH smoke harness | **Implemented** |
| Full Synthetic SNUH next-token training | **Implemented** |
| Full Synthetic SNUH evaluation and demo generation | **Implemented** |
| NHIS/HIRA production preprocessing | **Not implemented** |
| Real SNUH training / external validation in this public repo | **Not implemented** |

**One-line summary:**

- The public repo now supports an end-to-end **Synthetic SNUH 4-column FERMAT pipeline**: schema inspection, preprocessing, bin validation, dataset summary, mapping audit, smoke training, full next-token training, evaluation, and qualitative next-token demos.
- The best current public synthetic result is a **top-1 next-token accuracy of 40.83%** on full `data/synthetic_snuh`.

---

## B. Best Current Result

Best checkpoint:

- `FERMAT-synthetic-snuh-token-prediction-longer/ckpt_top1_best.pt`
- Checkpoint step: `4000`

Validation metrics on full `data/synthetic_snuh`:

| Metric | Value |
|--------|------:|
| Cross-entropy loss | `1.723160` |
| Perplexity | `5.602202` |
| Top-1 accuracy | `40.8291%` |
| Top-5 accuracy | `86.5737%` |
| Top-10 accuracy | `93.2485%` |

Token-type-specific top-1 accuracy:

| Token type | Top-1 |
|-----------|------:|
| DX | `47.8597%` |
| RX | `34.7514%` |
| PX | `94.9580%` |
| LAB | `40.6463%` |
| DTH | `8.4746%` |

Primary evidence:

- `logs/fermat_token_prediction_eval.md`
- `logs/fermat_token_prediction_eval.csv`
- `logs/fermat_next_token_examples.md`

---

## C. File Structure and Roles

### Core training / model code

| File | Role | Status |
|------|------|--------|
| `model.py` | FERMAT transformer, token-type embedding, forward pass, loss, generation | Implemented |
| `train.py` | Main training loop for 3-col/4-col data, checkpointing, config-driven experiments | Implemented |
| `utils.py` | `.bin` loader, patient-to-index mapping, batch construction for 3-col/4-col data | Implemented |

### Synthetic SNUH preprocessing / validation

| File | Role | Status |
|------|------|--------|
| `scripts/inspect_synthetic_snuh_schema.py` | Inspect Synthetic SNUH DuckDB schema | Implemented |
| `scripts/preprocess_synthetic_snuh_to_fermat.py` | Build 4-column FERMAT data from Synthetic SNUH DuckDB | Implemented |
| `scripts/check_fermat_bin.py` | Validate `.bin` files and token/type constraints | Implemented |
| `scripts/summarize_fermat_dataset.py` | Summarize row counts, patients, token-type distribution | Implemented |
| `scripts/audit_fermat_mapping.py` | Audit OMOP source tables against FERMAT token types | Implemented |
| `scripts/run_smoke_synthetic_snuh.sh` | End-to-end Synthetic SNUH smoke harness | Implemented |

### Token-prediction experiments

| File | Role | Status |
|------|------|--------|
| `config/train_fermat_synthetic_snuh.py` | 4-column Synthetic SNUH smoke-training config | Implemented |
| `config/train_fermat_synthetic_snuh_token_prediction.py` | Baseline full-data token-prediction config | Implemented |
| `config/train_fermat_synthetic_snuh_token_prediction_longer.py` | Longer-context / longer-training config | Implemented |
| `config/train_fermat_synthetic_snuh_token_prediction_large.py` | Larger-capacity comparison config | Implemented |
| `scripts/evaluate_token_prediction.py` | Validation CE / perplexity / top-k / per-type accuracy | Implemented |
| `scripts/demo_next_token_prediction.py` | Qualitative next-token examples decoded via `vocab.csv` | Implemented |

### Legacy / compatibility

| File | Role | Status |
|------|------|--------|
| `config/train_fermat_demo.py` | Delphi synthetic 3-column smoke-test config | Still supported |
| `config/train_fermat_kr.py` | Template for future Korean multimodal training | Template only |
| `config/ablation_configs.py` | Older ablation definition file | Design artifact; not central to current Synthetic SNUH path |

---

## D. Verified Data Interfaces

### D.1 Supported binary formats

**3-column (Delphi compatibility):**

```text
patient_id | age_in_days | token_id
```

**4-column (FERMAT):**

```text
patient_id | age_in_days | token_id | token_type
```

`utils.py:load_data()` auto-detects 3-column vs 4-column `.bin` data and `train.py` handles both paths.

### D.2 Token types in the current public implementation

| Name | Int value | Meaning | Status |
|------|----------:|---------|--------|
| PAD | 0 | Padding | Implemented |
| DX | 1 | Diagnosis | Implemented |
| RX | 2 | Prescription / drug exposure | Implemented |
| PX | 3 | Procedure | Implemented |
| LAB | 4 | Discretized numeric lab / measurement token | Implemented |
| LIFESTYLE | 5 | Lifestyle token type | Defined, but not central to current Synthetic SNUH run |
| DTH | 6 | Death event | Implemented |
| SEX | 7 | Static conditioning token | Implemented |
| NO_EVENT | 8 | Synthetic structural padding token | Implemented |

### D.3 Current Synthetic SNUH token conventions

Current Synthetic SNUH preprocessing creates:

- `SEX:<value>`
- `DX:<source_value>`
- `RX:<source_value>`
- `PX:<source_value>`
- `LAB:<measurement_source_value>:Q1/Q2/Q3`
- `DTH:DEATH`

For measurements:

- numeric LAB tertile cutpoints are derived from **train patients only**
- those cutpoints are then applied to validation patients

For death:

- current public v0.1 behavior uses a single token `DTH:DEATH`

---

## E. What Has Been Verified

### E.1 Synthetic SNUH 4-column pipeline

The following path has been executed successfully in this repo:

1. Inspect Synthetic SNUH DuckDB schema
2. Preprocess OMOP-like source tables into 4-column FERMAT bins
3. Validate `train.bin` / `val.bin`
4. Summarize dataset statistics
5. Audit source-table ↔ token-type mapping
6. Run 4-column smoke training
7. Run full next-token training on `data/synthetic_snuh`
8. Evaluate the trained checkpoint
9. Generate qualitative next-token prediction examples

Generated artifacts include:

- `data/synthetic_snuh/train.bin`
- `data/synthetic_snuh/val.bin`
- `data/synthetic_snuh/vocab.csv`
- `data/synthetic_snuh/patient_split.csv`
- `data/synthetic_snuh/dropped_events.csv`
- `logs/fermat_dataset_summary.md`
- `logs/fermat_mapping_audit.md`
- `logs/fermat_token_prediction_eval.md`
- `logs/fermat_next_token_examples.md`

### E.2 Mapping audit semantics

The implemented audit checks:

- source row counts for `person`, `condition_occurrence`, `drug_exposure`, `procedure_occurrence`, `measurement`, `death`
- FERMAT token-type counts from `train.bin` and `val.bin`
- expected mapping:
  - `person -> SEX`
  - `condition_occurrence -> DX`
  - `drug_exposure -> RX`
  - `procedure_occurrence -> PX`
  - `measurement -> LAB`
  - `death -> DTH`
- zero `person_id` overlap between train and val
- dropped-event explanations from `dropped_events.csv`
- ERROR if source `measurement > 0` but mapped `LAB == 0`
- ERROR if source `death > 0` but mapped `DTH == 0`

### E.3 Evaluation / demo path

The repo now supports:

- checkpoint loading from a trained token-prediction run
- validation CE / perplexity computation
- top-1 / top-5 / top-10 next-token accuracy
- token-type-specific top-1 reporting for DX/RX/PX/LAB/DTH
- decoded qualitative examples from validation patients using `vocab.csv`

---

## F. Experiment Progression

### F.1 Earlier public synthetic baseline

Baseline next-token checkpoint on full `data/synthetic_snuh`:

- Step: `2000`
- Top-1 accuracy: `36.79%`
- Top-5 accuracy: `85.10%`
- Top-10 accuracy: `92.42%`
- Perplexity: `6.91`

### F.2 What improved the current best result

The best-performing public synthetic run came from:

- longer context: `block_size = 256` instead of `128`
- longer optimization: `max_iters = 4000` instead of `2000`
- lower regularization: `dropout = 0.05`, `weight_decay = 0.05`

This produced the current best:

- Top-1 accuracy: `40.83%`
- Perplexity: `5.60`

In short:

- the largest observed gain came from giving the model **more longitudinal context** and **more training time**
- sampled validation CE continued improving through the longer run, and the final saved best/top-1 checkpoint came from the longer experiment path

---

## G. Known Gaps / Limitations

| Area | Current limitation |
|------|--------------------|
| NHIS/HIRA production ingestion | No raw NHIS/HIRA preprocessing pipeline is implemented in this public repo |
| Real SNUH / hospital external validation | Not included here |
| Vocabulary design for production claims data | Public Synthetic SNUH path is implemented, but production-scale token mapping policy is still a separate task |
| DTH prediction | Works structurally, but predictive accuracy remains low due to rarity |
| LIFESTYLE usage | Token type exists, but current public Synthetic SNUH emphasis is DX/RX/PX/LAB/DTH |
| Full paper-grade ablation matrix | Not yet run end-to-end in this repo |

---

## H. Immediate Next Priorities

| Priority | Task |
|----------|------|
| 1 | Decide whether to keep scaling the current Synthetic SNUH token-prediction path or start structured ablations from the new 40%+ baseline |
| 2 | Update documentation consistently across `README.md` and experiment logs |
| 3 | If needed, add experiment tracking tables for multiple token-prediction runs |
| 4 | Design production preprocessing for NHIS/HIRA and real hospital CDM paths |
| 5 | Expand evaluation beyond next-token metrics into downstream clinical forecasting tasks |

---

## I. Evidence Files

For the current public state, the most useful evidence files are:

- `README.md`
- `logs/fermat_dataset_summary.md`
- `logs/fermat_mapping_audit.md`
- `logs/fermat_token_prediction_eval.md`
- `logs/fermat_next_token_examples.md`
- `FERMAT-synthetic-snuh-token-prediction-longer/ckpt_top1_best.pt`
