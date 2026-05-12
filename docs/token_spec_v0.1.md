# FERMAT Token Specification v0.1

This document defines how OMOP CDM events are converted into FERMAT's
4-column event sequence. It is the **mapping rule layer** that the
existing FERMAT README's "token schema" leaves to implementation.

Scope: Synthetic SNUH OMOP CDM v5.4 (10,000 persons) and any OMOP v5.4
dataset that exposes the same tables. Adjustments for the BOBIC
flattened claims/screening tables are deferred to v0.2.

---

## 1. Input format

Every patient's clinical history is serialized as a stream of 4-column
rows:

```
patient_id | age_in_days | token_id | token_type
```

All four columns are stored as `uint32` in a contiguous binary file
(`train.bin`, `val.bin`).

- `patient_id`: dense integer assigned by `preprocess_*.py` (0..N-1).
- `age_in_days`: integer days from birth to event, computed per Section 4.
- `token_id`: globally unique integer (Section 3).
- `token_type`: integer in `0..8` mapped to `TokenType` enum (Section 2).

Rows are sorted by `(patient_id, age_in_days, deterministic_order)`.

---

## 2. Token types

Aligned with `model.py::TokenType`:

| name      | id | role                                  | included in loss |
|-----------|----|---------------------------------------|------------------|
| PAD       | 0  | sequence padding                      | no               |
| DX        | 1  | diagnosis (KCD / ICD-10)              | yes              |
| RX        | 2  | drug prescription                     | yes              |
| PX        | 3  | procedure                             | yes              |
| LAB       | 4  | discretized lab / screening result    | yes              |
| LIFESTYLE | 5  | screening questionnaire (smoking etc) | yes              |
| DTH       | 6  | death (terminal)                      | yes              |
| SEX       | 7  | sex (static, prepended)               | no               |
| NO_EVENT  | 8  | Delphi-style no-event padding         | no               |

The "ignore in loss" set is enforced via `FermatConfig.ignore_types`:
`[PAD, SEX, NO_EVENT]`. LAB is included in v0.1 to keep type-embedding
gradients flowing; the context-only ablation is deferred to v0.2.

---

## 3. Vocabulary and token_id assignment

- `token_id` is **globally unique across all token types**.
  `DX:I10` and `RX:I10` are different `token_id`s.
- `token_id = 0` is reserved for **PAD**.
- `token_id = 1` is reserved for the **no-event token** (Delphi convention,
  see `utils.py::get_batch` which shifts all tokens by +1).
- `token_id`s 2..N are assigned by `preprocess_*.py`, in descending order
  of frequency, with the following blocks:

  ```
  2                       no-event reserve (kept as id=1 after shift)
  SEX_BLOCK_START..       SEX tokens (male, female, unknown)
  DX_BLOCK_START..        DX tokens
  RX_BLOCK_START..        RX tokens
  PX_BLOCK_START..        PX tokens
  LAB_BLOCK_START..       LAB tokens (one per <code, quantile_bin>)
  LIFESTYLE_BLOCK_START.. LIFESTYLE tokens
  DTH_BLOCK_START..       DTH tokens
  ```

  Block boundaries are recorded in `vocab.csv` so downstream code can
  map a `token_id` back to `(token_type, source_value)`.

- Vocabulary cap for v0.1: **top-frequency 2,000 tokens** across all
  types combined. Source values not in the top-2000 are dropped (counted
  in `dropped_events.csv`).

---

## 4. age_in_days calculation

```
1. If `person.birth_datetime` is non-null:
       age_in_days = (event_date - birth_datetime.date()).days
2. Else if `year_of_birth` is non-null:
       month = month_of_birth if non-null else 7
       day   = day_of_birth   if non-null else 1
       birth_date = date(year_of_birth, month, day)
       age_in_days = (event_date - birth_date).days
3. Else: drop the patient entirely.
4. If `age_in_days < 0`: drop the event (data error).
5. If `age_in_days > 150 * 365`: drop the event (data error).
```

`age_in_days` is stored as `uint32`. Negative values are impossible by
construction.

---

## 5. OMOP table → token type mapping (v0.1)

| OMOP table              | type | preferred source column        | fallback                       | event date              |
|-------------------------|------|--------------------------------|--------------------------------|-------------------------|
| `person`                | SEX  | `gender_source_value`          | `gender_concept_id`            | birth                   |
| `condition_occurrence`  | DX   | `condition_source_value`       | `condition_concept_id`         | `condition_start_date`  |
| `drug_exposure`         | RX   | `drug_source_value`            | `drug_concept_id`              | `drug_exposure_start_date` |
| `procedure_occurrence`  | PX   | `procedure_source_value`       | `procedure_concept_id`         | `procedure_date`        |
| `measurement`           | LAB  | see Section 6                  | see Section 6                  | `measurement_date`      |
| `death`                 | DTH  | `cause_source_value`           | `cause_concept_id` or const    | `death_date`            |

Notes:
- "preferred source column" is used whenever it is non-null and non-empty
  in the source row. Otherwise the fallback column is used.
- The `source_value` strategy keeps Korean local codes (KCD, EDI, KD)
  intact when present. Mapping to standard ontologies (ATC, LOINC,
  SNOMED) is deferred to v0.2.
- All tokens carry a `token_type_name`-prefixed label in `vocab.csv` for
  debuggability: e.g., `DX:I10`, `RX:A10BA02`, `PX:M0010`,
  `LAB:LDL:Q3`, `DTH:I21`.

---

## 6. MEASUREMENT (LAB) handling

For each `measurement` row:

```
1. Choose the measurement key:
       key = measurement_source_value if non-null else str(measurement_concept_id)

2. If `value_as_concept_id` is non-null:
       bin_label = "C" + str(value_as_concept_id)
       token_label = f"LAB:{key}:{bin_label}"

3. Else if `value_as_number` is non-null:
       Look up quantile cutpoints for this `key` (computed once per
       dataset on train split only — see preprocess script).
       Assign Q1 / Q2 / Q3 based on tertile.
       token_label = f"LAB:{key}:{Q*}"

4. Else: drop the row.
```

- Unit (`unit_concept_id`, `unit_source_value`) is **not converted** in
  v0.1. It is stored in `vocab.csv` metadata as a warning that the same
  key may carry different units across rows. A v0.2 task is to either
  split by unit or harmonize.
- Quantile cutpoints are computed on the **training split only** to
  avoid leakage. Validation rows use the train cutpoints.
- A measurement key needs at least 30 non-null `value_as_number`
  observations in the training split to receive quantile binning;
  otherwise it is dropped (recorded in `dropped_events.csv`).

---

## 7. Same-day deterministic ordering

When multiple events share `(patient_id, age_in_days)`, ties are broken
by token type in this order:

```
SEX < DX < RX < PX < LAB < LIFESTYLE < DTH < NO_EVENT < PAD
```

Within a type, ties are broken by ascending `token_id`. This makes the
serialization reproducible and removes a source of training noise.

Note: `model.py::Fermat.forward` masks same-age tokens from attending
to each other (`mask_ties=True`), so ordering does not affect masked
attention. It only affects the final causal ordering of the predicted
sequence.

---

## 8. Loss participation summary

| token_type | counted in `loss_ce` | counted in `loss_dt` |
|------------|----------------------|----------------------|
| PAD        | no                   | no                   |
| DX         | yes                  | yes                  |
| RX         | yes                  | yes                  |
| PX         | yes                  | yes                  |
| LAB        | yes (v0.1)           | yes (v0.1)           |
| LIFESTYLE  | yes                  | yes                  |
| DTH        | yes                  | yes                  |
| SEX        | no                   | no                   |
| NO_EVENT   | no                   | no                   |

Enforced by `FermatConfig.ignore_types = [PAD, SEX, NO_EVENT]`. The
implementation in `model.py::Fermat.forward` masks predictions of
ignored types via `pass_tokens`.

---

## 9. Drop log

`preprocess_*.py` writes a `dropped_events.csv` with columns:

```
reason, table, source_value, count
```

Possible reasons:

- `birth_unknown_patient` — patient has no birth_datetime or year_of_birth
- `age_negative` — event date precedes birth
- `age_unrealistic` — age > 150 years
- `measurement_value_missing` — both value_as_concept_id and value_as_number null
- `measurement_below_threshold` — fewer than 30 numeric observations
- `not_in_top_2000` — source value not in top-frequency vocabulary
- `unknown_source_value` — null/empty in both preferred and fallback

This file is summarized in `summarize_fermat_dataset.py`.

---

## 10. Open decisions deferred to v0.2

- Unit harmonization for measurements (currently ignored).
- Clinical-cutpoint LOW/NORMAL/HIGH bins (currently quantile-only).
- VISIT_START / VISIT_END boundary tokens (currently no visit hierarchy).
- LIFESTYLE token wiring (Synthetic SNUH may not expose these; v0.1
  generates an empty LIFESTYLE block if no source is found).
- Mapping of source_value to standard vocabularies (ATC, LOINC, SNOMED).
- Sub-tokenization of long source values (currently kept as opaque
  strings in `vocab.csv`).

---

## 11. Compatibility with FERMAT v0.1 codebase

This spec is consistent with the existing `model.py` and `utils.py`:

- `utils.py::load_data` auto-detects 4-column files via `uint32` row
  alignment, so no model changes are required.
- `utils.py::get_batch` already returns `xt` (token type tensor) when
  the file has 4 columns; it returns `None` for 3-column files,
  preserving the Delphi smoke test.
- `model.py::Fermat.forward(idx, age, token_type, ...)` already accepts
  the type tensor and adds `wtype(token_type)` to the embedding.
- `train.py::_unpack_batch` defaults `xt` to `DX` when missing, so
  Delphi 3-column data continues to train without modification.

No model-side patches are required for v0.1. Patches, if any, are
restricted to `utils.py` for batch-construction edge cases (e.g.,
SEX-prefix preservation) and are recorded in `docs/changelog_v0.1.md`.
