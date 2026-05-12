# CDM Playground / SNUH CDM Schema Compatibility Checklist

A pre-flight checklist to confirm that a CDM environment matches the
assumptions in `docs/token_spec_v0.1.md` and the queries in
`sql/extract_cdm_playground_sample.sql`. Run before attempting any
FERMAT preprocessing on a new CDM.

For each item, record **YES / NO / UNKNOWN** with a one-line note.

---

## 1. Table presence

The seven tables FERMAT v0.1 reads. All must be present.

| OMOP table              | needed for token type     | YES / NO | note |
|-------------------------|---------------------------|----------|------|
| `person`                | SEX, age_in_days base     |          |      |
| `visit_occurrence`      | event filtering / summary |          |      |
| `condition_occurrence`  | DX                        |          |      |
| `drug_exposure`         | RX                        |          |      |
| `procedure_occurrence`  | PX                        |          |      |
| `measurement`           | LAB                       |          |      |
| `death`                 | DTH                       |          |      |

If `death` is missing: FERMAT can run, but no DTH tokens are produced.
Record this explicitly and note in the PI brief.

---

## 2. source_value columns and fill rates

Run these queries (PostgreSQL / DuckDB) and record both the column
existence and the **non-null fill rate**:

```sql
-- condition_occurrence
SELECT COUNT(*)                                                          AS n_rows,
       SUM(CASE WHEN condition_source_value IS NOT NULL
                  AND condition_source_value <> '' THEN 1 ELSE 0 END)    AS n_src,
       SUM(CASE WHEN condition_concept_id IS NOT NULL
                  AND condition_concept_id <> 0    THEN 1 ELSE 0 END)    AS n_concept
FROM condition_occurrence;
```

(Repeat for `drug_exposure.drug_source_value`,
`procedure_occurrence.procedure_source_value`,
`measurement.measurement_source_value`,
`death.cause_source_value`.)

| column                                | rows | source_value % | concept_id % | recommended primary |
|---------------------------------------|------|----------------|--------------|---------------------|
| `condition_source_value`              |      |                |              |                     |
| `drug_source_value`                   |      |                |              |                     |
| `procedure_source_value`              |      |                |              |                     |
| `measurement_source_value`            |      |                |              |                     |
| `death.cause_source_value`            |      |                |              |                     |

**Rule:** if `source_value %` ≥ 80, prefer source. If both are < 80,
record a `coverage_warning` in `dropped_events.csv` and proceed with the
higher one. This is FERMAT v0.1 policy; v0.2 will revisit.

---

## 3. concept_id fallback availability

For each domain, confirm a fallback path exists for rows where the
preferred source is null.

| domain      | fallback column          | fallback present | note |
|-------------|--------------------------|------------------|------|
| condition   | `condition_concept_id`   |                  |      |
| drug        | `drug_concept_id`        |                  |      |
| procedure   | `procedure_concept_id`   |                  |      |
| measurement | `measurement_concept_id` |                  |      |
| death       | `cause_concept_id`       |                  |      |

If `concept_id` is the standard vocabulary id (SNOMED / RxNorm / LOINC),
note this — v0.2 will use it for cross-site harmonization. v0.1 just
treats the integer as an opaque string.

---

## 4. Event date columns

The preprocess script reads exactly these date columns. Confirm they
exist and are populated.

| table                  | column                          | YES / NO | non-null % |
|------------------------|---------------------------------|----------|------------|
| `condition_occurrence` | `condition_start_date`          |          |            |
| `drug_exposure`        | `drug_exposure_start_date`      |          |            |
| `procedure_occurrence` | `procedure_date`                |          |            |
| `measurement`          | `measurement_date`              |          |            |
| `death`                | `death_date`                    |          |            |

The script falls back to nothing if the date is null. Such rows are
silently dropped (this is intentional). If non-null % is low for any
table, record it.

---

## 5. Birth date columns (person table)

`age_in_days` requires at least one of these:

| column             | YES / NO | non-null % | note |
|--------------------|----------|------------|------|
| `birth_datetime`   |          |            |      |
| `year_of_birth`    |          |            |      |
| `month_of_birth`   |          |            |      |
| `day_of_birth`     |          |            |      |

The script's rule: prefer `birth_datetime`, else build from
`year_of_birth (+ month_of_birth or 7 + day_of_birth or 1)`. Patients
with neither are dropped (logged in `dropped_events.csv`).

If `year_of_birth` non-null % is < 95, flag as a CDM quality issue.

---

## 6. Death table accessibility

Even if `death` is listed in the schema, IRB-restricted environments
sometimes block it.

```sql
SELECT COUNT(*) FROM death;
```

| check                                  | YES / NO | note |
|----------------------------------------|----------|------|
| `death` table exists                   |          |      |
| `SELECT` permission granted            |          |      |
| `death_date` non-null in >= 50%        |          |      |
| `cause_source_value` non-null in any % |          |      |

If `death` is blocked, FERMAT still runs without DTH tokens. Document
this in PI brief and `changelog_v0.1.md`.

---

## 7. Random sample extraction

Confirm the environment supports randomization at scale.

| approach                                | works | note |
|-----------------------------------------|-------|------|
| `ORDER BY RANDOM() LIMIT 10000`         |       |      |
| `TABLESAMPLE BERNOULLI(p)` (PostgreSQL) |       |      |
| `USING SAMPLE 10000 ROWS` (DuckDB)      |       |      |
| Pre-extracted cohort table              |       |      |

CDM Playground usually exposes only one. Pick the one that returns
quickly on `person` (which has the smallest row count among the
relevant tables).

---

## 8. Export path

How will the sample tables exit the analysis environment?

| option                                            | available | note |
|---------------------------------------------------|-----------|------|
| Direct DuckDB / Parquet / CSV file download       |           |      |
| Mounted shared volume                             |           |      |
| Approved researcher workstation copy              |           |      |
| Only result aggregates can leave (not raw rows)   |           |      |

The last case is consistent with BOBIC's air-lock policy described in
the meeting notes. In that case, **the entire FERMAT preprocess +
training must happen inside the analysis environment**, and only
model checkpoints and summary statistics are exported. Confirm with
the data operator (정집민 단장님 / 보건의료정보원) before extraction.

---

## 9. Output target

Record the planned target DuckDB filename and path.

| field                        | value |
|------------------------------|-------|
| target DuckDB filename       | `data/synthetic_snuh_raw.duckdb` |
| target table names           | `person`, `visit_occurrence`, `condition_occurrence`, `drug_exposure`, `procedure_occurrence`, `measurement`, `death` |
| sample size (persons)        | 10,000 |
| date filter                  | none in v0.1 |
| diagnosis filter             | none in v0.1 |

The default path is what `scripts/preprocess_synthetic_snuh_to_fermat.py`
expects. Do not rename without also passing `--duckdb` on the CLI.

---

## 10. Sign-off

| checked by | date | environment | result |
|------------|------|-------------|--------|
|            |      |             |        |

Attach this completed checklist to the PI brief.
