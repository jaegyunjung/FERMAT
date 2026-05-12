-- =============================================================================
-- FERMAT v0.1: Sample extraction from CDM Playground / SNUH CDM
-- =============================================================================
--
-- Purpose
--   Extract a random sample of patients with their core OMOP CDM tables, so the
--   same preprocess_synthetic_snuh_to_fermat.py can run with minimal changes.
--
-- Target environment
--   - CDM Playground (Synthetic SNUH closed-environment variant) OR
--   - SNUH OMOP CDM (after IRB-exempt approval)
--   - OMOP CDM v5.4 schema assumed.
--   - Dialects: PostgreSQL / DuckDB / Spark SQL. Adjust LIMIT / RAND() as needed.
--
-- Conventions
--   - "sample_cohort" is materialized first; all event tables JOIN on it.
--   - Source-value columns are preferred (token_spec_v0.1 §5).
--   - Output is intended to be exported to DuckDB tables of the same name so the
--     existing preprocess script works with a single --duckdb argument.
--
-- Run order
--   1. SET @sample_size (or replace inline below)
--   2. CREATE TABLE sample_cohort
--   3. CREATE TABLE person_sample, condition_occurrence_sample, ...
--   4. Export each *_sample table as <table> in target DuckDB.
--
-- Caution
--   These queries are intentionally simple. They do NOT filter by date range,
--   diagnosis, or completeness. Tune per CDM Playground operator's guidance.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 0. Parameters
-- -----------------------------------------------------------------------------
-- Adjust to taste. Use the smallest value that still gives a meaningful smoke
-- test. v0.1 target: 10,000 persons.
--
-- PostgreSQL: replace :sample_size with literal, or use psql -v sample_size=...
-- DuckDB:     SET VARIABLE sample_size = 10000;
-- -----------------------------------------------------------------------------


-- -----------------------------------------------------------------------------
-- 1. sample_cohort — random person_id selection
-- -----------------------------------------------------------------------------
CREATE TABLE sample_cohort AS
SELECT person_id
FROM person
WHERE year_of_birth IS NOT NULL
  AND year_of_birth BETWEEN 1900 AND 2025
ORDER BY RANDOM()           -- PostgreSQL / DuckDB
LIMIT 10000;                -- replace with :sample_size


-- -----------------------------------------------------------------------------
-- 2. person_sample
-- -----------------------------------------------------------------------------
CREATE TABLE person_sample AS
SELECT
    p.person_id,
    p.gender_concept_id,
    p.gender_source_value,
    p.year_of_birth,
    p.month_of_birth,
    p.day_of_birth,
    p.birth_datetime
FROM person p
JOIN sample_cohort s USING (person_id);


-- -----------------------------------------------------------------------------
-- 3. visit_occurrence_sample
--   Used only for event filtering / summary in v0.1, not for tokens.
-- -----------------------------------------------------------------------------
CREATE TABLE visit_occurrence_sample AS
SELECT
    v.person_id,
    v.visit_occurrence_id,
    v.visit_concept_id,
    v.visit_source_value,
    v.visit_start_date,
    v.visit_end_date
FROM visit_occurrence v
JOIN sample_cohort s USING (person_id);


-- -----------------------------------------------------------------------------
-- 4. condition_occurrence_sample  →  DX tokens
-- -----------------------------------------------------------------------------
CREATE TABLE condition_occurrence_sample AS
SELECT
    c.person_id,
    c.condition_concept_id,
    c.condition_source_value,
    c.condition_start_date
FROM condition_occurrence c
JOIN sample_cohort s USING (person_id)
WHERE c.condition_start_date IS NOT NULL;


-- -----------------------------------------------------------------------------
-- 5. drug_exposure_sample  →  RX tokens
-- -----------------------------------------------------------------------------
CREATE TABLE drug_exposure_sample AS
SELECT
    d.person_id,
    d.drug_concept_id,
    d.drug_source_value,
    d.drug_exposure_start_date
FROM drug_exposure d
JOIN sample_cohort s USING (person_id)
WHERE d.drug_exposure_start_date IS NOT NULL;


-- -----------------------------------------------------------------------------
-- 6. procedure_occurrence_sample  →  PX tokens
-- -----------------------------------------------------------------------------
CREATE TABLE procedure_occurrence_sample AS
SELECT
    pr.person_id,
    pr.procedure_concept_id,
    pr.procedure_source_value,
    pr.procedure_date
FROM procedure_occurrence pr
JOIN sample_cohort s USING (person_id)
WHERE pr.procedure_date IS NOT NULL;


-- -----------------------------------------------------------------------------
-- 7. measurement_sample  →  LAB tokens (quantile 3-bin per token_spec §6)
-- -----------------------------------------------------------------------------
CREATE TABLE measurement_sample AS
SELECT
    m.person_id,
    m.measurement_concept_id,
    m.measurement_source_value,
    m.measurement_date,
    m.value_as_number,
    m.value_as_concept_id,
    m.unit_concept_id,
    m.unit_source_value
FROM measurement m
JOIN sample_cohort s USING (person_id)
WHERE m.measurement_date IS NOT NULL;


-- -----------------------------------------------------------------------------
-- 8. death_sample  →  DTH tokens
-- -----------------------------------------------------------------------------
CREATE TABLE death_sample AS
SELECT
    d.person_id,
    d.death_date,
    d.cause_concept_id,
    d.cause_source_value
FROM death d
JOIN sample_cohort s USING (person_id)
WHERE d.death_date IS NOT NULL;


-- -----------------------------------------------------------------------------
-- 9. Sanity counts (run after extraction; copy these numbers into PI brief)
-- -----------------------------------------------------------------------------
-- SELECT 'sample_cohort'           AS tbl, COUNT(*) FROM sample_cohort
-- UNION ALL SELECT 'person_sample',          COUNT(*) FROM person_sample
-- UNION ALL SELECT 'visit_occurrence_sample',COUNT(*) FROM visit_occurrence_sample
-- UNION ALL SELECT 'condition_occurrence_sample', COUNT(*) FROM condition_occurrence_sample
-- UNION ALL SELECT 'drug_exposure_sample',   COUNT(*) FROM drug_exposure_sample
-- UNION ALL SELECT 'procedure_occurrence_sample', COUNT(*) FROM procedure_occurrence_sample
-- UNION ALL SELECT 'measurement_sample',     COUNT(*) FROM measurement_sample
-- UNION ALL SELECT 'death_sample',           COUNT(*) FROM death_sample;


-- -----------------------------------------------------------------------------
-- 10. Export hint
-- -----------------------------------------------------------------------------
-- DuckDB:
--   COPY (SELECT * FROM person_sample) TO 'person.parquet' (FORMAT PARQUET);
--   etc. Then create a fresh DuckDB and CREATE TABLE person AS SELECT * FROM
--   read_parquet('person.parquet'); for each table.
-- Or, more simply:
--   ATTACH 'fermat_export.duckdb' AS target;
--   CREATE TABLE target.person                AS SELECT * FROM person_sample;
--   CREATE TABLE target.visit_occurrence      AS SELECT * FROM visit_occurrence_sample;
--   CREATE TABLE target.condition_occurrence  AS SELECT * FROM condition_occurrence_sample;
--   CREATE TABLE target.drug_exposure         AS SELECT * FROM drug_exposure_sample;
--   CREATE TABLE target.procedure_occurrence  AS SELECT * FROM procedure_occurrence_sample;
--   CREATE TABLE target.measurement           AS SELECT * FROM measurement_sample;
--   CREATE TABLE target.death                 AS SELECT * FROM death_sample;
--   DETACH target;
--
-- After export, point the FERMAT preprocess script at the resulting DuckDB:
--   python scripts/preprocess_synthetic_snuh_to_fermat.py \
--       --duckdb data/synthetic_snuh_raw.duckdb \
--       --out_dir data/synthetic_snuh
