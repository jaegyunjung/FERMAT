# Task 17: Genomic LAB Token Feasibility Audit

Run from the SNUH Pod under `/home/khdp-user/workspace/fermat-data` after
extracting the bundle:

```bash
python scripts/audit_snuh_genomic_variant_feasibility.py
```

The audit writes progress and failures to `logs/task17_run.log` and writes
checkpointed outputs under `outputs/task17_genomic_variant_audit`:

- `candidate_molecular_concepts.csv`
- `lab_gen_test_candidates.csv`
- `variant_ontology_review_only.csv`
- `lab_gen_observed_test_candidates.csv`
- `lab_gen_numeric_unit_candidates.csv`
- `lab_gen_categorical_result_candidates.csv`
- `variant_value_as_concept_hits.csv`
- `source_value_genomic_regex_top_values.csv`
- `source_value_scan_status.csv`
- `summary.json`

The default measurement scan is limited to likely genomic LAB test concepts
such as LOINC Lab Test, EDI Meas Class, SNUBH generated Lab Test, OMOP
Extension Lab Test, SNOMED Observable Entity, and CIEL Test. Variant ontologies
such as OMOP Genomic, ClinVar, OncoKB, JAX, CIViC, and NCIt are written for
review but are not scanned as test concepts by default.

The intended tokenization direction is LAB-like: a candidate test concept plus
its observed result value, not separate GEN_TEST / GEN_RESULT / GEN_PROC token
types.

## Post-2021 biomarker report-location finding

Follow-up Pod checks on 2026-06-23 showed a split between pre-2021 report-like
sources and post-2021 weak clinical-summary sources:

- 2004-2020 molecular/NGS/pathology-like biomarker artifacts are primarily
  `note` based. Existing parser outputs point to `source_table=note`,
  `source_id=note_id`, `note_source_value2=M ...` or pathology accessions, and
  note titles such as `분자병리 검사` and `차세대염기서열검사`.
- Those direct note titles continue through 2020 but were not found after the
  first few days of 2021 in the current CDM extract. General `note` rows
  continue through 2025, so this is not a whole-note-table cutoff.
- The only non-system schema visible in the database was `cdm2024_official`.
  Candidate report/text tables were limited to `note`, empty `note_nlp`,
  `bio_signal`, `specimen`, and `condition_occurrence` text extension fields.
- `bio_signal` samples were respiratory waveform files, not biomarker reports.
  `note_source_value4` and `ext_format_*` were dominated by non-biomarker
  structured formats such as ECG, body composition, and breast laterality
  fields.

Current interpretation: the authoritative molecular pathology or NGS report
route is available in `note` through 2020, but a post-2021 canonical report
route has not been identified inside the current CDM extract.

Post-2021 biomarker information is still present, but mainly as clinical-summary
text in:

```text
observation.observation_source_value = 기타
observation.observation_concept_id = 1340204
```

Stricter local parsing of that form found 336 classified event rows, 186
persons, and 275 unique person-token-value tuples. High-confidence examples
include `EGFR EXON19DEL`, `EGFR L858R`, `EGFR WT`, `ALK WT/positive`,
`PD-L1 TPS/CPS/%`, `BRAF V600E/WT`, `HER2`, `MSS/MMRd`, and a small number of
BRCA-related status mentions. These should be treated as weak clinical-summary
biomarker tokens, not as canonical report-derived tokens.

`condition_occurrence.ext_cond_source_value_cc_text` contains a small post-2021
`BRCA mutation` diagnosis/problem-list signal, but it is not a molecular report
result source.

## Form-discovery principle

Do not build Task 17 integration around hard-coded EGFR/ALK/KRAS searches alone.
Those terms were useful for finding the location of post-2021 biomarker-bearing
text, but the durable unit of discovery should be the source/form signature
first, followed by cautious value parsing within that form.

The Pod-side helper:

```bash
python scripts/discover_snuh_task17_biomarker_forms.py
```

profiles biomarker-bearing forms by table and source signature:

- `observation`: source value plus observation concept ID
- `note`: note title plus source value
- `condition_occurrence`: condition source value plus condition concept ID

It intentionally excludes broad `RET`/`MET` substring matching because those
terms match common non-biomarker text such as interpretation, preterm, ureter,
metastasis, and endometrial. Use its output to decide which forms should feed a
token parser, rather than treating marker keyword hits as the final artifact.

By default the script does not rescan the OMOP `measurement` table. It reads
Task 15 ETL frequency files from:

```text
/home/khdp-user/workspace/fermat-data/etl/patient_100pct_seed_42
```

Specifically it intersects genomic LAB test candidates with
`train_numeric_lab_stats.parquet` and `train_lab_categorical_frequency.parquet`.
This keeps the audit aligned with the current LAB tokenizer and avoids repeated
long database scans.

The default run also checks source/value text columns for top genomic regex
matches in a bounded patient sample. The default sample is 1% of patients,
selected by deterministic hash:

- `measurement.measurement_source_value`
- `measurement.value_source_value`
- `procedure_occurrence.procedure_source_value`
- `observation.observation_source_value`
- `observation.value_source_value`

These checks are top-value queries with a separate 5-minute statement timeout.
If a column times out, the run records the timeout in
`source_value_scan_status.csv` and continues. The source-value counts are sample
counts, not full-cohort counts.

Only use `--db-measurement-scan` if a slow, chunked direct DB scan is
intentionally needed.

Useful knobs:

```bash
python scripts/audit_snuh_genomic_variant_feasibility.py \
  --etl-dir /home/khdp-user/workspace/fermat-data/etl/patient_100pct_seed_42
```

To adjust source-value sampling:

```bash
python scripts/audit_snuh_genomic_variant_feasibility.py \
  --source-patient-buckets 2 \
  --source-statement-timeout 3min
```

To skip source/value text checks and only compare against Task 15 LAB frequency
files:

```bash
python scripts/audit_snuh_genomic_variant_feasibility.py --skip-source-value-scan
```

The script uses PostgreSQL `application_name`:

```text
fermat_genomic_variant_feasibility_audit_v2
```

If a run fails, check active database sessions with:

```bash
python - <<'PY'
import os, getpass, psycopg

password = os.environ.get("SNUH_CDM_PASSWORD") or getpass.getpass("SNUH CDM password: ")
conn = psycopg.connect(
    host=os.environ.get("SNUH_CDM_HOST", "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"),
    port=int(os.environ.get("SNUH_CDM_PORT", "5432")),
    dbname=os.environ.get("SNUH_CDM_DATABASE", "cdm"),
    user=os.environ.get("SNUH_CDM_USER", "jaegyun_jung"),
    password=password,
    sslmode=os.environ.get("SNUH_CDM_SSLMODE", "disable"),
    application_name="fermat_task17_check",
)
with conn.cursor() as cur:
    cur.execute("""
        SELECT pid, leader_pid, backend_type, client_addr, client_port,
               state, wait_event_type, wait_event,
               now() - query_start AS query_age
        FROM pg_stat_activity
        WHERE application_name = 'fermat_genomic_variant_feasibility_audit_v2'
        ORDER BY COALESCE(leader_pid, pid), pid;
    """)
    rows = cur.fetchall()
    print(rows if rows else "no task17 DB sessions")
PY
```
