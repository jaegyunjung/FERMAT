#!/usr/bin/env python3
"""Extract split-aware inputs for the four-arm HbA1c ADM CCW analysis.

Every expensive database result is checkpointed immediately to a private Pod
parquet.  A later failure can resume from the last completed query.  Public
analysis files contain FERMAT dense IDs, not OMOP person_id values.

This is an SNUH adaptation of Ko et al. (JAMA Network Open 2026): fasting
plasma glucose is omitted because fasting status is not encoded in SNUH.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
try:
    from psycopg import sql
except (ImportError, ModuleNotFoundError):  # local self-test does not need DB
    sql = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_snuh_task30_adm_timing_feasibility import (  # noqa: E402
    connect,
    execute,
    load_development_patient_map,
    prepare_output,
    query_df,
    upload_patient_map,
    upload_id_table,
    write_csv,
    write_json,
)
from audit_snuh_task30_hba1c_adm_timing_feasibility import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_REUSE_FROM,
    build_hba1c_cohort,
    load_reused_artifacts,
)


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_OUTPUT_DIR = POD_ROOT / "task30" / "outputs" / "hba1c_adm_ccw_inputs"
CHECKPOINT_SCHEMA_VERSION = "ccw-inputs-v3-monthly-clinical-nonadm"
COMPATIBLE_CHECKPOINT_SCHEMA_VERSIONS = {
    "ccw-inputs-v2-no-generic-measurement",
    CHECKPOINT_SCHEMA_VERSION,
}
DEFAULT_PHENOTYPE_GROUP_PLAN = (
    POD_ROOT
    / "task19"
    / "outputs"
    / "patient_phenotype_labels_wide"
    / "phenotype_group_concept_map.csv"
)
BASELINE_TABLES = {
    "visit": ("visit_occurrence", "visit_start_date", "visit_occurrence_id"),
    "condition": ("condition_occurrence", "condition_start_date", "condition_occurrence_id"),
}
COMORBIDITY_PHENOTYPES = (
    "atrial_fibrillation",
    "coronary_artery_disease",
    "chronic_kidney_disease",
    "diabetes",
    "dyslipidemia",
    "hypertension",
    "ischemic_stroke",
    "asthma",
    "depression_or_mood_disorder",
    "epilepsy",
    "retinal_disorder",
)
REQUIRED_COMORBIDITY_PHENOTYPES = {
    "atrial_fibrillation",
    "coronary_artery_disease",
    "chronic_kidney_disease",
    "diabetes",
    "dyslipidemia",
    "hypertension",
    "ischemic_stroke",
}
COMEDICATION_ATC_ROOTS = {
    "ace_inhibitors": ("C09A", "C09B"),
    "arbs": ("C09C", "C09D"),
    "beta_blockers": ("C07",),
    "calcium_channel_blockers": ("C08", "C07FB", "C09BB", "C09DB"),
    "loop_diuretics": ("C03C",),
    "thiazides": ("C03A", "C07B", "C07D"),
    "other_diuretics": ("C03DA", "C03E", "C03X", "C07C", "C08G"),
    "nitrates": ("C01DA",),
    "other_antihypertensives": ("C02",),
    "digoxin": ("C01AA",),
    "antiarrhythmics": ("C01B", "C01C"),
    "obstructive_airway_drugs": ("R03",),
    "statins": ("C10AA",),
    "other_lipid_lowering": ("C10AB", "C10AC", "C10AX"),
    # The supplement prints B01AF for antiplatelets and B10AE/B10AF for oral
    # anticoagulants. Those are internally inconsistent/nonexistent ATC roots;
    # use the clinically valid B01AC, B01AE and B01AF families instead.
    "antiplatelet_drugs": ("B01AC",),
    "oral_anticoagulants": ("B01AA", "B01AE", "B01AF", "B01AX"),
    "heparin": ("B01AB",),
    "nsaids": ("M01A",),
    "oral_steroids": ("A07EA",),
    "opioids": ("N02A",),
    "antidepressants": ("N06A",),
    "antipsychotics": ("N05A",),
    "anticonvulsants": ("N03",),
    "benzodiazepines": ("N05CD",),
    "anxiolytics_hypnotics": ("N05B",),
}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--reuse-from", type=Path, default=DEFAULT_REUSE_FROM)
    parser.add_argument(
        "--phenotype-group-plan", type=Path, default=DEFAULT_PHENOTYPE_GROUP_PLAN
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--checkpoint-reuse-from",
        type=Path,
        help="Copy only compatible completed v2 query checkpoints into a new v3 output",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--study-start", default="2013-01-01")
    parser.add_argument("--entry-end", default="2022-12-31")
    parser.add_argument("--db-end-date", default="2025-02-05")
    parser.add_argument("--minimum-age", type=int, default=18)
    parser.add_argument("--washout-days", type=int, default=365)
    parser.add_argument("--host", default="pg-2vge6u.vpc-cdb-kr.gov-ntruss.com")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="cdm")
    parser.add_argument("--user", default="jaegyun_jung")
    parser.add_argument("--schema", default="cdm2024_official")
    parser.add_argument("--sslmode", default="disable")
    parser.add_argument("--statement-timeout", default="0")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def atomic_parquet(frame, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    log(f"[WRITE] {path} rows={len(frame):,}")


def checkpointed_query(path, resume, query, required_columns, key_columns, expected_rows=None):
    """Load a validated completed query or run and atomically save it."""
    path = Path(path)
    if resume and path.is_file():
        frame = pd.read_parquet(path)
        source = "checkpoint"
    else:
        frame = query()
        source = "query"

    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} {path} missing columns: {missing}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(
            f"{source} {path} rows={len(frame):,}; expected={expected_rows:,}"
        )
    if frame[list(key_columns)].isna().any().any():
        raise ValueError(f"{source} {path} contains null keys")
    if frame.duplicated(list(key_columns)).any():
        raise ValueError(f"{source} {path} contains duplicate keys")

    if source == "checkpoint":
        log(f"[RESUME] reused query checkpoint {path.name} rows={len(frame):,}")
    else:
        atomic_parquet(frame, path)
    return frame


def prepare_checkpoint_version(raw_dir, resume):
    path = Path(raw_dir) / "checkpoint_schema_version.txt"
    if resume and path.is_file():
        observed = path.read_text(encoding="utf-8").strip()
        if observed not in COMPATIBLE_CHECKPOINT_SCHEMA_VERSIONS:
            raise ValueError(
                f"checkpoint schema mismatch: observed={observed!r}, "
                f"compatible={sorted(COMPATIBLE_CHECKPOINT_SCHEMA_VERSIONS)!r}"
            )
        if observed != CHECKPOINT_SCHEMA_VERSION:
            log(
                f"[MIGRATE] preserving compatible v2 checkpoints; "
                f"new/changed queries use new filenames"
            )
    path.write_text(CHECKPOINT_SCHEMA_VERSION + "\n", encoding="utf-8")
    log(f"[WRITE] {path}")


def import_compatible_checkpoints(source_dir, target_dir):
    source = Path(source_dir) / "raw_private"
    target = Path(target_dir)
    if not source.is_dir():
        raise NotADirectoryError(source)
    compatible = [
        "eligible_person_ids.parquet",
        "demographics_observation.parquet",
        "index_hba1c.parquet",
        "baseline_visit.parquet",
        "baseline_condition.parquet",
        "monthly_visit.parquet",
        "monthly_condition.parquet",
        "monthly_hba1c.parquet",
    ]
    copied = []
    for filename in compatible:
        source_path = source / filename
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        target_path = target / filename
        if not target_path.exists():
            shutil.copy2(source_path, target_path)
        copied.append(filename)
    log(f"[REUSE IMPORT] copied compatible checkpoints={','.join(copied)}")


def load_dense_patient_map(data_dir):
    path = Path(data_dir) / "patient_id_map.parquet"
    frame = pd.read_parquet(path, columns=["person_id", "patient_id_dense", "split"])
    frame = frame.loc[frame["split"].astype(str).isin(["train", "val"])].copy()
    for column in ("person_id", "patient_id_dense"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    if frame["person_id"].duplicated().any() or frame["patient_id_dense"].duplicated().any():
        raise ValueError("development patient map contains duplicate IDs")
    return frame


def load_comorbidity_concept_map(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    plan = pd.read_csv(path)
    if "phenotype" not in plan.columns:
        raise ValueError(f"{path} missing phenotype column")
    selected = plan.loc[plan["phenotype"].astype(str).isin(COMORBIDITY_PHENOTYPES)].copy()
    observed = set(selected["phenotype"].astype(str))
    absent = sorted(REQUIRED_COMORBIDITY_PHENOTYPES - observed)
    if absent:
        raise ValueError(f"required reviewed phenotypes absent from {path}: {absent}")
    if "condition_concept_id" in selected.columns:
        mapping = selected[["phenotype", "condition_concept_id"]].rename(
            columns={"phenotype": "feature"}
        )
        mapping["condition_concept_id"] = pd.to_numeric(
            mapping["condition_concept_id"], errors="raise"
        ).astype("int64")
        mapping = mapping.drop_duplicates()
    elif "concept_ids" in selected.columns:
        rows = []
        for item in selected.itertuples(index=False):
            for value in str(item.concept_ids).split("|"):
                value = value.strip()
                if value:
                    rows.append(
                        {"feature": str(item.phenotype), "condition_concept_id": int(value)}
                    )
        mapping = pd.DataFrame(rows).drop_duplicates()
    else:
        raise ValueError(f"{path} has neither condition_concept_id nor concept_ids")
    if mapping.empty:
        raise ValueError("reviewed phenotype plan produced no condition concepts")
    return mapping.sort_values(["feature", "condition_concept_id"]).reset_index(drop=True)


def upload_comorbidity_map(conn, mapping):
    execute(conn, "DROP TABLE IF EXISTS tmp_s2_comorbidity_concepts")
    execute(
        conn,
        """
        CREATE TEMP TABLE tmp_s2_comorbidity_concepts (
            feature text NOT NULL,
            condition_concept_id bigint NOT NULL,
            PRIMARY KEY(feature, condition_concept_id)
        ) ON COMMIT PRESERVE ROWS
        """,
        label="create reviewed comorbidity concept map",
    )
    with conn.cursor() as cursor:
        with cursor.copy(
            "COPY tmp_s2_comorbidity_concepts (feature, condition_concept_id) FROM STDIN"
        ) as copy:
            for row in mapping.itertuples(index=False):
                copy.write_row((str(row.feature), int(row.condition_concept_id)))
    conn.commit()
    execute(
        conn,
        "CREATE INDEX ON tmp_s2_comorbidity_concepts(condition_concept_id)",
    )


def create_comedication_map(conn, schema_name):
    schema = sql.Identifier(schema_name)
    requested_roots = [
        (feature, code)
        for feature, codes in COMEDICATION_ATC_ROOTS.items()
        for code in codes
    ]
    values = sql.SQL(", ").join(
        sql.SQL("({}, {})").format(sql.Literal(feature), sql.Literal(code))
        for feature, code in requested_roots
    )
    execute(conn, "DROP TABLE IF EXISTS tmp_s2_comedication_concepts")
    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE tmp_s2_comedication_concepts ON COMMIT PRESERVE ROWS AS
            WITH requested(feature, atc_code) AS (VALUES {values}),
            roots AS (
                SELECT r.feature, r.atc_code, c.concept_id AS root_concept_id
                FROM requested r
                JOIN {schema}.concept c
                  ON c.vocabulary_id = 'ATC'
                 AND c.concept_code = r.atc_code
                 AND c.invalid_reason IS NULL
            )
            SELECT DISTINCT roots.feature, roots.atc_code, roots.root_concept_id,
                   ca.descendant_concept_id AS drug_concept_id
            FROM roots
            JOIN {schema}.concept_ancestor ca
              ON ca.ancestor_concept_id = roots.root_concept_id
            JOIN {schema}.concept d
              ON d.concept_id = ca.descendant_concept_id
             AND d.domain_id = 'Drug'
             AND d.invalid_reason IS NULL
            """
        ).format(values=values, schema=schema),
        label="resolve prespecified non-ADM comedication classes",
    )
    execute(
        conn,
        "CREATE INDEX ON tmp_s2_comedication_concepts(drug_concept_id)",
    )
    coverage = query_df(
        conn,
        """
        SELECT requested.feature, requested.atc_code,
               COUNT(DISTINCT m.root_concept_id)::bigint AS root_concepts,
               COUNT(DISTINCT m.drug_concept_id)::bigint AS descendant_drug_concepts
        FROM (VALUES
        """
        + ",".join("(%s, %s)" for _ in requested_roots)
        + """
        ) AS requested(feature, atc_code)
        LEFT JOIN tmp_s2_comedication_concepts m
          ON m.feature = requested.feature
        GROUP BY requested.feature, requested.atc_code
        ORDER BY requested.feature
        """,
        tuple(
            value
            for item in requested_roots
            for value in item
        ),
        label="audit comedication concept coverage",
    )
    empty = coverage.loc[coverage["descendant_drug_concepts"].eq(0), "feature"].tolist()
    if empty:
        raise ValueError(f"ATC comedication classes resolved no drug descendants: {empty}")
    return coverage


def upload_private_cohort(conn, private):
    execute(conn, "DROP TABLE IF EXISTS tmp_s2_timing")
    execute(
        conn,
        """
        CREATE TEMP TABLE tmp_s2_timing (
            person_id bigint PRIMARY KEY,
            split text NOT NULL,
            index_date date NOT NULL,
            age integer NOT NULL,
            first_adm_date date,
            death_date date
        ) ON COMMIT PRESERVE ROWS
        """,
        label="create resumable HbA1c cohort table",
    )
    with conn.cursor() as cursor:
        with cursor.copy(
            "COPY tmp_s2_timing (person_id, split, index_date, age, first_adm_date, death_date) FROM STDIN"
        ) as copy:
            for row in private.itertuples(index=False):
                copy.write_row(
                    (
                        int(row.person_id),
                        str(row.split),
                        pd.Timestamp(row.index_date).date(),
                        int(row.age),
                        None if pd.isna(row.first_adm_date) else pd.Timestamp(row.first_adm_date).date(),
                        None if pd.isna(row.death_date) else pd.Timestamp(row.death_date).date(),
                    )
                )
    conn.commit()
    execute(conn, "CREATE INDEX ON tmp_s2_timing(split)")


def query_private_cohort(conn):
    return query_df(
        conn,
        """
        SELECT person_id, split, index_date, age, first_adm_date, death_date
        FROM tmp_s2_timing
        ORDER BY split, person_id
        """,
        label="checkpoint eligible cohort immediately after HbA1c scan",
    )


def create_month_grid(conn):
    execute(
        conn,
        """
        CREATE TEMP TABLE tmp_s2_month_grid ON COMMIT PRESERVE ROWS AS
        SELECT t.person_id, g.month,
               (t.index_date + make_interval(months => g.month))::date AS month_start,
               (t.index_date + make_interval(months => g.month + 1))::date AS month_end
        FROM tmp_s2_timing t
        CROSS JOIN generate_series(0, 11) AS g(month)
        """,
        label="create 12-month patient grid",
    )
    execute(conn, "CREATE UNIQUE INDEX ON tmp_s2_month_grid(person_id, month)")


def extract_demographics_and_observation(conn, schema_name):
    schema = sql.Identifier(schema_name)
    return query_df(
        conn,
        sql.SQL(
            """
            WITH observation_end AS (
                SELECT t.person_id,
                       MAX(o.observation_period_end_date)::date AS observation_end_date
                FROM tmp_s2_timing t
                JOIN {}.observation_period o
                  ON o.person_id = t.person_id
                 AND o.observation_period_start_date <= t.index_date
                 AND o.observation_period_end_date >= t.index_date
                GROUP BY t.person_id
            )
            SELECT t.person_id, p.gender_concept_id,
                   EXTRACT(YEAR FROM t.index_date)::int AS index_year,
                   (t.index_date - make_date(
                       p.year_of_birth,
                       CASE WHEN p.month_of_birth BETWEEN 1 AND 12
                            THEN p.month_of_birth ELSE 7 END,
                       CASE WHEN p.day_of_birth BETWEEN 1 AND 28
                            THEN p.day_of_birth ELSE 1 END
                   ))::int AS index_age_days,
                   o.observation_end_date
            FROM tmp_s2_timing t
            JOIN {}.person p USING(person_id)
            JOIN observation_end o USING(person_id)
            """
        ).format(schema, schema),
        label="extract demographics and observation end",
    )


def extract_index_hba1c(conn, schema_name):
    schema = sql.Identifier(schema_name)
    return query_df(
        conn,
        sql.SQL(
            """
            SELECT t.person_id, AVG(m.value_as_number)::double precision AS index_hba1c
            FROM tmp_s2_timing t
            JOIN {}.measurement m
              ON m.person_id = t.person_id
             AND m.measurement_date = t.index_date
             AND m.measurement_concept_id = 3004410
             AND m.unit_concept_id = 8554
             AND m.value_as_number >= 6.5
            GROUP BY t.person_id
            """
        ).format(schema),
        label="extract HbA1c value at time zero",
    )


def extract_baseline_count(conn, schema_name, prefix, table, date_column, id_column):
    schema = sql.Identifier(schema_name)
    return query_df(
        conn,
        sql.SQL(
            """
            SELECT t.person_id,
                   COUNT(x.{id})::bigint AS {rows_alias},
                   COUNT(DISTINCT x.{date})::bigint AS {days_alias}
            FROM tmp_s2_timing t
            LEFT JOIN {schema}.{table} x
              ON x.person_id = t.person_id
             AND x.{date} >= t.index_date - INTERVAL '365 days'
             AND x.{date} < t.index_date
            GROUP BY t.person_id
            """
        ).format(
            id=sql.Identifier(id_column),
            rows_alias=sql.Identifier(f"{prefix}_rows_1y"),
            days_alias=sql.Identifier(f"{prefix}_days_1y"),
            date=sql.Identifier(date_column),
            schema=schema,
            table=sql.Identifier(table),
        ),
        label=f"extract baseline {prefix} utilization",
    )


def extract_monthly_count(conn, schema_name, prefix, table, date_column, id_column):
    schema = sql.Identifier(schema_name)
    return query_df(
        conn,
        sql.SQL(
            """
            SELECT g.person_id, g.month,
                   COUNT(x.{id})::bigint AS {rows_alias},
                   COUNT(DISTINCT x.{date})::bigint AS {days_alias}
            FROM tmp_s2_month_grid g
            LEFT JOIN {schema}.{table} x
              ON x.person_id = g.person_id
             AND x.{date} >= g.month_start
             AND x.{date} < g.month_end
            GROUP BY g.person_id, g.month
            ORDER BY g.person_id, g.month
            """
        ).format(
            id=sql.Identifier(id_column),
            rows_alias=sql.Identifier(f"{prefix}_rows"),
            days_alias=sql.Identifier(f"{prefix}_days"),
            date=sql.Identifier(date_column),
            schema=schema,
            table=sql.Identifier(table),
        ),
        label=f"extract monthly {prefix} utilization",
    )


def extract_baseline_nonadm_drug_count(conn, schema_name):
    schema = sql.Identifier(schema_name)
    return query_df(
        conn,
        sql.SQL(
            """
            SELECT t.person_id,
                   COUNT(d.drug_exposure_id) FILTER (
                       WHERE adm.drug_concept_id IS NULL
                   )::bigint AS nonadm_drug_rows_1y,
                   COUNT(DISTINCT d.drug_exposure_start_date) FILTER (
                       WHERE adm.drug_concept_id IS NULL
                   )::bigint AS nonadm_drug_days_1y
            FROM tmp_s2_timing t
            LEFT JOIN {schema}.drug_exposure d
              ON d.person_id = t.person_id
             AND d.drug_exposure_start_date >= t.index_date - INTERVAL '365 days'
             AND d.drug_exposure_start_date < t.index_date
            LEFT JOIN tmp_s2_adm_concepts adm
              ON adm.drug_concept_id = d.drug_concept_id
            GROUP BY t.person_id
            """
        ).format(schema=schema),
        label="extract baseline non-ADM drug utilization",
    )


def extract_monthly_nonadm_drug_count(conn, schema_name):
    schema = sql.Identifier(schema_name)
    return query_df(
        conn,
        sql.SQL(
            """
            SELECT g.person_id, g.month,
                   COUNT(d.drug_exposure_id) FILTER (
                       WHERE adm.drug_concept_id IS NULL
                   )::bigint AS nonadm_drug_rows,
                   COUNT(DISTINCT d.drug_exposure_start_date) FILTER (
                       WHERE adm.drug_concept_id IS NULL
                   )::bigint AS nonadm_drug_days
            FROM tmp_s2_month_grid g
            LEFT JOIN {schema}.drug_exposure d
              ON d.person_id = g.person_id
             AND d.drug_exposure_start_date >= g.month_start
             AND d.drug_exposure_start_date < g.month_end
            LEFT JOIN tmp_s2_adm_concepts adm
              ON adm.drug_concept_id = d.drug_concept_id
            GROUP BY g.person_id, g.month
            ORDER BY g.person_id, g.month
            """
        ).format(schema=schema),
        label="extract monthly non-ADM drug utilization",
    )


def extract_baseline_visit_types(conn, schema_name):
    schema = sql.Identifier(schema_name)
    return query_df(
        conn,
        sql.SQL(
            """
            SELECT t.person_id,
                   COUNT(v.visit_occurrence_id) FILTER (
                       WHERE v.visit_concept_id = 9201
                   )::bigint AS inpatient_visits_1y,
                   COUNT(v.visit_occurrence_id) FILTER (
                       WHERE v.visit_concept_id IN (9202, 9203)
                   )::bigint AS ambulatory_ed_visits_1y
            FROM tmp_s2_timing t
            LEFT JOIN {schema}.visit_occurrence v
              ON v.person_id = t.person_id
             AND v.visit_start_date >= t.index_date - INTERVAL '365 days'
             AND v.visit_start_date < t.index_date
            GROUP BY t.person_id
            """
        ).format(schema=schema),
        label="extract baseline inpatient and ambulatory/ED use",
    )


def extract_monthly_visit_types(conn, schema_name):
    schema = sql.Identifier(schema_name)
    return query_df(
        conn,
        sql.SQL(
            """
            SELECT g.person_id, g.month,
                   COUNT(v.visit_occurrence_id) FILTER (
                       WHERE v.visit_concept_id = 9201
                   )::bigint AS inpatient_visits,
                   COUNT(v.visit_occurrence_id) FILTER (
                       WHERE v.visit_concept_id IN (9202, 9203)
                   )::bigint AS ambulatory_ed_visits
            FROM tmp_s2_month_grid g
            LEFT JOIN {schema}.visit_occurrence v
              ON v.person_id = g.person_id
             AND v.visit_start_date >= g.month_start
             AND v.visit_start_date < g.month_end
            GROUP BY g.person_id, g.month
            ORDER BY g.person_id, g.month
            """
        ).format(schema=schema),
        label="extract monthly inpatient and ambulatory/ED use",
    )


def extract_baseline_feature_flags(conn, schema_name, kind):
    schema = sql.Identifier(schema_name)
    if kind == "condition":
        table = sql.Identifier("condition_occurrence")
        date = sql.Identifier("condition_start_date")
        concept = sql.Identifier("condition_concept_id")
        mapping = sql.Identifier("tmp_s2_comorbidity_concepts")
    elif kind == "medication":
        table = sql.Identifier("drug_exposure")
        date = sql.Identifier("drug_exposure_start_date")
        concept = sql.Identifier("drug_concept_id")
        mapping = sql.Identifier("tmp_s2_comedication_concepts")
    else:
        raise ValueError(kind)
    return query_df(
        conn,
        sql.SQL(
            """
            SELECT t.person_id, m.feature, 1::smallint AS present
            FROM tmp_s2_timing t
            JOIN {schema}.{table} x
              ON x.person_id = t.person_id
             AND x.{date} >= t.index_date - INTERVAL '365 days'
             AND x.{date} < t.index_date
            JOIN {mapping} m ON m.{concept} = x.{concept}
            GROUP BY t.person_id, m.feature
            ORDER BY t.person_id, m.feature
            """
        ).format(
            schema=schema, table=table, date=date, mapping=mapping, concept=concept
        ),
        label=f"extract baseline {kind} flags",
    )


def extract_monthly_feature_flags(conn, schema_name, kind):
    schema = sql.Identifier(schema_name)
    if kind == "condition":
        table = sql.Identifier("condition_occurrence")
        date = sql.Identifier("condition_start_date")
        concept = sql.Identifier("condition_concept_id")
        mapping = sql.Identifier("tmp_s2_comorbidity_concepts")
    elif kind == "medication":
        table = sql.Identifier("drug_exposure")
        date = sql.Identifier("drug_exposure_start_date")
        concept = sql.Identifier("drug_concept_id")
        mapping = sql.Identifier("tmp_s2_comedication_concepts")
    else:
        raise ValueError(kind)
    return query_df(
        conn,
        sql.SQL(
            """
            SELECT g.person_id, g.month, m.feature, 1::smallint AS present
            FROM tmp_s2_month_grid g
            JOIN {schema}.{table} x
              ON x.person_id = g.person_id
             AND x.{date} >= g.month_start
             AND x.{date} < g.month_end
            JOIN {mapping} m ON m.{concept} = x.{concept}
            GROUP BY g.person_id, g.month, m.feature
            ORDER BY g.person_id, g.month, m.feature
            """
        ).format(
            schema=schema, table=table, date=date, mapping=mapping, concept=concept
        ),
        label=f"extract monthly {kind} flags",
    )


def extract_monthly_hba1c(conn, schema_name):
    schema = sql.Identifier(schema_name)
    return query_df(
        conn,
        sql.SQL(
            """
            SELECT g.person_id, g.month,
                   AVG(m.value_as_number)::double precision AS month_hba1c
            FROM tmp_s2_month_grid g
            LEFT JOIN {}.measurement m
              ON m.person_id = g.person_id
             AND m.measurement_date >= g.month_start
             AND m.measurement_date < g.month_end
             AND m.measurement_concept_id = 3004410
             AND m.unit_concept_id = 8554
             AND m.value_as_number IS NOT NULL
            GROUP BY g.person_id, g.month
            ORDER BY g.person_id, g.month
            """
        ).format(schema),
        label="extract monthly HbA1c",
    )


def pivot_baseline_flags(frame, patient_ids, prefix, features):
    index = pd.Index(pd.Series(patient_ids, dtype="int64").unique(), name="person_id")
    if frame.empty:
        wide = pd.DataFrame(index=index)
    else:
        wide = frame.pivot(index="person_id", columns="feature", values="present")
        wide = wide.reindex(index=index)
    expected = [f"{prefix}_{feature}" for feature in features]
    wide = wide.rename(columns=lambda value: f"{prefix}_{value}")
    return wide.reindex(columns=expected, fill_value=0).fillna(0).astype("int8").reset_index()


def pivot_monthly_flags(frame, patient_ids, prefix, features):
    grid = pd.MultiIndex.from_product(
        [pd.Series(patient_ids, dtype="int64").unique(), range(12)],
        names=["person_id", "month"],
    )
    if frame.empty:
        wide = pd.DataFrame(index=grid)
    else:
        wide = frame.pivot(
            index=["person_id", "month"], columns="feature", values="present"
        ).reindex(grid)
    expected = [f"{prefix}_{feature}" for feature in features]
    wide = wide.rename(columns=lambda value: f"{prefix}_{value}")
    return wide.reindex(columns=expected, fill_value=0).fillna(0).astype("int8").reset_index()


def build_lagged_monthly(monthly, patients, condition_features, medication_features):
    monthly = monthly.sort_values(["patient_key", "month"]).copy()
    patient_index = patients.set_index("patient_key")
    baseline_hba1c = patient_index["index_hba1c"]
    monthly["month_hba1c"] = monthly.groupby("patient_key")["month_hba1c"].ffill()
    monthly["lag_hba1c"] = monthly.groupby("patient_key")["month_hba1c"].shift(1)
    monthly["lag_hba1c"] = monthly["lag_hba1c"].fillna(
        monthly["patient_key"].map(baseline_hba1c)
    )
    for prefix in BASELINE_TABLES:
        monthly[f"lag_{prefix}_rows_30d"] = (
            monthly.groupby("patient_key")[f"{prefix}_rows"].shift(1).fillna(0)
        )
        monthly[f"lag_{prefix}_days_30d"] = (
            monthly.groupby("patient_key")[f"{prefix}_days"].shift(1).fillna(0)
        )
    for value in ("nonadm_drug_rows", "nonadm_drug_days", "inpatient_visits", "ambulatory_ed_visits"):
        monthly[f"lag_{value}_30d"] = (
            monthly.groupby("patient_key")[value].shift(1).fillna(0)
        )
    for feature in condition_features:
        raw = f"month_cond_{feature}"
        baseline = f"baseline_cond_{feature}"
        history = monthly.groupby("patient_key")[raw].cummax()
        lagged = history.groupby(monthly["patient_key"]).shift(1)
        monthly[f"lag_cond_{feature}"] = lagged.fillna(
            monthly["patient_key"].map(patient_index[baseline])
        )
    for feature in medication_features:
        raw = f"month_med_{feature}"
        baseline = f"baseline_med_{feature}"
        monthly[f"lag_med_{feature}"] = (
            monthly.groupby("patient_key")[raw].shift(1).fillna(
                monthly["patient_key"].map(patient_index[baseline])
            )
        )
    monthly["lag_age"] = monthly["patient_key"].map(patient_index["age"]) + monthly["month"] / 12.0
    monthly["lag_calendar_year"] = (
        monthly["patient_key"].map(patient_index["index_year"]) + monthly["month"] / 12.0
    )
    return monthly


def validate_outputs(patients, monthly):
    if patients["patient_key"].duplicated().any():
        raise ValueError("duplicate patient_key in patient output")
    if set(patients["split"].astype(str).unique()) - {"train", "val"}:
        raise ValueError("non-development split reached CCW input")
    expected = len(patients) * 12
    if len(monthly) != expected:
        raise ValueError(f"monthly rows={len(monthly):,}; expected={expected:,}")
    if monthly[["patient_key", "month"]].duplicated().any():
        raise ValueError("duplicate patient-month row")
    if monthly["month"].min() != 0 or monthly["month"].max() != 11:
        raise ValueError("month grid is not exactly 0..11")
    if patients["person_id"].isna().any() or patients["patient_key"].isna().any():
        raise ValueError("patient mapping is incomplete")


def assemble_outputs(
    private,
    patient_parts,
    monthly_parts,
    dense_map,
    condition_features=(),
    medication_features=(),
):
    """Assemble, rename, lag, and validate exactly as the Pod path does."""
    patients = private.copy()
    for part in patient_parts:
        patients = patients.merge(part, on="person_id", how="left", validate="one_to_one")
    patients = patients.merge(
        dense_map, on=["person_id", "split"], how="left", validate="one_to_one"
    )
    patients = patients.rename(columns={"patient_id_dense": "patient_key"})
    patients["gender_concept_id"] = (
        patients["gender_concept_id"].fillna(0).astype("int64").astype(str)
    )
    patients["observation_end_date"] = pd.to_datetime(patients["observation_end_date"])

    monthly = monthly_parts[0].copy()
    for part in monthly_parts[1:]:
        monthly = monthly.merge(
            part, on=["person_id", "month"], how="left", validate="one_to_one"
        )
    monthly = monthly.merge(
        patients[["person_id", "patient_key", "split", "index_hba1c"]],
        on="person_id",
        how="left",
        validate="many_to_one",
    )
    monthly = build_lagged_monthly(
        monthly, patients, condition_features, medication_features
    )
    validate_outputs(patients, monthly)
    return patients, monthly


def self_test():
    private = pd.DataFrame(
        {
            "person_id": [1, 2],
            "split": ["train", "val"],
            "index_date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "age": [55, 65],
            "first_adm_date": pd.to_datetime(["2020-02-01", pd.NaT]),
            "death_date": pd.to_datetime([pd.NaT, "2021-01-01"]),
        }
    )
    dense_map = pd.DataFrame(
        {"person_id": [1, 2], "patient_id_dense": [10, 20], "split": ["train", "val"]}
    )
    patient_parts = [
        pd.DataFrame(
            {
                "person_id": [1, 2],
                "gender_concept_id": [8507, 8532],
                "index_year": [2020, 2020],
                "index_age_days": [20000, 24000],
                "observation_end_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            }
        ),
        pd.DataFrame({"person_id": [1, 2], "index_hba1c": [6.6, 7.2]}),
    ]
    for prefix in BASELINE_TABLES:
        patient_parts.append(
            pd.DataFrame(
                {
                    "person_id": [1, 2],
                    f"{prefix}_rows_1y": [2, 3],
                    f"{prefix}_days_1y": [1, 2],
                }
            )
        )
    patient_parts.extend(
        [
            pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "nonadm_drug_rows_1y": [4, 5],
                    "nonadm_drug_days_1y": [2, 3],
                }
            ),
            pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "inpatient_visits_1y": [0, 1],
                    "ambulatory_ed_visits_1y": [2, 3],
                }
            ),
        ]
    )
    patient_parts.extend(
        [
            pivot_baseline_flags(
                pd.DataFrame({"person_id": [1], "feature": ["hypertension"], "present": [1]}),
                private["person_id"],
                "baseline_cond",
                ["hypertension"],
            ),
            pivot_baseline_flags(
                pd.DataFrame({"person_id": [2], "feature": ["statins"], "present": [1]}),
                private["person_id"],
                "baseline_med",
                ["statins"],
            ),
        ]
    )

    monthly_parts = []
    for prefix in BASELINE_TABLES:
        rows = []
        for person_id in (1, 2):
            for month in range(12):
                rows.append(
                    {
                        "person_id": person_id,
                        "month": month,
                        f"{prefix}_rows": month,
                        f"{prefix}_days": min(month, 2),
                    }
                )
        monthly_parts.append(pd.DataFrame(rows))
    monthly_parts.append(
        pd.DataFrame(
            [
                {
                    "person_id": person_id,
                    "month": month,
                    "nonadm_drug_rows": month,
                    "nonadm_drug_days": min(month, 2),
                }
                for person_id in (1, 2)
                for month in range(12)
            ]
        )
    )
    monthly_parts.extend(
        [
            pivot_monthly_flags(
                pd.DataFrame(
                    {"person_id": [2], "month": [1], "feature": ["hypertension"], "present": [1]}
                ),
                private["person_id"],
                "month_cond",
                ["hypertension"],
            ),
            pivot_monthly_flags(
                pd.DataFrame(
                    {"person_id": [1], "month": [1], "feature": ["statins"], "present": [1]}
                ),
                private["person_id"],
                "month_med",
                ["statins"],
            ),
        ]
    )
    monthly_parts.append(
        pd.DataFrame(
            [
                {
                    "person_id": person_id,
                    "month": month,
                    "inpatient_visits": int(month % 5 == 0),
                    "ambulatory_ed_visits": month % 3,
                }
                for person_id in (1, 2)
                for month in range(12)
            ]
        )
    )
    rows = []
    for person_id in (1, 2):
        for month in range(12):
            rows.append({"person_id": person_id, "month": month, "month_hba1c": np.nan})
    monthly_parts.append(pd.DataFrame(rows))

    patients, monthly = assemble_outputs(
        private,
        patient_parts,
        monthly_parts,
        dense_map,
        ["hypertension"],
        ["statins"],
    )
    assert monthly.loc[(monthly.patient_key == 10) & (monthly.month == 0), "lag_hba1c"].iloc[0] == 6.6
    assert monthly.loc[(monthly.patient_key == 10) & (monthly.month == 1), "lag_visit_rows_30d"].iloc[0] == 0
    assert "patient_id_dense" not in patients.columns
    assert "lag_nonadm_drug_rows_30d" in monthly.columns
    assert "lag_drug_rows_30d" not in monthly.columns
    assert monthly.loc[(monthly.patient_key == 10) & (monthly.month == 0), "lag_cond_hypertension"].iloc[0] == 1
    assert monthly.loc[(monthly.patient_key == 10) & (monthly.month == 2), "lag_med_statins"].iloc[0] == 1
    assert patients["patient_key"].tolist() == [10, 20]

    with TemporaryDirectory() as directory:
        prepare_checkpoint_version(directory, False)
        prepare_checkpoint_version(directory, True)
        checkpoint = Path(directory) / "query.parquet"
        sample = pd.DataFrame({"person_id": [1, 2], "value": [3, 4]})
        checkpointed_query(
            checkpoint,
            False,
            lambda: sample.copy(),
            ["person_id", "value"],
            ["person_id"],
            2,
        )

        def must_not_run():
            raise AssertionError("resume executed a completed query")

        resumed = checkpointed_query(
            checkpoint,
            True,
            must_not_run,
            ["person_id", "value"],
            ["person_id"],
            2,
        )
        assert resumed.equals(sample)

        input_dir = Path(directory) / "runner_inputs"
        input_dir.mkdir()
        patients.drop(columns=["person_id"]).to_parquet(
            input_dir / "ccw_patients.parquet", index=False
        )
        monthly.drop(columns=["person_id", "index_hba1c"]).to_parquet(
            input_dir / "ccw_monthly_covariates.parquet", index=False
        )
        from run_snuh_task30_hba1c_adm_ccw import (  # noqa: E402
            attach_monthly_covariates,
            load_inputs,
        )
        from snuh_task30_adm_ccw_core import build_month_intervals, clone_patients  # noqa: E402

        loaded_patients, loaded_monthly, _, _, spec, _ = load_inputs(input_dir)
        intervals = attach_monthly_covariates(
            build_month_intervals(clone_patients(loaded_patients)),
            loaded_monthly,
            spec["time_varying_numeric"],
        )
        assert not intervals.empty
    print(
        "SELF_TEST_OK rows=24 merge=verified rename=verified "
        "validation=verified checkpoint_resume=verified runner_interface=verified"
    )


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if sql is None:
        raise RuntimeError("psycopg is required for a Pod database extraction")
    if args.resume:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        prepare_output(args.output_dir, args.overwrite)
    raw_dir = args.output_dir / "raw_private"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_reuse_from is not None:
        if args.resume:
            raise ValueError("use either --resume or --checkpoint-reuse-from, not both")
        import_compatible_checkpoints(args.checkpoint_reuse_from, raw_dir)
    reuse_queries = bool(args.resume or args.checkpoint_reuse_from is not None)
    prepare_checkpoint_version(raw_dir, reuse_queries)
    private_path = raw_dir / "eligible_person_ids.parquet"
    dense_map = load_dense_patient_map(args.data_dir)
    comorbidity_map = load_comorbidity_concept_map(args.phenotype_group_plan)
    condition_features = sorted(comorbidity_map["feature"].unique().tolist())
    medication_features = sorted(COMEDICATION_ATC_ROOTS)
    write_csv(
        comorbidity_map.groupby("feature", as_index=False)
        .agg(condition_concepts=("condition_concept_id", "nunique")),
        args.output_dir / "comorbidity_concept_coverage.csv",
    )

    reused, adm_ids, type1_ids = load_reused_artifacts(args.reuse_from, args.output_dir)
    development_map, patient_map_path, observed_splits = load_development_patient_map(args.data_dir)

    with connect(args) as conn:
        upload_patient_map(conn, development_map)
        # This table is required even on resume: changed v3 drug-utilization
        # queries explicitly remove the treatment exposure from covariates.
        upload_id_table(conn, "tmp_s2_adm_concepts", "drug_concept_id", adm_ids)
        upload_comorbidity_map(conn, comorbidity_map)
        comedication_coverage = create_comedication_map(conn, args.schema)
        write_csv(
            comedication_coverage,
            args.output_dir / "comedication_concept_coverage.csv",
        )
        if reuse_queries and private_path.is_file():
            private = pd.read_parquet(private_path)
            upload_private_cohort(conn, private)
            log(f"[RESUME] reused expensive HbA1c cohort checkpoint rows={len(private):,}")
        else:
            build_hba1c_cohort(conn, args, adm_ids, type1_ids, args.output_dir)
            private = query_private_cohort(conn)
            atomic_parquet(private, private_path)

        create_month_grid(conn)
        patient_count = len(private)
        month_count = patient_count * 12
        patient_parts = [
            checkpointed_query(
                raw_dir / "demographics_observation.parquet",
                reuse_queries,
                lambda: extract_demographics_and_observation(conn, args.schema),
                [
                    "person_id",
                    "gender_concept_id",
                    "index_year",
                    "index_age_days",
                    "observation_end_date",
                ],
                ["person_id"],
                patient_count,
            ),
            checkpointed_query(
                raw_dir / "index_hba1c.parquet",
                reuse_queries,
                lambda: extract_index_hba1c(conn, args.schema),
                ["person_id", "index_hba1c"],
                ["person_id"],
                patient_count,
            ),
        ]
        for prefix, (table, date_column, id_column) in BASELINE_TABLES.items():
            patient_parts.append(
                checkpointed_query(
                    raw_dir / f"baseline_{prefix}.parquet",
                    reuse_queries,
                    lambda prefix=prefix, table=table, date_column=date_column, id_column=id_column: extract_baseline_count(
                        conn, args.schema, prefix, table, date_column, id_column
                    ),
                    ["person_id", f"{prefix}_rows_1y", f"{prefix}_days_1y"],
                    ["person_id"],
                    patient_count,
                )
            )
        patient_parts.extend(
            [
                checkpointed_query(
                    raw_dir / "baseline_nonadm_drug_v3.parquet",
                    reuse_queries,
                    lambda: extract_baseline_nonadm_drug_count(conn, args.schema),
                    ["person_id", "nonadm_drug_rows_1y", "nonadm_drug_days_1y"],
                    ["person_id"],
                    patient_count,
                ),
                checkpointed_query(
                    raw_dir / "baseline_visit_types_v3.parquet",
                    reuse_queries,
                    lambda: extract_baseline_visit_types(conn, args.schema),
                    ["person_id", "inpatient_visits_1y", "ambulatory_ed_visits_1y"],
                    ["person_id"],
                    patient_count,
                ),
            ]
        )
        baseline_condition_sparse = checkpointed_query(
            raw_dir / "baseline_comorbidity_flags_v3.parquet",
            reuse_queries,
            lambda: extract_baseline_feature_flags(conn, args.schema, "condition"),
            ["person_id", "feature", "present"],
            ["person_id", "feature"],
        )
        baseline_medication_sparse = checkpointed_query(
            raw_dir / "baseline_comedication_flags_v3.parquet",
            reuse_queries,
            lambda: extract_baseline_feature_flags(conn, args.schema, "medication"),
            ["person_id", "feature", "present"],
            ["person_id", "feature"],
        )
        patient_parts.extend(
            [
                pivot_baseline_flags(
                    baseline_condition_sparse,
                    private["person_id"],
                    "baseline_cond",
                    condition_features,
                ),
                pivot_baseline_flags(
                    baseline_medication_sparse,
                    private["person_id"],
                    "baseline_med",
                    medication_features,
                ),
            ]
        )

        monthly_parts = []
        for prefix, (table, date_column, id_column) in BASELINE_TABLES.items():
            monthly_parts.append(
                checkpointed_query(
                    raw_dir / f"monthly_{prefix}.parquet",
                    reuse_queries,
                    lambda prefix=prefix, table=table, date_column=date_column, id_column=id_column: extract_monthly_count(
                        conn, args.schema, prefix, table, date_column, id_column
                    ),
                    ["person_id", "month", f"{prefix}_rows", f"{prefix}_days"],
                    ["person_id", "month"],
                    month_count,
                )
            )
        monthly_parts.extend(
            [
                checkpointed_query(
                    raw_dir / "monthly_nonadm_drug_v3.parquet",
                    reuse_queries,
                    lambda: extract_monthly_nonadm_drug_count(conn, args.schema),
                    ["person_id", "month", "nonadm_drug_rows", "nonadm_drug_days"],
                    ["person_id", "month"],
                    month_count,
                ),
                checkpointed_query(
                    raw_dir / "monthly_visit_types_v3.parquet",
                    reuse_queries,
                    lambda: extract_monthly_visit_types(conn, args.schema),
                    ["person_id", "month", "inpatient_visits", "ambulatory_ed_visits"],
                    ["person_id", "month"],
                    month_count,
                ),
            ]
        )
        monthly_condition_sparse = checkpointed_query(
            raw_dir / "monthly_comorbidity_flags_v3.parquet",
            reuse_queries,
            lambda: extract_monthly_feature_flags(conn, args.schema, "condition"),
            ["person_id", "month", "feature", "present"],
            ["person_id", "month", "feature"],
        )
        monthly_medication_sparse = checkpointed_query(
            raw_dir / "monthly_comedication_flags_v3.parquet",
            reuse_queries,
            lambda: extract_monthly_feature_flags(conn, args.schema, "medication"),
            ["person_id", "month", "feature", "present"],
            ["person_id", "month", "feature"],
        )
        monthly_parts.extend(
            [
                pivot_monthly_flags(
                    monthly_condition_sparse,
                    private["person_id"],
                    "month_cond",
                    condition_features,
                ),
                pivot_monthly_flags(
                    monthly_medication_sparse,
                    private["person_id"],
                    "month_med",
                    medication_features,
                ),
            ]
        )
        monthly_parts.append(
            checkpointed_query(
                raw_dir / "monthly_hba1c.parquet",
                reuse_queries,
                lambda: extract_monthly_hba1c(conn, args.schema),
                ["person_id", "month", "month_hba1c"],
                ["person_id", "month"],
                month_count,
            )
        )

    patients, monthly = assemble_outputs(
        private,
        patient_parts,
        monthly_parts,
        dense_map,
        condition_features,
        medication_features,
    )

    public_patients = patients.drop(columns=["person_id"]).copy()
    public_monthly = monthly.drop(columns=["person_id", "index_hba1c"]).copy()
    atomic_parquet(public_patients, args.output_dir / "ccw_patients.parquet")
    atomic_parquet(public_monthly, args.output_dir / "ccw_monthly_covariates.parquet")
    covariate_spec = {
        "baseline_numeric": [
            "age",
            "index_year",
            "index_hba1c",
            "visit_rows_1y",
            "visit_days_1y",
            "condition_rows_1y",
            "condition_days_1y",
            "nonadm_drug_rows_1y",
            "nonadm_drug_days_1y",
            "inpatient_visits_1y",
            "ambulatory_ed_visits_1y",
            *[f"baseline_cond_{value}" for value in condition_features],
            *[f"baseline_med_{value}" for value in medication_features],
        ],
        "baseline_categorical": ["gender_concept_id"],
        "time_varying_numeric": [
            "lag_hba1c",
            "lag_visit_rows_30d",
            "lag_visit_days_30d",
            "lag_condition_rows_30d",
            "lag_condition_days_30d",
            "lag_nonadm_drug_rows_30d",
            "lag_nonadm_drug_days_30d",
            "lag_inpatient_visits_30d",
            "lag_ambulatory_ed_visits_30d",
            *[f"lag_cond_{value}" for value in condition_features],
            *[f"lag_med_{value}" for value in medication_features],
        ],
        "treatment_exposure_excluded_from_drug_utilization": True,
        "excluded_redundant_time_varying_covariates": [
            "lag_age",
            "lag_calendar_year",
        ],
        "condition_source": str(args.phenotype_group_plan),
        "condition_features": condition_features,
        "comedication_atc_roots": COMEDICATION_ATC_ROOTS,
    }
    write_json(covariate_spec, args.output_dir / "covariate_spec.json")
    summary = (
        public_patients.groupby("split")
        .agg(
            patients=("patient_key", "size"),
            deaths=("death_date", lambda x: int(x.notna().sum())),
        )
        .reset_index()
    )
    # Timing counts need each patient's own calendar index.
    index = pd.to_datetime(public_patients["index_date"])
    adm = pd.to_datetime(public_patients["first_adm_date"])
    for months in (3, 6, 12):
        flag = adm.notna() & (adm <= index + pd.DateOffset(months=months))
        counts = flag.groupby(public_patients["split"]).sum().astype(int)
        summary[f"initiated_{months}m"] = summary["split"].map(counts).fillna(0).astype(int)
    write_csv(summary, args.output_dir / "input_summary.csv")
    manifest = {
        "status": "CCW_INPUTS_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "patient_map": str(patient_map_path),
        "observed_splits": observed_splits,
        "included_splits": ["train", "val"],
        "test_queried": False,
        "private_checkpoint": str(private_path),
        "checkpoint_reuse_from": (
            None if args.checkpoint_reuse_from is None else str(args.checkpoint_reuse_from)
        ),
        "private_checkpoint_contains_person_id": True,
        "public_outputs_contain_person_id": False,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "query_checkpoints": sorted(str(path) for path in raw_dir.glob("*.parquet")),
        "strategies": ["3m", "6m", "12m", "no initiation within 12m"],
        "outcome": "all-cause death",
        "covariate_spec": str(args.output_dir / "covariate_spec.json"),
        "covariate_scope": (
            "SNUH-adapted demographics, index/monthly HbA1c, explicit reviewed "
            "comorbidities, prespecified ATC comedications, health-care use, and "
            "non-ADM drug utilization"
        ),
        "treatment_exposure_excluded_from_drug_utilization": True,
    }
    write_json(manifest, args.output_dir / "manifest.json")
    log(f"[COMPLETE] {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
