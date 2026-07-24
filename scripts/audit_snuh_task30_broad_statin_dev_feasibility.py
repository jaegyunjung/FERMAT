#!/usr/bin/env python3
"""Split-aware development feasibility audit for a broad statin scenario.

The audit uses only FERMAT train and validation patients when querying clinical
events.  Test outcomes are not read.  It evaluates adults with either recorded
dyslipidaemia or an elevated LDL measurement, no earlier statin, and no earlier
CAD or ischaemic stroke.  Counts are reported by train, validation, and their
combined development population, with a 15/85 projection for the locked test
split.

This is an aggregate feasibility audit, not a causal-effect analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import pandas as pd
except (ImportError, ModuleNotFoundError):  # self-test does not need pandas
    pd = None

try:
    import psycopg
    from psycopg import sql
except (ImportError, ModuleNotFoundError):  # self-test does not need psycopg
    psycopg = None
    sql = None

from audit_snuh_task30_acei_ccb_mace_feasibility import (
    csv_text,
    load_registry,
    log,
    password,
    prepare_output,
    query_df,
    require_dependencies,
    write_csv,
    write_json,
)


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_CONCEPT_MAP = (
    POD_ROOT / "task19" / "outputs" / "patient_phenotype_labels_wide"
    / "phenotype_group_concept_map.csv"
)
DEFAULT_STATIN_AUDIT = (
    POD_ROOT / "task30" / "outputs" / "ckd_statin_feasibility_20260719_022424"
)
DEFAULT_OUTPUT_DIR = POD_ROOT / "task30" / "outputs" / "broad_statin_dev_feasibility"
PHENOTYPES = ("dyslipidemia", "coronary_artery_disease", "ischemic_stroke")
HORIZONS = (365, 1095, 1826)
DEV_SPLITS = ("train", "val")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--concept-map", type=Path, default=DEFAULT_CONCEPT_MAP)
    parser.add_argument("--statin-audit-dir", type=Path, default=DEFAULT_STATIN_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--minimum-age", type=int, default=40)
    parser.add_argument("--maximum-age", type=int, default=79)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--landmark-days", type=int, default=30)
    parser.add_argument("--ldl-mg-dl", type=float, default=100.0)
    parser.add_argument("--ldl-mmol-l", type=float, default=2.6)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--db-end-date", default="2025-02-05")
    parser.add_argument("--host", default=os.environ.get("SNUH_CDM_HOST", "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SNUH_CDM_PORT", "5432")))
    parser.add_argument("--dbname", default=os.environ.get("SNUH_CDM_DATABASE", "cdm"))
    parser.add_argument("--user", default=os.environ.get("SNUH_CDM_USER", "jaegyun_jung"))
    parser.add_argument("--schema", default=os.environ.get("SNUH_CDM_SCHEMA", "cdm2024_official"))
    parser.add_argument("--sslmode", default=os.environ.get("SNUH_CDM_SSLMODE", "disable"))
    parser.add_argument("--statement-timeout", default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def projected_test_count(dev_count, test_fraction=0.15):
    return float(dev_count) * float(test_fraction) / (1.0 - float(test_fraction))


def self_test():
    assert abs(projected_test_count(850) - 150.0) < 1e-9
    assert "RX:" + str(12345) == "RX:12345"
    print("SELF_TEST_OK")


def connect_database(args):
    conn = psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=password(),
        sslmode=args.sslmode,
        connect_timeout=15,
        application_name="fermat_task30_broad_statin_dev_feasibility",
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=6,
    )
    with conn.cursor() as cur:
        cur.execute("SELECT set_config('statement_timeout', %s, false)", (args.statement_timeout,))
    return conn


def load_patient_map(data_dir):
    path = data_dir / "patient_id_map.parquet"
    frame = pd.read_parquet(path, columns=["patient_id_dense", "person_id", "split"])
    if frame["person_id"].duplicated().any():
        raise ValueError("patient_id_map.parquet contains duplicate person_id values")
    if not set(frame["split"].dropna().unique()).issubset({"train", "val", "test"}):
        raise ValueError("patient map contains unexpected split values")
    counts = frame.groupby("split", sort=True).size().rename("patients").reset_index()
    dev = frame.loc[frame["split"].isin(DEV_SPLITS), ["person_id", "split"]].copy()
    dev["person_id"] = pd.to_numeric(dev["person_id"], errors="raise").astype("int64")
    return dev, counts, path


def load_phenotype_ids(path):
    frame = pd.read_csv(path)
    required = {"phenotype", "condition_concept_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Concept map missing columns: {sorted(missing)}")
    frame = frame.loc[frame["phenotype"].isin(PHENOTYPES)].copy()
    frame["condition_concept_id"] = pd.to_numeric(
        frame["condition_concept_id"], errors="raise"
    ).astype("int64")
    missing_phenotypes = sorted(set(PHENOTYPES) - set(frame["phenotype"]))
    if missing_phenotypes:
        raise ValueError(f"Missing reviewed phenotypes: {missing_phenotypes}")
    ids = {
        name: sorted(
            frame.loc[frame["phenotype"].eq(name), "condition_concept_id"]
            .drop_duplicates().astype(int).tolist()
        )
        for name in PHENOTYPES
    }
    coverage = (
        frame.groupby("phenotype", sort=True)["condition_concept_id"]
        .nunique().rename("reviewed_condition_concepts").reset_index()
    )
    return ids, coverage


def load_statin_artifacts(audit_dir, data_dir):
    inventory_path = audit_dir / "observed_plain_statin_tokens.csv"
    ldl_path = audit_dir / "ldl_loinc_concepts.csv"
    if not inventory_path.is_file() or not ldl_path.is_file():
        raise FileNotFoundError(f"Required statin audit files are missing under {audit_dir}")
    inventory = pd.read_csv(inventory_path)
    ldl = pd.read_csv(ldl_path)
    registry, registry_path = load_registry(data_dir)
    keys = set(registry["token_key"].astype(str))
    inventory["expected_concept_token_key"] = (
        "RX:" + pd.to_numeric(inventory["drug_concept_id"], errors="raise").astype("int64").astype(str)
    )
    inventory["concept_token_in_registry"] = inventory["expected_concept_token_key"].isin(keys)
    all_ids = sorted(pd.to_numeric(inventory["drug_concept_id"], errors="raise").astype(int).unique())
    covered_ids = sorted(
        pd.to_numeric(
            inventory.loc[inventory["concept_token_in_registry"], "drug_concept_id"],
            errors="raise",
        ).astype(int).unique()
    )
    valid_ldl = sorted(
        pd.to_numeric(
            ldl.loc[ldl["invalid_reason"].isna(), "concept_id"], errors="raise"
        ).astype(int).unique()
    )
    if not all_ids or not covered_ids or not valid_ldl:
        raise RuntimeError(
            f"Resolved IDs are empty: all_statin={len(all_ids)}, "
            f"covered_statin={len(covered_ids)}, ldl={len(valid_ldl)}"
        )
    exposure_rows = pd.to_numeric(inventory["exposure_rows"], errors="coerce").fillna(0)
    covered_rows = exposure_rows.where(inventory["concept_token_in_registry"], 0)
    coverage = pd.DataFrame(
        [{
            "mapping_rule": "RX:<drug_concept_id>",
            "distinct_product_rows": int(len(inventory)),
            "distinct_drug_concept_ids": int(len(all_ids)),
            "covered_drug_concept_ids": int(len(covered_ids)),
            "exposure_rows": int(exposure_rows.sum()),
            "covered_exposure_rows": int(covered_rows.sum()),
            "exposure_row_coverage": float(covered_rows.sum() / exposure_rows.sum())
            if exposure_rows.sum() else 0.0,
        }]
    )
    return inventory, ldl, coverage, all_ids, covered_ids, valid_ldl, registry_path


def execute(conn, statement, params=None, label=None):
    started = time.time()
    if label:
        log(f"[START] {label}")
    with conn.cursor() as cur:
        cur.execute(statement, params or ())
    conn.commit()
    if label:
        log(f"[DONE] {label}: seconds={time.time() - started:,.1f}")


def upload_dev_map(conn, frame):
    execute(
        conn,
        "CREATE TEMP TABLE tmp_t30_dev_map "
        "(person_id bigint PRIMARY KEY, split text NOT NULL) ON COMMIT PRESERVE ROWS",
        label="create development patient map",
    )
    started = time.time()
    log(f"[START] upload train+val patient map rows={len(frame):,}")
    with conn.cursor() as cur:
        with cur.copy("COPY tmp_t30_dev_map (person_id, split) FROM STDIN") as copy:
            for row in frame.itertuples(index=False):
                copy.write_row((int(row.person_id), str(row.split)))
    conn.commit()
    execute(conn, "ANALYZE tmp_t30_dev_map")
    log(f"[DONE] upload train+val patient map: seconds={time.time() - started:,.1f}")


def build_temp_tables(conn, args, phenotype_ids, statin_ids, covered_ids, ldl_ids):
    s = sql.Identifier(args.schema)
    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE tmp_t30_candidate ON COMMIT PRESERVE ROWS AS
            WITH dys AS (
                SELECT p.person_id, p.split,
                       MIN(c.condition_start_date)::date AS index_date
                FROM tmp_t30_dev_map p
                JOIN {}.condition_occurrence c USING (person_id)
                WHERE c.condition_concept_id = ANY(%s)
                  AND c.condition_start_date IS NOT NULL
                  AND c.condition_start_date <= %s::date
                GROUP BY p.person_id, p.split
            ), ldl AS (
                SELECT p.person_id, p.split,
                       MIN(m.measurement_date)::date AS index_date
                FROM tmp_t30_dev_map p
                JOIN {}.measurement m USING (person_id)
                LEFT JOIN {}.concept u ON u.concept_id = m.unit_concept_id
                WHERE m.measurement_concept_id = ANY(%s)
                  AND m.measurement_date IS NOT NULL
                  AND m.measurement_date <= %s::date
                  AND m.value_as_number IS NOT NULL
                  AND (
                       ((lower(COALESCE(u.concept_name, '')) LIKE '%%milligram per deciliter%%'
                          OR lower(COALESCE(m.unit_source_value, '')) ~ 'mg\\s*/?\\s*dl')
                         AND m.value_as_number >= %s)
                    OR ((lower(COALESCE(u.concept_name, '')) LIKE '%%millimole per liter%%'
                          OR lower(COALESCE(m.unit_source_value, '')) ~ 'mmol\\s*/?\\s*l')
                         AND m.value_as_number >= %s)
                  )
                GROUP BY p.person_id, p.split
            )
            SELECT person_id, split, 'recorded_dyslipidemia'::text AS eligibility_definition,
                   index_date FROM dys
            UNION ALL
            SELECT person_id, split, 'ldl_threshold'::text AS eligibility_definition,
                   index_date FROM ldl
            """
        ).format(s, s, s),
        (
            phenotype_ids["dyslipidemia"], args.db_end_date,
            ldl_ids, args.db_end_date, args.ldl_mg_dl, args.ldl_mmol_l,
        ),
        label="build train+val dyslipidemia and LDL candidates",
    )
    execute(conn, "CREATE INDEX ON tmp_t30_candidate(person_id)")
    execute(conn, "CREATE INDEX ON tmp_t30_candidate(split, eligibility_definition)")
    execute(conn, "ANALYZE tmp_t30_candidate")

    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE tmp_t30_statin ON COMMIT PRESERVE ROWS AS
            SELECT
                c.person_id, c.split, c.eligibility_definition, c.index_date,
                (MIN(d.drug_exposure_start_date) FILTER (
                    WHERE d.drug_exposure_start_date >= c.index_date
                ))::date AS first_statin_after,
                COALESCE(bool_or(d.drug_exposure_start_date < c.index_date), false)
                    AS prior_statin
            FROM tmp_t30_candidate c
            LEFT JOIN {}.drug_exposure d
              ON d.person_id = c.person_id
             AND d.drug_concept_id = ANY(%s)
             AND d.drug_exposure_start_date IS NOT NULL
            GROUP BY c.person_id, c.split, c.eligibility_definition, c.index_date
            """
        ).format(s),
        (statin_ids,),
        label="attach first and prior statin dates",
    )
    execute(conn, "CREATE INDEX ON tmp_t30_statin(person_id)")
    execute(conn, "ANALYZE tmp_t30_statin")

    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE tmp_t30_first_editable ON COMMIT PRESERVE ROWS AS
            SELECT s.person_id, s.split, s.eligibility_definition, s.index_date,
                   s.first_statin_after, s.prior_statin,
                   COALESCE(bool_or(d.drug_concept_id = ANY(%s)), false)
                       AS first_start_editable
            FROM tmp_t30_statin s
            LEFT JOIN {}.drug_exposure d
              ON d.person_id = s.person_id
             AND d.drug_exposure_start_date = s.first_statin_after
             AND d.drug_concept_id = ANY(%s)
            GROUP BY s.person_id, s.split, s.eligibility_definition, s.index_date,
                     s.first_statin_after, s.prior_statin
            """
        ).format(s),
        (covered_ids, statin_ids),
        label="mark first statin starts editable in FERMAT",
    )
    execute(conn, "CREATE INDEX ON tmp_t30_first_editable(person_id)")

    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE tmp_t30_annotated ON COMMIT PRESERVE ROWS AS
            WITH people AS (SELECT DISTINCT person_id FROM tmp_t30_candidate),
            obs AS (
                SELECT o.person_id,
                       MIN(o.observation_period_start_date)::date AS observation_start,
                       MAX(o.observation_period_end_date)::date AS observation_end
                FROM {}.observation_period o JOIN people p USING (person_id)
                GROUP BY o.person_id
            ), cad AS (
                SELECT co.person_id, MIN(co.condition_start_date)::date AS first_cad
                FROM {}.condition_occurrence co JOIN people p USING (person_id)
                WHERE co.condition_concept_id = ANY(%s)
                  AND co.condition_start_date IS NOT NULL
                GROUP BY co.person_id
            ), stroke AS (
                SELECT co.person_id, MIN(co.condition_start_date)::date AS first_stroke
                FROM {}.condition_occurrence co JOIN people p USING (person_id)
                WHERE co.condition_concept_id = ANY(%s)
                  AND co.condition_start_date IS NOT NULL
                GROUP BY co.person_id
            ), deaths AS (
                SELECT d.person_id, MIN(d.death_date)::date AS death_date
                FROM {}.death d JOIN people p USING (person_id)
                GROUP BY d.person_id
            )
            SELECT s.*,
                   date_part('year', age(s.index_date,
                     make_date(p.year_of_birth,
                       CASE WHEN p.month_of_birth BETWEEN 1 AND 12 THEN p.month_of_birth ELSE 7 END,
                       CASE WHEN p.day_of_birth BETWEEN 1 AND 28 THEN p.day_of_birth ELSE 1 END
                     )))::integer AS age,
                   obs.observation_start, obs.observation_end,
                   cad.first_cad, stroke.first_stroke, deaths.death_date
            FROM tmp_t30_first_editable s
            JOIN {}.person p USING (person_id)
            LEFT JOIN obs USING (person_id)
            LEFT JOIN cad USING (person_id)
            LEFT JOIN stroke USING (person_id)
            LEFT JOIN deaths USING (person_id)
            """
        ).format(s, s, s, s, s),
        (phenotype_ids["coronary_artery_disease"], phenotype_ids["ischemic_stroke"]),
        label="attach age, observation, outcomes, and death",
    )
    execute(conn, "CREATE INDEX ON tmp_t30_annotated(split, eligibility_definition)")
    execute(conn, "ANALYZE tmp_t30_annotated")


def flow_query(args):
    return """
        WITH stages AS (
            SELECT split, eligibility_definition, 'candidate'::text AS stage
            FROM tmp_t30_annotated
            UNION ALL
            SELECT split, eligibility_definition, 'age_range' FROM tmp_t30_annotated
            WHERE age BETWEEN %(min_age)s AND %(max_age)s
            UNION ALL
            SELECT split, eligibility_definition, 'lookback' FROM tmp_t30_annotated
            WHERE age BETWEEN %(min_age)s AND %(max_age)s
              AND observation_start <= index_date - %(lookback)s
              AND observation_end >= index_date
            UNION ALL
            SELECT split, eligibility_definition, 'statin_naive' FROM tmp_t30_annotated
            WHERE age BETWEEN %(min_age)s AND %(max_age)s
              AND observation_start <= index_date - %(lookback)s
              AND observation_end >= index_date
              AND NOT prior_statin
            UNION ALL
            SELECT split, eligibility_definition, 'primary_prevention' FROM tmp_t30_annotated
            WHERE age BETWEEN %(min_age)s AND %(max_age)s
              AND observation_start <= index_date - %(lookback)s
              AND observation_end >= index_date
              AND NOT prior_statin
              AND (first_cad IS NULL OR first_cad > index_date)
              AND (first_stroke IS NULL OR first_stroke > index_date)
            UNION ALL
            SELECT split, eligibility_definition, 'day30_landmark' FROM tmp_t30_annotated
            WHERE age BETWEEN %(min_age)s AND %(max_age)s
              AND observation_start <= index_date - %(lookback)s
              AND observation_end >= index_date + %(landmark)s
              AND NOT prior_statin
              AND (first_cad IS NULL OR first_cad > index_date + %(landmark)s)
              AND (first_stroke IS NULL OR first_stroke > index_date + %(landmark)s)
              AND (death_date IS NULL OR death_date > index_date + %(landmark)s)
        ), expanded AS (
            SELECT split, eligibility_definition, stage FROM stages
            UNION ALL
            SELECT 'train_val', eligibility_definition, stage FROM stages
        )
        SELECT split, eligibility_definition, stage, COUNT(*)::bigint AS patients
        FROM expanded
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
    """


def count_query(args):
    return """
        WITH eligible AS (
            SELECT * FROM tmp_t30_annotated
            WHERE age BETWEEN %(min_age)s AND %(max_age)s
              AND observation_start <= index_date - %(lookback)s
              AND observation_end >= index_date
              AND NOT prior_statin
              AND (first_cad IS NULL OR first_cad > index_date)
              AND (first_stroke IS NULL OR first_stroke > index_date)
        ), analysis_rows AS (
            SELECT e.*, 'DAY0_EDIT_ALIGNED'::text AS analysis_design,
                   e.index_date AS time_zero,
                   CASE WHEN first_statin_after = index_date
                        THEN 'INITIATE' ELSE 'NO_INITIATION' END AS strategy_group
            FROM eligible e
            UNION ALL
            SELECT e.*, 'DAY30_LANDMARK_EDIT_ALIGNED',
                   (e.index_date + %(landmark)s)::date AS time_zero,
                   CASE WHEN first_statin_after BETWEEN index_date AND index_date + %(landmark)s
                        THEN 'INITIATE' ELSE 'NO_INITIATION' END AS strategy_group
            FROM eligible e
            WHERE observation_end >= index_date + %(landmark)s
              AND (death_date IS NULL OR death_date > index_date + %(landmark)s)
              AND (first_cad IS NULL OR first_cad > index_date + %(landmark)s)
              AND (first_stroke IS NULL OR first_stroke > index_date + %(landmark)s)
        ), expanded AS (
            SELECT * FROM analysis_rows
            UNION ALL
            SELECT person_id, 'train_val'::text AS split, eligibility_definition, index_date,
                   first_statin_after, prior_statin, first_start_editable, age,
                   observation_start, observation_end, first_cad, first_stroke, death_date,
                   analysis_design, time_zero, strategy_group
            FROM analysis_rows
        ), horizon AS (
            SELECT unnest(ARRAY[365, 1095, 1826])::integer AS horizon_days
        )
        SELECT split, eligibility_definition, analysis_design, strategy_group,
               h.horizon_days,
               COUNT(*)::bigint AS patients,
               COUNT(*) FILTER (
                   WHERE strategy_group = 'INITIATE' AND first_start_editable
               )::bigint AS editable_initiators,
               COUNT(*) FILTER (
                   WHERE observation_end >= time_zero + h.horizon_days
                      OR death_date BETWEEN time_zero + 1 AND time_zero + h.horizon_days
                      OR first_cad BETWEEN time_zero + 1 AND time_zero + h.horizon_days
                      OR first_stroke BETWEEN time_zero + 1 AND time_zero + h.horizon_days
               )::bigint AS observed_or_event_through_horizon,
               COUNT(*) FILTER (
                   WHERE first_cad BETWEEN time_zero + 1 AND time_zero + h.horizon_days
                     AND (death_date IS NULL OR first_cad <= death_date)
               )::bigint AS broad_cad_events,
               COUNT(*) FILTER (
                   WHERE first_stroke BETWEEN time_zero + 1 AND time_zero + h.horizon_days
                     AND (death_date IS NULL OR first_stroke <= death_date)
               )::bigint AS ischemic_stroke_events,
               COUNT(*) FILTER (
                   WHERE death_date BETWEEN time_zero + 1 AND time_zero + h.horizon_days
               )::bigint AS deaths
        FROM expanded CROSS JOIN horizon h
        GROUP BY 1, 2, 3, 4, 5
        ORDER BY 1, 2, 3, 4, 5
    """


def top_token_query(schema):
    s = sql.Identifier(schema)
    return sql.SQL(
        """
        WITH eligible AS (
            SELECT * FROM tmp_t30_annotated
            WHERE age BETWEEN %(min_age)s AND %(max_age)s
              AND observation_start <= index_date - %(lookback)s
              AND observation_end >= index_date + %(landmark)s
              AND NOT prior_statin
              AND (death_date IS NULL OR death_date > index_date + %(landmark)s)
              AND (first_cad IS NULL OR first_cad > index_date + %(landmark)s)
              AND (first_stroke IS NULL OR first_stroke > index_date + %(landmark)s)
              AND first_statin_after BETWEEN index_date AND index_date + %(landmark)s
        )
        SELECT e.split, e.eligibility_definition, d.drug_concept_id,
               c.concept_name AS drug_concept_name,
               ('RX:' || d.drug_concept_id::text) AS token_key,
               (d.drug_concept_id = ANY(%(covered_statin)s)) AS token_in_registry,
               COUNT(DISTINCT (e.person_id, e.eligibility_definition))::bigint AS initiators
        FROM eligible e
        JOIN {}.drug_exposure d
          ON d.person_id = e.person_id
         AND d.drug_exposure_start_date = e.first_statin_after
         AND d.drug_concept_id = ANY(%(all_statin)s)
        LEFT JOIN {}.concept c ON c.concept_id = d.drug_concept_id
        GROUP BY 1, 2, 3, 4, 5, 6
        ORDER BY e.split, e.eligibility_definition, initiators DESC, d.drug_concept_id
        """
    ).format(s, s)


def add_projection(counts, test_fraction):
    frame = counts.copy()
    dev = frame["split"].eq("train_val")
    for column in (
        "patients", "editable_initiators", "observed_or_event_through_horizon",
        "broad_cad_events", "ischemic_stroke_events", "deaths",
    ):
        frame[f"projected_test_{column}"] = pd.NA
        frame.loc[dev, f"projected_test_{column}"] = (
            pd.to_numeric(frame.loc[dev, column], errors="raise")
            .map(lambda value: projected_test_count(value, test_fraction))
            .round(1)
        )
    return frame


def write_summary(path, study, patient_counts, token_coverage, flow, counts, tokens):
    dev_counts = counts.loc[counts["split"].eq("train_val")].copy()
    top_tokens = (
        tokens.sort_values(["split", "eligibility_definition", "initiators"],
                           ascending=[True, True, False])
        .groupby(["split", "eligibility_definition"], sort=True, group_keys=False)
        .head(10)
    )
    sections = [
        "## STATUS",
        "COMPLETE_BROAD_STATIN_DEV_FEASIBILITY_NOT_CAUSAL",
        "## STUDY_DEFINITION",
        json.dumps(study, indent=2, ensure_ascii=False),
        "## FERMAT_PATIENT_MAP_COUNTS",
        csv_text(patient_counts),
        "## STATIN_RX_CONCEPT_TOKEN_COVERAGE",
        csv_text(token_coverage),
        "## TRAIN_VAL_COHORT_FLOW",
        csv_text(flow.loc[flow["split"].eq("train_val")]),
        "## TRAIN_VAL_COUNTS_WITH_PROJECTED_TEST",
        csv_text(dev_counts),
        "## TOP_EXACT_RX_TOKENS_IN_TRAIN_AND_VAL",
        csv_text(top_tokens),
        "## CLAIM_BOUNDARY",
        "TEST_OUTCOMES_NOT_READ; PROJECTED_TEST_COUNTS_ARE_15_OVER_85_SCALING_NOT_OBSERVED_TEST_RESULTS",
        "BROAD_CAD_IS_NOT_ACUTE_MI_OR_MACE; COUNTS_ARE_UNADJUSTED_AND_NOT_CAUSAL_EFFECTS",
    ]
    text = "\n".join(sections) + "\n"
    path.write_text(text, encoding="utf-8")
    print(text, end="", flush=True)


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    require_dependencies()
    if not (0 < args.test_fraction < 1):
        raise ValueError("--test-fraction must be between 0 and 1")
    if args.minimum_age < 18 or args.maximum_age < args.minimum_age:
        raise ValueError("Invalid age range")
    for name in ("data_dir", "concept_map", "statin_audit_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    prepare_output(args.output_dir, args.overwrite)
    started = datetime.now(timezone.utc)

    dev_map, patient_counts, patient_map_path = load_patient_map(args.data_dir)
    phenotype_ids, phenotype_coverage = load_phenotype_ids(args.concept_map)
    inventory, ldl, token_coverage, statin_ids, covered_ids, ldl_ids, registry_path = (
        load_statin_artifacts(args.statin_audit_dir, args.data_dir)
    )
    write_csv(patient_counts, args.output_dir / "fermat_patient_map_counts.csv")
    write_csv(phenotype_coverage, args.output_dir / "phenotype_concept_coverage.csv")
    write_csv(inventory, args.output_dir / "statin_product_token_mapping.csv")
    write_csv(ldl, args.output_dir / "ldl_loinc_concepts.csv")
    write_csv(token_coverage, args.output_dir / "statin_rx_concept_token_coverage.csv")

    study = {
        "population": f"FERMAT train+val patients age {args.minimum_age}-{args.maximum_age}",
        "eligibility": [
            "recorded dyslipidemia OR LDL >= configured threshold (audited separately)",
            f"at least {args.lookback_days} days observed before index",
            "no statin before index",
            "no broad CAD or ischemic stroke on/before index",
        ],
        "strategy": f"statin initiation by day {args.landmark_days} versus no initiation by that landmark",
        "test_data_policy": "test patient outcomes are not queried; counts are projected from train+val",
        "test_fraction": args.test_fraction,
        "causal_effect_estimated": False,
    }
    write_json(study, args.output_dir / "study_definition.json")

    params = {
        "min_age": args.minimum_age,
        "max_age": args.maximum_age,
        "lookback": args.lookback_days,
        "landmark": args.landmark_days,
    }
    with connect_database(args) as conn:
        upload_dev_map(conn, dev_map)
        build_temp_tables(conn, args, phenotype_ids, statin_ids, covered_ids, ldl_ids)
        flow = query_df(conn, flow_query(args), params, label="split-aware development cohort flow")
        write_csv(flow, args.output_dir / "cohort_flow_by_split.csv")
        counts = query_df(conn, count_query(args), params, label="train+val strategy and event counts")
        counts = add_projection(counts, args.test_fraction)
        write_csv(counts, args.output_dir / "feasibility_counts_by_split.csv")
        token_statement = top_token_query(args.schema)
        token_query_params = {
            **params,
            "covered_statin": covered_ids,
            "all_statin": statin_ids,
        }
        tokens = query_df(
            conn, token_statement, token_query_params,
            label="exact RX tokens among day-30 initiators",
        )
        write_csv(tokens, args.output_dir / "exact_rx_token_initiators_by_split.csv")

    diagnostics = {
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "patient_map": str(patient_map_path),
        "concept_map": str(args.concept_map),
        "statin_audit_dir": str(args.statin_audit_dir),
        "token_registry": str(registry_path),
        "development_splits": list(DEV_SPLITS),
        "test_outcomes_read": False,
    }
    write_json(diagnostics, args.output_dir / "diagnostics.json")
    write_summary(
        args.output_dir / "return_summary.txt",
        study, patient_counts, token_coverage, flow, counts, tokens,
    )
    log(f"[COMPLETE] {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
