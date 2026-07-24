#!/usr/bin/env python3
"""Train/val feasibility audit for an HbA1c-defined ADM-timing scenario.

This is a restricted version of Ko et al., JAMA Network Open 2026:
time zero is the first observed HbA1c >=6.5%, while the 3/6/12-month
antidiabetic-medication strategies are unchanged.

The script reuses the already-completed ADM inventory and type-1-diabetes
concept files.  It scans the measurement table once to construct the HbA1c
threshold cohort.  It does not read test patients and does not estimate a
causal effect.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    import pandas as pd
except (ImportError, ModuleNotFoundError):  # self-test does not need pandas
    pd = None

try:
    from psycopg import sql
except (ImportError, ModuleNotFoundError):  # self-test does not need psycopg
    sql = None

from audit_snuh_task30_adm_timing_feasibility import (
    HBA1C_CONCEPT_ID,
    HBA1C_UNIT_CONCEPT_ID,
    add_combined_and_projection,
    complete_grid,
    connect,
    csv_text,
    execute,
    load_development_patient_map,
    prepare_output,
    query_df,
    upload_id_table,
    upload_patient_map,
    write_csv,
    write_json,
)


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_REUSE_FROM = (
    POD_ROOT
    / "task30"
    / "outputs"
    / "adm_timing_feasibility_20260719_043915"
)
DEFAULT_OUTPUT_DIR = POD_ROOT / "task30" / "outputs" / "hba1c_adm_timing_feasibility"

REUSE_FILES = (
    "observed_adm_inventory.csv",
    "adm_rx_token_coverage.csv",
    "adm_atc_roots.csv",
    "type1_diabetes_root.csv",
    "type1_diabetes_descendants.csv",
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--reuse-from", type=Path, default=DEFAULT_REUSE_FROM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--study-start", default="2013-01-01")
    parser.add_argument("--entry-end", default="2022-12-31")
    parser.add_argument("--db-end-date", default="2025-02-05")
    parser.add_argument("--minimum-age", type=int, default=18)
    parser.add_argument("--washout-days", type=int, default=365)
    parser.add_argument(
        "--host",
        default="pg-2vge6u.vpc-cdb-kr.gov-ntruss.com",
    )
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="cdm")
    parser.add_argument("--user", default="jaegyun_jung")
    parser.add_argument("--schema", default="cdm2024_official")
    parser.add_argument("--sslmode", default="disable")
    parser.add_argument(
        "--statement-timeout",
        default="3600000",
        help="PostgreSQL timeout in milliseconds; default is 60 minutes.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test():
    assert HBA1C_CONCEPT_ID == 3004410
    assert HBA1C_UNIT_CONCEPT_ID == 8554
    assert round(850 * 15 / 85) == 150
    print("SELF_TEST_OK")


def require_dependencies():
    missing = []
    if pd is None:
        missing.append("pandas")
    if sql is None:
        missing.append("psycopg")
    if missing:
        raise RuntimeError("Missing Pod Python dependencies: " + ", ".join(missing))


def load_reused_artifacts(reuse_from: Path, output_dir: Path):
    if not reuse_from.is_dir():
        raise NotADirectoryError(f"Reuse directory not found: {reuse_from}")
    frames = {}
    for filename in REUSE_FILES:
        source = reuse_from / filename
        if not source.is_file():
            raise FileNotFoundError(f"Required reuse file is missing: {source}")
        frame = pd.read_csv(source)
        frames[filename] = frame
        write_csv(frame, output_dir / filename)

    adm = frames["observed_adm_inventory.csv"]
    if "drug_concept_id" not in adm.columns:
        raise ValueError("observed_adm_inventory.csv has no drug_concept_id")
    adm_ids = (
        pd.to_numeric(adm["drug_concept_id"], errors="raise")
        .dropna()
        .astype("int64")
        .unique()
        .tolist()
    )

    type1 = frames["type1_diabetes_descendants.csv"]
    if "condition_concept_id" not in type1.columns:
        raise ValueError("type1_diabetes_descendants.csv has no condition_concept_id")
    type1_ids = (
        pd.to_numeric(type1["condition_concept_id"], errors="raise")
        .dropna()
        .astype("int64")
        .unique()
        .tolist()
    )
    if not adm_ids or not type1_ids:
        raise ValueError("Reused ADM or type-1 concept list is empty")
    return frames, adm_ids, type1_ids


def build_hba1c_cohort(conn, args, adm_ids, type1_ids, output_dir):
    upload_id_table(conn, "tmp_s2_adm_concepts", "drug_concept_id", adm_ids)
    upload_id_table(
        conn,
        "tmp_s2_type1_concepts",
        "condition_concept_id",
        type1_ids,
    )
    params = {
        "hba1c": HBA1C_CONCEPT_ID,
        "hba1c_unit": HBA1C_UNIT_CONCEPT_ID,
        "study_start": args.study_start,
        "entry_end": args.entry_end,
        "db_end": args.db_end_date,
        "washout": args.washout_days,
        "minimum_age": args.minimum_age,
    }
    schema = sql.Identifier(args.schema)

    # This is the only full measurement-table pass.  Rows below 6.5% are
    # discarded first; the database then retains only one minimum date per person.
    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE tmp_s2_first_threshold ON COMMIT PRESERVE ROWS AS
            WITH first_qualifying_hba1c AS MATERIALIZED (
                SELECT m.person_id,
                       MIN(m.measurement_date)::date AS lab_date
                FROM {}.measurement m
                WHERE m.measurement_concept_id = %(hba1c)s
                  AND m.unit_concept_id = %(hba1c_unit)s
                  AND m.value_as_number IS NOT NULL
                  AND m.value_as_number >= 6.5
                  AND m.measurement_date BETWEEN
                      (%(study_start)s::date - %(washout)s::int)
                      AND %(entry_end)s::date
                GROUP BY m.person_id
            )
            SELECT p.person_id, p.split, d.lab_date
            FROM first_qualifying_hba1c d
            JOIN tmp_s2_patient_map p USING(person_id)
            """
        ).format(schema),
        params,
        label="scan HbA1c once and find first value >=6.5%",
    )
    execute(conn, "CREATE UNIQUE INDEX ON tmp_s2_first_threshold(person_id)")
    threshold_count = query_df(
        conn,
        """
        SELECT split, COUNT(*)::bigint AS patients,
               MIN(lab_date)::text AS first_index_date,
               MAX(lab_date)::text AS last_index_date
        FROM tmp_s2_first_threshold
        GROUP BY split
        ORDER BY split
        """,
        label="summarize first HbA1c threshold dates",
    )
    write_csv(threshold_count, output_dir / "first_hba1c_threshold_counts.csv")

    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE tmp_s2_adult_index ON COMMIT PRESERVE ROWS AS
            SELECT t.person_id, t.split, t.lab_date AS index_date,
                   EXTRACT(YEAR FROM age(t.lab_date, make_date(
                       p.year_of_birth,
                       CASE WHEN p.month_of_birth BETWEEN 1 AND 12
                            THEN p.month_of_birth ELSE 7 END,
                       CASE WHEN p.day_of_birth BETWEEN 1 AND 28
                            THEN p.day_of_birth ELSE 1 END
                   )))::int AS age
            FROM tmp_s2_first_threshold t
            JOIN {}.person p USING(person_id)
            WHERE t.lab_date BETWEEN %(study_start)s::date AND %(entry_end)s::date
              AND EXTRACT(YEAR FROM age(t.lab_date, make_date(
                    p.year_of_birth,
                    CASE WHEN p.month_of_birth BETWEEN 1 AND 12
                         THEN p.month_of_birth ELSE 7 END,
                    CASE WHEN p.day_of_birth BETWEEN 1 AND 28
                         THEN p.day_of_birth ELSE 1 END
                  ))) >= %(minimum_age)s
            """
        ).format(schema),
        params,
        label="apply age and entry-period criteria",
    )
    execute(conn, "CREATE UNIQUE INDEX ON tmp_s2_adult_index(person_id)")

    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE tmp_s2_observed_washout ON COMMIT PRESERVE ROWS AS
            SELECT DISTINCT i.*
            FROM tmp_s2_adult_index i
            JOIN {}.observation_period o USING(person_id)
            WHERE o.observation_period_start_date <= i.index_date - %(washout)s::int
              AND o.observation_period_end_date >= i.index_date
            """
        ).format(schema),
        params,
        label="require one year of observation before time zero",
    )
    execute(conn, "CREATE UNIQUE INDEX ON tmp_s2_observed_washout(person_id)")

    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE tmp_s2_no_prior_adm ON COMMIT PRESERVE ROWS AS
            SELECT i.*
            FROM tmp_s2_observed_washout i
            WHERE NOT EXISTS (
                SELECT 1
                FROM {}.drug_exposure d
                JOIN tmp_s2_adm_concepts a USING(drug_concept_id)
                WHERE d.person_id = i.person_id
                  AND d.drug_exposure_start_date < i.index_date
            )
            """
        ).format(schema),
        label="exclude prior antidiabetic medication",
    )
    execute(conn, "CREATE UNIQUE INDEX ON tmp_s2_no_prior_adm(person_id)")

    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE tmp_s2_eligible ON COMMIT PRESERVE ROWS AS
            SELECT i.*
            FROM tmp_s2_no_prior_adm i
            WHERE NOT EXISTS (
                SELECT 1
                FROM {}.condition_occurrence c
                JOIN tmp_s2_type1_concepts t USING(condition_concept_id)
                WHERE c.person_id = i.person_id
                  AND c.condition_start_date < i.index_date
            )
            """
        ).format(schema),
        label="exclude prior type 1 diabetes",
    )
    execute(conn, "CREATE UNIQUE INDEX ON tmp_s2_eligible(person_id)")

    partial_flow = query_df(
        conn,
        """
        SELECT split, stage, patients
        FROM (
            SELECT split, 'FIRST_HBA1C_GE_6_5_IN_SCAN'::text AS stage,
                   COUNT(*)::bigint AS patients
            FROM tmp_s2_first_threshold GROUP BY split
            UNION ALL
            SELECT split, 'ADULT_ENTRY_2013_2022', COUNT(*)::bigint
            FROM tmp_s2_adult_index GROUP BY split
            UNION ALL
            SELECT split, 'PLUS_1Y_OBSERVATION_WASHOUT', COUNT(*)::bigint
            FROM tmp_s2_observed_washout GROUP BY split
            UNION ALL
            SELECT split, 'PLUS_NO_PRIOR_ADM', COUNT(*)::bigint
            FROM tmp_s2_no_prior_adm GROUP BY split
            UNION ALL
            SELECT split, 'PLUS_NO_PRIOR_TYPE1_FINAL', COUNT(*)::bigint
            FROM tmp_s2_eligible GROUP BY split
        ) q
        ORDER BY stage, split
        """,
        label="save cohort counts before treatment-timing query",
    )
    write_csv(partial_flow, output_dir / "cohort_flow_before_timing.csv")

    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE tmp_s2_timing ON COMMIT PRESERVE ROWS AS
            WITH first_adm AS (
                SELECT e.person_id,
                       MIN(x.drug_exposure_start_date)::date AS first_adm_date
                FROM tmp_s2_eligible e
                JOIN {}.drug_exposure x
                  ON x.person_id = e.person_id
                 AND x.drug_exposure_start_date >= e.index_date
                 AND x.drug_exposure_start_date <= %(db_end)s::date
                JOIN tmp_s2_adm_concepts a USING(drug_concept_id)
                GROUP BY e.person_id
            ),
            first_death AS (
                SELECT e.person_id,
                       MIN(x.death_date)::date AS death_date
                FROM tmp_s2_eligible e
                JOIN {}.death x
                  ON x.person_id = e.person_id
                 AND x.death_date > e.index_date
                 AND x.death_date <= %(db_end)s::date
                GROUP BY e.person_id
            )
            SELECT e.*, a.first_adm_date, d.death_date,
                   CASE
                     WHEN a.first_adm_date <= e.index_date + INTERVAL '3 months'
                       THEN 'INIT_0_3M'
                     WHEN a.first_adm_date <= e.index_date + INTERVAL '6 months'
                       THEN 'INIT_GT3_6M'
                     WHEN a.first_adm_date <= e.index_date + INTERVAL '12 months'
                       THEN 'INIT_GT6_12M'
                     ELSE 'NO_INIT_WITHIN_12M'
                   END AS observed_timing_bin
            FROM tmp_s2_eligible e
            LEFT JOIN first_adm a USING(person_id)
            LEFT JOIN first_death d USING(person_id)
            """
        ).format(schema, schema),
        params,
        label="derive observed ADM timing and mortality",
    )
    execute(conn, "CREATE UNIQUE INDEX ON tmp_s2_timing(person_id)")
    return threshold_count


def collect_results(conn, args):
    flow = query_df(
        conn,
        """
        SELECT split, stage, patients
        FROM (
            SELECT split, '01_FERMAT_TRAIN_VAL_PATIENTS'::text AS stage,
                   COUNT(*)::bigint AS patients
            FROM tmp_s2_patient_map GROUP BY split
            UNION ALL
            SELECT split, '02_FIRST_HBA1C_GE_6_5_IN_SCAN', COUNT(*)::bigint
            FROM tmp_s2_first_threshold GROUP BY split
            UNION ALL
            SELECT split, '03_ADULT_ENTRY_2013_2022', COUNT(*)::bigint
            FROM tmp_s2_adult_index GROUP BY split
            UNION ALL
            SELECT split, '04_PLUS_1Y_OBSERVATION_WASHOUT', COUNT(*)::bigint
            FROM tmp_s2_observed_washout GROUP BY split
            UNION ALL
            SELECT split, '05_PLUS_NO_PRIOR_ADM', COUNT(*)::bigint
            FROM tmp_s2_no_prior_adm GROUP BY split
            UNION ALL
            SELECT split, '06_PLUS_NO_PRIOR_TYPE1_FINAL', COUNT(*)::bigint
            FROM tmp_s2_eligible GROUP BY split
        ) q
        ORDER BY stage, split
        """,
        label="aggregate HbA1c cohort flow",
    )
    flow_stages = [
        "01_FERMAT_TRAIN_VAL_PATIENTS",
        "02_FIRST_HBA1C_GE_6_5_IN_SCAN",
        "03_ADULT_ENTRY_2013_2022",
        "04_PLUS_1Y_OBSERVATION_WASHOUT",
        "05_PLUS_NO_PRIOR_ADM",
        "06_PLUS_NO_PRIOR_TYPE1_FINAL",
    ]
    flow = complete_grid(flow, "stage", flow_stages, ["patients"])
    flow = add_combined_and_projection(flow, ["stage"], ["patients"])

    timing = query_df(
        conn,
        """
        SELECT split, observed_timing_bin,
               COUNT(*)::bigint AS patients,
               COUNT(*) FILTER (WHERE death_date <= index_date + 365)::bigint AS deaths_1y,
               COUNT(*) FILTER (WHERE death_date <= index_date + 1095)::bigint AS deaths_3y,
               COUNT(*) FILTER (WHERE death_date <= index_date + 1826)::bigint AS deaths_5y,
               COUNT(*) FILTER (WHERE index_date + 365 <= %(db_end)s::date)::bigint AS full_followup_1y,
               COUNT(*) FILTER (WHERE index_date + 1095 <= %(db_end)s::date)::bigint AS full_followup_3y,
               COUNT(*) FILTER (WHERE index_date + 1826 <= %(db_end)s::date)::bigint AS full_followup_5y
        FROM tmp_s2_timing
        GROUP BY split, observed_timing_bin
        ORDER BY observed_timing_bin, split
        """,
        {"db_end": args.db_end_date},
        label="aggregate observed treatment-timing bins",
    )
    timing_columns = [
        "patients", "deaths_1y", "deaths_3y", "deaths_5y",
        "full_followup_1y", "full_followup_3y", "full_followup_5y",
    ]
    timing = complete_grid(
        timing,
        "observed_timing_bin",
        ["INIT_0_3M", "INIT_GT3_6M", "INIT_GT6_12M", "NO_INIT_WITHIN_12M"],
        timing_columns,
    )
    timing = add_combined_and_projection(
        timing, ["observed_timing_bin"], timing_columns
    )

    strategies = query_df(
        conn,
        """
        WITH clones AS (
            SELECT t.*, v.strategy,
                   CASE v.strategy
                     WHEN 'INIT_WITHIN_3M'
                       THEN t.first_adm_date <= t.index_date + INTERVAL '3 months'
                     WHEN 'INIT_WITHIN_6M'
                       THEN t.first_adm_date <= t.index_date + INTERVAL '6 months'
                     WHEN 'INIT_WITHIN_12M'
                       THEN t.first_adm_date <= t.index_date + INTERVAL '12 months'
                     WHEN 'NO_INIT_WITHIN_12M'
                       THEN t.first_adm_date IS NULL
                         OR t.first_adm_date > t.index_date + INTERVAL '12 months'
                   END AS fulfilled
            FROM tmp_s2_timing t
            CROSS JOIN (VALUES
                ('INIT_WITHIN_3M'::text),
                ('INIT_WITHIN_6M'::text),
                ('INIT_WITHIN_12M'::text),
                ('NO_INIT_WITHIN_12M'::text)
            ) v(strategy)
        )
        SELECT split, strategy,
               COUNT(*)::bigint AS eligible_clones,
               COUNT(*) FILTER (WHERE fulfilled)::bigint AS observed_fulfillers,
               COUNT(*) FILTER (
                   WHERE fulfilled AND death_date <= index_date + 365
               )::bigint AS fulfiller_deaths_1y,
               COUNT(*) FILTER (
                   WHERE fulfilled AND death_date <= index_date + 1095
               )::bigint AS fulfiller_deaths_3y,
               COUNT(*) FILTER (
                   WHERE fulfilled AND death_date <= index_date + 1826
               )::bigint AS fulfiller_deaths_5y
        FROM clones
        GROUP BY split, strategy
        ORDER BY strategy, split
        """,
        label="aggregate four treatment-strategy clones",
    )
    strategy_columns = [
        "eligible_clones", "observed_fulfillers", "fulfiller_deaths_1y",
        "fulfiller_deaths_3y", "fulfiller_deaths_5y",
    ]
    strategies = complete_grid(
        strategies,
        "strategy",
        ["INIT_WITHIN_3M", "INIT_WITHIN_6M", "INIT_WITHIN_12M", "NO_INIT_WITHIN_12M"],
        strategy_columns,
    )
    strategies = add_combined_and_projection(
        strategies, ["strategy"], strategy_columns
    )
    return flow, timing, strategies


def write_summary(path, study, coverage, flow, strategies, limitations):
    sections = [
        "## STATUS",
        "HBA1C_RESTRICTED_FEASIBILITY_ONLY_NOT_CAUSAL_EFFECT",
        "## STUDY_DEFINITION",
        json.dumps(study, ensure_ascii=False, indent=2),
        "## LIMITATIONS",
        json.dumps(limitations, ensure_ascii=False, indent=2),
        "## ADM_RX_TOKEN_COVERAGE_REUSED",
        csv_text(coverage),
        "## TRAIN_VAL_COHORT_FLOW_AND_PROJECTED_TEST",
        csv_text(flow.loc[flow["split"].eq("train_val")]),
        "## STRATEGY_FULFILLMENT_AND_PROJECTED_TEST",
        csv_text(strategies.loc[strategies["split"].eq("train_val")]),
        "## INTERPRETATION",
        (
            "These are unadjusted feasibility counts, not causal risks. "
            "The treatment strategies match the JAMA Network Open scenario, "
            "but eligibility is restricted to patients newly crossing HbA1c 6.5%. "
            "The final causal curves require clone-censor-weight after the design is frozen."
        ),
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
    prepare_output(args.output_dir, args.overwrite)
    started = datetime.now(timezone.utc)

    study = {
        "source_scenario": "Ko et al., JAMA Network Open 2026",
        "scenario_role": "clinically motivated FERMAT counterfactual validation scenario; not a literal reproduction",
        "time_zero": "first observed HbA1c >=6.5%",
        "change_from_source_study": "fasting-plasma-glucose >=126 mg/dL route omitted because fasting status is not encoded in SNUH",
        "age": ">=18 years",
        "washout": f"{args.washout_days} days",
        "exclusions": ["prior antidiabetic medication", "prior type 1 diabetes"],
        "strategies": [
            "initiate ADM within 3 months",
            "initiate ADM within 6 months",
            "initiate ADM within 12 months",
            "no ADM initiation within 12 months",
        ],
        "feasibility_outcome": "all-cause mortality",
        "splits_queried": ["train", "val"],
        "test_data_queried": False,
        "causal_effect_estimated": False,
    }
    limitations = {
        "mace": "Acute MI and the source paper's stroke definition are not yet reviewed for SNUH; broad CAD is not substituted.",
        "fermat_alignment": "The strategies occur after time zero; exact FERMAT validation still requires future treatment-token conditioning.",
        "test_projection": "Projected test counts equal observed train+val counts multiplied by 15/85; they are not actual test counts.",
    }
    write_json(study, args.output_dir / "study_definition.json")
    write_json(limitations, args.output_dir / "limitations.json")

    try:
        reused, adm_ids, type1_ids = load_reused_artifacts(
            args.reuse_from, args.output_dir
        )
        patient_map, patient_map_path, observed_splits = (
            load_development_patient_map(args.data_dir)
        )
        split_counts = (
            patient_map.groupby("split").size().rename("patients").reset_index()
        )
        write_csv(split_counts, args.output_dir / "development_split_counts.csv")

        with connect(args) as conn:
            upload_patient_map(conn, patient_map)
            threshold_count = build_hba1c_cohort(
                conn, args, adm_ids, type1_ids, args.output_dir
            )
            flow, timing, strategies = collect_results(conn, args)
            write_csv(flow, args.output_dir / "cohort_flow.csv")
            write_csv(timing, args.output_dir / "observed_timing_bins.csv")
            write_csv(
                strategies, args.output_dir / "strategy_fulfillment_counts.csv"
            )

        diagnostics = {
            "started_utc": started.isoformat(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "reuse_from": str(args.reuse_from),
            "patient_map": str(patient_map_path),
            "observed_split_values": observed_splits,
            "queried_split_values": ["train", "val"],
            "test_rows_loaded": 0,
            "measurement_full_passes_in_this_script": 1,
            "projected_test_multiplier": 15.0 / 85.0,
        }
        write_json(diagnostics, args.output_dir / "diagnostics.json")
        write_summary(
            args.output_dir / "return_summary.txt",
            study,
            reused["adm_rx_token_coverage.csv"],
            flow,
            strategies,
            limitations,
        )
        print(f"[COMPLETE] {args.output_dir}", flush=True)
        return 0
    except Exception as exc:
        failure = {
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "started_utc": started.isoformat(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(failure, args.output_dir / "failure.json")
        (args.output_dir / "return_summary.txt").write_text(
            "## STATUS\nFAILED\n"
            f"error_type={type(exc).__name__}\n"
            f"error={exc}\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
