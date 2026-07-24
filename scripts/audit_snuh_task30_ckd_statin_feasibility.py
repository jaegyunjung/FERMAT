#!/usr/bin/env python3
"""CPU-only feasibility audit for a CKD statin-initiation scenario.

This script answers a deliberately narrow question before any FERMAT run:
does SNUH contain enough older, statin-naive patients with CKD and
hyperlipidaemia to compare statin initiation with non-initiation?

It produces aggregate counts only.  It does not estimate a treatment effect.
The raw groups printed here must not be interpreted causally.  Later answer-key
curves must use the same prediction time zero as the FERMAT edit and must handle
treatment crossover with an appropriate target-trial analysis.
"""

from __future__ import annotations

import argparse
import json
import os
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
    add_registry_coverage,
    csv_text,
    load_registry,
    log,
    observed_product_ingredients,
    password,
    prepare_output,
    query_df,
    require_dependencies,
    resolve_roots,
    select_valid_root_ids,
    token_coverage_summary,
    write_csv,
    write_json,
)


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_CONCEPT_MAP = (
    POD_ROOT
    / "task19"
    / "outputs"
    / "patient_phenotype_labels_wide"
    / "phenotype_group_concept_map.csv"
)
DEFAULT_OUTPUT_DIR = POD_ROOT / "task30" / "outputs" / "ckd_statin_feasibility"

STATIN_ROOTS = (("PLAIN_STATIN", "ATC", "C10AA"),)
LDL_LOINC_CODES = ("2089-1", "18262-6", "13457-7")
PHENOTYPES = ("chronic_kidney_disease", "dyslipidemia", "coronary_artery_disease", "ischemic_stroke")
HORIZONS = (365, 1095, 1826)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--concept-map", type=Path, default=DEFAULT_CONCEPT_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Reuse completed inventory, LDL, and cohort-flow files from a failed run.",
    )
    parser.add_argument("--minimum-age", type=int, default=60)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--landmark-days", type=int, default=30)
    parser.add_argument("--ldl-mg-dl", type=float, default=100.0)
    parser.add_argument("--ldl-mmol-l", type=float, default=2.6)
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


def age_band(age):
    if age < 60:
        return "LT60"
    if age < 75:
        return "60_74"
    if age < 85:
        return "75_84"
    return "GE85"


def qualifying_ldl(value, unit_name, unit_source, mg_dl=100.0, mmol_l=2.6):
    unit = f"{unit_name or ''} {unit_source or ''}".lower().replace(" ", "")
    if "mg/dl" in unit or "milligramperdeciliter" in unit:
        return float(value) >= mg_dl
    if "mmol/l" in unit or "millimoleperliter" in unit:
        return float(value) >= mmol_l
    return False


def self_test():
    assert age_band(60) == "60_74"
    assert age_band(75) == "75_84"
    assert age_band(85) == "GE85"
    assert qualifying_ldl(100, "milligram per deciliter", None)
    assert not qualifying_ldl(99.9, None, "mg/dL")
    assert qualifying_ldl(2.6, None, "mmol/L")
    assert not qualifying_ldl(100, "unknown", None)
    print("SELF_TEST_OK")


def connect_database(args):
    """Open a read-only connection with keepalives for long aggregate queries."""
    conn = psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=password(),
        sslmode=args.sslmode,
        connect_timeout=15,
        application_name="fermat_task30_ckd_statin_feasibility",
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=6,
    )
    with conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute("SELECT set_config('statement_timeout', %s, false)", (args.statement_timeout,))
    return conn


def load_phenotype_ids(path: Path):
    frame = pd.read_csv(path)
    required = {"phenotype", "condition_concept_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Concept map is missing columns: {sorted(missing)}")
    frame = frame.loc[frame["phenotype"].isin(PHENOTYPES)].copy()
    frame["condition_concept_id"] = pd.to_numeric(
        frame["condition_concept_id"], errors="raise"
    ).astype("int64")
    found = set(frame["phenotype"])
    missing_phenotypes = sorted(set(PHENOTYPES) - found)
    if missing_phenotypes:
        raise ValueError(f"Concept map has no reviewed concepts for: {missing_phenotypes}")
    ids = {
        name: sorted(
            frame.loc[frame["phenotype"].eq(name), "condition_concept_id"]
            .drop_duplicates()
            .astype(int)
            .tolist()
        )
        for name in PHENOTYPES
    }
    coverage = (
        frame.groupby("phenotype", sort=True)["condition_concept_id"]
        .nunique()
        .rename("reviewed_condition_concepts")
        .reset_index()
    )
    return ids, coverage


def load_resume_artifacts(path: Path, output_dir: Path):
    filenames = (
        "statin_concept_root.csv",
        "observed_statin_inventory_all.csv",
        "observed_statin_product_ingredients.csv",
        "observed_plain_statin_tokens.csv",
        "statin_rx_token_coverage.csv",
        "ldl_loinc_concepts.csv",
        "cohort_flow.csv",
    )
    frames = {}
    for filename in filenames:
        source = path / filename
        if not source.is_file():
            raise FileNotFoundError(f"Resume file is missing: {source}")
        frame = pd.read_csv(source)
        frames[filename] = frame
        write_csv(frame, output_dir / filename)
    all_ids = set(
        pd.to_numeric(
            frames["observed_statin_inventory_all.csv"]["drug_concept_id"],
            errors="raise",
        ).astype(int)
    )
    plain_ids = set(
        pd.to_numeric(
            frames["observed_plain_statin_tokens.csv"]["drug_concept_id"],
            errors="raise",
        ).astype(int)
    )
    return {
        "plain_inventory": frames["observed_plain_statin_tokens.csv"],
        "token_coverage": frames["statin_rx_token_coverage.csv"],
        "ldl": frames["ldl_loinc_concepts.csv"],
        "flow": frames["cohort_flow.csv"],
        "combination_ids": all_ids - plain_ids,
    }


def resolve_ldl_concepts(conn, schema):
    statement = sql.SQL(
        """
        SELECT concept_id, concept_name, vocabulary_id, concept_code,
               domain_id, standard_concept, invalid_reason
        FROM {}.concept
        WHERE vocabulary_id = 'LOINC'
          AND concept_code = ANY(%s)
          AND domain_id = 'Measurement'
        ORDER BY concept_code, invalid_reason NULLS FIRST, concept_id
        """
    ).format(sql.Identifier(schema))
    return query_df(conn, statement, (list(LDL_LOINC_CODES),), label="resolve LDL LOINC concepts")


def observed_statin_inventory(conn, schema, root_id):
    statement = sql.SQL(
        """
        SELECT
            d.drug_concept_id,
            c.concept_name AS drug_concept_name,
            c.vocabulary_id,
            c.concept_class_id,
            COALESCE(NULLIF(btrim(d.drug_source_value), ''), d.drug_concept_id::text)
                AS fermat_source_value,
            d.drug_source_concept_id,
            COUNT(*)::bigint AS exposure_rows,
            COUNT(DISTINCT d.person_id)::bigint AS patients,
            MIN(d.drug_exposure_start_date)::text AS first_exposure_date,
            MAX(d.drug_exposure_start_date)::text AS last_exposure_date
        FROM {}.concept_ancestor ca
        JOIN {}.drug_exposure d
          ON d.drug_concept_id = ca.descendant_concept_id
         AND d.drug_exposure_start_date IS NOT NULL
        LEFT JOIN {}.concept c ON c.concept_id = d.drug_concept_id
        WHERE ca.ancestor_concept_id = %s
        GROUP BY 1, 2, 3, 4, 5, 6
        ORDER BY exposure_rows DESC, d.drug_concept_id, fermat_source_value
        """
    ).format(sql.Identifier(schema), sql.Identifier(schema), sql.Identifier(schema))
    frame = query_df(conn, statement, (root_id,), label="inventory observed plain-statin products")
    frame.insert(0, "treatment_class", "PLAIN_STATIN")
    return frame


def cohort_statement(schema):
    s = sql.Identifier(schema)
    return sql.SQL(
        """
        WITH
        ckd AS (
            SELECT person_id, MIN(condition_start_date)::date AS ckd_date
            FROM {}.condition_occurrence
            WHERE condition_concept_id = ANY(%(ckd)s)
              AND condition_start_date IS NOT NULL
            GROUP BY person_id
        ),
        dyslipidemia AS (
            SELECT person_id, MIN(condition_start_date)::date AS dyslipidemia_date
            FROM {}.condition_occurrence
            WHERE condition_concept_id = ANY(%(dyslipidemia)s)
              AND condition_start_date IS NOT NULL
            GROUP BY person_id
        ),
        qualifying_ldl AS (
            SELECT m.person_id, MIN(m.measurement_date)::date AS ldl_date
            FROM {}.measurement m
            JOIN ckd ON ckd.person_id = m.person_id
            LEFT JOIN {}.concept u ON u.concept_id = m.unit_concept_id
            WHERE m.measurement_concept_id = ANY(%(ldl)s)
              AND m.measurement_date >= ckd.ckd_date
              AND m.value_as_number IS NOT NULL
              AND (
                    ((lower(COALESCE(u.concept_name, '')) LIKE '%%milligram per deciliter%%'
                       OR lower(COALESCE(m.unit_source_value, '')) ~ 'mg\\s*/?\\s*dl')
                      AND m.value_as_number >= %(ldl_mg_dl)s)
                 OR ((lower(COALESCE(u.concept_name, '')) LIKE '%%millimole per liter%%'
                       OR lower(COALESCE(m.unit_source_value, '')) ~ 'mmol\\s*/?\\s*l')
                      AND m.value_as_number >= %(ldl_mmol_l)s)
              )
            GROUP BY m.person_id
        ),
        candidate AS (
            SELECT c.person_id, 'recorded_dyslipidemia'::text AS eligibility_definition,
                   GREATEST(c.ckd_date, d.dyslipidemia_date)::date AS index_date
            FROM ckd c JOIN dyslipidemia d USING (person_id)
            UNION ALL
            SELECT c.person_id, 'ldl_threshold'::text AS eligibility_definition,
                   l.ldl_date::date AS index_date
            FROM ckd c JOIN qualifying_ldl l USING (person_id)
        ),
        candidate_people AS (
            SELECT DISTINCT person_id FROM candidate
        ),
        observation AS (
            SELECT o.person_id,
                   MIN(o.observation_period_start_date)::date AS observation_start,
                   MAX(o.observation_period_end_date)::date AS observation_end
            FROM {}.observation_period o
            JOIN candidate_people cp USING (person_id)
            GROUP BY o.person_id
        ),
        statin_by_candidate AS (
            SELECT
                c.person_id,
                c.eligibility_definition,
                c.index_date,
                (MIN(d.drug_exposure_start_date) FILTER (
                    WHERE d.drug_exposure_start_date >= c.index_date
                ))::date AS first_statin_after,
                COALESCE(bool_or(d.drug_exposure_start_date < c.index_date), false)
                    AS prior_statin
            FROM candidate c
            LEFT JOIN {}.drug_exposure d
              ON d.person_id = c.person_id
             AND d.drug_concept_id = ANY(%(statin)s)
             AND d.drug_exposure_start_date IS NOT NULL
            GROUP BY c.person_id, c.eligibility_definition, c.index_date
        ),
        cad AS (
            SELECT co.person_id, MIN(co.condition_start_date)::date AS first_cad
            FROM {}.condition_occurrence co
            JOIN candidate_people cp USING (person_id)
            WHERE co.condition_concept_id = ANY(%(cad)s)
              AND co.condition_start_date IS NOT NULL
            GROUP BY co.person_id
        ),
        stroke AS (
            SELECT co.person_id, MIN(co.condition_start_date)::date AS first_stroke
            FROM {}.condition_occurrence co
            JOIN candidate_people cp USING (person_id)
            WHERE co.condition_concept_id = ANY(%(stroke)s)
              AND co.condition_start_date IS NOT NULL
            GROUP BY co.person_id
        ),
        deaths AS (
            SELECT d.person_id, MIN(d.death_date)::date AS death_date
            FROM {}.death d
            JOIN candidate_people cp USING (person_id)
            GROUP BY d.person_id
        ),
        annotated AS (
            SELECT
                c.*,
                date_part('year', age(c.index_date,
                    make_date(p.year_of_birth,
                              COALESCE(NULLIF(p.month_of_birth, 0), 7),
                              COALESCE(NULLIF(p.day_of_birth, 0), 1))))::integer AS age,
                o.observation_start,
                o.observation_end,
                cad.first_cad,
                stroke.first_stroke,
                deaths.death_date
            FROM statin_by_candidate c
            JOIN {}.person p USING (person_id)
            LEFT JOIN observation o USING (person_id)
            LEFT JOIN cad USING (person_id)
            LEFT JOIN stroke USING (person_id)
            LEFT JOIN deaths USING (person_id)
            WHERE c.index_date <= %(db_end)s
        ),
        classified AS (
            SELECT a.*,
                   (first_statin_after - index_date) AS days_to_statin,
                   CASE WHEN age < 75 THEN '60_74'
                        WHEN age < 85 THEN '75_84' ELSE 'GE85' END AS age_group,
                   (first_cad IS NOT NULL AND first_cad <= index_date) AS prior_cad,
                   (first_stroke IS NOT NULL AND first_stroke <= index_date) AS prior_stroke
            FROM annotated a
        ),
        eligible AS (
            SELECT * FROM classified
            WHERE age >= %(minimum_age)s
              AND observation_start <= index_date - %(lookback_days)s
              AND observation_end >= index_date
              AND NOT prior_statin
              AND NOT prior_cad
              AND NOT prior_stroke
        ),
        analysis_rows AS (
            SELECT
                e.*,
                'DAY0_EDIT_ALIGNED'::text AS analysis_design,
                e.index_date AS prediction_time_zero,
                CASE WHEN e.first_statin_after = e.index_date
                     THEN 'INITIATE_DAY0'
                     ELSE 'NO_INITIATION_DAY0'
                END AS strategy_group
            FROM eligible e
            UNION ALL
            SELECT
                e.*,
                'DAY30_LANDMARK_EDIT_ALIGNED'::text AS analysis_design,
                (e.index_date + %(landmark_days)s)::date AS prediction_time_zero,
                CASE WHEN e.first_statin_after BETWEEN e.index_date
                                                   AND e.index_date + %(landmark_days)s
                     THEN 'INITIATE_BY_LANDMARK'
                     ELSE 'NO_INITIATION_BY_LANDMARK'
                END AS strategy_group
            FROM eligible e
            WHERE e.observation_end >= e.index_date + %(landmark_days)s
              AND (e.death_date IS NULL OR e.death_date > e.index_date + %(landmark_days)s)
              AND (e.first_cad IS NULL OR e.first_cad > e.index_date + %(landmark_days)s)
              AND (e.first_stroke IS NULL OR e.first_stroke > e.index_date + %(landmark_days)s)
        ),
        strata AS (
            SELECT *, 'ALL'::text AS reported_age_group FROM analysis_rows
            UNION ALL
            SELECT *, age_group AS reported_age_group FROM analysis_rows
        ),
        horizon AS (
            SELECT unnest(ARRAY[365, 1095, 1826])::integer AS horizon_days
        )
        SELECT
            eligibility_definition,
            analysis_design,
            reported_age_group AS age_group,
            strategy_group,
            h.horizon_days,
            COUNT(*)::bigint AS patients,
            COUNT(*) FILTER (
                WHERE observation_end >= prediction_time_zero + h.horizon_days
                   OR (death_date BETWEEN prediction_time_zero + 1 AND prediction_time_zero + h.horizon_days)
                   OR (first_cad BETWEEN prediction_time_zero + 1 AND prediction_time_zero + h.horizon_days)
                   OR (first_stroke BETWEEN prediction_time_zero + 1 AND prediction_time_zero + h.horizon_days)
            )::bigint AS observed_or_event_through_horizon,
            COUNT(*) FILTER (
                WHERE first_cad BETWEEN prediction_time_zero + 1 AND prediction_time_zero + h.horizon_days
                  AND (death_date IS NULL OR first_cad <= death_date)
            )::bigint AS broad_cad_events,
            COUNT(*) FILTER (
                WHERE first_stroke BETWEEN prediction_time_zero + 1 AND prediction_time_zero + h.horizon_days
                  AND (death_date IS NULL OR first_stroke <= death_date)
            )::bigint AS ischemic_stroke_events,
            COUNT(*) FILTER (
                WHERE death_date BETWEEN prediction_time_zero + 1 AND prediction_time_zero + h.horizon_days
            )::bigint AS deaths,
            MIN(prediction_time_zero)::text AS earliest_time_zero,
            MAX(prediction_time_zero)::text AS latest_time_zero
        FROM strata CROSS JOIN horizon h
        GROUP BY 1, 2, 3, 4, 5
        ORDER BY 1, 2, 3, 4, 5
        """
    ).format(s, s, s, s, s, s, s, s, s, s)


def flow_statement(schema):
    s = sql.Identifier(schema)
    return sql.SQL(
        """
        WITH ckd AS (
            SELECT DISTINCT person_id FROM {}.condition_occurrence
            WHERE condition_concept_id = ANY(%(ckd)s)
        ), dys AS (
            SELECT DISTINCT person_id FROM {}.condition_occurrence
            WHERE condition_concept_id = ANY(%(dyslipidemia)s)
        )
        SELECT 'patients_with_reviewed_ckd'::text AS stage, COUNT(*)::bigint AS patients FROM ckd
        UNION ALL
        SELECT 'patients_with_ckd_and_recorded_dyslipidemia', COUNT(*)::bigint
        FROM ckd JOIN dys USING (person_id)
        """
    ).format(s, s)


def write_summary(path, study, coverage, flow, counts):
    all_age = counts.loc[counts["age_group"].eq("ALL")].copy()
    sections = [
        "## STATUS",
        "FEASIBILITY_ONLY_NOT_CAUSAL_EFFECT",
        "## STUDY_DEFINITION",
        json.dumps(study, ensure_ascii=False, indent=2),
        "## STATIN_RX_TOKEN_COVERAGE",
        csv_text(coverage),
        "## BASIC_COHORT_FLOW",
        csv_text(flow),
        "## ELIGIBLE_GROUP_COUNTS_AND_EVENTS_ALL_AGES",
        csv_text(all_age),
        "## INTERPRETATION_RULE",
        (
            "Proceed if one edit-aligned design has enough patients and outcome events in both groups. "
            "These unadjusted counts are not treatment effects. The later "
            "answer-key must start follow-up at the same prediction_time_zero used by FERMAT."
        ),
        "## ENDPOINT_LABEL_WARNING",
        "coronary_artery_disease is the reviewed broad CAD phenotype; it is not acute MI or MACE.",
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
    if args.landmark_days < 1:
        raise ValueError("--landmark-days must be positive")
    if args.resume_from is not None:
        args.resume_from = args.resume_from.expanduser().resolve()
        if not args.resume_from.is_dir():
            raise NotADirectoryError(f"--resume-from is not a directory: {args.resume_from}")
    prepare_output(args.output_dir, args.overwrite)
    started = datetime.now(timezone.utc)

    phenotype_ids, phenotype_coverage = load_phenotype_ids(args.concept_map)
    write_csv(phenotype_coverage, args.output_dir / "phenotype_concept_coverage.csv")

    study = {
        "question": "older statin-naive CKD patients: compare edit-aligned statin initiation versus non-initiation",
        "eligibility_definitions_audited": [
            "reviewed CKD plus reviewed recorded dyslipidemia",
            "reviewed CKD plus LDL >= configured threshold",
        ],
        "minimum_age": args.minimum_age,
        "lookback_days": args.lookback_days,
        "edit_aligned_designs": {
            "DAY0_EDIT_ALIGNED": "edit a same-day RX token; prediction starts the next day",
            "DAY30_LANDMARK_EDIT_ALIGNED": "edit an RX token in the preceding 30 days; prediction starts at the day-30 landmark",
        },
        "landmark_days": args.landmark_days,
        "primary_prevention_exclusions": ["prior broad CAD", "prior ischemic stroke"],
        "outcomes": ["all-cause death", "ischemic stroke", "broad CAD"],
        "causal_effect_estimated": False,
    }
    write_json(study, args.output_dir / "study_definition.json")

    with connect_database(args) as conn:
        registry, registry_path = load_registry(args.data_dir)
        if args.resume_from is not None:
            resumed = load_resume_artifacts(args.resume_from, args.output_dir)
            plain_inventory = resumed["plain_inventory"]
            token_coverage = resumed["token_coverage"]
            ldl = resumed["ldl"]
            flow = resumed["flow"]
            combination_ids = resumed["combination_ids"]
            log(f"[RESUME] reused completed files from {args.resume_from}")
        else:
            statin_roots = resolve_roots(
                conn, args.schema, STATIN_ROOTS, "Drug", "resolve plain-statin ATC root"
            )
            write_csv(statin_roots, args.output_dir / "statin_concept_root.csv")
            root_ids = select_valid_root_ids(statin_roots, ["PLAIN_STATIN"])
            if "PLAIN_STATIN" not in root_ids:
                raise RuntimeError("Could not resolve valid ATC C10AA plain-statin root")

            inventory = observed_statin_inventory(conn, args.schema, root_ids["PLAIN_STATIN"])
            ingredients = observed_product_ingredients(
                conn, args.schema, inventory["drug_concept_id"].dropna().astype(int).unique().tolist()
            )
            ingredient_counts = (
                ingredients.groupby("drug_concept_id")["ingredient_concept_id"].nunique()
                if not ingredients.empty else pd.Series(dtype="int64")
            )
            combination_ids = set(ingredient_counts.loc[ingredient_counts > 1].index.astype(int))
            plain_inventory = inventory.loc[
                ~inventory["drug_concept_id"].astype(int).isin(combination_ids)
            ].copy()
            plain_inventory = add_registry_coverage(plain_inventory, registry)
            token_coverage = token_coverage_summary(plain_inventory)
            write_csv(inventory, args.output_dir / "observed_statin_inventory_all.csv")
            write_csv(ingredients, args.output_dir / "observed_statin_product_ingredients.csv")
            write_csv(plain_inventory, args.output_dir / "observed_plain_statin_tokens.csv")
            write_csv(token_coverage, args.output_dir / "statin_rx_token_coverage.csv")

            ldl = resolve_ldl_concepts(conn, args.schema)
            write_csv(ldl, args.output_dir / "ldl_loinc_concepts.csv")

        valid_ldl = ldl.loc[ldl["invalid_reason"].isna(), "concept_id"].astype(int).unique().tolist()
        if not valid_ldl:
            raise RuntimeError("No valid LDL LOINC measurement concepts resolved")

        statin_ids = plain_inventory["drug_concept_id"].dropna().astype(int).unique().tolist()
        if not statin_ids:
            raise RuntimeError("No observed single-ingredient plain statin products")
        params = {
            "ckd": phenotype_ids["chronic_kidney_disease"],
            "dyslipidemia": phenotype_ids["dyslipidemia"],
            "cad": phenotype_ids["coronary_artery_disease"],
            "stroke": phenotype_ids["ischemic_stroke"],
            "ldl": valid_ldl,
            "statin": statin_ids,
            "ldl_mg_dl": args.ldl_mg_dl,
            "ldl_mmol_l": args.ldl_mmol_l,
            "minimum_age": args.minimum_age,
            "lookback_days": args.lookback_days,
            "landmark_days": args.landmark_days,
            "db_end": args.db_end_date,
        }
        if args.resume_from is None:
            flow = query_df(conn, flow_statement(args.schema), params, label="basic CKD cohort flow")
            write_csv(flow, args.output_dir / "cohort_flow.csv")
        try:
            counts = query_df(
                conn, cohort_statement(args.schema), params,
                label="eligible strategy groups and 1/3/5-year events",
            )
        except Exception as exc:
            failure = {
                "status": "FAILED_ELIGIBLE_STRATEGY_QUERY",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_prior_outputs": [
                    "phenotype_concept_coverage.csv",
                    "study_definition.json",
                    "statin_concept_root.csv",
                    "observed_statin_inventory_all.csv",
                    "observed_statin_product_ingredients.csv",
                    "observed_plain_statin_tokens.csv",
                    "statin_rx_token_coverage.csv",
                    "ldl_loinc_concepts.csv",
                    "cohort_flow.csv",
                ],
            }
            write_json(failure, args.output_dir / "failure.json")
            (args.output_dir / "return_summary.txt").write_text(
                "## STATUS\nFAILED_ELIGIBLE_STRATEGY_QUERY\n"
                f"error_type={type(exc).__name__}\nerror={exc}\n",
                encoding="utf-8",
            )
            raise
        write_csv(counts, args.output_dir / "feasibility_counts.csv")

    diagnostics = {
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "concept_map": str(args.concept_map),
        "token_registry": str(registry_path),
        "resume_from": str(args.resume_from) if args.resume_from is not None else None,
        "combination_products_excluded": len(combination_ids),
        "ldl_loinc_codes": list(LDL_LOINC_CODES),
        "horizons_days": list(HORIZONS),
    }
    write_json(diagnostics, args.output_dir / "diagnostics.json")
    write_summary(
        args.output_dir / "return_summary.txt", study, token_coverage, flow, counts
    )
    log(f"[COMPLETE] {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
