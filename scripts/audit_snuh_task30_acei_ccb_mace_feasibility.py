#!/usr/bin/env python3
"""Audit whether an ACEi-vs-DHP-CCB MACE analysis is feasible on SNUH CDM.

The script performs aggregate, read-only analyses against the OMOP CDM and
checks whether the corresponding prescription source values exist as FERMAT
RX tokens.  It does not export patient-level rows.

The endpoint used for event-count feasibility is deliberately labelled
``MI_OR_STROKE_PROXY``.  Cardiovascular death is not silently replaced by
all-cause death; death-table coverage is reported separately so that the final
MACE definition can be matched to the investigator's previous study.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import io
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

try:
    import pandas as pd
except (ImportError, ModuleNotFoundError):  # pragma: no cover - Pod dependency
    pd = None

try:
    import psycopg
    from psycopg import sql
except (ImportError, ModuleNotFoundError):  # pragma: no cover - Pod dependency
    psycopg = None
    sql = None


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_OUTPUT_DIR = (
    POD_ROOT / "task30" / "outputs" / "acei_ccb_mace_feasibility_20260714"
)
DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
APPLICATION_NAME = "fermat_task30_acei_ccb_feasibility"

# ATC roots.  C09AA is plain ACE inhibitors.  C08CA is dihydropyridine CCBs
# (for example amlodipine), kept separate from non-DHP CCBs in C08D.
DRUG_CLASS_ROOTS = (
    ("ACEI", "ATC", "C09AA"),
    ("DHP_CCB", "ATC", "C08CA"),
    ("NON_DHP_CCB_AUDIT_ONLY", "ATC", "C08D"),
)
ANTIHYPERTENSIVE_ROOTS = (
    ("ANTIHYPERTENSIVES", "ATC", "C02"),
    ("DIURETICS", "ATC", "C03"),
    ("BETA_BLOCKERS", "ATC", "C07"),
    ("CALCIUM_CHANNEL_BLOCKERS", "ATC", "C08"),
    ("RAS_AGENTS", "ATC", "C09"),
)

# Standard SNOMED roots used only to count feasibility.  The resolved OMOP
# concepts and all descendants are written to disk for review.
CONDITION_ROOTS = (
    ("HYPERTENSION", "SNOMED", "38341003"),
    ("ACUTE_MI", "SNOMED", "57054005"),
    ("ISCHEMIC_STROKE", "SNOMED", "422504002"),
    ("ANY_STROKE", "SNOMED", "230690007"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--db-end-date", default="2025-02-05")
    parser.add_argument("--host", default=os.environ.get("SNUH_CDM_HOST", DEFAULT_HOST))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("SNUH_CDM_PORT", "5432"))
    )
    parser.add_argument("--dbname", default=os.environ.get("SNUH_CDM_DATABASE", "cdm"))
    parser.add_argument("--user", default=os.environ.get("SNUH_CDM_USER", "jaegyun_jung"))
    parser.add_argument("--schema", default=os.environ.get("SNUH_CDM_SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--sslmode", default=os.environ.get("SNUH_CDM_SSLMODE", "disable"))
    parser.add_argument("--statement-timeout", default="0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message: str):
    print(message, flush=True)


def require_dependencies():
    missing = []
    if pd is None:
        missing.append("pandas")
    if psycopg is None:
        missing.append("psycopg")
    if missing:
        raise RuntimeError("Missing Pod Python dependencies: " + ", ".join(missing))


def prepare_output(path: Path, overwrite: bool):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists and is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)


def password():
    value = os.environ.get("SNUH_CDM_PASSWORD")
    if value:
        return value
    return getpass.getpass("SNUH_CDM_PASSWORD: ")


def connect(args):
    conn = psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=password(),
        sslmode=args.sslmode,
        connect_timeout=15,
        application_name=APPLICATION_NAME,
    )
    with conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute("SELECT set_config('statement_timeout', %s, false)", (args.statement_timeout,))
    return conn


def query_df(conn, statement, params=None, label=None):
    started = time.time()
    if label:
        log(f"[START] {label}")
    with conn.cursor() as cur:
        cur.execute(statement, params or ())
        columns = [item.name for item in cur.description]
        rows = cur.fetchall()
    frame = pd.DataFrame(rows, columns=columns)
    if label:
        log(f"[DONE] {label}: rows={len(frame):,}, seconds={time.time() - started:,.1f}")
    return frame


def write_csv(frame, path: Path):
    frame.to_csv(path, index=False)
    log(f"[WRITE] {path} rows={len(frame):,}")


def write_json(value, path: Path):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    log(f"[WRITE] {path}")


def csv_text(frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().rstrip()


def write_return_summary(
    path: Path,
    endpoint_note,
    roots,
    coverage,
    inventory,
    flow,
    summary,
    by_year,
    death_coverage,
):
    target_inventory = inventory.loc[
        inventory["treatment_class"].isin(["ACEI", "DHP_CCB"])
    ].copy()
    target_inventory = (
        target_inventory.sort_values(
            ["treatment_class", "exposure_rows"], ascending=[True, False]
        )
        .groupby("treatment_class", sort=True, group_keys=False)
        .head(20)
    )
    token_columns = [
        "treatment_class",
        "drug_concept_id",
        "drug_concept_name",
        "expected_token_key",
        "token_in_fermat_registry",
        "exposure_rows",
        "patients",
    ]
    root_columns = [
        "root_name",
        "requested_vocabulary_id",
        "requested_concept_code",
        "root_concept_id",
        "root_concept_name",
        "invalid_reason",
    ]
    sections = [
        "## ENDPOINT_DEFINITION_STATUS",
        json.dumps(endpoint_note, ensure_ascii=False, indent=2),
        "## CONCEPT_ROOTS",
        csv_text(roots[root_columns]),
        "## DRUG_TOKEN_COVERAGE",
        csv_text(coverage),
        "## TOP_20_TOKENS_PER_TARGET_CLASS",
        csv_text(target_inventory[token_columns]),
        "## COHORT_FLOW",
        csv_text(flow),
        "## COHORT_SUMMARY",
        csv_text(summary),
        "## COHORT_BY_INDEX_YEAR",
        csv_text(by_year),
        "## DEATH_CAUSE_COVERAGE",
        csv_text(death_coverage),
    ]
    text = "\n".join(sections) + "\n"
    path.write_text(text, encoding="utf-8")
    log(f"[WRITE] {path}")
    print(text, end="", flush=True)


def values_sql(rows):
    placeholders = sql.SQL(", ").join(
        sql.SQL("({}, {}, {})").format(sql.Placeholder(), sql.Placeholder(), sql.Placeholder())
        for _ in rows
    )
    params = [value for row in rows for value in row]
    return placeholders, params


def resolve_roots(conn, schema: str, rows, domain: str, label: str):
    placeholders, params = values_sql(rows)
    statement = sql.SQL(
        """
        WITH requested(root_name, vocabulary_id, concept_code) AS (
            VALUES {}
        )
        SELECT
            r.root_name,
            r.vocabulary_id AS requested_vocabulary_id,
            r.concept_code AS requested_concept_code,
            c.concept_id AS root_concept_id,
            c.concept_name AS root_concept_name,
            c.domain_id,
            c.concept_class_id,
            c.standard_concept,
            c.invalid_reason
        FROM requested r
        LEFT JOIN {}.concept c
          ON c.vocabulary_id = r.vocabulary_id
         AND upper(c.concept_code) = upper(r.concept_code)
         AND c.domain_id = %s
        ORDER BY r.root_name, c.invalid_reason NULLS FIRST, c.concept_id
        """
    ).format(placeholders, sql.Identifier(schema))
    return query_df(conn, statement, params + [domain], label=label)


def select_valid_root_ids(root_frame, required_names):
    resolved = {}
    for name in required_names:
        subset = root_frame.loc[
            (root_frame["root_name"] == name)
            & root_frame["root_concept_id"].notna()
            & root_frame["invalid_reason"].isna()
        ]
        if subset.empty:
            continue
        standard = subset.loc[subset["standard_concept"].fillna("") == "S"]
        chosen = standard.iloc[0] if not standard.empty else subset.iloc[0]
        resolved[name] = int(chosen["root_concept_id"])
    return resolved


def descendants(conn, schema: str, root_ids: dict[str, int], domain: str, label: str):
    if not root_ids:
        return pd.DataFrame()
    rows = [(name, concept_id) for name, concept_id in root_ids.items()]
    placeholders = sql.SQL(", ").join(
        sql.SQL("({}, {})").format(sql.Placeholder(), sql.Placeholder()) for _ in rows
    )
    params = [value for row in rows for value in row]
    statement = sql.SQL(
        """
        WITH roots(root_name, root_concept_id) AS (VALUES {}),
        members AS (
            SELECT r.root_name, r.root_concept_id, r.root_concept_id AS member_concept_id
            FROM roots r
            UNION
            SELECT r.root_name, r.root_concept_id, ca.descendant_concept_id
            FROM roots r
            JOIN {}.concept_ancestor ca
              ON ca.ancestor_concept_id = r.root_concept_id
        )
        SELECT DISTINCT
            m.root_name,
            m.root_concept_id,
            c.concept_id AS member_concept_id,
            c.concept_name AS member_concept_name,
            c.vocabulary_id,
            c.concept_code,
            c.domain_id,
            c.concept_class_id,
            c.standard_concept,
            c.invalid_reason
        FROM members m
        JOIN {}.concept c ON c.concept_id = m.member_concept_id
        WHERE c.domain_id = %s
          AND c.invalid_reason IS NULL
        ORDER BY m.root_name, c.concept_id
        """
    ).format(placeholders, sql.Identifier(schema), sql.Identifier(schema))
    return query_df(conn, statement, params + [domain], label=label)


def ids_for(frame, name):
    if frame.empty:
        return []
    return sorted(
        frame.loc[frame["root_name"] == name, "member_concept_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )


def load_registry(data_dir: Path):
    for filename in ("token_registry.csv", "vocab.csv"):
        path = data_dir / filename
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                frame = pd.DataFrame(list(csv.DictReader(handle)))
            if "token_key" not in frame.columns:
                raise ValueError(f"{path} has no token_key column")
            return frame, path
    raise FileNotFoundError(f"No token_registry.csv or vocab.csv under {data_dir}")


def observed_antihypertensive_inventory(
    conn,
    schema: str,
    target_root_ids: dict[str, int],
    broad_root_ids: dict[str, int],
):
    """Classify only Drug Products that actually occur in SNUH drug_exposure."""
    roots = [
        (name, "TARGET", concept_id) for name, concept_id in target_root_ids.items()
    ] + [
        (name, "BROAD_WASHOUT", concept_id)
        for name, concept_id in broad_root_ids.items()
    ]
    placeholders = sql.SQL(", ").join(
        sql.SQL("({}, {}, {})").format(
            sql.Placeholder(), sql.Placeholder(), sql.Placeholder()
        )
        for _ in roots
    )
    params = [value for row in roots for value in row]
    statement = sql.SQL(
        """
        WITH roots(root_name, root_kind, root_concept_id) AS (VALUES {})
        SELECT
            r.root_kind,
            r.root_name,
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
        FROM roots r
        JOIN {}.concept_ancestor ca
          ON ca.ancestor_concept_id = r.root_concept_id
        JOIN {}.drug_exposure d
          ON d.drug_concept_id = ca.descendant_concept_id
         AND d.drug_exposure_start_date IS NOT NULL
        LEFT JOIN {}.concept c ON c.concept_id = d.drug_concept_id
        GROUP BY 1, 2, 3, 4, 5, 6, 7, 8
        ORDER BY r.root_kind, r.root_name, exposure_rows DESC,
                 d.drug_concept_id, fermat_source_value
        """
    ).format(
        placeholders,
        sql.Identifier(schema),
        sql.Identifier(schema),
        sql.Identifier(schema),
    )
    return query_df(
        conn,
        statement,
        params,
        label="classify antihypertensive products actually observed in SNUH",
    )


def observed_product_ingredients(conn, schema: str, product_ids):
    if not product_ids:
        return pd.DataFrame(
            columns=[
                "drug_concept_id",
                "drug_concept_name",
                "ingredient_concept_id",
                "ingredient_concept_name",
            ]
        )
    statement = sql.SQL(
        """
        SELECT DISTINCT
            ds.drug_concept_id,
            product.concept_name AS drug_concept_name,
            ds.ingredient_concept_id,
            ingredient.concept_name AS ingredient_concept_name
        FROM {}.drug_strength ds
        LEFT JOIN {}.concept product
          ON product.concept_id = ds.drug_concept_id
        LEFT JOIN {}.concept ingredient
          ON ingredient.concept_id = ds.ingredient_concept_id
        WHERE ds.drug_concept_id = ANY(%s)
        ORDER BY ds.drug_concept_id, ds.ingredient_concept_id
        """
    ).format(
        sql.Identifier(schema),
        sql.Identifier(schema),
        sql.Identifier(schema),
    )
    return query_df(
        conn,
        statement,
        (list(product_ids),),
        label="resolve ingredients for observed ACEi and CCB products",
    )


def add_registry_coverage(inventory, registry):
    frame = inventory.copy()
    if frame.empty:
        frame["expected_token_key"] = []
        frame["token_in_fermat_registry"] = []
        return frame
    keys = set(registry["token_key"].astype(str))
    frame["expected_token_key"] = "RX:" + frame["fermat_source_value"].astype(str)
    frame["token_in_fermat_registry"] = frame["expected_token_key"].isin(keys)
    return frame


def token_coverage_summary(inventory):
    if inventory.empty:
        return pd.DataFrame(
            columns=[
                "treatment_class", "exposure_rows", "patients_summed_not_unique",
                "distinct_source_tokens", "registry_source_tokens", "covered_exposure_rows",
                "exposure_row_coverage",
            ]
        )
    rows = []
    for treatment_class, group in inventory.groupby("treatment_class", sort=True):
        exposure_rows = int(group["exposure_rows"].sum())
        covered = int(
            group.loc[group["token_in_fermat_registry"], "exposure_rows"].sum()
        )
        rows.append(
            {
                "treatment_class": treatment_class,
                "exposure_rows": exposure_rows,
                "patients_summed_not_unique": int(group["patients"].sum()),
                "distinct_source_tokens": int(group["expected_token_key"].nunique()),
                "registry_source_tokens": int(
                    group.loc[group["token_in_fermat_registry"], "expected_token_key"].nunique()
                ),
                "covered_exposure_rows": covered,
                "exposure_row_coverage": covered / exposure_rows if exposure_rows else 0.0,
            }
        )
    return pd.DataFrame(rows)


def cohort_ctes(schema: str):
    s = sql.Identifier(schema)
    return sql.SQL(
        """
        antihtn_days AS (
            SELECT
                d.person_id,
                d.drug_exposure_start_date::date AS index_date,
                bool_or(d.drug_concept_id = ANY(%(acei)s)) AS has_acei,
                bool_or(d.drug_concept_id = ANY(%(dhp)s)) AS has_dhp_ccb,
                bool_or(
                    NOT (d.drug_concept_id = ANY(%(acei)s))
                    AND NOT (d.drug_concept_id = ANY(%(dhp)s))
                ) AS has_other_antihypertensive,
                COUNT(DISTINCT d.drug_concept_id)::integer AS distinct_drug_concepts
            FROM {}.drug_exposure d
            WHERE d.drug_exposure_start_date BETWEEN DATE '1900-01-01' AND %(db_end)s
              AND d.drug_concept_id = ANY(%(all_antihtn)s)
            GROUP BY d.person_id, d.drug_exposure_start_date::date
        ),
        ordered_days AS (
            SELECT
                a.*,
                lag(index_date) OVER (PARTITION BY person_id ORDER BY index_date)
                    AS previous_antihypertensive_date
            FROM antihtn_days a
        ),
        target_new_user_days AS (
            SELECT
                o.*,
                CASE WHEN has_acei THEN 'ACEI' ELSE 'DHP_CCB' END AS treatment_class,
                (has_acei AND has_dhp_ccb) AS started_both_target_classes,
                (
                    previous_antihypertensive_date IS NULL
                    OR previous_antihypertensive_date
                        < index_date - (%(lookback)s * INTERVAL '1 day')
                ) AS passed_drug_washout
            FROM ordered_days o
            WHERE (has_acei OR has_dhp_ccb)
        ),
        first_candidate AS (
            SELECT DISTINCT ON (person_id)
                *
            FROM target_new_user_days
            WHERE passed_drug_washout
            ORDER BY person_id, index_date
        ),
        candidate_flags AS (
            SELECT
                f.*,
                EXISTS (
                    SELECT 1
                    FROM {}.observation_period op
                    WHERE op.person_id = f.person_id
                      AND op.observation_period_start_date
                            <= f.index_date - (%(lookback)s * INTERVAL '1 day')
                      AND op.observation_period_end_date >= f.index_date
                ) AS has_observation_lookback,
                EXISTS (
                    SELECT 1
                    FROM {}.condition_occurrence c
                    WHERE c.person_id = f.person_id
                      AND c.condition_concept_id = ANY(%(hypertension)s)
                      AND c.condition_start_date
                            BETWEEN f.index_date - (%(lookback)s * INTERVAL '1 day')
                                AND f.index_date
                ) AS has_recent_hypertension,
                EXISTS (
                    SELECT 1
                    FROM {}.condition_occurrence c
                    WHERE c.person_id = f.person_id
                      AND c.condition_concept_id = ANY(%(mace_proxy)s)
                      AND c.condition_start_date < f.index_date
                ) AS has_prior_mi_or_stroke,
                (
                    SELECT MAX(op.observation_period_end_date)
                    FROM {}.observation_period op
                    WHERE op.person_id = f.person_id
                      AND op.observation_period_start_date <= f.index_date
                      AND op.observation_period_end_date >= f.index_date
                ) AS observation_end,
                d.death_date
            FROM first_candidate f
            LEFT JOIN (
                SELECT person_id, MIN(death_date)::date AS death_date
                FROM {}.death
                GROUP BY person_id
            ) d ON d.person_id = f.person_id
        ),
        final_cohort AS (
            SELECT
                c.*,
                LEAST(
                    COALESCE(c.observation_end, %(db_end)s),
                    COALESCE(c.death_date, %(db_end)s),
                    %(db_end)s
                )::date AS followup_end
            FROM candidate_flags c
            WHERE NOT c.started_both_target_classes
              AND NOT c.has_other_antihypertensive
              AND c.has_observation_lookback
              AND c.has_recent_hypertension
              AND NOT c.has_prior_mi_or_stroke
        ),
        outcome_dates AS (
            SELECT
                f.person_id,
                MIN(c.condition_start_date)::date AS first_mi_or_stroke_date
            FROM final_cohort f
            JOIN {}.condition_occurrence c
              ON c.person_id = f.person_id
             AND c.condition_concept_id = ANY(%(mace_proxy)s)
             AND c.condition_start_date > f.index_date
             AND c.condition_start_date <= f.followup_end
            GROUP BY f.person_id
        ),
        analysed AS (
            SELECT f.*, o.first_mi_or_stroke_date
            FROM final_cohort f
            LEFT JOIN outcome_dates o ON o.person_id = f.person_id
        )
        """
    ).format(s, s, s, s, s, s, s)


def cohort_params(args, class_ids, condition_ids, all_antihtn):
    mi_or_stroke = sorted(
        set(condition_ids["ACUTE_MI"])
        | set(condition_ids["ISCHEMIC_STROKE"])
        | set(condition_ids["ANY_STROKE"])
    )
    return {
        "acei": class_ids["ACEI"] or [-1],
        "dhp": class_ids["DHP_CCB"] or [-1],
        "all_antihtn": all_antihtn or [-1],
        "hypertension": condition_ids["HYPERTENSION"] or [-1],
        "mace_proxy": mi_or_stroke or [-1],
        "lookback": int(args.lookback_days),
        "db_end": date.fromisoformat(args.db_end_date),
    }


def cohort_audit(conn, args, class_ids, condition_ids, all_antihtn):
    statement = sql.SQL(
        """
        WITH {}
        SELECT
            'FLOW'::text AS section,
            NULL::integer AS index_year,
            treatment_class,
            COUNT(*)::bigint AS washout_passed_candidates,
            COUNT(*) FILTER (WHERE started_both_target_classes)::bigint
                AS same_day_acei_and_dhp_ccb,
            COUNT(*) FILTER (WHERE has_other_antihypertensive)::bigint
                AS same_day_other_antihypertensive,
            COUNT(*) FILTER (WHERE NOT has_observation_lookback)::bigint
                AS insufficient_observation_lookback,
            COUNT(*) FILTER (WHERE NOT has_recent_hypertension)::bigint
                AS no_recent_hypertension_diagnosis,
            COUNT(*) FILTER (WHERE has_prior_mi_or_stroke)::bigint
                AS prior_mi_or_stroke,
            COUNT(*) FILTER (
                WHERE NOT started_both_target_classes
                  AND NOT has_other_antihypertensive
                  AND has_observation_lookback
                  AND has_recent_hypertension
                  AND NOT has_prior_mi_or_stroke
            )::bigint AS final_eligible_patients,
            NULL::bigint AS observable_1y_or_event,
            NULL::bigint AS mi_or_stroke_within_1y,
            NULL::bigint AS observable_3y_or_event,
            NULL::bigint AS mi_or_stroke_within_3y,
            NULL::bigint AS observable_5y_or_event,
            NULL::bigint AS mi_or_stroke_within_5y,
            NULL::bigint AS all_cause_death_within_1y,
            NULL::bigint AS all_cause_death_within_3y,
            NULL::bigint AS all_cause_death_within_5y,
            NULL::text AS earliest_index_date,
            NULL::text AS latest_index_date
        FROM candidate_flags
        GROUP BY treatment_class

        UNION ALL

        SELECT
            'SUMMARY'::text AS section,
            NULL::integer AS index_year,
            treatment_class,
            NULL::bigint AS washout_passed_candidates,
            NULL::bigint AS same_day_acei_and_dhp_ccb,
            NULL::bigint AS same_day_other_antihypertensive,
            NULL::bigint AS insufficient_observation_lookback,
            NULL::bigint AS no_recent_hypertension_diagnosis,
            NULL::bigint AS prior_mi_or_stroke,
            COUNT(*)::bigint AS final_eligible_patients,
            COUNT(*) FILTER (
                WHERE followup_end >= index_date + INTERVAL '1 year'
                   OR first_mi_or_stroke_date <= index_date + INTERVAL '1 year'
            )::bigint AS observable_1y_or_event,
            COUNT(*) FILTER (
                WHERE first_mi_or_stroke_date <= index_date + INTERVAL '1 year'
            )::bigint AS mi_or_stroke_within_1y,
            COUNT(*) FILTER (
                WHERE followup_end >= index_date + INTERVAL '3 years'
                   OR first_mi_or_stroke_date <= index_date + INTERVAL '3 years'
            )::bigint AS observable_3y_or_event,
            COUNT(*) FILTER (
                WHERE first_mi_or_stroke_date <= index_date + INTERVAL '3 years'
            )::bigint AS mi_or_stroke_within_3y,
            COUNT(*) FILTER (
                WHERE followup_end >= index_date + INTERVAL '5 years'
                   OR first_mi_or_stroke_date <= index_date + INTERVAL '5 years'
            )::bigint AS observable_5y_or_event,
            COUNT(*) FILTER (
                WHERE first_mi_or_stroke_date <= index_date + INTERVAL '5 years'
            )::bigint AS mi_or_stroke_within_5y,
            COUNT(*) FILTER (
                WHERE death_date > index_date
                  AND death_date <= index_date + INTERVAL '1 year'
            )::bigint AS all_cause_death_within_1y,
            COUNT(*) FILTER (
                WHERE death_date > index_date
                  AND death_date <= index_date + INTERVAL '3 years'
            )::bigint AS all_cause_death_within_3y,
            COUNT(*) FILTER (
                WHERE death_date > index_date
                  AND death_date <= index_date + INTERVAL '5 years'
            )::bigint AS all_cause_death_within_5y,
            MIN(index_date)::text AS earliest_index_date,
            MAX(index_date)::text AS latest_index_date
        FROM analysed
        GROUP BY treatment_class

        UNION ALL

        SELECT
            'BY_YEAR'::text AS section,
            EXTRACT(YEAR FROM index_date)::integer AS index_year,
            treatment_class,
            NULL::bigint AS washout_passed_candidates,
            NULL::bigint AS same_day_acei_and_dhp_ccb,
            NULL::bigint AS same_day_other_antihypertensive,
            NULL::bigint AS insufficient_observation_lookback,
            NULL::bigint AS no_recent_hypertension_diagnosis,
            NULL::bigint AS prior_mi_or_stroke,
            COUNT(*)::bigint AS final_eligible_patients,
            NULL::bigint AS observable_1y_or_event,
            NULL::bigint AS mi_or_stroke_within_1y,
            NULL::bigint AS observable_3y_or_event,
            NULL::bigint AS mi_or_stroke_within_3y,
            COUNT(*) FILTER (
                WHERE followup_end >= index_date + INTERVAL '5 years'
                   OR first_mi_or_stroke_date <= index_date + INTERVAL '5 years'
            )::bigint AS observable_5y_or_event,
            COUNT(*) FILTER (
                WHERE first_mi_or_stroke_date <= index_date + INTERVAL '5 years'
            )::bigint AS mi_or_stroke_within_5y,
            NULL::bigint AS all_cause_death_within_1y,
            NULL::bigint AS all_cause_death_within_3y,
            NULL::bigint AS all_cause_death_within_5y,
            NULL::text AS earliest_index_date,
            NULL::text AS latest_index_date
        FROM analysed
        GROUP BY EXTRACT(YEAR FROM index_date), treatment_class

        ORDER BY section, index_year NULLS FIRST, treatment_class
        """
    ).format(cohort_ctes(args.schema))
    return query_df(
        conn,
        statement,
        cohort_params(args, class_ids, condition_ids, all_antihtn),
        label="single-pass ACEi vs DHP-CCB cohort audit",
    )


def death_profile(conn, schema: str):
    columns = query_df(
        conn,
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = 'death'
        ORDER BY ordinal_position
        """,
        (schema,),
        label="death table columns",
    )
    names = set(columns["column_name"].astype(str))
    cause_concept = "cause_concept_id" if "cause_concept_id" in names else None
    cause_source = "cause_source_value" if "cause_source_value" in names else None
    if not cause_concept and not cause_source:
        return columns, pd.DataFrame(
            [{"death_rows": None, "cause_concept_populated": None, "cause_source_populated": None}]
        ), pd.DataFrame()

    s = sql.Identifier(schema)
    concept_expr = (
        sql.SQL("COUNT(*) FILTER (WHERE cause_concept_id IS NOT NULL AND cause_concept_id <> 0)")
        if cause_concept
        else sql.SQL("NULL::bigint")
    )
    source_expr = (
        sql.SQL("COUNT(*) FILTER (WHERE NULLIF(btrim(cause_source_value), '') IS NOT NULL)")
        if cause_source
        else sql.SQL("NULL::bigint")
    )
    profile = query_df(
        conn,
        sql.SQL(
            "SELECT COUNT(*)::bigint AS death_rows, {}::bigint AS cause_concept_populated, "
            "{}::bigint AS cause_source_populated FROM {}.death"
        ).format(concept_expr, source_expr, s),
        label="death cause coverage",
    )
    causes = pd.DataFrame()
    if cause_concept:
        causes = query_df(
            conn,
            sql.SQL(
                """
                SELECT d.cause_concept_id, c.concept_name, c.vocabulary_id,
                       c.concept_code, COUNT(*)::bigint AS deaths
                FROM {}.death d
                LEFT JOIN {}.concept c ON c.concept_id = d.cause_concept_id
                WHERE d.cause_concept_id IS NOT NULL AND d.cause_concept_id <> 0
                GROUP BY 1, 2, 3, 4
                ORDER BY deaths DESC
                LIMIT 100
                """
            ).format(s, s),
            label="top populated death causes",
        )
    return columns, profile, causes


def main():
    args = parse_args()
    require_dependencies()
    if args.lookback_days < 1:
        raise ValueError("--lookback-days must be positive")
    prepare_output(args.output_dir, args.overwrite)
    started = datetime.now(timezone.utc)

    endpoint_note = {
        "final_mace_definition_status": "NOT_YET_FIXED",
        "feasibility_event_proxy": "first diagnosis of acute MI or stroke after treatment start",
        "cardiovascular_death": (
            "Not included in the proxy. Death-table cause coverage is reported separately. "
            "Match the final definition to the investigator's previous abstract before modeling."
        ),
        "treatment_comparison": "ACEI monotherapy-class start versus DHP_CCB monotherapy-class start",
        "why_dhp_ccb": "Non-DHP CCB is audited separately because it is not interchangeable with amlodipine-type CCB.",
    }
    write_json(endpoint_note, args.output_dir / "endpoint_definition_status.json")

    with connect(args) as conn:
        drug_roots = resolve_roots(
            conn, args.schema, DRUG_CLASS_ROOTS, "Drug", "resolve ACEi and CCB ATC roots"
        )
        antihtn_roots = resolve_roots(
            conn, args.schema, ANTIHYPERTENSIVE_ROOTS, "Drug", "resolve antihypertensive ATC roots"
        )
        condition_roots = resolve_roots(
            conn, args.schema, CONDITION_ROOTS, "Condition", "resolve hypertension and outcome roots"
        )
        all_roots = pd.concat([drug_roots, antihtn_roots, condition_roots], ignore_index=True)
        write_csv(all_roots, args.output_dir / "concept_roots.csv")

        drug_root_ids = select_valid_root_ids(drug_roots, [row[0] for row in DRUG_CLASS_ROOTS])
        antihtn_root_ids = select_valid_root_ids(
            antihtn_roots, [row[0] for row in ANTIHYPERTENSIVE_ROOTS]
        )
        condition_root_ids = select_valid_root_ids(
            condition_roots, [row[0] for row in CONDITION_ROOTS]
        )
        required_drug_roots = {"ACEI", "DHP_CCB"}
        missing = {
            "drug_roots": sorted(required_drug_roots - set(drug_root_ids)),
            "antihypertensive_roots": sorted(
                set(row[0] for row in ANTIHYPERTENSIVE_ROOTS) - set(antihtn_root_ids)
            ),
            "condition_roots": sorted(
                set(row[0] for row in CONDITION_ROOTS) - set(condition_root_ids)
            ),
        }
        if any(missing.values()):
            write_json(missing, args.output_dir / "missing_required_roots.json")
            raise RuntimeError(f"Required OMOP roots were not resolved: {missing}")

        observed_drugs = observed_antihypertensive_inventory(
            conn,
            args.schema,
            drug_root_ids,
            antihtn_root_ids,
        )
        write_csv(
            observed_drugs,
            args.output_dir / "observed_antihypertensive_inventory.csv",
        )

        target_raw = observed_drugs.loc[
            observed_drugs["root_kind"] == "TARGET"
        ].copy()
        target_product_ids = sorted(
            target_raw["drug_concept_id"].dropna().astype(int).unique().tolist()
        )
        ingredients = observed_product_ingredients(
            conn, args.schema, target_product_ids
        )
        write_csv(
            ingredients,
            args.output_dir / "observed_target_product_ingredients.csv",
        )
        ingredient_counts = (
            ingredients.groupby("drug_concept_id")["ingredient_concept_id"].nunique()
            if not ingredients.empty
            else pd.Series(dtype="int64")
        )
        combination_ids = set(
            ingredient_counts.loc[ingredient_counts > 1].index.astype(int).tolist()
        )
        if combination_ids:
            log(
                "[MONOTHERAPY_FILTER] excluded observed target products with multiple "
                f"ingredients: {len(combination_ids):,}"
            )
        target = target_raw.loc[
            ~target_raw["drug_concept_id"].astype(int).isin(combination_ids)
        ].copy()
        target = target.rename(columns={"root_name": "treatment_class"})

        condition_members = descendants(
            conn,
            args.schema,
            condition_root_ids,
            "Condition",
            "resolve hypertension and outcome descendants",
        )
        drug_class_concepts = target[
            [
                "treatment_class",
                "drug_concept_id",
                "drug_concept_name",
                "vocabulary_id",
                "concept_class_id",
            ]
        ].drop_duplicates()
        write_csv(
            drug_class_concepts,
            args.output_dir / "drug_class_concepts.csv",
        )
        write_csv(condition_members, args.output_dir / "condition_endpoint_concepts.csv")

        class_ids = {
            name: sorted(
                target.loc[
                    target["treatment_class"] == name, "drug_concept_id"
                ].dropna().astype(int).unique().tolist()
            )
            for name, _, _ in DRUG_CLASS_ROOTS
        }
        condition_ids = {name: ids_for(condition_members, name) for name in condition_root_ids}
        all_antihtn = sorted(
            observed_drugs.loc[
                observed_drugs["root_kind"] == "BROAD_WASHOUT",
                "drug_concept_id",
            ].dropna().astype(int).unique().tolist()
        )
        class_size_guard = {
            "meaning": "counts below are products actually observed in SNUH drug_exposure",
            "SNUH_OBSERVED_ACEI_PRODUCTS": len(class_ids["ACEI"]),
            "SNUH_OBSERVED_DHP_CCB_PRODUCTS": len(class_ids["DHP_CCB"]),
            "SNUH_OBSERVED_NON_DHP_CCB_PRODUCTS": len(
                class_ids["NON_DHP_CCB_AUDIT_ONLY"]
            ),
            "SNUH_OBSERVED_ANTIHYPERTENSIVE_PRODUCTS": len(all_antihtn),
            "OBSERVED_TARGET_COMBINATION_PRODUCTS_EXCLUDED": len(combination_ids),
            "guard_limits": {
                "each_target_class": 10000,
                "all_antihypertensive": 50000,
            },
        }
        write_json(class_size_guard, args.output_dir / "class_size_guard.json")
        log("[CLASS_SIZE] " + json.dumps(class_size_guard, ensure_ascii=False))
        if not class_ids["ACEI"] or not class_ids["DHP_CCB"] or not all_antihtn:
            raise RuntimeError(
                "Resolved class membership is empty for ACEI, DHP_CCB, or all antihypertensives"
            )
        if (
            len(class_ids["ACEI"]) > 10000
            or len(class_ids["DHP_CCB"]) > 10000
            or len(all_antihtn) > 50000
        ):
            raise RuntimeError(
                "Observed SNUH class membership is implausibly broad; stopped before cohort query. "
                f"See class_size_guard.json: {class_size_guard}"
            )

        registry, registry_path = load_registry(args.data_dir)
        inventory = add_registry_coverage(target, registry)
        write_csv(inventory, args.output_dir / "drug_token_inventory.csv")
        coverage = token_coverage_summary(inventory)
        write_csv(coverage, args.output_dir / "drug_token_coverage_summary.csv")

        cohort_results = cohort_audit(conn, args, class_ids, condition_ids, all_antihtn)
        flow = cohort_results.loc[cohort_results["section"] == "FLOW"].drop(
            columns=["section", "index_year"]
        ).dropna(axis="columns", how="all")
        summary = cohort_results.loc[cohort_results["section"] == "SUMMARY"].drop(
            columns=["section", "index_year"]
        ).dropna(axis="columns", how="all")
        by_year = cohort_results.loc[cohort_results["section"] == "BY_YEAR"].drop(
            columns=["section"]
        ).dropna(axis="columns", how="all")
        write_csv(flow, args.output_dir / "cohort_flow.csv")
        write_csv(summary, args.output_dir / "cohort_summary.csv")
        write_csv(by_year, args.output_dir / "cohort_by_index_year.csv")

        death_columns, death_coverage, death_causes = death_profile(conn, args.schema)
        write_csv(death_columns, args.output_dir / "death_table_columns.csv")
        write_csv(death_coverage, args.output_dir / "death_cause_coverage.csv")
        write_csv(death_causes, args.output_dir / "death_cause_top100.csv")

    finished = datetime.now(timezone.utc)
    manifest = {
        "status": "complete",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "schema": args.schema,
        "db_end_date": args.db_end_date,
        "lookback_days": args.lookback_days,
        "data_dir": str(args.data_dir),
        "registry_path": str(registry_path),
        "output_dir": str(args.output_dir),
        "patient_level_rows_exported": False,
        "class_member_counts": {key: len(value) for key, value in class_ids.items()},
        "condition_member_counts": {key: len(value) for key, value in condition_ids.items()},
        "all_antihypertensive_member_count": len(all_antihtn),
    }
    write_json(manifest, args.output_dir / "run_manifest.json")
    write_return_summary(
        args.output_dir / "return_summary.txt",
        endpoint_note,
        all_roots,
        coverage,
        inventory,
        flow,
        summary,
        by_year,
        death_coverage,
    )
    log("[COMPLETE] Aggregate ACEi-vs-DHP-CCB feasibility audit finished.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
