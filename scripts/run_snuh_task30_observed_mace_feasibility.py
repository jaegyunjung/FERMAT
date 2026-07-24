#!/usr/bin/env python3
"""Draw observed post-treatment cardiovascular event curves for Task 30.

This is deliberately an observed-data feasibility analysis, not a FERMAT or
counterfactual model.  It reuses the SNUH-observed drug products written by the
Task 30 v3 mapping stage, builds a treatment-new-user cohort in indexed
temporary PostgreSQL tables, and calculates cumulative incidence of acute MI
or stroke after ACEi versus DHP-CCB initiation.  All-cause death is handled as
a competing event.

The current endpoint is not labelled final MACE because cardiovascular death
has not yet been defined from the investigator's previous abstract.  No
patient identifiers are written to disk.
"""

from __future__ import annotations

import argparse
import getpass
import io
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except (ImportError, ModuleNotFoundError):  # pragma: no cover - Pod dependency
    np = None
    pd = None

try:
    import psycopg
    from psycopg import sql
except (ImportError, ModuleNotFoundError):  # pragma: no cover - Pod dependency
    psycopg = None
    sql = None


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_MAPPING_DIR = (
    POD_ROOT
    / "task30"
    / "outputs"
    / "acei_ccb_mace_feasibility_v3_20260714_014127"
)
DEFAULT_OUTPUT_DIR = (
    POD_ROOT / "task30" / "outputs" / "observed_mi_stroke_curves"
)
DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
APPLICATION_NAME = "fermat_task30_observed_mace_feasibility"
LANDMARK_DAYS = (365, 1095, 1826)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mapping-dir", type=Path, default=DEFAULT_MAPPING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--db-end-date", default="2025-02-05")
    parser.add_argument("--max-curve-days", type=int, default=1826)
    parser.add_argument("--host", default=os.environ.get("SNUH_CDM_HOST", DEFAULT_HOST))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("SNUH_CDM_PORT", "5432"))
    )
    parser.add_argument("--dbname", default=os.environ.get("SNUH_CDM_DATABASE", "cdm"))
    parser.add_argument("--user", default=os.environ.get("SNUH_CDM_USER", "jaegyun_jung"))
    parser.add_argument("--schema", default=os.environ.get("SNUH_CDM_SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--sslmode", default=os.environ.get("SNUH_CDM_SSLMODE", "disable"))
    parser.add_argument("--statement-timeout", default="0")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def require_dependencies():
    missing = []
    if np is None or pd is None:
        missing.append("numpy/pandas")
    if psycopg is None:
        missing.append("psycopg")
    if missing:
        raise RuntimeError("Missing Pod dependencies: " + ", ".join(missing))


def prepare_output(path: Path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"{path} exists and is not empty; use a new output directory")
    path.mkdir(parents=True, exist_ok=True)


def password():
    value = os.environ.get("SNUH_CDM_PASSWORD")
    return value if value else getpass.getpass("SNUH_CDM_PASSWORD: ")


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
        autocommit=True,
    )
    with conn.cursor() as cur:
        cur.execute("SELECT set_config('statement_timeout', %s, false)", (args.statement_timeout,))
    return conn


def execute(conn, statement, params=None, label=None):
    started = time.time()
    if label:
        log(f"[START] {label}")
    with conn.cursor() as cur:
        cur.execute(statement, params or ())
    if label:
        log(f"[DONE] {label}: seconds={time.time() - started:,.1f}")


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


def write_csv(frame, path):
    frame.to_csv(path, index=False)
    log(f"[WRITE] {path}: rows={len(frame):,}")


def write_json(value, path):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    log(f"[WRITE] {path}")


def load_mapping(mapping_dir: Path):
    paths = {
        "products": mapping_dir / "observed_antihypertensive_inventory.csv",
        "targets": mapping_dir / "drug_class_concepts.csv",
        "conditions": mapping_dir / "condition_endpoint_concepts.csv",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    products = pd.read_csv(paths["products"])
    targets = pd.read_csv(paths["targets"])
    conditions = pd.read_csv(paths["conditions"])

    required_products = {"root_kind", "drug_concept_id"}
    required_targets = {"treatment_class", "drug_concept_id"}
    required_conditions = {"root_name", "member_concept_id"}
    if not required_products.issubset(products.columns):
        raise ValueError(f"{paths['products']} lacks {sorted(required_products)}")
    if not required_targets.issubset(targets.columns):
        raise ValueError(f"{paths['targets']} lacks {sorted(required_targets)}")
    if not required_conditions.issubset(conditions.columns):
        raise ValueError(f"{paths['conditions']} lacks {sorted(required_conditions)}")

    broad = set(
        products.loc[
            products["root_kind"] == "BROAD_WASHOUT", "drug_concept_id"
        ].dropna().astype(int)
    )
    acei = set(
        targets.loc[targets["treatment_class"] == "ACEI", "drug_concept_id"]
        .dropna()
        .astype(int)
    )
    dhp = set(
        targets.loc[targets["treatment_class"] == "DHP_CCB", "drug_concept_id"]
        .dropna()
        .astype(int)
    )
    broad.update(acei)
    broad.update(dhp)

    product_flags = pd.DataFrame(
        [
            {
                "drug_concept_id": concept_id,
                "is_acei": concept_id in acei,
                "is_dhp_ccb": concept_id in dhp,
            }
            for concept_id in sorted(broad)
        ]
    )
    if not acei or not dhp or product_flags.empty:
        raise ValueError(
            f"Observed mappings are incomplete: ACEI={len(acei)}, DHP_CCB={len(dhp)}, broad={len(broad)}"
        )

    condition_sets = {}
    for name, group in conditions.groupby("root_name"):
        condition_sets[str(name)] = sorted(
            group["member_concept_id"].dropna().astype(int).unique().tolist()
        )
    required_roots = {"HYPERTENSION", "ACUTE_MI", "ANY_STROKE"}
    missing = sorted(required_roots - set(condition_sets))
    if missing:
        raise ValueError(f"Missing condition roots in mapping: {missing}")

    return product_flags, condition_sets, {key: str(value) for key, value in paths.items()}


def create_product_flags(conn, frame):
    execute(
        conn,
        """
        CREATE TEMP TABLE t30_product_flags (
            drug_concept_id bigint PRIMARY KEY,
            is_acei boolean NOT NULL,
            is_dhp_ccb boolean NOT NULL
        ) ON COMMIT PRESERVE ROWS
        """,
        label="create temporary observed-drug map",
    )
    rows = [
        (int(row.drug_concept_id), bool(row.is_acei), bool(row.is_dhp_ccb))
        for row in frame.itertuples(index=False)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO t30_product_flags VALUES (%s, %s, %s)",
            rows,
        )
        cur.execute("ANALYZE t30_product_flags")
    log(f"[TEMP MAP] observed antihypertensive products={len(rows):,}")


def table_count(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT COUNT(*)::bigint FROM {}").format(sql.Identifier(table_name)))
        return int(cur.fetchone()[0])


def record_stage(conn, stages, output_dir, stage, table_name):
    count = table_count(conn, table_name)
    stages.append({"stage": stage, "rows": count, "table": table_name})
    write_csv(pd.DataFrame(stages), output_dir / "stage_counts.csv")
    log(f"[STAGE COUNT] {stage}: {count:,}")


def build_cohort_tables(conn, args, condition_sets, output_dir):
    stages = []
    schema = sql.Identifier(args.schema)
    db_end = date.fromisoformat(args.db_end_date)

    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE t30_antihtn_days ON COMMIT PRESERVE ROWS AS
            SELECT
                d.person_id,
                d.drug_exposure_start_date::date AS index_date,
                bool_or(p.is_acei) AS has_acei,
                bool_or(p.is_dhp_ccb) AS has_dhp_ccb,
                bool_or(NOT p.is_acei AND NOT p.is_dhp_ccb) AS has_other
            FROM {}.drug_exposure d
            JOIN t30_product_flags p ON p.drug_concept_id = d.drug_concept_id
            WHERE d.drug_exposure_start_date BETWEEN DATE '1900-01-01' AND %s
            GROUP BY d.person_id, d.drug_exposure_start_date::date
            """
        ).format(schema),
        (db_end,),
        label="collect observed antihypertensive treatment days",
    )
    execute(
        conn,
        "CREATE INDEX ON t30_antihtn_days (person_id, index_date)",
        label="index treatment days",
    )
    record_stage(conn, stages, output_dir, "antihypertensive_days", "t30_antihtn_days")

    execute(
        conn,
        """
        CREATE TEMP TABLE t30_ordered_days ON COMMIT PRESERVE ROWS AS
        SELECT
            a.*,
            lag(index_date) OVER (PARTITION BY person_id ORDER BY index_date)
                AS previous_antihypertensive_date
        FROM t30_antihtn_days a
        """,
        label="order treatment days and calculate prior-use gap",
    )
    execute(
        conn,
        "CREATE INDEX ON t30_ordered_days (person_id, index_date)",
        label="index ordered treatment days",
    )

    execute(
        conn,
        """
        CREATE TEMP TABLE t30_candidates ON COMMIT PRESERVE ROWS AS
        SELECT DISTINCT ON (person_id)
            person_id,
            index_date,
            CASE WHEN has_acei THEN 'ACEI' ELSE 'DHP_CCB' END AS treatment_class
        FROM t30_ordered_days
        WHERE has_acei <> has_dhp_ccb
          AND NOT has_other
          AND (
                previous_antihypertensive_date IS NULL
                OR previous_antihypertensive_date
                    < index_date - (%s * INTERVAL '1 day')
          )
        ORDER BY person_id, index_date
        """,
        (int(args.lookback_days),),
        label="select ACEi or DHP-CCB new-user candidates",
    )
    execute(
        conn,
        "CREATE UNIQUE INDEX ON t30_candidates (person_id)",
        label="index candidates",
    )
    record_stage(conn, stages, output_dir, "new_user_candidates", "t30_candidates")

    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE t30_observed ON COMMIT PRESERVE ROWS AS
            SELECT
                c.person_id,
                c.index_date,
                c.treatment_class,
                MAX(op.observation_period_end_date)::date AS observation_end
            FROM t30_candidates c
            JOIN {}.observation_period op
              ON op.person_id = c.person_id
             AND op.observation_period_start_date
                    <= c.index_date - (%s * INTERVAL '1 day')
             AND op.observation_period_end_date >= c.index_date
            GROUP BY c.person_id, c.index_date, c.treatment_class
            """
        ).format(schema),
        (int(args.lookback_days),),
        label="require one-year database observation before treatment",
    )
    execute(
        conn,
        "CREATE UNIQUE INDEX ON t30_observed (person_id)",
        label="index observed candidates",
    )
    record_stage(conn, stages, output_dir, "with_observation_lookback", "t30_observed")

    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE t30_hypertension ON COMMIT PRESERVE ROWS AS
            SELECT DISTINCT o.person_id
            FROM t30_observed o
            JOIN {}.condition_occurrence c
              ON c.person_id = o.person_id
             AND c.condition_concept_id = ANY(%s)
             AND c.condition_start_date
                    BETWEEN o.index_date - (%s * INTERVAL '1 day') AND o.index_date
            """
        ).format(schema),
        (condition_sets["HYPERTENSION"], int(args.lookback_days)),
        label="confirm recent hypertension diagnosis",
    )
    execute(conn, "CREATE UNIQUE INDEX ON t30_hypertension (person_id)")
    record_stage(conn, stages, output_dir, "with_recent_hypertension", "t30_hypertension")

    outcome_ids = sorted(
        set(condition_sets["ACUTE_MI"]) | set(condition_sets["ANY_STROKE"])
    )
    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE t30_prior_outcome ON COMMIT PRESERVE ROWS AS
            SELECT DISTINCT o.person_id
            FROM t30_observed o
            JOIN {}.condition_occurrence c
              ON c.person_id = o.person_id
             AND c.condition_concept_id = ANY(%s)
             AND c.condition_start_date < o.index_date
            """
        ).format(schema),
        (outcome_ids,),
        label="identify prior MI or stroke",
    )
    execute(conn, "CREATE UNIQUE INDEX ON t30_prior_outcome (person_id)")

    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE t30_death ON COMMIT PRESERVE ROWS AS
            SELECT d.person_id, MIN(d.death_date)::date AS death_date
            FROM {}.death d
            JOIN t30_observed o ON o.person_id = d.person_id
            GROUP BY d.person_id
            """
        ).format(schema),
        label="collect death dates for observed candidates",
    )
    execute(conn, "CREATE UNIQUE INDEX ON t30_death (person_id)")

    execute(
        conn,
        """
        CREATE TEMP TABLE t30_final ON COMMIT PRESERVE ROWS AS
        SELECT
            o.*,
            d.death_date,
            LEAST(
                o.observation_end,
                COALESCE(d.death_date, %s),
                %s
            )::date AS followup_end
        FROM t30_observed o
        JOIN t30_hypertension h ON h.person_id = o.person_id
        LEFT JOIN t30_prior_outcome p ON p.person_id = o.person_id
        LEFT JOIN t30_death d ON d.person_id = o.person_id
        WHERE p.person_id IS NULL
        """,
        (db_end, db_end),
        label="build final observed treatment cohort",
    )
    execute(conn, "CREATE UNIQUE INDEX ON t30_final (person_id)")
    record_stage(conn, stages, output_dir, "final_cohort", "t30_final")

    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE t30_outcomes ON COMMIT PRESERVE ROWS AS
            SELECT
                f.person_id,
                MIN(c.condition_start_date)::date AS first_outcome_date,
                MIN(c.condition_start_date) FILTER (
                    WHERE c.condition_concept_id = ANY(%s)
                )::date AS first_mi_date,
                MIN(c.condition_start_date) FILTER (
                    WHERE c.condition_concept_id = ANY(%s)
                )::date AS first_stroke_date
            FROM t30_final f
            JOIN {}.condition_occurrence c
              ON c.person_id = f.person_id
             AND c.condition_concept_id = ANY(%s)
             AND c.condition_start_date > f.index_date
             AND c.condition_start_date <= f.followup_end
            GROUP BY f.person_id
            """
        ).format(schema),
        (
            condition_sets["ACUTE_MI"],
            condition_sets["ANY_STROKE"],
            outcome_ids,
        ),
        label="collect first post-treatment MI or stroke",
    )
    execute(conn, "CREATE UNIQUE INDEX ON t30_outcomes (person_id)")

    execute(
        conn,
        """
        CREATE TEMP TABLE t30_survival ON COMMIT PRESERVE ROWS AS
        SELECT
            f.treatment_class,
            GREATEST(
                1,
                CASE
                    WHEN o.first_outcome_date IS NOT NULL
                         AND (f.death_date IS NULL OR o.first_outcome_date <= f.death_date)
                        THEN o.first_outcome_date - f.index_date
                    WHEN f.death_date IS NOT NULL
                         AND f.death_date > f.index_date
                         AND f.death_date <= f.followup_end
                        THEN f.death_date - f.index_date
                    ELSE f.followup_end - f.index_date
                END
            )::integer AS duration_days,
            CASE
                WHEN o.first_outcome_date IS NOT NULL
                     AND (f.death_date IS NULL OR o.first_outcome_date <= f.death_date)
                    THEN 'MI_OR_STROKE'
                WHEN f.death_date IS NOT NULL
                     AND f.death_date > f.index_date
                     AND f.death_date <= f.followup_end
                    THEN 'DEATH'
                ELSE 'CENSOR'
            END AS event_type,
            (o.first_mi_date IS NOT NULL)::integer AS had_mi,
            (o.first_stroke_date IS NOT NULL)::integer AS had_stroke
        FROM t30_final f
        LEFT JOIN t30_outcomes o ON o.person_id = f.person_id
        WHERE f.followup_end > f.index_date
        """,
        label="build de-identified time-to-event rows",
    )
    record_stage(conn, stages, output_dir, "survival_rows", "t30_survival")
    return stages


def cumulative_incidence(survival_rows, max_day):
    output = []
    for treatment, group in survival_rows.groupby("treatment_class", sort=True):
        n_at_risk = int(len(group))
        survival = 1.0
        cif = 0.0
        counts = (
            group.groupby(["duration_days", "event_type"])
            .size()
            .unstack(fill_value=0)
        )
        for day in range(0, max_day + 1):
            if day == 0:
                d_outcome = d_death = d_censor = 0
            elif day in counts.index:
                row = counts.loc[day]
                d_outcome = int(row.get("MI_OR_STROKE", 0))
                d_death = int(row.get("DEATH", 0))
                d_censor = int(row.get("CENSOR", 0))
                if n_at_risk > 0:
                    previous_survival = survival
                    cif += previous_survival * d_outcome / n_at_risk
                    survival *= 1.0 - (d_outcome + d_death) / n_at_risk
                    n_at_risk -= d_outcome + d_death + d_censor
            else:
                d_outcome = d_death = d_censor = 0
            output.append(
                {
                    "treatment_class": treatment,
                    "day": day,
                    "patients_at_risk_after_day": n_at_risk,
                    "mi_or_stroke_events_on_day": d_outcome,
                    "competing_deaths_on_day": d_death,
                    "censored_on_day": d_censor,
                    "event_free_survival": survival,
                    "mi_or_stroke_cumulative_incidence": cif,
                }
            )
    return pd.DataFrame(output)


def cohort_summary(survival_rows):
    rows = []
    for treatment, group in survival_rows.groupby("treatment_class", sort=True):
        rows.append(
            {
                "treatment_class": treatment,
                "patients": int(len(group)),
                "mi_or_stroke_events": int((group["event_type"] == "MI_OR_STROKE").sum()),
                "competing_deaths": int((group["event_type"] == "DEATH").sum()),
                "censored": int((group["event_type"] == "CENSOR").sum()),
                "patients_with_mi": int(group["had_mi"].sum()),
                "patients_with_stroke": int(group["had_stroke"].sum()),
                "median_followup_days": float(group["duration_days"].median()),
                "max_followup_days": int(group["duration_days"].max()),
                "curve_computable": bool((group["event_type"] == "MI_OR_STROKE").any()),
            }
        )
    return pd.DataFrame(rows)


def curve_landmarks(curve):
    return curve.loc[curve["day"].isin(LANDMARK_DAYS)].reset_index(drop=True)


def plot_curve(curve, path):
    try:
        import matplotlib.pyplot as plt
    except (ImportError, ModuleNotFoundError):
        log("[WARN] matplotlib unavailable; CSV curves were still written")
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for treatment, group in curve.groupby("treatment_class", sort=True):
        ax.step(
            group["day"],
            group["mi_or_stroke_cumulative_incidence"],
            where="post",
            label=treatment,
        )
    ax.set_xlabel("Days after first-line treatment initiation")
    ax.set_ylabel("Cumulative incidence of acute MI or stroke")
    ax.set_xlim(0, int(curve["day"].max()))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    log(f"[WRITE] {path}")
    return path


def csv_text(frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().rstrip()


def main():
    args = parse_args()
    require_dependencies()
    prepare_output(args.output_dir)
    started = datetime.now(timezone.utc)

    endpoint_status = {
        "curve": "observed cumulative incidence after ACEi versus DHP-CCB initiation",
        "included_events": ["acute myocardial infarction", "stroke"],
        "competing_event": "all-cause death",
        "final_mace_status": "NOT_FINAL_UNTIL_CARDIOVASCULAR_DEATH_DEFINITION_IS_MATCHED",
        "uses_fermat": False,
        "uses_cox": False,
        "causal_effect_claim": False,
    }
    write_json(endpoint_status, args.output_dir / "endpoint_status.json")
    product_flags, condition_sets, mapping_paths = load_mapping(args.mapping_dir)
    mapping_summary = {
        "mapping_dir": str(args.mapping_dir),
        "observed_antihypertensive_products": int(len(product_flags)),
        "observed_acei_products": int(product_flags["is_acei"].sum()),
        "observed_dhp_ccb_products": int(product_flags["is_dhp_ccb"].sum()),
        "mapping_files": mapping_paths,
    }
    write_json(mapping_summary, args.output_dir / "mapping_summary.json")

    with connect(args) as conn:
        create_product_flags(conn, product_flags)
        build_cohort_tables(conn, args, condition_sets, args.output_dir)
        survival_rows = query_df(
            conn,
            """
            SELECT treatment_class, duration_days, event_type, had_mi, had_stroke
            FROM t30_survival
            ORDER BY treatment_class, duration_days, event_type
            """,
            label="load de-identified survival rows for curve calculation",
        )

    if survival_rows.empty:
        raise RuntimeError("No eligible ACEi or DHP-CCB survival rows were produced")
    survival_rows["duration_days"] = survival_rows["duration_days"].astype(int)
    survival_rows["had_mi"] = survival_rows["had_mi"].astype(int)
    survival_rows["had_stroke"] = survival_rows["had_stroke"].astype(int)

    summary = cohort_summary(survival_rows)
    curve = cumulative_incidence(survival_rows, int(args.max_curve_days))
    landmarks = curve_landmarks(curve)
    write_csv(summary, args.output_dir / "cohort_summary.csv")
    write_csv(curve, args.output_dir / "observed_mi_stroke_cumulative_incidence.csv")
    write_csv(landmarks, args.output_dir / "observed_mi_stroke_landmarks.csv")
    plot_curve(curve, args.output_dir / "observed_mi_stroke_cumulative_incidence.png")

    finished = datetime.now(timezone.utc)
    manifest = {
        "status": "complete",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "schema": args.schema,
        "db_end_date": args.db_end_date,
        "lookback_days": args.lookback_days,
        "max_curve_days": args.max_curve_days,
        "output_dir": str(args.output_dir),
        "patient_identifiers_written": False,
    }
    write_json(manifest, args.output_dir / "run_manifest.json")
    return_text = "\n".join(
        [
            "## ENDPOINT_STATUS",
            json.dumps(endpoint_status, ensure_ascii=False, indent=2),
            "## MAPPING_SUMMARY",
            json.dumps(mapping_summary, ensure_ascii=False, indent=2),
            "## STAGE_COUNTS",
            (args.output_dir / "stage_counts.csv").read_text().rstrip(),
            "## COHORT_SUMMARY",
            csv_text(summary),
            "## CURVE_LANDMARKS",
            csv_text(landmarks),
        ]
    ) + "\n"
    return_path = args.output_dir / "return_summary.txt"
    return_path.write_text(return_text)
    log(f"[WRITE] {return_path}")
    print(return_text, end="", flush=True)
    log("[COMPLETE] Observed MI-or-stroke cumulative-incidence feasibility finished")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
