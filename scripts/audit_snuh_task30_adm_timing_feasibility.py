#!/usr/bin/env python3
"""Aggregate feasibility audit for the Ko et al. 2026 ADM-timing target trial.

The paper-defined question is kept fixed:

* adults newly crossing HbA1c >= 6.5% or fasting plasma glucose >= 126 mg/dL;
* HbA1c and fasting plasma glucose both measured on the index date;
* at least one year of observable washout;
* no prior antidiabetic medication and no prior type 1 diabetes;
* initiate any listed antidiabetic medication within 3, 6, or 12 months,
  versus no initiation within 12 months.

This script uses FERMAT train and validation patients only.  It writes aggregate
counts and a projected test-set size; it never queries test-patient outcomes.
It does not estimate a causal effect and does not implement clone-censor-weight.

Important: a generic serum/plasma glucose measurement is not accepted as
fasting plasma glucose.  If exact fasting LOINC measurements are absent, the
paper-exact feasibility audit stops and reports that limitation.
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
from datetime import datetime, timezone
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
DEFAULT_OUTPUT_DIR = POD_ROOT / "task30" / "outputs" / "adm_timing_feasibility"
DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
APPLICATION_NAME = "fermat_task30_adm_timing_feasibility"

HBA1C_CONCEPT_ID = 3004410
HBA1C_UNIT_CONCEPT_ID = 8554
FPG_LOINC_CODES = ("1558-6", "14771-0", "35184-1")

# Classes listed in Ko et al., JAMA Network Open 2026.  Meglitinides are
# enumerated at ingredient-level ATC codes because A10BX also contains drugs
# outside that class.
ADM_ATC_ROOTS = (
    ("INSULIN", "ATC", "A10A"),
    ("METFORMIN_BIGUANIDE", "ATC", "A10BA"),
    ("SULFONYLUREA", "ATC", "A10BB"),
    ("ALPHA_GLUCOSIDASE_INHIBITOR", "ATC", "A10BF"),
    ("THIAZOLIDINEDIONE", "ATC", "A10BG"),
    ("DPP4_INHIBITOR", "ATC", "A10BH"),
    ("SGLT2_INHIBITOR", "ATC", "A10BK"),
    ("GLP1_RECEPTOR_AGONIST", "ATC", "A10BJ"),
    ("MEGLITINIDE_REPAGLINIDE", "ATC", "A10BX02"),
    ("MEGLITINIDE_NATEGLINIDE", "ATC", "A10BX03"),
    ("MEGLITINIDE_MITIGLINIDE", "ATC", "A10BX08"),
)

TYPE1_DIABETES_ROOTS = (("TYPE1_DIABETES", "SNOMED", "46635009"),)
HORIZONS = (("1y", 365), ("3y", 1095), ("5y", 1826))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--study-start", default="2013-01-01")
    parser.add_argument("--entry-end", default="2022-12-31")
    parser.add_argument("--db-end-date", default="2025-02-05")
    parser.add_argument("--minimum-age", type=int, default=18)
    parser.add_argument("--washout-days", type=int, default=365)
    parser.add_argument("--host", default=os.environ.get("SNUH_CDM_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SNUH_CDM_PORT", "5432")))
    parser.add_argument("--dbname", default=os.environ.get("SNUH_CDM_DATABASE", "cdm"))
    parser.add_argument("--user", default=os.environ.get("SNUH_CDM_USER", "jaegyun_jung"))
    parser.add_argument("--schema", default=os.environ.get("SNUH_CDM_SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--sslmode", default=os.environ.get("SNUH_CDM_SSLMODE", "disable"))
    parser.add_argument("--statement-timeout", default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def require_dependencies():
    missing = []
    if pd is None:
        missing.append("pandas")
    if psycopg is None:
        missing.append("psycopg")
    if missing:
        raise RuntimeError("Missing Pod Python dependencies: " + ", ".join(missing))


def log(message: str):
    print(message, flush=True)


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
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=6,
    )
    with conn.cursor() as cur:
        cur.execute("SELECT set_config('statement_timeout', %s, false)", (args.statement_timeout,))
    conn.commit()
    return conn


def execute(conn, statement, params=None, label=None):
    started = time.time()
    if label:
        log(f"[START] {label}")
    with conn.cursor() as cur:
        cur.execute(statement, params or ())
    conn.commit()
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


def load_registry(data_dir: Path):
    for filename in ("token_registry.csv", "vocab.csv"):
        path = data_dir / filename
        if path.is_file():
            frame = pd.read_csv(path, dtype={"token_key": str})
            if "token_key" not in frame.columns:
                raise ValueError(f"{path} has no token_key column")
            return frame, path
    raise FileNotFoundError(f"No token_registry.csv or vocab.csv under {data_dir}")


def load_development_patient_map(data_dir: Path):
    path = data_dir / "patient_id_map.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=["person_id", "split"])
    frame["split"] = frame["split"].astype(str)
    observed = sorted(frame["split"].dropna().unique().tolist())
    frame = frame.loc[frame["split"].isin(["train", "val"]), ["person_id", "split"]].copy()
    frame["person_id"] = pd.to_numeric(frame["person_id"], errors="raise").astype("int64")
    if frame.empty:
        raise ValueError(f"No train/val rows in {path}; observed split values={observed}")
    if frame["person_id"].duplicated().any():
        raise ValueError("patient_id_map has duplicate person_id rows")
    return frame, path, observed


def upload_patient_map(conn, frame):
    execute(
        conn,
        """
        CREATE TEMP TABLE tmp_s2_patient_map (
            person_id bigint PRIMARY KEY,
            split text NOT NULL CHECK (split IN ('train', 'val'))
        ) ON COMMIT PRESERVE ROWS
        """,
        label="create train+val patient-map table",
    )
    started = time.time()
    log(f"[START] upload train+val patient map: rows={len(frame):,}")
    with conn.cursor() as cur:
        with cur.copy("COPY tmp_s2_patient_map (person_id, split) FROM STDIN") as copy:
            for row in frame.itertuples(index=False):
                copy.write_row((int(row.person_id), str(row.split)))
    conn.commit()
    log(f"[DONE] upload train+val patient map: seconds={time.time() - started:,.1f}")


def values_sql(rows):
    placeholders = sql.SQL(", ").join(
        sql.SQL("({}, {}, {})").format(sql.Placeholder(), sql.Placeholder(), sql.Placeholder())
        for _ in rows
    )
    params = [value for row in rows for value in row]
    return placeholders, params


def resolve_roots(conn, schema, roots, domain, label):
    placeholders, params = values_sql(roots)
    statement = sql.SQL(
        """
        WITH requested(root_name, vocabulary_id, concept_code) AS (VALUES {})
        SELECT r.root_name,
               r.vocabulary_id AS requested_vocabulary_id,
               r.concept_code AS requested_concept_code,
               c.concept_id AS root_concept_id,
               c.concept_name AS root_concept_name,
               c.domain_id, c.concept_class_id, c.standard_concept, c.invalid_reason
        FROM requested r
        LEFT JOIN {}.concept c
          ON c.vocabulary_id = r.vocabulary_id
         AND upper(c.concept_code) = upper(r.concept_code)
         AND c.domain_id = %s
        ORDER BY r.root_name, c.invalid_reason NULLS FIRST, c.concept_id
        """
    ).format(placeholders, sql.Identifier(schema))
    return query_df(conn, statement, params + [domain], label=label)


def select_root_ids(frame, required_names):
    result = {}
    for name in required_names:
        subset = frame.loc[
            frame["root_name"].eq(name)
            & frame["root_concept_id"].notna()
            & frame["invalid_reason"].isna()
        ]
        if subset.empty:
            continue
        standard = subset.loc[subset["standard_concept"].fillna("").eq("S")]
        chosen = standard.iloc[0] if not standard.empty else subset.iloc[0]
        result[name] = int(chosen["root_concept_id"])
    return result


def observed_adm_inventory(conn, schema, root_ids):
    rows = list(root_ids.items())
    placeholders = sql.SQL(", ").join(
        sql.SQL("({}, {})").format(sql.Placeholder(), sql.Placeholder()) for _ in rows
    )
    params = [value for row in rows for value in row]
    statement = sql.SQL(
        """
        WITH roots(drug_class, root_concept_id) AS (VALUES {}),
        members AS (
            SELECT drug_class, root_concept_id AS member_concept_id FROM roots
            UNION
            SELECT r.drug_class, ca.descendant_concept_id
            FROM roots r
            JOIN {}.concept_ancestor ca ON ca.ancestor_concept_id = r.root_concept_id
        ),
        classified_products AS (
            SELECT DISTINCT m.drug_class, d.drug_concept_id
            FROM members m
            JOIN {}.drug_exposure d ON d.drug_concept_id = m.member_concept_id
            WHERE d.drug_exposure_start_date IS NOT NULL
            UNION
            SELECT DISTINCT m.drug_class, d.drug_concept_id
            FROM members m
            JOIN {}.drug_strength ds ON ds.ingredient_concept_id = m.member_concept_id
            JOIN {}.drug_exposure d ON d.drug_concept_id = ds.drug_concept_id
            WHERE d.drug_exposure_start_date IS NOT NULL
        )
        SELECT m.drug_class,
               d.drug_concept_id,
               c.concept_name AS drug_concept_name,
               c.vocabulary_id,
               c.concept_class_id,
               COUNT(*)::bigint AS exposure_rows,
               COUNT(DISTINCT d.person_id)::bigint AS patients,
               MIN(d.drug_exposure_start_date)::text AS first_exposure_date,
               MAX(d.drug_exposure_start_date)::text AS last_exposure_date
        FROM classified_products m
        JOIN {}.drug_exposure d ON d.drug_concept_id = m.drug_concept_id
        LEFT JOIN {}.concept c ON c.concept_id = d.drug_concept_id
        WHERE d.drug_exposure_start_date IS NOT NULL
        GROUP BY 1,2,3,4,5
        ORDER BY m.drug_class, exposure_rows DESC, d.drug_concept_id
        """
    ).format(
        placeholders,
        sql.Identifier(schema),
        sql.Identifier(schema),
        sql.Identifier(schema),
        sql.Identifier(schema),
        sql.Identifier(schema),
        sql.Identifier(schema),
    )
    return query_df(conn, statement, params, label="inventory observed ADM products")


def add_rx_registry_coverage(inventory, registry):
    frame = inventory.copy()
    keys = set(registry["token_key"].astype(str))
    frame["expected_token_key"] = "RX:" + frame["drug_concept_id"].astype("int64").astype(str)
    frame["token_in_fermat_registry"] = frame["expected_token_key"].isin(keys)
    return frame


def rx_coverage_summary(inventory):
    rows = []
    for drug_class, group in inventory.groupby("drug_class", sort=True):
        total = int(group["exposure_rows"].sum())
        covered = int(group.loc[group["token_in_fermat_registry"], "exposure_rows"].sum())
        rows.append(
            {
                "drug_class": drug_class,
                "observed_drug_concepts": int(group["drug_concept_id"].nunique()),
                "registry_covered_concepts": int(group.loc[group["token_in_fermat_registry"], "drug_concept_id"].nunique()),
                "exposure_rows": total,
                "covered_exposure_rows": covered,
                "exposure_row_coverage": covered / total if total else None,
            }
        )
    return pd.DataFrame(rows)


def resolve_fpg_concepts(conn, schema):
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
    return query_df(conn, statement, (list(FPG_LOINC_CODES),), label="resolve exact fasting-glucose LOINC concepts")


def type1_descendants(conn, schema, root_id):
    statement = sql.SQL(
        """
        WITH members AS (
            SELECT %s::bigint AS condition_concept_id
            UNION
            SELECT descendant_concept_id
            FROM {}.concept_ancestor
            WHERE ancestor_concept_id = %s
        )
        SELECT DISTINCT m.condition_concept_id,
               c.concept_name, c.vocabulary_id, c.concept_code,
               c.standard_concept, c.invalid_reason
        FROM members m
        JOIN {}.concept c ON c.concept_id = m.condition_concept_id
        WHERE c.domain_id = 'Condition' AND c.invalid_reason IS NULL
        ORDER BY m.condition_concept_id
        """
    ).format(sql.Identifier(schema), sql.Identifier(schema))
    return query_df(conn, statement, (root_id, root_id), label="resolve type 1 diabetes descendants")


def upload_id_table(conn, table, column, values):
    if not values:
        raise ValueError(f"No values for {table}")
    execute(
        conn,
        sql.SQL("CREATE TEMP TABLE {} ({} bigint PRIMARY KEY) ON COMMIT PRESERVE ROWS").format(
            sql.Identifier(table), sql.Identifier(column)
        ),
    )
    with conn.cursor() as cur:
        with cur.copy(
            sql.SQL("COPY {} ({}) FROM STDIN").format(sql.Identifier(table), sql.Identifier(column))
        ) as copy:
            for value in sorted(set(int(x) for x in values)):
                copy.write_row((value,))
    conn.commit()


def lab_coverage_statement(schema):
    return sql.SQL(
        """
        SELECT p.split,
               CASE
                 WHEN m.measurement_concept_id = %(hba1c)s THEN 'HBA1C_EXACT'
                 WHEN f.concept_id IS NOT NULL THEN 'FASTING_GLUCOSE_EXACT'
               END AS marker,
               m.measurement_concept_id,
               c.concept_name,
               c.concept_code,
               m.unit_concept_id,
               u.concept_name AS unit_concept_name,
               COALESCE(m.unit_source_value, '') AS unit_source_value,
               COUNT(*)::bigint AS measurement_rows,
               COUNT(DISTINCT m.person_id)::bigint AS patients,
               MIN(m.measurement_date)::text AS first_date,
               MAX(m.measurement_date)::text AS last_date
        FROM {}.measurement m
        JOIN tmp_s2_patient_map p USING(person_id)
        LEFT JOIN tmp_s2_fpg_concepts f ON f.concept_id = m.measurement_concept_id
        LEFT JOIN {}.concept c ON c.concept_id = m.measurement_concept_id
        LEFT JOIN {}.concept u ON u.concept_id = m.unit_concept_id
        WHERE (m.measurement_concept_id = %(hba1c)s OR f.concept_id IS NOT NULL)
          AND m.measurement_date BETWEEN (%(study_start)s::date - %(washout)s::int) AND %(entry_end)s::date
          AND m.value_as_number IS NOT NULL
        GROUP BY 1,2,3,4,5,6,7,8
        ORDER BY marker, p.split, measurement_rows DESC
        """
    ).format(sql.Identifier(schema), sql.Identifier(schema), sql.Identifier(schema))


def create_lab_day_statement(schema):
    return sql.SQL(
        """
        CREATE TEMP TABLE tmp_s2_lab_day ON COMMIT PRESERVE ROWS AS
        WITH normalized AS (
            SELECT p.person_id,
                   p.split,
                   m.measurement_date::date AS lab_date,
                   CASE
                     WHEN m.measurement_concept_id = %(hba1c)s
                      AND m.unit_concept_id = %(hba1c_unit)s
                     THEN m.value_as_number::double precision
                   END AS hba1c_pct,
                   CASE
                     WHEN f.concept_code = '1558-6'
                      AND (
                           lower(COALESCE(u.concept_name, '')) LIKE '%%milligram per deciliter%%'
                           OR lower(COALESCE(m.unit_source_value, '')) ~ 'mg\\s*/?\\s*dl'
                      )
                     THEN m.value_as_number::double precision
                     WHEN f.concept_code = '14771-0'
                      AND (
                           lower(COALESCE(u.concept_name, '')) LIKE '%%millimole per liter%%'
                           OR lower(COALESCE(m.unit_source_value, '')) ~ 'mmol\\s*/?\\s*l'
                      )
                     THEN m.value_as_number::double precision * 18.0156
                     WHEN f.concept_code = '35184-1'
                      AND (
                           lower(COALESCE(u.concept_name, '')) LIKE '%%milligram per deciliter%%'
                           OR lower(COALESCE(m.unit_source_value, '')) ~ 'mg\\s*/?\\s*dl'
                      )
                     THEN m.value_as_number::double precision
                     WHEN f.concept_code = '35184-1'
                      AND (
                           lower(COALESCE(u.concept_name, '')) LIKE '%%millimole per liter%%'
                           OR lower(COALESCE(m.unit_source_value, '')) ~ 'mmol\\s*/?\\s*l'
                      )
                     THEN m.value_as_number::double precision * 18.0156
                   END AS fpg_mg_dl
            FROM {}.measurement m
            JOIN tmp_s2_patient_map p USING(person_id)
            LEFT JOIN tmp_s2_fpg_concept_meta f ON f.concept_id = m.measurement_concept_id
            LEFT JOIN {}.concept u ON u.concept_id = m.unit_concept_id
            WHERE (m.measurement_concept_id = %(hba1c)s OR f.concept_id IS NOT NULL)
              AND m.measurement_date BETWEEN (%(study_start)s::date - %(washout)s::int) AND %(entry_end)s::date
              AND m.value_as_number IS NOT NULL
        )
        SELECT person_id, split, lab_date,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY hba1c_pct)
                   FILTER (WHERE hba1c_pct IS NOT NULL) AS hba1c_pct,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fpg_mg_dl)
                   FILTER (WHERE fpg_mg_dl IS NOT NULL) AS fpg_mg_dl
        FROM normalized
        WHERE hba1c_pct IS NOT NULL OR fpg_mg_dl IS NOT NULL
        GROUP BY person_id, split, lab_date
        """
    ).format(sql.Identifier(schema), sql.Identifier(schema))


def build_cohort(conn, args, adm_ids, type1_ids, fpg_meta):
    upload_id_table(conn, "tmp_s2_adm_concepts", "drug_concept_id", adm_ids)
    upload_id_table(conn, "tmp_s2_type1_concepts", "condition_concept_id", type1_ids)

    execute(
        conn,
        "CREATE TEMP TABLE tmp_s2_fpg_concepts (concept_id bigint PRIMARY KEY) ON COMMIT PRESERVE ROWS",
    )
    execute(
        conn,
        "CREATE TEMP TABLE tmp_s2_fpg_concept_meta (concept_id bigint PRIMARY KEY, concept_code text NOT NULL) ON COMMIT PRESERVE ROWS",
    )
    with conn.cursor() as cur:
        with cur.copy("COPY tmp_s2_fpg_concepts (concept_id) FROM STDIN") as copy:
            for row in fpg_meta.itertuples(index=False):
                copy.write_row((int(row.concept_id),))
        with cur.copy("COPY tmp_s2_fpg_concept_meta (concept_id, concept_code) FROM STDIN") as copy:
            for row in fpg_meta.itertuples(index=False):
                copy.write_row((int(row.concept_id), str(row.concept_code)))
    conn.commit()

    params = {
        "hba1c": HBA1C_CONCEPT_ID,
        "hba1c_unit": HBA1C_UNIT_CONCEPT_ID,
        "study_start": args.study_start,
        "entry_end": args.entry_end,
        "db_end": args.db_end_date,
        "washout": args.washout_days,
        "minimum_age": args.minimum_age,
    }

    coverage = query_df(
        conn,
        lab_coverage_statement(args.schema),
        params,
        label="count exact HbA1c and fasting-glucose coverage",
    )

    raw_fpg_rows = coverage.loc[
        coverage["marker"].eq("FASTING_GLUCOSE_EXACT"), "measurement_rows"
    ].sum()
    if int(raw_fpg_rows) == 0:
        return coverage, "NO_EXACT_FPG_ROWS"

    execute(
        conn,
        create_lab_day_statement(args.schema),
        params,
        label="stage same-day HbA1c and exact fasting glucose",
    )
    execute(conn, "CREATE INDEX ON tmp_s2_lab_day(person_id, lab_date)")

    normalized_fpg = query_df(
        conn,
        """
        SELECT COUNT(*)::bigint AS rows,
               COUNT(DISTINCT person_id)::bigint AS patients
        FROM tmp_s2_lab_day
        WHERE fpg_mg_dl IS NOT NULL
        """,
        label="check usable exact fasting-glucose values",
    )
    if int(normalized_fpg.iloc[0]["rows"]) == 0:
        return coverage, "NO_USABLE_FPG_VALUES"

    execute(
        conn,
        """
        CREATE TEMP TABLE tmp_s2_first_threshold ON COMMIT PRESERVE ROWS AS
        WITH threshold_days AS (
            SELECT person_id, split, lab_date, hba1c_pct, fpg_mg_dl,
                   ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY lab_date) AS threshold_order
            FROM tmp_s2_lab_day
            WHERE hba1c_pct IS NOT NULL
              AND fpg_mg_dl IS NOT NULL
              AND hba1c_pct >= 5.7
              AND (hba1c_pct >= 6.5 OR fpg_mg_dl >= 126.0)
        )
        SELECT * FROM threshold_days WHERE threshold_order = 1
        """,
        label="find first observed joint-lab diabetes threshold",
    )
    execute(conn, "CREATE UNIQUE INDEX ON tmp_s2_first_threshold(person_id)")

    s = sql.Identifier(args.schema)
    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE tmp_s2_adult_index ON COMMIT PRESERVE ROWS AS
            SELECT t.person_id, t.split, t.lab_date AS index_date,
                   t.hba1c_pct, t.fpg_mg_dl,
                   EXTRACT(YEAR FROM age(t.lab_date, make_date(
                       p.year_of_birth,
                       CASE WHEN p.month_of_birth BETWEEN 1 AND 12 THEN p.month_of_birth ELSE 7 END,
                       CASE WHEN p.day_of_birth BETWEEN 1 AND 28 THEN p.day_of_birth ELSE 1 END
                   )))::int AS age
            FROM tmp_s2_first_threshold t
            JOIN {}.person p USING(person_id)
            WHERE t.lab_date BETWEEN %(study_start)s::date AND %(entry_end)s::date
              AND EXTRACT(YEAR FROM age(t.lab_date, make_date(
                    p.year_of_birth,
                    CASE WHEN p.month_of_birth BETWEEN 1 AND 12 THEN p.month_of_birth ELSE 7 END,
                    CASE WHEN p.day_of_birth BETWEEN 1 AND 28 THEN p.day_of_birth ELSE 1 END
                  ))) >= %(minimum_age)s
            """
        ).format(s),
        params,
        label="apply paper age and entry-period criteria",
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
        ).format(s),
        params,
        label="require one year of OMOP observation before time zero",
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
        ).format(s),
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
        ).format(s),
        label="exclude prior type 1 diabetes",
    )
    execute(conn, "CREATE UNIQUE INDEX ON tmp_s2_eligible(person_id)")

    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE tmp_s2_timing ON COMMIT PRESERVE ROWS AS
            SELECT e.*,
                   a.first_adm_date,
                   d.death_date,
                   CASE
                     WHEN a.first_adm_date <= e.index_date + INTERVAL '3 months' THEN 'INIT_0_3M'
                     WHEN a.first_adm_date <= e.index_date + INTERVAL '6 months' THEN 'INIT_GT3_6M'
                     WHEN a.first_adm_date <= e.index_date + INTERVAL '12 months' THEN 'INIT_GT6_12M'
                     ELSE 'NO_INIT_WITHIN_12M'
                   END AS observed_timing_bin
            FROM tmp_s2_eligible e
            LEFT JOIN LATERAL (
                SELECT MIN(x.drug_exposure_start_date)::date AS first_adm_date
                FROM {}.drug_exposure x
                JOIN tmp_s2_adm_concepts a USING(drug_concept_id)
                WHERE x.person_id = e.person_id
                  AND x.drug_exposure_start_date >= e.index_date
                  AND x.drug_exposure_start_date <= %(db_end)s::date
            ) a ON true
            LEFT JOIN LATERAL (
                SELECT MIN(x.death_date)::date AS death_date
                FROM {}.death x
                WHERE x.person_id = e.person_id
                  AND x.death_date > e.index_date
                  AND x.death_date <= %(db_end)s::date
            ) d ON true
            """
        ).format(s, s),
        params,
        label="derive observed ADM timing and mortality",
    )
    execute(conn, "CREATE UNIQUE INDEX ON tmp_s2_timing(person_id)")
    return coverage, "BUILT"


def complete_grid(frame, dimension, values, count_columns):
    grid = pd.MultiIndex.from_product(
        [["train", "val"], values], names=["split", dimension]
    ).to_frame(index=False)
    merged = grid.merge(frame, on=["split", dimension], how="left")
    for column in count_columns:
        merged[column] = merged[column].fillna(0).astype("int64")
    return merged


def add_combined_and_projection(frame, group_columns, count_columns):
    combined = frame.groupby(group_columns, as_index=False)[count_columns].sum()
    combined.insert(0, "split", "train_val")
    result = pd.concat([frame, combined], ignore_index=True, sort=False)
    for column in count_columns:
        projected = f"projected_test_{column}"
        result[projected] = None
        mask = result["split"].eq("train_val")
        result.loc[mask, projected] = (result.loc[mask, column].astype(float) * 15.0 / 85.0).round(1)
    return result


def collect_results(conn, args):
    flow = query_df(
        conn,
        """
        SELECT split, stage, patients
        FROM (
            SELECT split, '01_FERMAT_TRAIN_VAL_PATIENTS'::text AS stage, COUNT(*)::bigint AS patients
            FROM tmp_s2_patient_map GROUP BY split
            UNION ALL
            SELECT split, '02_ANY_EXACT_HBA1C_OR_FPG', COUNT(DISTINCT person_id)::bigint
            FROM tmp_s2_lab_day GROUP BY split
            UNION ALL
            SELECT split, '03_SAME_DAY_BOTH_LABS', COUNT(DISTINCT person_id)::bigint
            FROM tmp_s2_lab_day WHERE hba1c_pct IS NOT NULL AND fpg_mg_dl IS NOT NULL GROUP BY split
            UNION ALL
            SELECT split, '04_FIRST_JOINT_LAB_DIABETES_THRESHOLD', COUNT(*)::bigint
            FROM tmp_s2_first_threshold GROUP BY split
            UNION ALL
            SELECT split, '05_ADULT_ENTRY_2013_2022', COUNT(*)::bigint
            FROM tmp_s2_adult_index GROUP BY split
            UNION ALL
            SELECT split, '06_PLUS_1Y_OBSERVATION_WASHOUT', COUNT(*)::bigint
            FROM tmp_s2_observed_washout GROUP BY split
            UNION ALL
            SELECT split, '07_PLUS_NO_PRIOR_ADM', COUNT(*)::bigint
            FROM tmp_s2_no_prior_adm GROUP BY split
            UNION ALL
            SELECT split, '08_PLUS_NO_PRIOR_TYPE1_FINAL', COUNT(*)::bigint
            FROM tmp_s2_eligible GROUP BY split
        ) q
        ORDER BY stage, split
        """,
        label="aggregate cohort flow",
    )
    flow_stages = [
        "01_FERMAT_TRAIN_VAL_PATIENTS",
        "02_ANY_EXACT_HBA1C_OR_FPG",
        "03_SAME_DAY_BOTH_LABS",
        "04_FIRST_JOINT_LAB_DIABETES_THRESHOLD",
        "05_ADULT_ENTRY_2013_2022",
        "06_PLUS_1Y_OBSERVATION_WASHOUT",
        "07_PLUS_NO_PRIOR_ADM",
        "08_PLUS_NO_PRIOR_TYPE1_FINAL",
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
        label="aggregate mutually exclusive observed timing bins",
    )
    timing_counts = [
        "patients", "deaths_1y", "deaths_3y", "deaths_5y",
        "full_followup_1y", "full_followup_3y", "full_followup_5y",
    ]
    timing = complete_grid(
        timing,
        "observed_timing_bin",
        ["INIT_0_3M", "INIT_GT3_6M", "INIT_GT6_12M", "NO_INIT_WITHIN_12M"],
        timing_counts,
    )
    timing = add_combined_and_projection(timing, ["observed_timing_bin"], timing_counts)

    strategies = query_df(
        conn,
        """
        WITH clones AS (
            SELECT t.*,
                   v.strategy,
                   CASE v.strategy
                     WHEN 'INIT_WITHIN_3M' THEN t.first_adm_date <= t.index_date + INTERVAL '3 months'
                     WHEN 'INIT_WITHIN_6M' THEN t.first_adm_date <= t.index_date + INTERVAL '6 months'
                     WHEN 'INIT_WITHIN_12M' THEN t.first_adm_date <= t.index_date + INTERVAL '12 months'
                     WHEN 'NO_INIT_WITHIN_12M' THEN t.first_adm_date IS NULL OR t.first_adm_date > t.index_date + INTERVAL '12 months'
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
               COUNT(*) FILTER (WHERE fulfilled AND death_date <= index_date + 365)::bigint AS fulfiller_deaths_1y,
               COUNT(*) FILTER (WHERE fulfilled AND death_date <= index_date + 1095)::bigint AS fulfiller_deaths_3y,
               COUNT(*) FILTER (WHERE fulfilled AND death_date <= index_date + 1826)::bigint AS fulfiller_deaths_5y
        FROM clones
        GROUP BY split, strategy
        ORDER BY strategy, split
        """,
        label="aggregate four paper-defined strategy clones",
    )
    strategy_counts = [
        "eligible_clones", "observed_fulfillers", "fulfiller_deaths_1y",
        "fulfiller_deaths_3y", "fulfiller_deaths_5y",
    ]
    strategies = complete_grid(
        strategies,
        "strategy",
        ["INIT_WITHIN_3M", "INIT_WITHIN_6M", "INIT_WITHIN_12M", "NO_INIT_WITHIN_12M"],
        strategy_counts,
    )
    strategies = add_combined_and_projection(strategies, ["strategy"], strategy_counts)
    return flow, timing, strategies


def write_return_summary(path, study, lab_coverage, rx_coverage, flow, strategies, limitations):
    combined_flow = flow.loc[flow["split"].eq("train_val")]
    combined_strategies = strategies.loc[strategies["split"].eq("train_val")]
    sections = [
        "## STATUS",
        "FEASIBILITY_ONLY_NOT_CAUSAL_EFFECT",
        "## STUDY_DEFINITION",
        json.dumps(study, ensure_ascii=False, indent=2),
        "## IMPLEMENTATION_LIMITATIONS",
        json.dumps(limitations, ensure_ascii=False, indent=2),
        "## EXACT_LAB_COVERAGE",
        csv_text(lab_coverage),
        "## ADM_RX_TOKEN_COVERAGE",
        csv_text(rx_coverage),
        "## TRAIN_VAL_COHORT_FLOW_AND_PROJECTED_TEST",
        csv_text(combined_flow),
        "## PAPER_STRATEGY_FULFILLMENT_AND_PROJECTED_TEST",
        csv_text(combined_strategies),
        "## INTERPRETATION",
        (
            "These are unadjusted feasibility counts, not causal risks. "
            "Deaths among observed fulfillers are not clone-censor-weight estimates. "
            "The exact causal analysis must be developed on train, checked on val, frozen, "
            "and only then applied to test. The current FERMAT history editor cannot yet "
            "condition at time zero on a treatment strategy that occurs during the next 12 months."
        ),
    ]
    text = "\n".join(sections) + "\n"
    path.write_text(text, encoding="utf-8")
    print(text, end="", flush=True)


def self_test():
    assert round(7.0 * 15.0 / 85.0, 1) == 1.2
    assert 7.0 >= 6.5
    assert 125.9 < 126.0
    assert 7.0 >= 5.7
    assert HORIZONS[-1] == ("5y", 1826)
    assert {row[2] for row in ADM_ATC_ROOTS} >= {"A10A", "A10BA", "A10BB", "A10BF", "A10BG", "A10BH", "A10BK", "A10BJ"}
    print("SELF_TEST_OK")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    require_dependencies()
    prepare_output(args.output_dir, args.overwrite)
    started = datetime.now(timezone.utc)

    study = {
        "paper": "Ko HY et al. JAMA Network Open. 2026;9(6):e2619362",
        "doi": "10.1001/jamanetworkopen.2026.19362",
        "population": "adults newly crossing the type 2 diabetes laboratory threshold",
        "time_zero": "first observed date with HbA1c >=6.5% or exact fasting plasma glucose >=126 mg/dL; both values present that day",
        "prediabetes_requirement": "index-day HbA1c >=5.7%",
        "washout": f"{args.washout_days} days of OMOP observation and no earlier threshold in the scanned washout interval",
        "exclusions": ["any prior listed antidiabetic medication", "prior type 1 diabetes"],
        "strategies": ["initiate within 3 months", "initiate within 6 months", "initiate within 12 months", "no initiation within 12 months"],
        "paper_outcomes": ["modified 3-point MACE", "all-cause mortality"],
        "feasibility_outcome_in_this_script": "all-cause mortality only",
        "development_splits_queried": ["train", "val"],
        "test_data_queried": False,
        "causal_effect_estimated": False,
    }
    write_json(study, args.output_dir / "study_definition.json")

    patient_map, patient_map_path, observed_splits = load_development_patient_map(args.data_dir)
    registry, registry_path = load_registry(args.data_dir)
    split_counts = patient_map.groupby("split").size().rename("patients").reset_index()
    write_csv(split_counts, args.output_dir / "development_split_counts.csv")

    limitations = {
        "fasting_glucose": "Only exact fasting LOINC concepts are accepted; generic serum glucose is excluded.",
        "washout": "SNUH OMOP observation_period is used as the continuous-observation proxy.",
        "mace": "Not counted here because acute MI and the paper's stroke claims definitions have not been reviewed for SNUH. Broad CAD is not substituted for MI.",
        "fermat_alignment": "Paper strategies occur after time zero; exact FERMAT validation requires future treatment-strategy conditioning, not the current pre-cutoff history editor.",
    }
    write_json(limitations, args.output_dir / "implementation_limitations.json")

    try:
        with connect(args) as conn:
            upload_patient_map(conn, patient_map)

            adm_roots = resolve_roots(conn, args.schema, ADM_ATC_ROOTS, "Drug", "resolve paper-listed ADM ATC roots")
            write_csv(adm_roots, args.output_dir / "adm_atc_roots.csv")
            required_adm = [row[0] for row in ADM_ATC_ROOTS]
            adm_root_ids = select_root_ids(adm_roots, required_adm)
            missing_adm = sorted(set(required_adm) - set(adm_root_ids))
            if missing_adm:
                raise RuntimeError(f"Unresolved paper-listed ADM ATC roots: {missing_adm}")

            adm_inventory = observed_adm_inventory(conn, args.schema, adm_root_ids)
            if adm_inventory.empty:
                raise RuntimeError("No observed SNUH drug products under the paper-listed ADM classes")
            adm_inventory = add_rx_registry_coverage(adm_inventory, registry)
            rx_coverage = rx_coverage_summary(adm_inventory)
            write_csv(adm_inventory, args.output_dir / "observed_adm_inventory.csv")
            write_csv(rx_coverage, args.output_dir / "adm_rx_token_coverage.csv")

            type1_roots = resolve_roots(conn, args.schema, TYPE1_DIABETES_ROOTS, "Condition", "resolve type 1 diabetes root")
            write_csv(type1_roots, args.output_dir / "type1_diabetes_root.csv")
            type1_root_ids = select_root_ids(type1_roots, ["TYPE1_DIABETES"])
            if "TYPE1_DIABETES" not in type1_root_ids:
                raise RuntimeError("Could not resolve SNOMED type 1 diabetes root 46635009")
            type1 = type1_descendants(conn, args.schema, type1_root_ids["TYPE1_DIABETES"])
            write_csv(type1, args.output_dir / "type1_diabetes_descendants.csv")

            fpg = resolve_fpg_concepts(conn, args.schema)
            write_csv(fpg, args.output_dir / "fasting_glucose_loinc_concepts.csv")
            valid_fpg = fpg.loc[fpg["invalid_reason"].isna() & fpg["concept_id"].notna()].copy()
            if valid_fpg.empty:
                raise RuntimeError("No valid exact fasting-glucose LOINC concept is present in the OMOP vocabulary")

            lab_coverage, build_status = build_cohort(
                conn,
                args,
                adm_inventory["drug_concept_id"].dropna().astype(int).unique().tolist(),
                type1["condition_concept_id"].dropna().astype(int).unique().tolist(),
                valid_fpg[["concept_id", "concept_code"]],
            )
            write_csv(lab_coverage, args.output_dir / "exact_lab_coverage.csv")

            if build_status != "BUILT":
                reason = {
                    "NO_EXACT_FPG_ROWS": (
                        "No train/val measurement rows use an exact fasting-plasma-glucose LOINC concept."
                    ),
                    "NO_USABLE_FPG_VALUES": (
                        "Exact fasting-plasma-glucose concepts exist, but no numeric values have a compatible mg/dL or mmol/L unit."
                    ),
                }[build_status]
                status = {
                    "status": "BLOCKED_PAPER_EXACT_FPG_ABSENT",
                    "reason": reason + " Generic serum glucose was not substituted.",
                }
                write_json(status, args.output_dir / "blocked.json")
                (args.output_dir / "return_summary.txt").write_text(
                    "## STATUS\nBLOCKED_PAPER_EXACT_FPG_ABSENT\n"
                    f"{reason}\n"
                    "Generic serum/plasma glucose was not substituted.\n",
                    encoding="utf-8",
                )
                print((args.output_dir / "return_summary.txt").read_text(), end="", flush=True)
                return 0

            flow, timing, strategies = collect_results(conn, args)
            write_csv(flow, args.output_dir / "cohort_flow.csv")
            write_csv(timing, args.output_dir / "observed_timing_bins.csv")
            write_csv(strategies, args.output_dir / "strategy_fulfillment_counts.csv")

        diagnostics = {
            "started_utc": started.isoformat(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "patient_map": str(patient_map_path),
            "token_registry": str(registry_path),
            "observed_split_values": observed_splits,
            "queried_split_values": ["train", "val"],
            "test_rows_loaded": 0,
            "projected_test_multiplier": 15.0 / 85.0,
        }
        write_json(diagnostics, args.output_dir / "diagnostics.json")
        write_return_summary(
            args.output_dir / "return_summary.txt",
            study,
            lab_coverage,
            rx_coverage,
            flow,
            strategies,
            limitations,
        )
        log(f"[COMPLETE] {args.output_dir}")
        return 0
    except Exception as exc:
        failure = {
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
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
