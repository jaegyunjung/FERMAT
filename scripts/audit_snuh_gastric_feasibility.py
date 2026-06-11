"""Run aggregate gastric-cancer feasibility checks on the SNUH OMOP CDM."""

import getpass
import os
import re
import sys
from datetime import date

import psycopg
from psycopg import sql


HOST = os.environ.get(
    "SNUH_CDM_HOST",
    "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com",
)
PORT = int(os.environ.get("SNUH_CDM_PORT", "5432"))
DATABASE = os.environ.get("SNUH_CDM_DATABASE", "cdm")
USER = os.environ.get("SNUH_CDM_USER", "jaegyun_jung")
SCHEMA = os.environ.get("SNUH_CDM_SCHEMA", "cdm2024_official")
SSL_MODE = os.environ.get("SNUH_CDM_SSLMODE", "disable")
STATEMENT_TIMEOUT = os.environ.get("SNUH_CDM_TIMEOUT", "30min")

CARD_END_DATE = date(2023, 12, 31)
VALID_DATE_MIN = date(1900, 1, 1)

PRIMARY_CODES = (
    "C160", "C161", "C162", "C163", "C164", "C165", "C166", "C168",
    "C169", "C1690", "C1691", "C1699",
)
SENSITIVITY_CODES = ("C788", "C7880", "D371")


def normalize_code(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def print_rows(columns, rows):
    if not rows:
        print("(none)")
        return
    widths = [
        max(len(str(column)), *(len(str(row[i])) for row in rows))
        for i, column in enumerate(columns)
    ]
    print("  ".join(str(column).ljust(widths[i]) for i, column in enumerate(columns)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))


def fetch(cursor, query, params=None):
    cursor.execute(query, params or ())
    columns = [column.name for column in cursor.description]
    return columns, cursor.fetchall()


def find_target_concepts(cursor, codes):
    columns, rows = fetch(
        cursor,
        sql.SQL(
            """
            SELECT
                concept_id,
                vocabulary_id,
                concept_code,
                concept_name,
                standard_concept,
                invalid_reason
            FROM {}.concept
            WHERE regexp_replace(
                upper(concept_code), '[^A-Z0-9]', '', 'g'
            ) = ANY(%s)
            ORDER BY concept_code, vocabulary_id, concept_id
            """
        ).format(sql.Identifier(SCHEMA)),
        (list(codes),),
    )
    source_ids = [row[0] for row in rows]

    mapped_columns = (
        "source_concept_id",
        "source_vocabulary",
        "source_code",
        "standard_concept_id",
        "standard_vocabulary",
        "standard_code",
        "standard_name",
    )
    mapped_rows = []
    standard_ids = []
    if source_ids:
        mapped_columns, mapped_rows = fetch(
            cursor,
            sql.SQL(
                """
                SELECT DISTINCT
                    src.concept_id AS source_concept_id,
                    src.vocabulary_id AS source_vocabulary,
                    src.concept_code AS source_code,
                    dst.concept_id AS standard_concept_id,
                    dst.vocabulary_id AS standard_vocabulary,
                    dst.concept_code AS standard_code,
                    dst.concept_name AS standard_name
                FROM {}.concept AS src
                JOIN {}.concept_relationship AS rel
                  ON rel.concept_id_1 = src.concept_id
                 AND rel.relationship_id = 'Maps to'
                 AND rel.invalid_reason IS NULL
                JOIN {}.concept AS dst
                  ON dst.concept_id = rel.concept_id_2
                 AND dst.invalid_reason IS NULL
                WHERE src.concept_id = ANY(%s)
                ORDER BY src.concept_code, src.vocabulary_id, dst.concept_id
                """
            ).format(
                sql.Identifier(SCHEMA),
                sql.Identifier(SCHEMA),
                sql.Identifier(SCHEMA),
            ),
            (source_ids,),
        )
        standard_ids = sorted({row[3] for row in mapped_rows})

    return (columns, rows), (mapped_columns, mapped_rows), source_ids, standard_ids


def run_definition(cursor, label, codes, source_ids, standard_ids, db_end_date):
    print(f"\n=== {label}: condition counts by independent path ===")
    columns, rows = fetch(
        cursor,
        sql.SQL(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE condition_source_concept_id = ANY(%s)
                ) AS source_concept_events,
                COUNT(DISTINCT person_id) FILTER (
                    WHERE condition_source_concept_id = ANY(%s)
                ) AS source_concept_patients,
                COUNT(*) FILTER (
                    WHERE condition_concept_id = ANY(%s)
                ) AS standard_concept_events,
                COUNT(DISTINCT person_id) FILTER (
                    WHERE condition_concept_id = ANY(%s)
                ) AS standard_concept_patients
            FROM {}.condition_occurrence
            """
        ).format(sql.Identifier(SCHEMA)),
        (
            source_ids or [-1],
            source_ids or [-1],
            standard_ids or [-1],
            standard_ids or [-1],
        ),
    )
    print_rows(columns, rows)

    print(f"\n=== {label}: incident feasibility ===")
    query = sql.SQL(
        """
        WITH target_events AS (
            SELECT
                person_id,
                condition_start_date
            FROM {}.condition_occurrence
            WHERE condition_start_date
                      BETWEEN %s AND %s
              AND (
                    condition_source_concept_id = ANY(%s)
                 OR condition_concept_id = ANY(%s)
              )
        ),
        first_event AS (
            SELECT
                person_id,
                MIN(condition_start_date) AS index_date
            FROM target_events
            GROUP BY person_id
        ),
        eligible_period AS (
            SELECT DISTINCT ON (fe.person_id)
                fe.person_id,
                fe.index_date,
                op.observation_period_start_date AS observation_start,
                op.observation_period_end_date AS observation_end,
                d.death_date
            FROM first_event AS fe
            JOIN {}.observation_period AS op
              ON op.person_id = fe.person_id
             AND op.observation_period_start_date
                     BETWEEN %s AND %s
             AND op.observation_period_end_date
                     BETWEEN %s AND %s
             AND op.observation_period_end_date
                     >= op.observation_period_start_date
             AND op.observation_period_start_date <= fe.index_date
             AND op.observation_period_end_date >= fe.index_date
            LEFT JOIN {}.death AS d
              ON d.person_id = fe.person_id
            ORDER BY
                fe.person_id,
                op.observation_period_start_date,
                op.observation_period_end_date DESC
        ),
        followup AS (
            SELECT
                *,
                LEAST(
                    observation_end,
                    COALESCE(death_date, %s),
                    %s
                ) AS followup_end
            FROM eligible_period
        )
        SELECT
            COUNT(*) AS indexed_patients,
            COUNT(*) FILTER (
                WHERE observation_start <= index_date - INTERVAL '365 days'
            ) AS with_1y_lookback,
            COUNT(*) FILTER (
                WHERE observation_start <= index_date - INTERVAL '365 days'
                  AND followup_end >= index_date + INTERVAL '5 years'
            ) AS with_1y_lookback_and_5y_followup,
            COUNT(*) FILTER (
                WHERE observation_start <= index_date - INTERVAL '365 days'
                  AND followup_end >= index_date + INTERVAL '10 years'
            ) AS with_1y_lookback_and_10y_followup,
            COUNT(*) FILTER (
                WHERE observation_start <= index_date - INTERVAL '365 days'
                  AND death_date IS NOT NULL
                  AND death_date <= index_date + INTERVAL '5 years'
            ) AS deaths_within_5y,
            COUNT(*) FILTER (
                WHERE observation_start <= index_date - INTERVAL '365 days'
                  AND death_date IS NOT NULL
                  AND death_date <= index_date + INTERVAL '10 years'
            ) AS deaths_within_10y,
            MIN(index_date)::text AS earliest_index,
            MAX(index_date)::text AS latest_index
        FROM followup
        """
    ).format(
        sql.Identifier(SCHEMA),
        sql.Identifier(SCHEMA),
        sql.Identifier(SCHEMA),
    )

    common_params = (
        VALID_DATE_MIN,
        db_end_date,
        source_ids or [-1],
        standard_ids or [-1],
        VALID_DATE_MIN,
        db_end_date,
        VALID_DATE_MIN,
        db_end_date,
    )

    for cutoff_label, cutoff_date in (
        ("Dataset Card cutoff", CARD_END_DATE),
        ("Actual DB cutoff", db_end_date),
    ):
        columns, rows = fetch(
            cursor,
            query,
            common_params + (cutoff_date, cutoff_date),
        )
        print(cutoff_label, cutoff_date)
        print_rows(columns, rows)

    print(f"\n=== {label}: first events by year with 1-year lookback ===")
    columns, rows = fetch(
        cursor,
        sql.SQL(
            """
            WITH first_event AS (
                SELECT
                    person_id,
                    MIN(condition_start_date) AS index_date
                FROM {}.condition_occurrence
                WHERE condition_start_date BETWEEN %s AND %s
                  AND (
                        condition_source_concept_id = ANY(%s)
                     OR condition_concept_id = ANY(%s)
                  )
                GROUP BY person_id
            )
            SELECT
                EXTRACT(YEAR FROM fe.index_date)::integer AS index_year,
                COUNT(*) AS patients
            FROM first_event AS fe
            WHERE EXISTS (
                SELECT 1
                FROM {}.observation_period AS op
                WHERE op.person_id = fe.person_id
                  AND op.observation_period_start_date
                          BETWEEN %s AND %s
                  AND op.observation_period_end_date
                          BETWEEN %s AND %s
                  AND op.observation_period_end_date
                          >= op.observation_period_start_date
                  AND op.observation_period_start_date
                          <= fe.index_date - INTERVAL '365 days'
                  AND op.observation_period_end_date >= fe.index_date
            )
            GROUP BY EXTRACT(YEAR FROM fe.index_date)
            ORDER BY index_year
            """
        ).format(sql.Identifier(SCHEMA), sql.Identifier(SCHEMA)),
        (
            VALID_DATE_MIN,
            db_end_date,
            source_ids or [-1],
            standard_ids or [-1],
            VALID_DATE_MIN,
            db_end_date,
            VALID_DATE_MIN,
            db_end_date,
        ),
    )
    print_rows(columns, rows)


def main():
    password = os.environ.get("SNUH_CDM_PASSWORD")
    if not password:
        password = getpass.getpass("SNUH CDM password: ")

    try:
        with psycopg.connect(
            host=HOST,
            port=PORT,
            dbname=DATABASE,
            user=USER,
            password=password,
            sslmode=SSL_MODE,
            connect_timeout=15,
            application_name="fermat_gastric_feasibility",
            options=f"-c statement_timeout={STATEMENT_TIMEOUT}",
        ) as connection:
            with connection.cursor() as cursor:
                print("Connected.")
                print(f"Schema: {SCHEMA}")
                print(f"Statement timeout: {STATEMENT_TIMEOUT}")

                print("\n=== Actual database profile ===")
                columns, rows = fetch(
                    cursor,
                    sql.SQL(
                        """
                        SELECT
                            (SELECT COUNT(*) FROM {}.person) AS persons,
                            (SELECT COUNT(*) FROM {}.condition_occurrence)
                                AS condition_events,
                            (SELECT MIN(observation_period_start_date)::text
                             FROM {}.observation_period)
                                AS raw_min_observation_start,
                            (SELECT MAX(observation_period_end_date)::text
                             FROM {}.observation_period)
                                AS raw_max_observation_end,
                            (SELECT COUNT(*)
                             FROM {}.observation_period
                             WHERE observation_period_start_date
                                       NOT BETWEEN %s AND %s
                                OR observation_period_end_date
                                       NOT BETWEEN %s AND %s
                                OR observation_period_end_date
                                       < observation_period_start_date)
                                AS invalid_observation_rows
                        """
                    ).format(
                        *(sql.Identifier(SCHEMA) for _ in range(5))
                    ),
                    (
                        VALID_DATE_MIN,
                        date(2100, 12, 31),
                        VALID_DATE_MIN,
                        date(2100, 12, 31),
                    ),
                )
                print_rows(columns, rows)

                columns, rows = fetch(
                    cursor,
                    sql.SQL(
                        """
                        SELECT MAX(observation_period_end_date)::text
                        FROM {}.observation_period
                        WHERE observation_period_end_date
                                  BETWEEN %s AND %s
                        """
                    ).format(sql.Identifier(SCHEMA)),
                    (VALID_DATE_MIN, date(2100, 12, 31)),
                )
                db_end_date = date.fromisoformat(rows[0][0])
                print("Actual valid DB end date:", db_end_date)
                print("Dataset Card end date:", CARD_END_DATE)

                for label, codes in (
                    ("PRIMARY", PRIMARY_CODES),
                    ("SENSITIVITY", SENSITIVITY_CODES),
                ):
                    print(f"\n=== {label}: matching OMOP source concepts ===")
                    (
                        source_result,
                        mapped_result,
                        source_ids,
                        standard_ids,
                    ) = find_target_concepts(cursor, codes)
                    print_rows(*source_result)

                    print(f"\n=== {label}: Maps to standard concepts ===")
                    print_rows(*mapped_result)
                    print("Source concept IDs:", source_ids)
                    print("Standard concept IDs:", standard_ids)

                    run_definition(
                        cursor,
                        label,
                        codes,
                        source_ids,
                        standard_ids,
                        db_end_date,
                    )
        return 0
    except psycopg.Error as error:
        print("Database query failed.", file=sys.stderr)
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
