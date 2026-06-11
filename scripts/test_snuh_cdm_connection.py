"""Test direct PostgreSQL access to the SNUH OMOP CDM.

The password is read with getpass and is not saved or printed.
"""

import getpass
import os
import sys

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

CORE_TABLES = (
    "person",
    "observation_period",
    "visit_occurrence",
    "condition_occurrence",
    "drug_exposure",
    "procedure_occurrence",
    "measurement",
    "death",
    "concept",
    "concept_ancestor",
    "concept_relationship",
)


def main():
    password = os.environ.get("SNUH_CDM_PASSWORD")
    if not password:
        password = getpass.getpass("SNUH CDM password: ")

    print(f"Connecting to {HOST}:{PORT}/{DATABASE} as {USER} ...")
    print(f"SSL mode: {SSL_MODE}")

    try:
        with psycopg.connect(
            host=HOST,
            port=PORT,
            dbname=DATABASE,
            user=USER,
            password=password,
            sslmode=SSL_MODE,
            connect_timeout=15,
            application_name="fermat_connection_test",
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        current_user,
                        current_database(),
                        current_schema(),
                        version(),
                        pg_backend_pid()
                    """
                )
                user, database, current_schema, version, backend_pid = (
                    cursor.fetchone()
                )

                ssl_info = connection.info.pgconn.ssl_in_use
                print("Connection: OK")
                print(f"SSL in use: {bool(ssl_info)}")
                print(f"Current user: {user}")
                print(f"Current database: {database}")
                print(f"Current schema: {current_schema}")
                print(f"Backend PID: {backend_pid}")
                print(f"Server: {version}")

                cursor.execute(
                    """
                    SELECT
                        has_schema_privilege(%s, 'USAGE'),
                        has_schema_privilege(%s, 'CREATE')
                    """,
                    (SCHEMA, SCHEMA),
                )
                usage, create = cursor.fetchone()
                print()
                print(f"Schema {SCHEMA!r} USAGE: {usage}")
                print(f"Schema {SCHEMA!r} CREATE: {create}")

                cursor.execute(
                    """
                    SELECT
                        table_name,
                        has_table_privilege(
                            current_user,
                            quote_ident(table_schema)
                                || '.'
                                || quote_ident(table_name),
                            'SELECT'
                        ) AS can_select
                    FROM information_schema.tables
                    WHERE table_schema = %s
                      AND table_name = ANY(%s)
                    ORDER BY table_name
                    """,
                    (SCHEMA, list(CORE_TABLES)),
                )
                found = cursor.fetchall()
                found_names = {row[0] for row in found}

                print()
                print("Core OMOP tables:")
                for table_name in CORE_TABLES:
                    match = next(
                        (row for row in found if row[0] == table_name),
                        None,
                    )
                    if match:
                        print(
                            f"  {table_name:28s} present, "
                            f"SELECT={match[1]}"
                        )
                    else:
                        print(f"  {table_name:28s} MISSING")

                required = {
                    "person",
                    "observation_period",
                    "condition_occurrence",
                }
                if not required.issubset(found_names):
                    print(
                        "\nConnection succeeded, but required tables are "
                        "missing.",
                        file=sys.stderr,
                    )
                    return 2

                cursor.execute(
                    sql.SQL(
                        """
                        SELECT
                            (SELECT COUNT(*) FROM {}.person) AS persons,
                            (SELECT COUNT(*)
                             FROM {}.condition_occurrence)
                                AS condition_events,
                            (SELECT MIN(observation_period_start_date)::text
                             FROM {}.observation_period
                             WHERE observation_period_start_date
                                   BETWEEN DATE '1900-01-01'
                                       AND DATE '2100-12-31')
                                AS first_observation,
                            (SELECT MAX(observation_period_end_date)::text
                             FROM {}.observation_period
                             WHERE observation_period_end_date
                                   BETWEEN DATE '1900-01-01'
                                       AND DATE '2100-12-31')
                                AS last_observation,
                            (SELECT COUNT(*)
                             FROM {}.observation_period
                             WHERE observation_period_start_date
                                       < DATE '1900-01-01'
                                OR observation_period_start_date
                                       > DATE '2100-12-31'
                                OR observation_period_end_date
                                       < DATE '1900-01-01'
                                OR observation_period_end_date
                                       > DATE '2100-12-31')
                                AS invalid_observation_period_rows
                        """
                    ).format(
                        sql.Identifier(SCHEMA),
                        sql.Identifier(SCHEMA),
                        sql.Identifier(SCHEMA),
                        sql.Identifier(SCHEMA),
                        sql.Identifier(SCHEMA),
                    )
                )
                (
                    persons,
                    conditions,
                    first_date,
                    last_date,
                    invalid_periods,
                ) = cursor.fetchone()

                print()
                print("Basic aggregate check:")
                print(f"  persons: {persons:,}")
                print(f"  condition events: {conditions:,}")
                print(
                    "  valid observation range (1900-2100): "
                    f"{first_date} to {last_date}"
                )
                print(
                    "  observation_period rows outside 1900-2100: "
                    f"{invalid_periods:,}"
                )

        return 0
    except psycopg.Error as error:
        print("Connection/query failed.", file=sys.stderr)
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
