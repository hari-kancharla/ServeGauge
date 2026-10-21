"""DuckDB views over deterministic Parquet content; no network extensions."""

from pathlib import Path

import duckdb


def build_catalog(directory: Path, table_names: tuple[str, ...]) -> Path:
    target = directory / "catalog.duckdb"
    temporary = directory / "catalog.next.duckdb"
    temporary.unlink(missing_ok=True)
    with duckdb.connect(str(temporary), config={"enable_external_access": "true"}) as connection:
        for table in table_names:
            parquet = str((directory / f"{table}.parquet").resolve()).replace("'", "''")
            connection.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{parquet}')")
    temporary.replace(target)
    return target
