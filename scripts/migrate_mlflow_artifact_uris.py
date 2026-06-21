"""Migrate local MLflow artifact URIs to the MLflow artifact proxy scheme."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sqlite3


URI_COLUMNS = (
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("logged_models", "artifact_location"),
    ("model_versions", "storage_location"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace file:// artifact URIs below the configured artifact root "
            "with portable mlflow-artifacts:/ URIs."
        )
    )
    parser.add_argument("--database", type=Path, default=Path("mlflow.db"))
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("data/08_reporting/mlflow_artifacts"),
    )
    return parser.parse_args()


def migrate(database: Path, artifact_root: Path) -> dict[str, int]:
    database = database.resolve()
    artifact_root_uri = artifact_root.resolve().as_uri().rstrip("/")

    if not database.is_file():
        raise FileNotFoundError(f"MLflow database not found: {database}")
    if not artifact_root.resolve().is_dir():
        raise FileNotFoundError(
            f"MLflow artifact root not found: {artifact_root.resolve()}"
        )

    backup = database.with_suffix(database.suffix + ".before-artifact-proxy.bak")
    if not backup.exists():
        shutil.copy2(database, backup)

    changed: dict[str, int] = {}
    with sqlite3.connect(database) as connection:
        for table, column in URI_COLUMNS:
            cursor = connection.execute(
                f"""
                UPDATE {table}
                SET {column} = 'mlflow-artifacts:/' ||
                    ltrim(substr({column}, ?), '/')
                WHERE {column} LIKE ?
                """,
                (len(artifact_root_uri) + 1, f"{artifact_root_uri}%"),
            )
            changed[f"{table}.{column}"] = cursor.rowcount

    print(f"Database: {database}")
    print(f"Backup:   {backup}")
    print(f"Root:     {artifact_root_uri}")
    for field, count in changed.items():
        print(f"{field}: {count} row(s) migrated")
    return changed


if __name__ == "__main__":
    arguments = parse_args()
    migrate(arguments.database, arguments.artifact_root)
