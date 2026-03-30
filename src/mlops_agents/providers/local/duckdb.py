"""Local DuckDB data provider."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from mlops_agents.providers.protocols import Dataset

logger = structlog.get_logger()


class DuckDBData:
    """DuckDB-backed data provider for local development.

    Falls back to JSON file storage if DuckDB is not installed.
    Supports SQL queries over local datasets stored as JSON/CSV.
    """

    def __init__(self, base_dir: str = ".mlops/data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._db = None

    def _get_db(self):
        if self._db is None:
            try:
                import duckdb

                db_path = self.base_dir / "local.duckdb"
                self._db = duckdb.connect(str(db_path))
            except ImportError:
                logger.warning("duckdb not installed, using JSON-only mode")
                self._db = "json_fallback"
        return self._db

    async def query(self, sql: str) -> list[dict[str, Any]]:
        db = self._get_db()
        if db == "json_fallback":
            raise RuntimeError("SQL queries require duckdb. Install with: pip install duckdb")
        result = db.execute(sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    async def get_dataset(self, name: str, version: str = "latest") -> Dataset:
        dataset_dir = self.base_dir / name
        if not dataset_dir.exists():
            raise ValueError(f"Dataset not found: {name}")

        if version == "latest":
            versions = sorted(dataset_dir.glob("v*.json"))
            if not versions:
                raise ValueError(f"No versions found for dataset: {name}")
            version_file = versions[-1]
            version = version_file.stem
        else:
            version_file = dataset_dir / f"{version}.json"

        if not version_file.exists():
            raise ValueError(f"Version not found: {name}/{version}")

        data = json.loads(version_file.read_text())
        records = data.get("records", [])

        schema = {}
        if records:
            schema = {k: type(v).__name__ for k, v in records[0].items()}

        return Dataset(
            name=name,
            version=version,
            path=str(version_file),
            num_rows=len(records),
            num_columns=len(schema),
            schema=schema,
            metadata=data.get("metadata", {}),
        )

    async def save_dataset(self, data: list[dict[str, Any]], name: str) -> Dataset:
        dataset_dir = self.base_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(dataset_dir.glob("v*.json"))
        next_version = f"v{len(existing) + 1}"

        dataset_file = dataset_dir / f"{next_version}.json"
        dataset_file.write_text(
            json.dumps(
                {
                    "records": data,
                    "metadata": {"num_rows": len(data)},
                },
                indent=2,
                default=str,
            )
        )

        schema = {}
        if data:
            schema = {k: type(v).__name__ for k, v in data[0].items()}

        logger.info("data.local.save", name=name, version=next_version, rows=len(data))

        return Dataset(
            name=name,
            version=next_version,
            path=str(dataset_file),
            num_rows=len(data),
            num_columns=len(schema),
            schema=schema,
        )

    async def list_datasets(self) -> list[str]:
        return [
            d.name for d in self.base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]
