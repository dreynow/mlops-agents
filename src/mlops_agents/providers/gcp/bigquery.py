"""BigQuery data provider for feature data and dataset management.

Handles SQL queries over feature tables, dataset versioning via
timestamped tables, and paginated reads for large datasets.

IMPORTANT: All queries use parameterized queries to prevent SQL injection.
Never interpolate user input into query strings.

Requires: pip install mlops-agents[gcp]
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import structlog

from mlops_agents.providers.protocols import Dataset

logger = structlog.get_logger()

# Max rows per page when reading large datasets
DEFAULT_PAGE_SIZE = 10_000


class BigQueryData:
    """BigQuery-backed data provider.

    Maps the DataProvider protocol to BigQuery operations.
    Datasets are stored as tables in a configured BQ dataset.
    Versioning uses timestamped table suffixes.

    Usage:
        data = BigQueryData(project="my-project", dataset="ml_features")
        rows = await data.query("SELECT * FROM transactions WHERE amount > @threshold",
                                params={"threshold": 1000.0})
    """

    def __init__(
        self,
        project: str,
        dataset: str,
        location: str = "US",
    ):
        self._project = project
        self._dataset = dataset
        self._location = location
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google.cloud import bigquery
            except ImportError:
                raise RuntimeError(
                    "google-cloud-bigquery required for BigQueryData. "
                    "Install with: pip install mlops-agents[gcp]"
                )
            self._client = bigquery.Client(
                project=self._project,
                location=self._location,
            )
        return self._client

    @property
    def _full_dataset(self) -> str:
        return f"{self._project}.{self._dataset}"

    async def query(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a parameterized SQL query and return results as dicts.

        Uses BigQuery query parameters to prevent SQL injection.
        For large results, use pagination (automatically handled).

        Args:
            sql: SQL query with @param_name placeholders for parameters.
            params: Dict of parameter values. Types are inferred automatically.
        """
        client = self._get_client()
        from google.cloud import bigquery

        job_config = bigquery.QueryJobConfig()

        if params:
            query_params = []
            for name, value in params.items():
                if isinstance(value, int):
                    query_params.append(bigquery.ScalarQueryParameter(name, "INT64", value))
                elif isinstance(value, float):
                    query_params.append(bigquery.ScalarQueryParameter(name, "FLOAT64", value))
                elif isinstance(value, bool):
                    query_params.append(bigquery.ScalarQueryParameter(name, "BOOL", value))
                elif isinstance(value, str):
                    query_params.append(bigquery.ScalarQueryParameter(name, "STRING", value))
                elif isinstance(value, datetime):
                    query_params.append(bigquery.ScalarQueryParameter(name, "TIMESTAMP", value))
                else:
                    query_params.append(bigquery.ScalarQueryParameter(name, "STRING", str(value)))
            job_config.query_parameters = query_params

        logger.info("bigquery.query.start", sql=sql[:100], params=list((params or {}).keys()))

        loop = asyncio.get_event_loop()
        query_job = await loop.run_in_executor(
            None,
            lambda: client.query(sql, job_config=job_config),
        )

        # Wait for results with pagination
        rows = await loop.run_in_executor(None, lambda: list(query_job.result()))

        results = [dict(row) for row in rows]
        logger.info("bigquery.query.complete", rows=len(results))
        return results

    async def get_dataset(self, name: str, version: str = "latest") -> Dataset:
        """Get a dataset by name and version.

        Datasets are stored as BQ tables with naming convention:
          {dataset}.{name}          (latest / only version)
          {dataset}.{name}_v{N}     (versioned)
        """
        client = self._get_client()

        if version == "latest":
            table_id = await self._resolve_latest_table(name)
        else:
            table_id = f"{self._full_dataset}.{name}_{version}"

        loop = asyncio.get_event_loop()

        try:
            table = await loop.run_in_executor(
                None,
                lambda: client.get_table(table_id),
            )
        except Exception as e:
            if "NotFound" in type(e).__name__ or "404" in str(e):
                raise ValueError(f"Dataset not found: {name}/{version}")
            raise

        schema = {field.name: field.field_type for field in table.schema}

        return Dataset(
            name=name,
            version=version,
            path=table_id,
            num_rows=table.num_rows,
            num_columns=len(table.schema),
            schema=schema,
            metadata={
                "created": table.created.isoformat() if table.created else "",
                "modified": table.modified.isoformat() if table.modified else "",
                "size_bytes": table.num_bytes,
            },
        )

    async def save_dataset(
        self,
        data: list[dict[str, Any]],
        name: str,
    ) -> Dataset:
        """Save a dataset as a new BQ table.

        Creates a versioned table: {dataset}.{name}_v{N}
        """
        client = self._get_client()
        from google.cloud import bigquery

        # Determine next version
        version = await self._next_version(name)
        table_id = f"{self._full_dataset}.{name}_{version}"

        logger.info("bigquery.save.start", table=table_id, rows=len(data))

        loop = asyncio.get_event_loop()

        # Create table and load data
        job_config = bigquery.LoadJobConfig(
            autodetect=True,
            write_disposition="WRITE_TRUNCATE",
        )

        load_job = await loop.run_in_executor(
            None,
            lambda: client.load_table_from_json(data, table_id, job_config=job_config),
        )

        # Wait for load to complete
        await loop.run_in_executor(None, load_job.result)

        # Get table metadata
        table = await loop.run_in_executor(
            None,
            lambda: client.get_table(table_id),
        )

        schema = {field.name: field.field_type for field in table.schema}

        logger.info("bigquery.save.complete", table=table_id, rows=table.num_rows)

        return Dataset(
            name=name,
            version=version,
            path=table_id,
            num_rows=table.num_rows,
            num_columns=len(schema),
            schema=schema,
        )

    async def list_datasets(self) -> list[str]:
        """List all dataset names in the BQ dataset."""
        client = self._get_client()

        loop = asyncio.get_event_loop()
        tables = await loop.run_in_executor(
            None,
            lambda: list(client.list_tables(f"{self._project}.{self._dataset}")),
        )

        # Extract unique base names (strip _v{N} suffixes)
        names = set()
        for table in tables:
            base_name = table.table_id
            # Strip version suffix: transactions_v3 -> transactions
            parts = base_name.rsplit("_v", 1)
            if len(parts) == 2 and parts[1].isdigit():
                names.add(parts[0])
            else:
                names.add(base_name)

        return sorted(names)

    async def _resolve_latest_table(self, name: str) -> str:
        """Find the latest versioned table for a dataset name."""
        client = self._get_client()

        loop = asyncio.get_event_loop()
        tables = await loop.run_in_executor(
            None,
            lambda: list(client.list_tables(f"{self._project}.{self._dataset}")),
        )

        # Find all tables matching {name}_v{N} pattern
        versions = []
        for table in tables:
            tid = table.table_id
            if tid == name:
                versions.append((0, tid))
            elif tid.startswith(f"{name}_v"):
                suffix = tid[len(f"{name}_v") :]
                if suffix.isdigit():
                    versions.append((int(suffix), tid))

        if not versions:
            raise ValueError(f"Dataset not found: {name}")

        # Return highest version
        _, table_name = max(versions, key=lambda x: x[0])
        return f"{self._full_dataset}.{table_name}"

    async def _next_version(self, name: str) -> str:
        """Determine the next version number for a dataset."""
        client = self._get_client()

        loop = asyncio.get_event_loop()

        try:
            tables = await loop.run_in_executor(
                None,
                lambda: list(client.list_tables(f"{self._project}.{self._dataset}")),
            )
        except Exception:
            return "v1"

        max_version = 0
        for table in tables:
            tid = table.table_id
            if tid.startswith(f"{name}_v"):
                suffix = tid[len(f"{name}_v") :]
                if suffix.isdigit():
                    max_version = max(max_version, int(suffix))

        return f"v{max_version + 1}"
