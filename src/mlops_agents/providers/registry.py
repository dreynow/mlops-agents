"""Provider registry - builds each provider independently from config.

Users specify which backend to use for each service:
  provider:
    compute: vertex_ai
    storage: gcs
    ml: local
    data: bigquery
    event_bus: local
    serving: local

Each provider is built independently. Mix and match freely.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from mlops_agents.core.config import ProviderConfig
from mlops_agents.core.event import EventBus, LocalAsyncEventBus
from mlops_agents.providers.protocols import (
    ComputeProvider,
    DataProvider,
    MLPlatformProvider,
    ServingProvider,
    StorageProvider,
)

logger = structlog.get_logger()


@dataclass
class Providers:
    """Container holding all instantiated providers.

    Agents receive this to access infrastructure without
    knowing which backend they're talking to.
    """

    compute: ComputeProvider
    storage: StorageProvider
    ml: MLPlatformProvider
    data: DataProvider
    event_bus: EventBus
    serving: ServingProvider


class ProviderRegistry:
    """Builds each provider independently from config."""

    @staticmethod
    def from_config(
        config: ProviderConfig,
        event_bus: EventBus | None = None,
    ) -> Providers:
        local_base = config.local.get("base_dir", ".mlops")

        providers = Providers(
            compute=ProviderRegistry._build_compute(config.compute, config, local_base),
            storage=ProviderRegistry._build_storage(config.storage, config, local_base),
            ml=ProviderRegistry._build_ml(config.ml, config, local_base),
            data=ProviderRegistry._build_data(config.data, config, local_base),
            event_bus=event_bus or LocalAsyncEventBus(),
            serving=ProviderRegistry._build_serving(config.serving, config),
        )

        logger.info(
            "providers.registry.built",
            compute=config.compute,
            storage=config.storage,
            ml=config.ml,
            data=config.data,
            serving=config.serving,
        )

        return providers

    # --- Compute ---

    @staticmethod
    def _build_compute(backend: str, config: ProviderConfig, local_base: str) -> ComputeProvider:
        if backend == "vertex_ai":
            from mlops_agents.providers.gcp.vertex import VertexAICompute

            settings = config.vertex_ai
            return VertexAICompute(
                project=settings.get("project", ""),
                location=settings.get("region", "us-central1"),
                staging_bucket=settings.get("staging_bucket", ""),
            )
        elif backend == "local":
            from mlops_agents.providers.local.compute import LocalDockerCompute

            return LocalDockerCompute(artifacts_dir=f"{local_base}/artifacts")
        else:
            raise ValueError(f"Unknown compute provider: '{backend}'. Use 'local' or 'vertex_ai'.")

    # --- Storage ---

    @staticmethod
    def _build_storage(backend: str, config: ProviderConfig, local_base: str) -> StorageProvider:
        if backend == "gcs":
            from mlops_agents.providers.gcp.gcs import GCSStorage

            settings = config.gcs
            return GCSStorage(
                bucket=settings.get("bucket", ""),
                prefix=settings.get("prefix", ""),
                project=settings.get("project"),
            )
        elif backend == "local":
            from mlops_agents.providers.local.storage import LocalFileStorage

            return LocalFileStorage(base_dir=f"{local_base}/storage")
        else:
            raise ValueError(f"Unknown storage provider: '{backend}'. Use 'local' or 'gcs'.")

    # --- ML Platform ---

    @staticmethod
    def _build_ml(backend: str, config: ProviderConfig, local_base: str) -> MLPlatformProvider:
        if backend == "local":
            from mlops_agents.providers.local.mlflow import LocalMLPlatform

            settings = config.mlflow
            base_dir = settings.get("base_dir", f"{local_base}/mlplatform")
            return LocalMLPlatform(base_dir=base_dir)
        else:
            raise ValueError(f"Unknown ML platform provider: '{backend}'. Use 'local'.")

    # --- Data ---

    @staticmethod
    def _build_data(backend: str, config: ProviderConfig, local_base: str) -> DataProvider:
        if backend == "bigquery":
            from mlops_agents.providers.gcp.bigquery import BigQueryData

            settings = config.bigquery
            project = settings.get("project", "")
            if not project:
                raise ValueError("BigQuery provider requires 'project' in bigquery settings.")
            return BigQueryData(
                project=project,
                dataset=settings.get("dataset", "ml_features"),
                location=settings.get("location", "US"),
            )
        elif backend == "local":
            from mlops_agents.providers.local.duckdb import DuckDBData

            return DuckDBData(base_dir=f"{local_base}/data")
        else:
            raise ValueError(f"Unknown data provider: '{backend}'. Use 'local' or 'bigquery'.")

    # --- Serving ---

    @staticmethod
    def _build_serving(backend: str, config: ProviderConfig) -> ServingProvider:
        if backend == "local":
            from mlops_agents.providers.local.serving import LocalServing

            return LocalServing()
        else:
            raise ValueError(f"Unknown serving provider: '{backend}'. Use 'local'.")
