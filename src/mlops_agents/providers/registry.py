"""Provider registry - discovers and instantiates providers from config.

Central place to get the right provider implementations based on
the pipeline config. Supports "local" and "gcp" backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    """Creates the right provider set from config."""

    @staticmethod
    def from_config(
        config: ProviderConfig,
        event_bus: EventBus | None = None,
    ) -> Providers:
        backend = config.backend

        if backend == "local":
            return ProviderRegistry._build_local(config.local, event_bus)
        elif backend == "gcp":
            return ProviderRegistry._build_gcp(config.gcp, event_bus)
        else:
            raise ValueError(f"Unknown provider backend: {backend}. Use 'local' or 'gcp'.")

    @staticmethod
    def _build_local(
        settings: dict[str, Any],
        event_bus: EventBus | None = None,
    ) -> Providers:
        from mlops_agents.providers.local.compute import LocalDockerCompute
        from mlops_agents.providers.local.duckdb import DuckDBData
        from mlops_agents.providers.local.mlflow import LocalMLPlatform
        from mlops_agents.providers.local.serving import LocalServing
        from mlops_agents.providers.local.storage import LocalFileStorage

        base_dir = settings.get("base_dir", ".mlops")

        logger.info("providers.registry.local", base_dir=base_dir)

        return Providers(
            compute=LocalDockerCompute(artifacts_dir=f"{base_dir}/artifacts"),
            storage=LocalFileStorage(base_dir=f"{base_dir}/storage"),
            ml=LocalMLPlatform(base_dir=f"{base_dir}/mlplatform"),
            data=DuckDBData(base_dir=f"{base_dir}/data"),
            event_bus=event_bus or LocalAsyncEventBus(),
            serving=LocalServing(),
        )

    @staticmethod
    def _build_gcp(
        settings: dict[str, Any],
        event_bus: EventBus | None = None,
    ) -> Providers:
        """Build GCP providers.

        Tier 1 (implemented): GCS, BigQuery, Vertex AI
        Tier 2 (stubbed): Experiments, Pub/Sub, Cloud Run

        Tier 2 providers fall back to local implementations
        until their GCP versions are built.
        """
        project = settings.get("project_id")
        if not project:
            raise ValueError("GCP provider requires 'project_id' in gcp settings.")

        region = settings.get("region", "us-central1")
        staging_bucket = settings.get("staging_bucket", "")
        bq_dataset = settings.get("bigquery_dataset", "ml_features")
        bq_location = settings.get("bigquery_location", "US")

        logger.info(
            "providers.registry.gcp",
            project=project,
            region=region,
            bucket=staging_bucket,
            dataset=bq_dataset,
        )

        # Tier 1 - real GCP implementations
        from mlops_agents.providers.gcp.bigquery import BigQueryData
        from mlops_agents.providers.gcp.gcs import GCSStorage
        from mlops_agents.providers.gcp.vertex import VertexAICompute

        # Tier 2 - fall back to local implementations
        from mlops_agents.providers.local.mlflow import LocalMLPlatform
        from mlops_agents.providers.local.serving import LocalServing

        # Parse bucket name from gs:// URI if needed
        bucket_name = staging_bucket.replace("gs://", "").strip("/")

        return Providers(
            compute=VertexAICompute(
                project=project,
                location=region,
                staging_bucket=staging_bucket,
            ),
            storage=GCSStorage(
                bucket=bucket_name,
                project=project,
            ),
            ml=LocalMLPlatform(),  # Tier 2: use local until VertexAIExperiments is built
            data=BigQueryData(
                project=project,
                dataset=bq_dataset,
                location=bq_location,
            ),
            event_bus=event_bus or LocalAsyncEventBus(),  # Tier 2: local until PubSub is built
            serving=LocalServing(),  # Tier 2: local until CloudRun is built
        )
