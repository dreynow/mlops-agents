"""Cloud Run model serving provider (Tier 2 - stub).

This provider will manage model serving on Cloud Run, including:
  - Container deployment from GCR/Artifact Registry
  - Traffic splitting for canary deployments
  - Metrics collection via Cloud Monitoring
  - Automatic rollback based on error rate thresholds

Not yet implemented. LocalServing covers the deployment simulation
for development. Contribute this provider if you need real
container-based model serving.

Architecture when implemented:
  - Each model version = one Cloud Run revision
  - Canary = traffic split between two revisions
  - Metrics via Cloud Monitoring API (request count, latency, errors)
  - Rollback = shift 100% traffic back to previous revision
  - Health checks via configured endpoint
"""

from __future__ import annotations

from mlops_agents.providers.protocols import (
    DeployConfig,
    Endpoint,
    EndpointMetrics,
    ModelVersion,
)


class CloudRunServing:
    """Cloud Run model serving (not yet implemented).

    Use LocalServing for local development. This provider is planned
    for production model serving with canary deployment support.
    """

    def __init__(
        self,
        project: str,
        region: str = "us-central1",
        service_prefix: str = "mlops-model",
    ):
        self._project = project
        self._region = region
        self._service_prefix = service_prefix

    async def deploy(self, model: ModelVersion, config: DeployConfig) -> Endpoint:
        raise NotImplementedError("CloudRunServing is not yet implemented. Use LocalServing.")

    async def get_endpoint(self, endpoint_id: str) -> Endpoint:
        raise NotImplementedError("CloudRunServing is not yet implemented. Use LocalServing.")

    async def get_endpoint_metrics(self, endpoint_id: str, window_minutes: int = 15) -> EndpointMetrics:
        raise NotImplementedError("CloudRunServing is not yet implemented. Use LocalServing.")

    async def set_traffic(self, endpoint_id: str, split: dict[str, int]) -> None:
        raise NotImplementedError("CloudRunServing is not yet implemented. Use LocalServing.")

    async def undeploy(self, endpoint_id: str) -> None:
        raise NotImplementedError("CloudRunServing is not yet implemented. Use LocalServing.")
