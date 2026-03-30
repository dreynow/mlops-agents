"""Local model serving provider.

Tracks deployed models in memory. In a real setup this would manage
Docker containers or local FastAPI processes. For the framework demo,
it simulates deployment lifecycle and metrics.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone

import structlog

from mlops_agents.providers.protocols import (
    DeployConfig,
    Endpoint,
    EndpointMetrics,
    ModelVersion,
)

logger = structlog.get_logger()


class LocalServing:
    """In-memory model serving for local development.

    Simulates deployment lifecycle (creating -> ready -> serving)
    and generates synthetic metrics for canary evaluation.
    """

    def __init__(self):
        self._endpoints: dict[str, Endpoint] = {}
        self._metrics: dict[str, EndpointMetrics] = {}
        self._request_counts: dict[str, int] = {}

    async def deploy(self, model: ModelVersion, config: DeployConfig) -> Endpoint:
        endpoint_id = config.endpoint_name or f"ep-{uuid.uuid4().hex[:8]}"

        endpoint = Endpoint(
            endpoint_id=endpoint_id,
            url=f"http://localhost:{config.port}/predict",
            model_name=model.model_name,
            model_version=model.version,
            status="ready",
            traffic_split={model.version: 100},
        )

        self._endpoints[endpoint_id] = endpoint
        self._request_counts[endpoint_id] = 0

        logger.info(
            "serving.local.deploy",
            endpoint_id=endpoint_id,
            model=model.model_name,
            version=model.version,
        )
        return endpoint

    async def get_endpoint(self, endpoint_id: str) -> Endpoint:
        if endpoint_id not in self._endpoints:
            raise ValueError(f"Endpoint not found: {endpoint_id}")
        return self._endpoints[endpoint_id]

    async def get_endpoint_metrics(
        self, endpoint_id: str, window_minutes: int = 15
    ) -> EndpointMetrics:
        if endpoint_id not in self._endpoints:
            raise ValueError(f"Endpoint not found: {endpoint_id}")

        # Simulate realistic metrics
        request_count = self._request_counts.get(endpoint_id, 0) + random.randint(50, 200)
        self._request_counts[endpoint_id] = request_count
        error_count = int(request_count * random.uniform(0.001, 0.02))

        return EndpointMetrics(
            endpoint_id=endpoint_id,
            request_count=request_count,
            error_count=error_count,
            error_rate=error_count / max(request_count, 1),
            latency_p50_ms=random.uniform(5.0, 15.0),
            latency_p95_ms=random.uniform(20.0, 40.0),
            latency_p99_ms=random.uniform(40.0, 80.0),
            window_minutes=window_minutes,
        )

    async def set_traffic(self, endpoint_id: str, split: dict[str, int]) -> None:
        if endpoint_id not in self._endpoints:
            raise ValueError(f"Endpoint not found: {endpoint_id}")

        endpoint = self._endpoints[endpoint_id]
        endpoint.traffic_split = split

        logger.info("serving.local.traffic", endpoint_id=endpoint_id, split=split)

    async def undeploy(self, endpoint_id: str) -> None:
        if endpoint_id in self._endpoints:
            self._endpoints[endpoint_id].status = "deleted"
            del self._endpoints[endpoint_id]
            logger.info("serving.local.undeploy", endpoint_id=endpoint_id)
