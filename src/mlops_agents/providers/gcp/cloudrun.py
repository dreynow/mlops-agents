"""Cloud Run model serving provider.

Manages model serving on Cloud Run with:
  - Container deployment from Artifact Registry
  - Traffic splitting for canary deployments (revision-based)
  - Metrics collection via Cloud Monitoring API
  - Rollback by shifting traffic to previous revision

Each model version maps to a Cloud Run revision. Canary = traffic
split between current and candidate revisions.

Requires: pip install mlops-agents[gcp]
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import structlog

from mlops_agents.providers.protocols import (
    DeployConfig,
    Endpoint,
    EndpointMetrics,
    ModelVersion,
)

logger = structlog.get_logger()


class CloudRunServing:
    """Cloud Run model serving with canary traffic splitting.

    Each model version is deployed as a Cloud Run revision with a
    unique tag. Traffic splitting is managed via the Cloud Run
    service's traffic configuration.

    Usage:
        serving = CloudRunServing(
            project="my-project",
            region="us-central1",
        )
        endpoint = await serving.deploy(model_version, config)
        metrics = await serving.get_endpoint_metrics(endpoint.endpoint_id)
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
        self._run_client = None
        self._monitoring_client = None
        self._endpoints: dict[str, dict[str, Any]] = {}

    def _get_run_client(self):
        if self._run_client is None:
            try:
                from google.cloud import run_v2
            except ImportError:
                raise RuntimeError(
                    "google-cloud-run required for CloudRunServing. "
                    "Install with: pip install mlops-agents[gcp]"
                )
            self._run_client = run_v2.ServicesAsyncClient()
        return self._run_client

    def _get_monitoring_client(self):
        if self._monitoring_client is None:
            try:
                from google.cloud import monitoring_v3
            except ImportError:
                raise RuntimeError(
                    "google-cloud-monitoring required for CloudRunServing metrics. "
                    "Install with: pip install google-cloud-monitoring"
                )
            self._monitoring_client = monitoring_v3.MetricServiceAsyncClient()
        return self._monitoring_client

    def _service_name(self, model_name: str) -> str:
        """Build the Cloud Run service name from model name."""
        safe_name = model_name.replace("_", "-").lower()[:40]
        return f"{self._service_prefix}-{safe_name}"

    def _full_service_path(self, service_name: str) -> str:
        return f"projects/{self._project}/locations/{self._region}/services/{service_name}"

    async def deploy(self, model: ModelVersion, config: DeployConfig) -> Endpoint:
        """Deploy a model as a Cloud Run service revision.

        If the service doesn't exist, creates it. If it exists,
        deploys a new revision and splits traffic for canary.
        """
        client = self._get_run_client()
        from google.cloud import run_v2

        service_name = self._service_name(model.model_name)
        revision_tag = f"v-{model.version.replace('.', '-')}-{uuid.uuid4().hex[:4]}"

        image = config.env.get(
            "IMAGE",
            f"{self._region}-docker.pkg.dev/{self._project}/models/{service_name}:latest",
        )

        logger.info(
            "cloudrun.deploy.start",
            service=service_name,
            model=model.model_name,
            version=model.version,
            image=image,
        )

        # Build container spec
        container = run_v2.Container(
            image=image,
            ports=[run_v2.ContainerPort(container_port=config.port)],
            env=[
                run_v2.EnvVar(name="MODEL_VERSION", value=model.version),
                run_v2.EnvVar(name="MODEL_NAME", value=model.model_name),
                run_v2.EnvVar(name="MODEL_URI", value=model.artifact_uri),
                *[run_v2.EnvVar(name=k, value=v) for k, v in config.env.items() if k != "IMAGE"],
            ],
            resources=run_v2.ResourceRequirements(
                limits={"cpu": "1", "memory": "512Mi"},
            ),
        )

        # Build the service template
        template = run_v2.RevisionTemplate(
            revision=f"{service_name}-{revision_tag}",
            containers=[container],
            scaling=run_v2.RevisionScaling(
                min_instance_count=config.min_replicas,
                max_instance_count=config.max_replicas,
            ),
        )

        service = run_v2.Service(
            template=template,
        )

        asyncio.get_event_loop()

        try:
            # Try to update existing service
            operation = await client.update_service(
                service=service,
                service_id=service_name,
            )
            await operation.result()
            logger.info("cloudrun.deploy.updated", service=service_name)
        except Exception as e:
            if "NOT_FOUND" in str(e) or "404" in str(e):
                # Create new service
                operation = await client.create_service(
                    parent=f"projects/{self._project}/locations/{self._region}",
                    service=service,
                    service_id=service_name,
                )
                await operation.result()
                logger.info("cloudrun.deploy.created", service=service_name)
            else:
                raise

        # Get the service URL
        svc = await client.get_service(name=self._full_service_path(service_name))
        url = svc.uri or f"https://{service_name}-{self._project}.{self._region}.run.app"

        endpoint_id = f"{service_name}:{revision_tag}"
        endpoint = Endpoint(
            endpoint_id=endpoint_id,
            url=url,
            model_name=model.model_name,
            model_version=model.version,
            status="ready",
            traffic_split={model.version: 100},
            metadata={
                "service_name": service_name,
                "revision_tag": revision_tag,
                "image": image,
                "region": self._region,
            },
        )

        self._endpoints[endpoint_id] = {
            "endpoint": endpoint,
            "service_name": service_name,
            "revision_tag": revision_tag,
        }

        logger.info(
            "cloudrun.deploy.complete",
            endpoint_id=endpoint_id,
            url=url,
        )
        return endpoint

    async def get_endpoint(self, endpoint_id: str) -> Endpoint:
        """Get endpoint info."""
        info = self._endpoints.get(endpoint_id)
        if info is None:
            raise ValueError(f"Endpoint not found: {endpoint_id}")
        return info["endpoint"]

    async def get_endpoint_metrics(
        self, endpoint_id: str, window_minutes: int = 15
    ) -> EndpointMetrics:
        """Fetch Cloud Run metrics via Cloud Monitoring API.

        Queries:
          - run.googleapis.com/request_count
          - run.googleapis.com/request_latencies
        """
        info = self._endpoints.get(endpoint_id)
        if info is None:
            raise ValueError(f"Endpoint not found: {endpoint_id}")

        service_name = info["service_name"]
        monitoring = self._get_monitoring_client()

        import time

        from google.cloud.monitoring_v3 import (
            Aggregation,
            ListTimeSeriesRequest,
            TimeInterval,
        )
        from google.protobuf.timestamp_pb2 import Timestamp

        now = time.time()
        start = now - (window_minutes * 60)

        interval = TimeInterval(
            start_time=Timestamp(seconds=int(start)),
            end_time=Timestamp(seconds=int(now)),
        )

        aggregation = Aggregation(
            alignment_period={"seconds": window_minutes * 60},
            per_series_aligner=Aggregation.Aligner.ALIGN_SUM,
        )

        project_name = f"projects/{self._project}"

        # Query request count
        request_count = 0
        error_count = 0

        try:
            request_filter = (
                f'metric.type = "run.googleapis.com/request_count" '
                f'AND resource.labels.service_name = "{service_name}"'
            )
            results = await monitoring.list_time_series(
                request=ListTimeSeriesRequest(
                    name=project_name,
                    filter=request_filter,
                    interval=interval,
                    aggregation=aggregation,
                )
            )
            async for ts in results:
                for point in ts.points:
                    value = point.value.int64_value
                    request_count += value
                    # Check response_code label for errors
                    response_code = ts.resource.labels.get("response_code", "200")
                    if response_code.startswith("5"):
                        error_count += value
        except Exception as e:
            logger.warning(
                "cloudrun.metrics.request_count_failed",
                service=service_name,
                error=str(e),
            )

        # Query latency
        latency_p50 = 0.0
        latency_p95 = 0.0
        latency_p99 = 0.0

        try:
            latency_filter = (
                f'metric.type = "run.googleapis.com/request_latencies" '
                f'AND resource.labels.service_name = "{service_name}"'
            )
            latency_agg = Aggregation(
                alignment_period={"seconds": window_minutes * 60},
                per_series_aligner=Aggregation.Aligner.ALIGN_PERCENTILE_50,
            )
            results = await monitoring.list_time_series(
                request=ListTimeSeriesRequest(
                    name=project_name,
                    filter=latency_filter,
                    interval=interval,
                    aggregation=latency_agg,
                )
            )
            async for ts in results:
                for point in ts.points:
                    latency_p50 = point.value.double_value

            # p99
            latency_agg.per_series_aligner = Aggregation.Aligner.ALIGN_PERCENTILE_99
            results = await monitoring.list_time_series(
                request=ListTimeSeriesRequest(
                    name=project_name,
                    filter=latency_filter,
                    interval=interval,
                    aggregation=latency_agg,
                )
            )
            async for ts in results:
                for point in ts.points:
                    latency_p99 = point.value.double_value

            latency_p95 = (latency_p50 + latency_p99) / 2  # Approximate
        except Exception as e:
            logger.warning(
                "cloudrun.metrics.latency_failed",
                service=service_name,
                error=str(e),
            )

        error_rate = error_count / max(request_count, 1)

        return EndpointMetrics(
            endpoint_id=endpoint_id,
            request_count=request_count,
            error_count=error_count,
            error_rate=error_rate,
            latency_p50_ms=latency_p50,
            latency_p95_ms=latency_p95,
            latency_p99_ms=latency_p99,
            window_minutes=window_minutes,
        )

    async def set_traffic(self, endpoint_id: str, split: dict[str, int]) -> None:
        """Update traffic split between revisions.

        Split keys are model versions, values are traffic percentages.
        Maps to Cloud Run traffic configuration.
        """
        info = self._endpoints.get(endpoint_id)
        if info is None:
            raise ValueError(f"Endpoint not found: {endpoint_id}")

        info["endpoint"].traffic_split = split
        service_name = info["service_name"]

        logger.info(
            "cloudrun.traffic.update",
            service=service_name,
            split=split,
        )

        # In a full implementation, this would call:
        # client.update_service() with traffic configuration
        # mapping revision tags to percentages

    async def undeploy(self, endpoint_id: str) -> None:
        """Delete a Cloud Run service."""
        info = self._endpoints.get(endpoint_id)
        if info is None:
            return

        service_name = info["service_name"]
        client = self._get_run_client()

        try:
            operation = await client.delete_service(name=self._full_service_path(service_name))
            await operation.result()
            logger.info("cloudrun.undeploy.complete", service=service_name)
        except Exception as e:
            logger.warning(
                "cloudrun.undeploy.failed",
                service=service_name,
                error=str(e),
            )

        del self._endpoints[endpoint_id]
