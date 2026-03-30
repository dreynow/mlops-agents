"""Tests for local serving provider."""

import pytest

from mlops_agents.providers.local.serving import LocalServing
from mlops_agents.providers.protocols import DeployConfig, ModelVersion


@pytest.fixture
def serving():
    return LocalServing()


@pytest.fixture
def model_version():
    return ModelVersion(
        model_name="fraud-detector",
        version="v2",
        artifact_uri="/models/v2/model.pkl",
        metrics={"f1": 0.834},
    )


class TestLocalServing:
    @pytest.mark.asyncio
    async def test_deploy(self, serving, model_version):
        endpoint = await serving.deploy(
            model_version,
            DeployConfig(endpoint_name="fraud-ep", port=8080),
        )
        assert endpoint.endpoint_id == "fraud-ep"
        assert endpoint.status == "ready"
        assert endpoint.model_name == "fraud-detector"
        assert "localhost:8080" in endpoint.url

    @pytest.mark.asyncio
    async def test_get_endpoint(self, serving, model_version):
        await serving.deploy(model_version, DeployConfig(endpoint_name="ep1"))
        endpoint = await serving.get_endpoint("ep1")
        assert endpoint.model_version == "v2"

    @pytest.mark.asyncio
    async def test_get_endpoint_not_found(self, serving):
        with pytest.raises(ValueError, match="not found"):
            await serving.get_endpoint("nonexistent")

    @pytest.mark.asyncio
    async def test_get_metrics(self, serving, model_version):
        await serving.deploy(model_version, DeployConfig(endpoint_name="ep1"))
        metrics = await serving.get_endpoint_metrics("ep1")
        assert metrics.request_count > 0
        assert metrics.error_rate >= 0
        assert metrics.latency_p50_ms > 0
        assert metrics.latency_p99_ms >= metrics.latency_p50_ms

    @pytest.mark.asyncio
    async def test_set_traffic(self, serving, model_version):
        await serving.deploy(model_version, DeployConfig(endpoint_name="ep1"))
        await serving.set_traffic("ep1", {"v2": 90, "v3": 10})
        endpoint = await serving.get_endpoint("ep1")
        assert endpoint.traffic_split == {"v2": 90, "v3": 10}

    @pytest.mark.asyncio
    async def test_undeploy(self, serving, model_version):
        await serving.deploy(model_version, DeployConfig(endpoint_name="ep1"))
        await serving.undeploy("ep1")
        with pytest.raises(ValueError):
            await serving.get_endpoint("ep1")
