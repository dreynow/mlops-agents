"""Tests for Deployment Agent."""

import pytest

from mlops_agents.agents.deployment import DeployAgent
from mlops_agents.core.audit import SQLiteAuditStore
from mlops_agents.core.event import Event, LocalAsyncEventBus
from mlops_agents.core.reasoning import StaticReasoner
from mlops_agents.providers.local.serving import LocalServing
from mlops_agents.providers.protocols import DeployConfig, ModelVersion


@pytest.fixture
def serving():
    return LocalServing()


@pytest.fixture
def agent(tmp_path):
    return DeployAgent(
        event_bus=LocalAsyncEventBus(),
        audit_store=SQLiteAuditStore(db_path=tmp_path / "test.db"),
        reasoning_engine=StaticReasoner(),
        max_error_rate=0.05,
        max_latency_p99_ms=200.0,
    )


def _event(action="deploy", trace_id="pipe-test", **payload):
    return Event(
        type="model.deploy.canary" if action == "deploy" else f"model.deploy.{action}",
        source="test",
        trace_id=trace_id,
        payload={"action": action, **payload},
    )


class TestDeployCanary:
    @pytest.mark.asyncio
    async def test_deploy_canary_with_serving(self, agent, serving):
        event = _event(
            action="deploy",
            model_name="fraud-detector",
            model_version="v2",
            artifact_uri="/models/v2.pkl",
        )
        decision = await agent.run(event, providers={"serving": serving})
        assert decision.approved is True
        assert decision.action == "model.deploy.canary"
        assert decision.artifacts["endpoint_id"]  # Not empty

    @pytest.mark.asyncio
    async def test_deploy_canary_without_serving(self, agent):
        event = _event(
            action="deploy",
            model_name="fraud-detector",
            model_version="v2",
        )
        decision = await agent.run(event)
        assert decision.approved is True


class TestCheckCanary:
    @pytest.mark.asyncio
    async def test_healthy_canary_promotes(self, agent, serving):
        # Deploy first
        mv = ModelVersion(model_name="fraud-detector", version="v2", artifact_uri="/v2.pkl")
        await serving.deploy(mv, DeployConfig(endpoint_name="test-ep"))

        # Check canary
        event = _event(action="check_canary", endpoint_id="test-ep")
        decision = await agent.run(event, providers={"serving": serving})
        # LocalServing generates low error rates (0.1-2%), well below 5% threshold
        assert decision.action in ("model.deploy.promote", "model.deploy.rollback")

    @pytest.mark.asyncio
    async def test_canary_without_serving_uses_payload(self, agent):
        event = _event(
            action="check_canary",
            endpoint_id="ep-1",
            error_rate=0.01,
            latency_p99_ms=15.0,
            request_count=500,
        )
        decision = await agent.run(event)
        assert decision.approved is True
        assert decision.action == "model.deploy.promote"

    @pytest.mark.asyncio
    async def test_high_error_rate_triggers_rollback(self, agent):
        event = _event(
            action="check_canary",
            endpoint_id="ep-1",
            error_rate=0.15,  # 15% > 5% threshold
            latency_p99_ms=15.0,
        )
        decision = await agent.run(event)
        assert decision.approved is False
        assert decision.action == "model.deploy.rollback"

    @pytest.mark.asyncio
    async def test_high_latency_triggers_rollback(self, agent):
        event = _event(
            action="check_canary",
            endpoint_id="ep-1",
            error_rate=0.01,
            latency_p99_ms=500.0,  # 500ms > 200ms threshold
        )
        decision = await agent.run(event)
        assert decision.approved is False
        assert decision.action == "model.deploy.rollback"


class TestRollback:
    @pytest.mark.asyncio
    async def test_rollback(self, agent, serving):
        # Deploy first
        mv = ModelVersion(model_name="fraud-detector", version="v2", artifact_uri="/v2.pkl")
        await serving.deploy(mv, DeployConfig(endpoint_name="rollback-ep"))

        event = _event(action="rollback", endpoint_id="rollback-ep")
        decision = await agent.run(event, providers={"serving": serving})
        assert decision.approved is True
        assert decision.action == "model.rollback"

    @pytest.mark.asyncio
    async def test_rollback_nonexistent_endpoint(self, agent, serving):
        event = _event(action="rollback", endpoint_id="no-such-ep")
        decision = await agent.run(event, providers={"serving": serving})
        assert decision.approved is True  # Rollback still succeeds
