"""Tests for Monitoring Agent."""

import pytest

from mlops_agents.agents.monitoring import MonitorAgent
from mlops_agents.core.audit import SQLiteAuditStore
from mlops_agents.core.event import Event, LocalAsyncEventBus
from mlops_agents.core.reasoning import StaticReasoner


@pytest.fixture
def agent(tmp_path):
    return MonitorAgent(
        event_bus=LocalAsyncEventBus(),
        audit_store=SQLiteAuditStore(db_path=tmp_path / "test.db"),
        reasoning_engine=StaticReasoner(),
        psi_threshold=0.2,
        accuracy_drop_threshold=0.05,
        error_rate_threshold=0.1,
    )


def _event(trace_id="pipe-test", **payload):
    return Event(type="metrics.collect", source="test", trace_id=trace_id, payload=payload)


class TestMonitorNoDrift:
    @pytest.mark.asyncio
    async def test_healthy_system(self, agent):
        event = _event(
            psi_scores={"amount": 0.05, "hour": 0.08, "category": 0.03},
            baseline_accuracy=0.95,
            current_accuracy=0.94,
        )
        decision = await agent.run(event)
        assert decision.approved is True
        assert decision.metadata["severity"] == "info"
        assert decision.metadata["needs_retrain"] is False

    @pytest.mark.asyncio
    async def test_no_metrics_healthy(self, agent):
        event = _event()
        decision = await agent.run(event)
        assert decision.approved is True


class TestMonitorDrift:
    @pytest.mark.asyncio
    async def test_feature_drift_detected(self, agent):
        event = _event(
            psi_scores={"amount": 0.35, "hour": 0.05},  # amount drifted
        )
        decision = await agent.run(event)
        assert decision.approved is False
        assert decision.metadata["severity"] == "warning"
        assert "amount" in decision.metadata["drifted_features"]

    @pytest.mark.asyncio
    async def test_multiple_features_drifted(self, agent):
        event = _event(
            psi_scores={"amount": 0.35, "hour": 0.25, "category": 0.05},
        )
        decision = await agent.run(event)
        assert decision.approved is False
        assert len(decision.metadata["drifted_features"]) == 2


class TestMonitorDegradation:
    @pytest.mark.asyncio
    async def test_accuracy_degradation(self, agent):
        event = _event(
            baseline_accuracy=0.95,
            current_accuracy=0.88,  # 7% drop > 5% threshold
        )
        decision = await agent.run(event)
        assert decision.approved is False
        assert decision.metadata["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_high_error_rate(self, agent):
        event = _event(error_rate=0.15)  # 15% > 10% threshold
        decision = await agent.run(event)
        assert decision.approved is False
        assert decision.metadata["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_drift_plus_degradation_needs_retrain(self, agent):
        event = _event(
            psi_scores={"amount": 0.35},
            baseline_accuracy=0.95,
            current_accuracy=0.88,
        )
        decision = await agent.run(event)
        assert decision.metadata["needs_retrain"] is True
        assert decision.metadata["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_drift_without_degradation_no_retrain(self, agent):
        event = _event(
            psi_scores={"amount": 0.35},
            baseline_accuracy=0.95,
            current_accuracy=0.94,  # Within threshold
        )
        decision = await agent.run(event)
        assert decision.metadata["needs_retrain"] is False
        assert decision.metadata["severity"] == "warning"
