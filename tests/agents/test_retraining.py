"""Tests for Retraining Agent."""

import pytest

from mlops_agents.agents.retraining import RetrainAgent
from mlops_agents.core.audit import SQLiteAuditStore
from mlops_agents.core.event import Event, LocalAsyncEventBus
from mlops_agents.core.reasoning import StaticReasoner


@pytest.fixture
def agent(tmp_path):
    return RetrainAgent(
        event_bus=LocalAsyncEventBus(),
        audit_store=SQLiteAuditStore(db_path=tmp_path / "test.db"),
        reasoning_engine=StaticReasoner(),
        min_feedback_samples=50,
        min_drifted_features=1,
        full_retrain_drift_ratio=0.3,
    )


def _event(trace_id="pipe-test", **payload):
    return Event(type="model.retrain", source="test", trace_id=trace_id, payload=payload)


class TestRetrainDecision:
    @pytest.mark.asyncio
    async def test_drift_triggers_retrain(self, agent):
        event = _event(
            trigger_source="drift",
            model_name="fraud-detector",
            drifted_features=["amount", "hour"],
            total_features=10,
        )
        decision = await agent.run(event)
        assert decision.approved is True
        assert decision.metadata["strategy"] == "full_retrain"

    @pytest.mark.asyncio
    async def test_feedback_triggers_fine_tune(self, agent):
        event = _event(
            trigger_source="feedback",
            model_name="fraud-detector",
            feedback_count=100,
        )
        decision = await agent.run(event)
        assert decision.approved is True
        assert decision.metadata["strategy"] == "fine_tune"
        assert "feedback" in decision.metadata["data_window"]

    @pytest.mark.asyncio
    async def test_insufficient_feedback_no_retrain(self, agent):
        event = _event(
            trigger_source="feedback",
            model_name="fraud-detector",
            feedback_count=10,  # Below 50 threshold
        )
        decision = await agent.run(event)
        assert decision.approved is False
        assert decision.metadata["strategy"] == "skip"

    @pytest.mark.asyncio
    async def test_manual_trigger_always_retrains(self, agent):
        event = _event(trigger_source="manual", model_name="fraud-detector")
        decision = await agent.run(event)
        assert decision.approved is True

    @pytest.mark.asyncio
    async def test_accuracy_drop_triggers_retrain(self, agent):
        event = _event(
            trigger_source="degradation",
            model_name="fraud-detector",
            accuracy_drop=0.08,
        )
        decision = await agent.run(event)
        assert decision.approved is True


class TestRetrainStrategy:
    @pytest.mark.asyncio
    async def test_high_drift_ratio_full_retrain(self, agent):
        event = _event(
            trigger_source="drift",
            model_name="fraud-detector",
            drifted_features=["a", "b", "c", "d"],  # 4/10 = 40% > 30%
            total_features=10,
        )
        decision = await agent.run(event)
        assert decision.metadata["strategy"] == "full_retrain"
        assert decision.metadata["data_window"] == "all"

    @pytest.mark.asyncio
    async def test_low_drift_ratio_with_feedback_fine_tunes(self, agent):
        event = _event(
            trigger_source="drift",
            model_name="fraud-detector",
            drifted_features=["amount"],  # 1/10 = 10% < 30%
            total_features=10,
            feedback_count=75,
        )
        decision = await agent.run(event)
        assert decision.metadata["strategy"] == "fine_tune"

    @pytest.mark.asyncio
    async def test_low_drift_no_feedback_full_retrain_90d(self, agent):
        event = _event(
            trigger_source="drift",
            model_name="fraud-detector",
            drifted_features=["amount"],
            total_features=10,
        )
        decision = await agent.run(event)
        assert decision.metadata["strategy"] == "full_retrain"
        assert decision.metadata["data_window"] == "recent_90d"


class TestRetrainMetadata:
    @pytest.mark.asyncio
    async def test_retrain_reasons_tracked(self, agent):
        event = _event(
            trigger_source="drift",
            model_name="fraud-detector",
            drifted_features=["amount"],
            total_features=10,
            feedback_count=100,
            accuracy_drop=0.03,
        )
        decision = await agent.run(event)
        reasons = decision.metadata["retrain_reasons"]
        assert any("drifted" in r for r in reasons)
        assert any("feedback" in r for r in reasons)
        assert any("accuracy" in r for r in reasons)
