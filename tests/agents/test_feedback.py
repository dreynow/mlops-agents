"""Tests for Feedback Agent."""

import pytest

from mlops_agents.agents.feedback import FeedbackAgent
from mlops_agents.core.audit import SQLiteAuditStore
from mlops_agents.core.event import Event, LocalAsyncEventBus
from mlops_agents.core.reasoning import StaticReasoner


@pytest.fixture
def agent(tmp_path):
    return FeedbackAgent(
        event_bus=LocalAsyncEventBus(),
        audit_store=SQLiteAuditStore(db_path=tmp_path / "test.db"),
        reasoning_engine=StaticReasoner(),
        min_samples_for_retrain=50,
        min_pattern_frequency=5,
        min_agreement_rate=0.7,
    )


def _event(trace_id="pipe-test", **payload):
    return Event(type="feedback.analyze", source="test", trace_id=trace_id, payload=payload)


def _corrections(n, segment="general"):
    return [{"segment": segment, "original": 0, "corrected": 1} for _ in range(n)]


class TestFeedbackSufficiency:
    @pytest.mark.asyncio
    async def test_enough_feedback_approved(self, agent):
        event = _event(corrections=_corrections(60))
        decision = await agent.run(event)
        assert decision.approved is True
        assert decision.metadata["enough_for_retrain"] is True

    @pytest.mark.asyncio
    async def test_insufficient_feedback_rejected(self, agent):
        event = _event(corrections=_corrections(20))
        decision = await agent.run(event)
        assert decision.approved is False
        assert decision.metadata["enough_for_retrain"] is False

    @pytest.mark.asyncio
    async def test_flags_count_toward_total(self, agent):
        event = _event(
            corrections=_corrections(30),
            flags=[{"reason": "wrong"} for _ in range(25)],
        )
        decision = await agent.run(event)
        assert decision.metadata["total_feedback"] == 55
        assert decision.approved is True


class TestLabelQuality:
    @pytest.mark.asyncio
    async def test_good_agreement_passes(self, agent):
        event = _event(
            corrections=_corrections(60),
            agreement_scores=[0.85, 0.90, 0.78, 0.92],
        )
        decision = await agent.run(event)
        assert decision.approved is True
        assert decision.metadata["agreement_ok"] is True

    @pytest.mark.asyncio
    async def test_low_agreement_rejects(self, agent):
        event = _event(
            corrections=_corrections(60),
            agreement_scores=[0.50, 0.45, 0.60, 0.55],  # Avg 52.5% < 70%
        )
        decision = await agent.run(event)
        assert decision.approved is False
        assert decision.metadata["agreement_ok"] is False

    @pytest.mark.asyncio
    async def test_no_agreement_scores_passes(self, agent):
        event = _event(corrections=_corrections(60))
        decision = await agent.run(event)
        assert decision.metadata["agreement_ok"] is True


class TestErrorPatterns:
    @pytest.mark.asyncio
    async def test_detects_segment_patterns(self, agent):
        corrections = (
            _corrections(15, segment="high_value_transactions")
            + _corrections(3, segment="international")
            + _corrections(40, segment="general")
        )
        event = _event(corrections=corrections)
        decision = await agent.run(event)
        patterns = decision.metadata["error_patterns"]
        # high_value (15) and general (40) exceed min_pattern_frequency=5
        assert len(patterns) >= 2
        segment_names = [p["segment"] for p in patterns]
        assert "high_value_transactions" in segment_names

    @pytest.mark.asyncio
    async def test_no_patterns_below_threshold(self, agent):
        corrections = (
            _corrections(2, segment="a")
            + _corrections(3, segment="b")
            + _corrections(4, segment="c")
        )
        event = _event(corrections=corrections)
        decision = await agent.run(event)
        assert len(decision.metadata["error_patterns"]) == 0


class TestFeedbackMetadata:
    @pytest.mark.asyncio
    async def test_metadata_complete(self, agent):
        event = _event(
            model_name="fraud-detector",
            corrections=_corrections(60, segment="high_value"),
            flags=[{"reason": "suspicious"}],
        )
        decision = await agent.run(event)
        assert decision.metadata["corrections_count"] == 60
        assert decision.metadata["flags_count"] == 1
        assert decision.metadata["total_feedback"] == 61
        assert decision.artifacts["model_name"] == "fraud-detector"
