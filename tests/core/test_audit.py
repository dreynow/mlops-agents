"""Tests for AuditStore."""

import pytest

from mlops_agents.core.audit import SQLiteAuditStore
from mlops_agents.core.decision import Decision, PipelineTrace, ReasoningTrace


def _make_trace(confidence: float = 0.9) -> ReasoningTrace:
    return ReasoningTrace(
        observations=["Test observation"],
        analysis="Test analysis",
        conclusion="Test conclusion",
        confidence=confidence,
        model_used="test",
    )


def _make_decision(
    trace_id: str = "pipe-test-001",
    agent_name: str = "evaluation",
    action: str = "model.evaluate",
    approved: bool = True,
) -> Decision:
    return Decision(
        trace_id=trace_id,
        agent_name=agent_name,
        action=action,
        approved=approved,
        reasoning=_make_trace(),
        artifacts={"model": "gs://test/model.pkl"},
    )


class TestSQLiteAuditStore:
    @pytest.fixture
    def store(self, tmp_path):
        return SQLiteAuditStore(db_path=tmp_path / "test_audit.db")

    @pytest.mark.asyncio
    async def test_log_and_retrieve(self, store):
        decision = _make_decision()
        await store.log_decision(decision)

        decisions = await store.get_trace("pipe-test-001")
        assert len(decisions) == 1
        assert decisions[0].agent_name == "evaluation"
        assert decisions[0].approved is True

    @pytest.mark.asyncio
    async def test_retrieve_empty_trace(self, store):
        decisions = await store.get_trace("nonexistent")
        assert decisions == []

    @pytest.mark.asyncio
    async def test_multiple_decisions_same_trace(self, store):
        await store.log_decision(_make_decision(agent_name="cicd", action="data.validate"))
        await store.log_decision(_make_decision(agent_name="evaluation", action="model.evaluate"))
        await store.log_decision(_make_decision(agent_name="deployment", action="model.deploy"))

        decisions = await store.get_trace("pipe-test-001")
        assert len(decisions) == 3
        agents = [d.agent_name for d in decisions]
        assert "cicd" in agents
        assert "evaluation" in agents
        assert "deployment" in agents

    @pytest.mark.asyncio
    async def test_get_by_agent(self, store):
        await store.log_decision(_make_decision(trace_id="t1", agent_name="eval"))
        await store.log_decision(_make_decision(trace_id="t2", agent_name="eval"))
        await store.log_decision(_make_decision(trace_id="t3", agent_name="deploy"))

        eval_decisions = await store.get_decisions_by_agent("eval")
        assert len(eval_decisions) == 2

        deploy_decisions = await store.get_decisions_by_agent("deploy")
        assert len(deploy_decisions) == 1

    @pytest.mark.asyncio
    async def test_get_recent(self, store):
        for i in range(5):
            await store.log_decision(_make_decision(trace_id=f"t{i}", agent_name=f"agent-{i}"))

        recent = await store.get_recent(limit=3)
        assert len(recent) == 3

    @pytest.mark.asyncio
    async def test_artifacts_persisted(self, store):
        decision = _make_decision()
        await store.log_decision(decision)

        decisions = await store.get_trace("pipe-test-001")
        assert decisions[0].artifacts["model"] == "gs://test/model.pkl"

    @pytest.mark.asyncio
    async def test_reasoning_persisted(self, store):
        decision = _make_decision()
        await store.log_decision(decision)

        decisions = await store.get_trace("pipe-test-001")
        reasoning = decisions[0].reasoning
        assert reasoning.confidence == 0.9
        assert reasoning.model_used == "test"
        assert len(reasoning.observations) == 1

    @pytest.mark.asyncio
    async def test_escalation_persisted(self, store):
        decision = Decision(
            trace_id="pipe-esc",
            agent_name="deploy",
            action="model.deploy",
            approved=True,
            reasoning=_make_trace(0.55),
            escalate_to_human=True,
            escalation_reason="Low confidence on canary metrics",
        )
        await store.log_decision(decision)

        decisions = await store.get_trace("pipe-esc")
        assert decisions[0].escalate_to_human is True
        assert "Low confidence" in decisions[0].escalation_reason

    @pytest.mark.asyncio
    async def test_save_and_get_pipeline_trace(self, store):
        trace = PipelineTrace(
            trace_id="pipe-full",
            pipeline_name="fraud-detection",
        )
        await store.save_trace(trace)

        await store.log_decision(_make_decision(trace_id="pipe-full", agent_name="cicd"))
        await store.log_decision(_make_decision(trace_id="pipe-full", agent_name="eval"))

        result = await store.get_pipeline_trace("pipe-full")
        assert result is not None
        assert result.pipeline_name == "fraud-detection"
        assert len(result.decisions) == 2

    @pytest.mark.asyncio
    async def test_get_nonexistent_pipeline_trace(self, store):
        result = await store.get_pipeline_trace("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_idempotent_schema_init(self, store):
        # Calling multiple operations should auto-init schema only once
        await store.log_decision(_make_decision())
        await store.log_decision(_make_decision(trace_id="t2"))
        decisions = await store.get_recent()
        assert len(decisions) == 2
