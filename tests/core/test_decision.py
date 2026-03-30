"""Tests for Decision and ReasoningTrace."""

import pytest

from mlops_agents.core.decision import Decision, PipelineTrace, ReasoningTrace


class TestReasoningTrace:
    def test_create_basic(self):
        trace = ReasoningTrace(
            observations=["F1 score: 0.94", "No drift detected"],
            analysis="Model performance is strong.",
            conclusion="Proceed with deployment.",
            confidence=0.92,
            model_used="claude-sonnet-4-20250514",
        )
        assert trace.confidence == 0.92
        assert len(trace.observations) == 2
        assert trace.alternatives_considered == []

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            ReasoningTrace(
                observations=[],
                analysis="",
                conclusion="",
                confidence=1.5,  # Out of bounds
                model_used="test",
            )

        with pytest.raises(Exception):
            ReasoningTrace(
                observations=[],
                analysis="",
                conclusion="",
                confidence=-0.1,  # Out of bounds
                model_used="test",
            )

    def test_frozen(self):
        trace = ReasoningTrace(
            observations=["test"],
            analysis="test",
            conclusion="test",
            confidence=0.5,
            model_used="test",
        )
        with pytest.raises(Exception):
            trace.confidence = 0.9

    def test_serialization_roundtrip(self):
        trace = ReasoningTrace(
            observations=["obs1", "obs2"],
            analysis="analysis text",
            conclusion="conclusion text",
            confidence=0.85,
            alternatives_considered=["alt1"],
            model_used="gpt-4o",
        )
        json_str = trace.model_dump_json()
        restored = ReasoningTrace.model_validate_json(json_str)
        assert restored.confidence == trace.confidence
        assert restored.observations == trace.observations
        assert restored.model_used == "gpt-4o"


class TestDecision:
    def _make_trace(self, confidence: float = 0.9) -> ReasoningTrace:
        return ReasoningTrace(
            observations=["Test observation"],
            analysis="Test analysis",
            conclusion="Test conclusion",
            confidence=confidence,
            model_used="static-reasoner",
        )

    def test_create_basic(self):
        decision = Decision(
            trace_id="pipe-test-001",
            agent_name="evaluation",
            action="model.evaluate",
            approved=True,
            reasoning=self._make_trace(),
        )
        assert decision.approved is True
        assert decision.agent_name == "evaluation"
        assert decision.escalate_to_human is False
        assert decision.id  # Auto-generated

    def test_summary_approved(self):
        decision = Decision(
            trace_id="pipe-test-001",
            agent_name="evaluation",
            action="model.evaluate",
            approved=True,
            reasoning=self._make_trace(0.92),
        )
        summary = decision.summary()
        assert "GO" in summary
        assert "evaluation" in summary
        assert "92%" in summary

    def test_summary_rejected(self):
        decision = Decision(
            trace_id="pipe-test-001",
            agent_name="evaluation",
            action="model.evaluate",
            approved=False,
            reasoning=self._make_trace(0.3),
        )
        summary = decision.summary()
        assert "NO-GO" in summary

    def test_summary_escalated(self):
        decision = Decision(
            trace_id="pipe-test-001",
            agent_name="deployment",
            action="model.deploy",
            approved=True,
            reasoning=self._make_trace(0.6),
            escalate_to_human=True,
            escalation_reason="Low confidence",
        )
        assert decision.is_escalated()
        assert "ESCALATED" in decision.summary()

    def test_with_artifacts(self):
        decision = Decision(
            trace_id="pipe-test-001",
            agent_name="evaluation",
            action="model.evaluate",
            approved=True,
            reasoning=self._make_trace(),
            artifacts={
                "model_uri": "gs://bucket/model.pkl",
                "eval_report": "/reports/eval.html",
            },
        )
        assert "model_uri" in decision.artifacts
        assert decision.artifacts["model_uri"] == "gs://bucket/model.pkl"

    def test_frozen(self):
        decision = Decision(
            trace_id="pipe-test-001",
            agent_name="evaluation",
            action="model.evaluate",
            approved=True,
            reasoning=self._make_trace(),
        )
        with pytest.raises(Exception):
            decision.approved = False


class TestPipelineTrace:
    def _make_decision(self, agent: str, approved: bool = True) -> Decision:
        return Decision(
            trace_id="pipe-test-001",
            agent_name=agent,
            action=f"{agent}.action",
            approved=approved,
            reasoning=ReasoningTrace(
                observations=["obs"],
                analysis="analysis",
                conclusion="conclusion",
                confidence=0.9,
                model_used="test",
            ),
        )

    def test_create_and_add(self):
        trace = PipelineTrace(pipeline_name="fraud-detection")
        trace.add_decision(self._make_decision("cicd"))
        trace.add_decision(self._make_decision("evaluation"))
        assert len(trace.decisions) == 2
        assert trace.status == "running"

    def test_finalize(self):
        trace = PipelineTrace(pipeline_name="test")
        trace.add_decision(self._make_decision("cicd"))
        trace.finalize("completed")
        assert trace.status == "completed"
        assert trace.completed_at is not None

    def test_summary(self):
        trace = PipelineTrace(trace_id="pipe-abc", pipeline_name="test")
        trace.add_decision(self._make_decision("cicd"))
        trace.add_decision(self._make_decision("evaluation", approved=False))
        summary = trace.summary()
        assert "pipe-abc" in summary
        assert "cicd" in summary
        assert "evaluation" in summary
