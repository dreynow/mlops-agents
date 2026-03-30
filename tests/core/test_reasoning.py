"""Tests for ReasoningEngine implementations."""

import pytest

from mlops_agents.core.decision import ReasoningTrace
from mlops_agents.core.reasoning import StaticReasoner, _parse_reasoning_json


class TestStaticReasoner:
    @pytest.mark.asyncio
    async def test_default_approved(self):
        reasoner = StaticReasoner()
        trace = await reasoner.reason(
            observations=["F1: 0.94", "Latency: 12ms"],
            context={"model": "fraud-v12"},
            agent_name="evaluation",
            action="model.evaluate",
        )
        assert isinstance(trace, ReasoningTrace)
        assert trace.confidence == 0.9
        assert "Proceed" in trace.conclusion
        assert trace.model_used == "static-reasoner"
        assert len(trace.observations) == 2

    @pytest.mark.asyncio
    async def test_default_rejected(self):
        reasoner = StaticReasoner(default_confidence=0.3, default_approved=False)
        trace = await reasoner.reason(
            observations=["Error rate: 15%"],
            context={},
            agent_name="deployment",
            action="model.deploy",
        )
        assert trace.confidence == 0.3
        assert "Block" in trace.conclusion

    @pytest.mark.asyncio
    async def test_observations_passed_through(self):
        reasoner = StaticReasoner()
        obs = ["obs1", "obs2", "obs3"]
        trace = await reasoner.reason(obs, {}, "test", "test.action")
        assert trace.observations == obs


class TestParseReasoningJson:
    def test_valid_json(self):
        text = """
        {
            "observations": ["F1 improved by 2.3%"],
            "analysis": "Clear improvement.",
            "conclusion": "Promote model.",
            "confidence": 0.92,
            "alternatives_considered": ["Keep current model"]
        }
        """
        trace = _parse_reasoning_json(text, "test-model")
        assert trace.confidence == 0.92
        assert trace.conclusion == "Promote model."
        assert trace.model_used == "test-model"
        assert len(trace.alternatives_considered) == 1

    def test_json_with_code_fences(self):
        text = """```json
        {
            "observations": ["test"],
            "analysis": "test",
            "conclusion": "test",
            "confidence": 0.8
        }
        ```"""
        trace = _parse_reasoning_json(text, "test-model")
        assert trace.confidence == 0.8

    def test_invalid_json_returns_zero_confidence(self):
        trace = _parse_reasoning_json("not valid json at all", "test-model")
        assert trace.confidence == 0.0
        assert "Parse error" in trace.conclusion

    def test_missing_fields_use_defaults(self):
        text = '{"observations": ["test"]}'
        trace = _parse_reasoning_json(text, "test-model")
        assert trace.observations == ["test"]
        assert trace.confidence == 0.5  # Default
        assert trace.analysis == ""
        assert trace.conclusion == ""
        assert trace.alternatives_considered == []

    def test_confidence_as_string(self):
        text = '{"observations": [], "confidence": "0.85", "analysis": "", "conclusion": ""}'
        trace = _parse_reasoning_json(text, "test-model")
        assert trace.confidence == 0.85
