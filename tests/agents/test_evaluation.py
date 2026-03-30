"""Tests for the Evaluation Agent."""

import pytest

from mlops_agents.agents.evaluation import EvalAgent
from mlops_agents.core.audit import SQLiteAuditStore
from mlops_agents.core.config import EscalationConfig
from mlops_agents.core.event import Event, LocalAsyncEventBus
from mlops_agents.core.reasoning import StaticReasoner
from mlops_agents.providers.local.mlflow import LocalMLPlatform
from mlops_agents.providers.protocols import ModelArtifact


@pytest.fixture
def bus():
    return LocalAsyncEventBus()


@pytest.fixture
def store(tmp_path):
    return SQLiteAuditStore(db_path=tmp_path / "test.db")


@pytest.fixture
def ml(tmp_path):
    return LocalMLPlatform(base_dir=str(tmp_path / "ml"))


@pytest.fixture
def agent(bus, store):
    return EvalAgent(
        event_bus=bus,
        audit_store=store,
        reasoning_engine=StaticReasoner(default_confidence=0.92),
        escalation_config=EscalationConfig(default_confidence_threshold=0.7),
        min_improvement=0.005,
        max_fairness_delta=0.05,
        max_latency_p99_ms=100.0,
    )


def _candidate_event(
    f1=0.834,
    precision=0.891,
    recall=0.783,
    auc_roc=0.967,
    fairness_delta=0.02,
    latency_p99_ms=12.0,
    **extra_metrics,
):
    metrics = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc_roc": auc_roc,
        "fairness_delta": fairness_delta,
        "latency_p99_ms": latency_p99_ms,
        **extra_metrics,
    }
    return Event(
        type="model.evaluate",
        source="retraining-agent",
        trace_id="pipe-test",
        payload={
            "model_name": "fraud-detector",
            "run_id": "run-abc",
            "artifact_uri": "/models/v12/model.pkl",
            "metrics": metrics,
        },
    )


class TestEvalAgentFirstModel:
    """Tests for when there's no existing champion."""

    @pytest.mark.asyncio
    async def test_first_model_approved(self, agent):
        event = _candidate_event(f1=0.834)
        decision = await agent.run(event)

        assert decision.approved is True
        assert decision.agent_name == "evaluation"
        assert decision.metadata["is_first_model"] is True

    @pytest.mark.asyncio
    async def test_first_model_zero_f1_rejected(self, agent):
        event = _candidate_event(f1=0.0)
        decision = await agent.run(event)
        assert decision.approved is False


class TestEvalAgentWithChampion:
    """Tests for champion comparison."""

    @pytest.mark.asyncio
    async def test_improvement_approved(self, agent, ml):
        # Register champion
        await ml.register_model(
            ModelArtifact(
                model_name="fraud-detector",
                artifact_path="/v1.pkl",
                metrics={"f1": 0.821},
            )
        )
        await ml.promote_model("fraud-detector", "v1", "production")

        event = _candidate_event(f1=0.834)  # +0.013 improvement
        decision = await agent.run(event, providers={"ml": ml})

        assert decision.approved is True
        assert decision.metadata["improvement"] == pytest.approx(0.013, abs=0.001)

    @pytest.mark.asyncio
    async def test_regression_rejected(self, agent, ml):
        await ml.register_model(
            ModelArtifact(
                model_name="fraud-detector",
                artifact_path="/v1.pkl",
                metrics={"f1": 0.850},
            )
        )
        await ml.promote_model("fraud-detector", "v1", "production")

        event = _candidate_event(f1=0.834)  # -0.016 regression
        decision = await agent.run(event, providers={"ml": ml})

        assert decision.approved is False

    @pytest.mark.asyncio
    async def test_marginal_improvement_rejected(self, agent, ml):
        await ml.register_model(
            ModelArtifact(
                model_name="fraud-detector",
                artifact_path="/v1.pkl",
                metrics={"f1": 0.832},
            )
        )
        await ml.promote_model("fraud-detector", "v1", "production")

        # +0.002 is below min_improvement of 0.005
        event = _candidate_event(f1=0.834)
        decision = await agent.run(event, providers={"ml": ml})

        assert decision.approved is False


class TestEvalAgentFairness:
    @pytest.mark.asyncio
    async def test_fairness_violation_rejected(self, agent, ml):
        await ml.register_model(
            ModelArtifact(
                model_name="fraud-detector",
                artifact_path="/v1.pkl",
                metrics={"f1": 0.800},
            )
        )
        await ml.promote_model("fraud-detector", "v1", "production")

        event = _candidate_event(f1=0.850, fairness_delta=0.08)  # Above 0.05 threshold
        decision = await agent.run(event, providers={"ml": ml})

        assert decision.approved is False

    @pytest.mark.asyncio
    async def test_no_fairness_metrics_passes(self, agent, ml):
        await ml.register_model(
            ModelArtifact(
                model_name="fraud-detector",
                artifact_path="/v1.pkl",
                metrics={"f1": 0.800},
            )
        )
        await ml.promote_model("fraud-detector", "v1", "production")

        event = _candidate_event(f1=0.850, fairness_delta=0.0)
        decision = await agent.run(event, providers={"ml": ml})

        assert decision.approved is True


class TestEvalAgentLatency:
    @pytest.mark.asyncio
    async def test_latency_violation_rejected(self, agent, ml):
        await ml.register_model(
            ModelArtifact(
                model_name="fraud-detector",
                artifact_path="/v1.pkl",
                metrics={"f1": 0.800},
            )
        )
        await ml.promote_model("fraud-detector", "v1", "production")

        event = _candidate_event(f1=0.850, latency_p99_ms=150.0)  # Above 100ms SLA
        decision = await agent.run(event, providers={"ml": ml})

        assert decision.approved is False


class TestEvalAgentAuditTrail:
    @pytest.mark.asyncio
    async def test_decision_persisted(self, agent, store):
        event = _candidate_event()
        await agent.run(event)

        decisions = await store.get_trace("pipe-test")
        assert len(decisions) == 1
        assert decisions[0].agent_name == "evaluation"
        assert decisions[0].reasoning.confidence > 0

    @pytest.mark.asyncio
    async def test_artifacts_include_run_id(self, agent):
        event = _candidate_event()
        decision = await agent.run(event)
        assert decision.artifacts.get("candidate_run_id") == "run-abc"

    @pytest.mark.asyncio
    async def test_metadata_includes_metrics(self, agent):
        event = _candidate_event()
        decision = await agent.run(event)
        assert "candidate_metrics" in decision.metadata
        assert decision.metadata["candidate_metrics"]["f1"] == 0.834


class TestEvalAgentEvents:
    @pytest.mark.asyncio
    async def test_approved_emits_event(self, agent, bus):
        received = []

        async def handler(event: Event):
            received.append(event)

        await bus.subscribe("model.*.approved", handler)

        event = _candidate_event(f1=0.834)
        await agent.run(event)

        assert len(received) == 1
        assert received[0].payload["approved"] is True

    @pytest.mark.asyncio
    async def test_rejected_emits_event(self, agent, bus):
        received = []

        async def handler(event: Event):
            received.append(event)

        await bus.subscribe("model.evaluate.rejected", handler)

        event = _candidate_event(f1=0.0)
        await agent.run(event)

        assert len(received) == 1
        assert received[0].payload["approved"] is False
