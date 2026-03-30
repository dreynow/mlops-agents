"""Tests for BaseAgent."""

import pytest

from mlops_agents.core.agent import AgentContext, AuthorityError, BaseAgent
from mlops_agents.core.audit import SQLiteAuditStore
from mlops_agents.core.config import EscalationConfig
from mlops_agents.core.decision import Decision, ReasoningTrace
from mlops_agents.core.event import Event, LocalAsyncEventBus
from mlops_agents.core.reasoning import StaticReasoner


class MockEvalAgent(BaseAgent):
    """Concrete agent for testing."""

    name = "evaluation"
    authority = ["model.evaluate", "model.compare", "model.register"]
    description = "Evaluates model quality"

    async def decide(self, ctx: AgentContext) -> Decision:
        ctx.observe("F1 score: 0.94")
        ctx.observe("No fairness regression")

        trace = await self.reason(
            observations=ctx.observations,
            context={"model": "fraud-v12"},
            action=ctx.event.type,
        )

        return Decision(
            trace_id=ctx.trace_id,
            agent_name=self.name,
            action=ctx.event.type,
            approved=True,
            reasoning=trace,
            artifacts={"model_uri": "gs://test/model.pkl"},
        )


class MockDeployAgent(BaseAgent):
    """Deploy agent with hierarchical authority."""

    name = "deployment"
    authority = ["model.deploy.**", "model.rollback"]

    async def decide(self, ctx: AgentContext) -> Decision:
        trace = await self.reason([], {}, ctx.event.type)
        return Decision(
            trace_id=ctx.trace_id,
            agent_name=self.name,
            action=ctx.event.type,
            approved=True,
            reasoning=trace,
        )


class ErrorAgent(BaseAgent):
    """Agent that always raises an error in decide()."""

    name = "error-agent"
    authority = ["test.*"]

    async def decide(self, ctx: AgentContext) -> Decision:
        raise RuntimeError("Something went wrong")


class LowConfidenceAgent(BaseAgent):
    """Agent that returns low-confidence decisions."""

    name = "uncertain"
    authority = ["test.*"]

    async def decide(self, ctx: AgentContext) -> Decision:
        return Decision(
            trace_id=ctx.trace_id,
            agent_name=self.name,
            action=ctx.event.type,
            approved=True,
            reasoning=ReasoningTrace(
                observations=["Borderline metrics"],
                analysis="Unclear whether this is an improvement.",
                conclusion="Tentatively proceed.",
                confidence=0.45,
                model_used="test",
            ),
        )


@pytest.fixture
def bus():
    return LocalAsyncEventBus()


@pytest.fixture
def store(tmp_path):
    return SQLiteAuditStore(db_path=tmp_path / "test.db")


@pytest.fixture
def reasoner():
    return StaticReasoner()


class TestBaseAgentLifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, bus, store, reasoner):
        agent = MockEvalAgent(
            event_bus=bus,
            audit_store=store,
            reasoning_engine=reasoner,
        )

        event = Event(
            type="model.evaluate",
            source="training-agent",
            trace_id="pipe-test",
        )

        decision = await agent.run(event)

        assert decision.approved is True
        assert decision.agent_name == "evaluation"
        assert decision.trace_id == "pipe-test"
        assert decision.reasoning.confidence == 0.9

    @pytest.mark.asyncio
    async def test_decision_persisted_to_audit(self, bus, store, reasoner):
        agent = MockEvalAgent(event_bus=bus, audit_store=store, reasoning_engine=reasoner)
        event = Event(type="model.evaluate", source="test", trace_id="pipe-audit")

        await agent.run(event)

        decisions = await store.get_trace("pipe-audit")
        assert len(decisions) == 1
        assert decisions[0].agent_name == "evaluation"

    @pytest.mark.asyncio
    async def test_result_event_emitted(self, bus, store, reasoner):
        received = []

        async def handler(event: Event):
            received.append(event)

        await bus.subscribe("model.evaluate.approved", handler)

        agent = MockEvalAgent(event_bus=bus, audit_store=store, reasoning_engine=reasoner)
        event = Event(type="model.evaluate", source="test", trace_id="pipe-emit")

        await agent.run(event)

        assert len(received) == 1
        assert received[0].type == "model.evaluate.approved"
        assert received[0].payload["approved"] is True

    @pytest.mark.asyncio
    async def test_auto_trace_id_when_missing(self, bus, store, reasoner):
        agent = MockEvalAgent(event_bus=bus, audit_store=store, reasoning_engine=reasoner)
        event = Event(type="model.evaluate", source="test")  # No trace_id

        decision = await agent.run(event)
        assert decision.trace_id.startswith("pipe-")


class TestAuthorityValidation:
    @pytest.mark.asyncio
    async def test_exact_match(self, bus, store, reasoner):
        agent = MockEvalAgent(event_bus=bus, audit_store=store, reasoning_engine=reasoner)
        event = Event(type="model.evaluate", source="test")
        decision = await agent.run(event)
        assert decision.approved  # No AuthorityError

    @pytest.mark.asyncio
    async def test_unauthorized_action(self, bus, store, reasoner):
        agent = MockEvalAgent(event_bus=bus, audit_store=store, reasoning_engine=reasoner)
        event = Event(type="model.deploy", source="test")

        with pytest.raises(AuthorityError, match="lacks authority"):
            await agent.run(event)

    @pytest.mark.asyncio
    async def test_wildcard_authority(self, bus, store, reasoner):
        agent = MockDeployAgent(event_bus=bus, audit_store=store, reasoning_engine=reasoner)

        # model.deploy.* should match model.deploy.canary
        event = Event(type="model.deploy.canary", source="test")
        decision = await agent.run(event)
        assert decision.approved

    @pytest.mark.asyncio
    async def test_wildcard_no_match(self, bus, store, reasoner):
        agent = MockDeployAgent(event_bus=bus, audit_store=store, reasoning_engine=reasoner)

        # model.deploy.* should NOT match model.evaluate
        event = Event(type="model.evaluate", source="test")
        with pytest.raises(AuthorityError):
            await agent.run(event)

    @pytest.mark.asyncio
    async def test_exact_plus_wildcard(self, bus, store, reasoner):
        agent = MockDeployAgent(event_bus=bus, audit_store=store, reasoning_engine=reasoner)

        # "model.rollback" is exact in authority list
        event = Event(type="model.rollback", source="test")
        decision = await agent.run(event)
        assert decision.approved


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_decide_error_produces_nogo_decision(self, bus, store, reasoner):
        agent = ErrorAgent(event_bus=bus, audit_store=store, reasoning_engine=reasoner)
        event = Event(type="test.action", source="test", trace_id="pipe-err")

        decision = await agent.run(event)

        assert decision.approved is False
        assert decision.escalate_to_human is True
        assert "Something went wrong" in decision.reasoning.observations[0]

    @pytest.mark.asyncio
    async def test_error_decision_persisted(self, bus, store, reasoner):
        agent = ErrorAgent(event_bus=bus, audit_store=store, reasoning_engine=reasoner)
        event = Event(type="test.action", source="test", trace_id="pipe-err2")

        await agent.run(event)

        decisions = await store.get_trace("pipe-err2")
        assert len(decisions) == 1
        assert decisions[0].approved is False


class TestEscalation:
    @pytest.mark.asyncio
    async def test_low_confidence_triggers_escalation(self, bus, store):
        escalation = EscalationConfig(default_confidence_threshold=0.7)
        agent = LowConfidenceAgent(
            event_bus=bus,
            audit_store=store,
            reasoning_engine=StaticReasoner(),
            escalation_config=escalation,
        )
        event = Event(type="test.action", source="test", trace_id="pipe-esc")

        decision = await agent.run(event)

        assert decision.escalate_to_human is True
        assert "45%" in decision.escalation_reason
        assert "70%" in decision.escalation_reason

    @pytest.mark.asyncio
    async def test_high_confidence_no_escalation(self, bus, store, reasoner):
        escalation = EscalationConfig(default_confidence_threshold=0.7)
        agent = MockEvalAgent(
            event_bus=bus,
            audit_store=store,
            reasoning_engine=reasoner,  # StaticReasoner defaults to 0.9
            escalation_config=escalation,
        )
        event = Event(type="model.evaluate", source="test")

        decision = await agent.run(event)
        assert decision.escalate_to_human is False

    @pytest.mark.asyncio
    async def test_per_stage_threshold(self, bus, store):
        escalation = EscalationConfig(
            default_confidence_threshold=0.5,
            per_stage={"uncertain": 0.9},  # Higher threshold for this agent
        )
        agent = LowConfidenceAgent(
            event_bus=bus,
            audit_store=store,
            reasoning_engine=StaticReasoner(),
            escalation_config=escalation,
        )
        event = Event(type="test.action", source="test")

        decision = await agent.run(event)
        assert decision.escalate_to_human is True  # 0.45 < 0.9

    @pytest.mark.asyncio
    async def test_escalation_event_emitted(self, bus, store):
        received = []

        async def handler(event: Event):
            received.append(event)

        await bus.subscribe("human.escalation", handler)

        escalation = EscalationConfig(default_confidence_threshold=0.7)
        agent = LowConfidenceAgent(
            event_bus=bus,
            audit_store=store,
            reasoning_engine=StaticReasoner(),
            escalation_config=escalation,
        )
        event = Event(type="test.action", source="test")

        await agent.run(event)

        assert len(received) == 1
        assert received[0].type == "human.escalation"
        assert received[0].payload["escalated"] is True


class TestAgentContext:
    def test_observe(self):
        ctx = AgentContext(
            event=Event(type="test", source="test"),
            trace_id="pipe-ctx",
        )
        ctx.observe("F1: 0.94")
        ctx.observe("Latency: 12ms")

        assert len(ctx.observations) == 2
        assert ctx.observations[0] == "F1: 0.94"

    def test_observations_returns_copy(self):
        ctx = AgentContext(
            event=Event(type="test", source="test"),
            trace_id="pipe-ctx",
        )
        ctx.observe("obs1")
        obs = ctx.observations
        obs.append("obs2")  # Mutating the copy
        assert len(ctx.observations) == 1  # Original unchanged
