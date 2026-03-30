"""Tests for Event and EventBus."""

import pytest

from mlops_agents.core.event import Event, EventTypes, LocalAsyncEventBus


class TestEvent:
    def test_create_basic(self):
        event = Event(
            type="model.trained",
            source="training-agent",
            payload={"model_uri": "gs://bucket/model.pkl"},
            trace_id="pipe-001",
        )
        assert event.type == "model.trained"
        assert event.source == "training-agent"
        assert event.id  # Auto-generated
        assert event.timestamp  # Auto-generated

    def test_frozen(self):
        event = Event(type="test", source="test")
        with pytest.raises(Exception):
            event.type = "modified"


class TestEventTypes:
    def test_standard_types_exist(self):
        assert EventTypes.MODEL_TRAINED == "model.trained"
        assert EventTypes.MODEL_EVALUATED == "model.evaluated"
        assert EventTypes.DATA_VALIDATED == "data.validated"
        assert EventTypes.PIPELINE_FAILED == "pipeline.failed"
        assert EventTypes.HUMAN_APPROVED == "human.approved"
        assert EventTypes.FEEDBACK_COLLECTED == "feedback.collected"


class TestLocalAsyncEventBus:
    @pytest.fixture
    def bus(self):
        return LocalAsyncEventBus()

    @pytest.mark.asyncio
    async def test_publish_subscribe_exact(self, bus):
        received = []

        async def handler(event: Event):
            received.append(event)

        await bus.subscribe("model.trained", handler)
        event = Event(type="model.trained", source="test")
        await bus.publish(event)

        assert len(received) == 1
        assert received[0].type == "model.trained"

    @pytest.mark.asyncio
    async def test_publish_no_match(self, bus):
        received = []

        async def handler(event: Event):
            received.append(event)

        await bus.subscribe("model.trained", handler)
        await bus.publish(Event(type="data.validated", source="test"))

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_wildcard_subscribe(self, bus):
        received = []

        async def handler(event: Event):
            received.append(event)

        await bus.subscribe("model.*", handler)

        await bus.publish(Event(type="model.trained", source="test"))
        await bus.publish(Event(type="model.evaluated", source="test"))
        await bus.publish(Event(type="data.validated", source="test"))

        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_deep_wildcard(self, bus):
        received = []

        async def handler(event: Event):
            received.append(event)

        await bus.subscribe("model.deploy.*", handler)

        await bus.publish(Event(type="model.deploy.canary_started", source="test"))
        await bus.publish(Event(type="model.deploy.promoted", source="test"))
        await bus.publish(Event(type="model.trained", source="test"))

        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_multiple_handlers(self, bus):
        received_a = []
        received_b = []

        async def handler_a(event: Event):
            received_a.append(event)

        async def handler_b(event: Event):
            received_b.append(event)

        await bus.subscribe("model.trained", handler_a)
        await bus.subscribe("model.trained", handler_b)

        await bus.publish(Event(type="model.trained", source="test"))

        assert len(received_a) == 1
        assert len(received_b) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self, bus):
        received = []

        async def handler(event: Event):
            received.append(event)

        await bus.subscribe("model.trained", handler)
        await bus.publish(Event(type="model.trained", source="test"))
        assert len(received) == 1

        await bus.unsubscribe("model.trained", handler)
        await bus.publish(Event(type="model.trained", source="test"))
        assert len(received) == 1  # No new events

    @pytest.mark.asyncio
    async def test_history(self, bus):
        await bus.publish(Event(type="a", source="test"))
        await bus.publish(Event(type="b", source="test"))
        await bus.publish(Event(type="c", source="test"))

        assert len(bus.history) == 3
        assert bus.history[0].type == "a"
        assert bus.history[2].type == "c"

    @pytest.mark.asyncio
    async def test_history_limit(self, bus):
        bus._max_history = 5
        for i in range(10):
            await bus.publish(Event(type=f"event.{i}", source="test"))

        assert len(bus.history) == 5
        assert bus.history[0].type == "event.5"

    @pytest.mark.asyncio
    async def test_clear_history(self, bus):
        await bus.publish(Event(type="a", source="test"))
        bus.clear_history()
        assert len(bus.history) == 0

    @pytest.mark.asyncio
    async def test_handler_error_doesnt_break_others(self, bus):
        received = []

        async def bad_handler(event: Event):
            raise ValueError("boom")

        async def good_handler(event: Event):
            received.append(event)

        await bus.subscribe("test", bad_handler)
        await bus.subscribe("test", good_handler)

        await bus.publish(Event(type="test", source="test"))

        # Good handler still ran despite bad handler raising
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_trace_id_propagation(self, bus):
        received = []

        async def handler(event: Event):
            received.append(event)

        await bus.subscribe("model.trained", handler)

        event = Event(type="model.trained", source="test", trace_id="pipe-abc123")
        await bus.publish(event)

        assert received[0].trace_id == "pipe-abc123"
