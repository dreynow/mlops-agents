"""Event system for agent communication.

Agents NEVER call each other directly. All coordination flows through
the EventBus. Events use a dot-separated taxonomy (e.g. "model.trained",
"drift.detected") and carry typed payloads.
"""

from __future__ import annotations

import asyncio
import fnmatch
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Protocol, runtime_checkable

from pydantic import BaseModel, Field

import structlog

logger = structlog.get_logger()


class Event(BaseModel):
    """A single event in the pipeline."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: str = Field(
        description="Dot-separated event type (e.g. 'model.trained', 'drift.detected')"
    )
    source: str = Field(
        description="Agent or system that emitted this event"
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific data"
    )
    trace_id: str = Field(
        default="",
        description="Pipeline trace ID for correlation"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = {"frozen": True}


# Type alias for event handlers
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


@runtime_checkable
class EventBus(Protocol):
    """Protocol for event bus implementations.

    Local: AsyncIO queue-based
    GCP: Cloud Pub/Sub
    AWS: SNS/SQS (future)
    """

    async def publish(self, event: Event) -> None: ...

    async def subscribe(self, pattern: str, handler: EventHandler) -> None: ...

    async def unsubscribe(self, pattern: str, handler: EventHandler) -> None: ...


class LocalAsyncEventBus:
    """In-process event bus using asyncio for local development.

    Supports glob-style pattern matching on event types:
      "model.*"       -> matches model.trained, model.evaluated, etc.
      "*.failed"      -> matches pipeline.failed, model.failed, etc.
      "model.deploy.*"-> matches model.deploy.canary, model.deploy.promote
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._history: list[Event] = []
        self._max_history: int = 1000

    async def publish(self, event: Event) -> None:
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        logger.info("event.published", type=event.type, source=event.source, trace_id=event.trace_id)

        matching_handlers = self._get_matching_handlers(event.type)
        tasks = [handler(event) for handler in matching_handlers]

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        "event.handler_error",
                        event_type=event.type,
                        handler=matching_handlers[i].__name__,
                        error=str(result),
                    )

    async def subscribe(self, pattern: str, handler: EventHandler) -> None:
        self._handlers[pattern].append(handler)
        logger.debug("event.subscribed", pattern=pattern, handler=handler.__name__)

    async def unsubscribe(self, pattern: str, handler: EventHandler) -> None:
        if pattern in self._handlers:
            self._handlers[pattern] = [
                h for h in self._handlers[pattern] if h is not handler
            ]

    def _get_matching_handlers(self, event_type: str) -> list[EventHandler]:
        handlers = []
        for pattern, pattern_handlers in self._handlers.items():
            if fnmatch.fnmatch(event_type, pattern):
                handlers.extend(pattern_handlers)
        return handlers

    @property
    def history(self) -> list[Event]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()


# Standard event types as constants to avoid typos
class EventTypes:
    # Data lifecycle
    DATA_VALIDATED = "data.validated"
    DATA_DRIFT_DETECTED = "data.drift_detected"
    DATA_NEW_BATCH = "data.new_batch"

    # Model lifecycle
    MODEL_TRAINED = "model.trained"
    MODEL_EVALUATED = "model.evaluated"
    MODEL_REGISTERED = "model.registered"
    MODEL_DEPLOYED = "model.deployed"
    MODEL_DEGRADED = "model.degraded"
    MODEL_ROLLED_BACK = "model.rolled_back"

    # Deployment
    DEPLOY_CANARY_STARTED = "model.deploy.canary_started"
    DEPLOY_PROMOTED = "model.deploy.promoted"
    DEPLOY_ROLLBACK = "model.deploy.rollback"

    # Feedback
    FEEDBACK_COLLECTED = "feedback.collected"
    FEEDBACK_ANALYZED = "feedback.analyzed"

    # Pipeline
    PIPELINE_STARTED = "pipeline.started"
    PIPELINE_COMPLETED = "pipeline.completed"
    PIPELINE_FAILED = "pipeline.failed"

    # Human-in-the-loop
    HUMAN_APPROVED = "human.approved"
    HUMAN_REJECTED = "human.rejected"
    HUMAN_ESCALATION = "human.escalation"
