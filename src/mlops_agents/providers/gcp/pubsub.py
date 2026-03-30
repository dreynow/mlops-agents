"""Cloud Pub/Sub event bus provider (Tier 2 - stub).

This provider will replace the local AsyncIO event bus with
Cloud Pub/Sub for distributed agent communication across
multiple processes or services.

Not yet implemented. The LocalAsyncEventBus covers single-process
pipelines. Contribute this provider if you need cross-process
event routing.

Architecture when implemented:
  - One topic per event type prefix (e.g. "model", "data", "feedback")
  - Subscriptions with filters for specific event types
  - Push delivery for low-latency, pull for batch
  - Dead letter topics for failed handler retries
  - Message ordering by trace_id for pipeline consistency
"""

from __future__ import annotations

from mlops_agents.core.event import Event, EventHandler


class PubSubEventBus:
    """Cloud Pub/Sub event bus (not yet implemented).

    Use LocalAsyncEventBus for local development. This provider is planned
    for distributed pipeline execution across multiple services.
    """

    def __init__(self, project: str, topic_prefix: str = "mlops-events"):
        self._project = project
        self._topic_prefix = topic_prefix

    async def publish(self, event: Event) -> None:
        raise NotImplementedError("PubSubEventBus is not yet implemented. Use LocalAsyncEventBus.")

    async def subscribe(self, pattern: str, handler: EventHandler) -> None:
        raise NotImplementedError("PubSubEventBus is not yet implemented. Use LocalAsyncEventBus.")

    async def unsubscribe(self, pattern: str, handler: EventHandler) -> None:
        raise NotImplementedError("PubSubEventBus is not yet implemented. Use LocalAsyncEventBus.")
