"""BaseAgent - the core abstraction every agent extends.

The lifecycle:
  1. Receive event
  2. Validate authority (Kanoniv delegation if configured, local scope check otherwise)
  3. Gather context via providers
  4. Call decide() - subclass implements
  5. Log decision to audit store
  6. Emit result events

Agents NEVER call each other directly. All communication is event-driven.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any

import structlog

from mlops_agents.core.audit import AuditStore, SQLiteAuditStore
from mlops_agents.core.config import EscalationConfig
from mlops_agents.core.decision import Decision, ReasoningTrace
from mlops_agents.core.event import Event, EventBus, LocalAsyncEventBus
from mlops_agents.core.reasoning import ReasoningEngine, StaticReasoner

logger = structlog.get_logger()


class AuthorityError(Exception):
    """Agent attempted an action outside its delegated scope."""


class AgentContext:
    """Context passed to decide() - everything the agent needs to make a decision.

    Agents receive this instead of raw provider access to maintain
    the principle of least privilege.
    """

    def __init__(
        self,
        event: Event,
        trace_id: str,
        providers: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.event = event
        self.trace_id = trace_id
        self.providers = providers or {}
        self.config = config or {}
        self._observations: list[str] = []

    def observe(self, observation: str) -> None:
        """Record an observation for the reasoning trace."""
        self._observations.append(observation)

    @property
    def observations(self) -> list[str]:
        return list(self._observations)


class BaseAgent(ABC):
    """Abstract base for all MLOps agents.

    Subclasses implement decide() with their domain logic.
    The base class handles authority validation, audit logging,
    and event emission.
    """

    name: str = "base"
    authority: list[str] = []
    description: str = ""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        audit_store: AuditStore | None = None,
        reasoning_engine: ReasoningEngine | None = None,
        escalation_config: EscalationConfig | None = None,
        kanoniv_client: Any | None = None,
    ):
        self.event_bus: EventBus = event_bus or LocalAsyncEventBus()
        self.audit_store: AuditStore = audit_store or SQLiteAuditStore()
        self.reasoning_engine: ReasoningEngine = reasoning_engine or StaticReasoner()
        self.escalation_config = escalation_config or EscalationConfig()
        self._kanoniv = kanoniv_client

    async def run(self, event: Event, providers: dict[str, Any] | None = None) -> Decision:
        """Full agent lifecycle: validate -> decide -> audit -> emit."""
        trace_id = event.trace_id or f"pipe-{uuid.uuid4().hex[:8]}"

        log = logger.bind(agent=self.name, action=event.type, trace_id=trace_id)
        log.info("agent.run.start")

        # 1. Validate authority
        await self._validate_authority(event.type)

        # 2. Build context
        ctx = AgentContext(
            event=event,
            trace_id=trace_id,
            providers=providers,
        )

        # 3. Call decide() - subclass implements
        try:
            decision = await self.decide(ctx)
        except Exception as e:
            log.error("agent.decide.error", error=str(e))
            decision = self._error_decision(trace_id, event.type, e)

        # 4. Check if escalation is needed
        threshold = self.escalation_config.threshold_for(self.name)
        if decision.reasoning.confidence < threshold and not decision.escalate_to_human:
            decision = decision.model_copy(update={
                "escalate_to_human": True,
                "escalation_reason": (
                    f"Confidence {decision.reasoning.confidence:.0%} below "
                    f"threshold {threshold:.0%} for stage '{self.name}'"
                ),
            })

        # 5. Log to audit store
        await self.audit_store.log_decision(decision)

        # 6. Emit result event
        await self._emit_result(decision)

        log.info(
            "agent.run.complete",
            approved=decision.approved,
            confidence=decision.reasoning.confidence,
            escalated=decision.escalate_to_human,
        )

        return decision

    @abstractmethod
    async def decide(self, ctx: AgentContext) -> Decision:
        """Subclass implements: gather observations, reason, return Decision.

        Pattern:
            1. Use ctx.providers to gather data (metrics, models, etc.)
            2. Call ctx.observe() for each key finding
            3. Call self.reason() to get LLM reasoning
            4. Build and return a Decision
        """
        ...

    async def reason(
        self,
        observations: list[str],
        context: dict[str, Any],
        action: str,
    ) -> ReasoningTrace:
        """Convenience wrapper around the reasoning engine."""
        return await self.reasoning_engine.reason(
            observations=observations,
            context=context,
            agent_name=self.name,
            action=action,
        )

    async def _validate_authority(self, action: str) -> None:
        """Check if this agent has authority to handle this action.

        If Kanoniv is configured, delegates to cryptographic scope verification.
        Otherwise falls back to local authority list check.
        """
        if self._kanoniv is not None:
            await self._validate_kanoniv_authority(action)
        else:
            self._validate_local_authority(action)

    def _validate_local_authority(self, action: str) -> None:
        """Local authority check - matches action against declared scopes.

        Matching rules:
          - Exact match: "model.evaluate" matches "model.evaluate"
          - Glob wildcard: "model.deploy.*" matches "model.deploy.canary"
            (fnmatch semantics - "*" does NOT cross dots)
          - Use "model.**" or prefix check if you need deep matching

        For deep hierarchical matching (scope covers all sub-actions),
        use a trailing ".**" pattern or list each sub-scope explicitly.
        """
        if not self.authority:
            return  # No restrictions declared

        import fnmatch

        for scope in self.authority:
            if scope == action:
                return
            # Deep wildcard: "model.deploy.**" matches "model.deploy.canary.blue"
            if scope.endswith(".**"):
                prefix = scope[:-3]
                if action.startswith(prefix + ".") or action == prefix:
                    return
            # Standard glob: "model.deploy.*" matches "model.deploy.canary"
            elif "*" in scope:
                if fnmatch.fnmatch(action, scope):
                    return

        raise AuthorityError(
            f"Agent '{self.name}' lacks authority for '{action}'. "
            f"Declared scopes: {self.authority}"
        )

    async def _validate_kanoniv_authority(self, action: str) -> None:
        """Kanoniv-backed authority verification.

        Uses the Kanoniv delegation system for cryptographic scope
        enforcement. Each agent has a delegation token with specific
        scopes - the framework verifies before every action.
        """
        try:
            result = self._kanoniv.verify_scope(action)
            if not result.authorized:
                raise AuthorityError(
                    f"Kanoniv denied '{action}' for agent '{self.name}': {result.reason}"
                )
            logger.debug(
                "agent.authority.kanoniv_verified",
                agent=self.name,
                action=action,
                delegation_id=result.delegation_id,
            )
        except AttributeError:
            # Kanoniv client doesn't have verify_scope - fall back to local
            logger.warning("agent.authority.kanoniv_fallback", agent=self.name)
            self._validate_local_authority(action)

    async def _emit_result(self, decision: Decision) -> None:
        """Emit an event based on the decision outcome."""
        if decision.escalate_to_human:
            event_type = "human.escalation"
        elif decision.approved:
            event_type = f"{decision.action}.approved"
        else:
            event_type = f"{decision.action}.rejected"

        event = Event(
            type=event_type,
            source=self.name,
            payload={
                "decision_id": decision.id,
                "action": decision.action,
                "approved": decision.approved,
                "confidence": decision.reasoning.confidence,
                "conclusion": decision.reasoning.conclusion,
                "artifacts": decision.artifacts,
                "escalated": decision.escalate_to_human,
            },
            trace_id=decision.trace_id,
        )
        await self.event_bus.publish(event)

    def _error_decision(self, trace_id: str, action: str, error: Exception) -> Decision:
        """Create a NO-GO decision when decide() raises an exception."""
        return Decision(
            trace_id=trace_id,
            agent_name=self.name,
            action=action,
            approved=False,
            reasoning=ReasoningTrace(
                observations=[f"Agent error: {type(error).__name__}: {error}"],
                analysis="Agent encountered an unhandled error during decision-making.",
                conclusion=f"Blocking due to error: {error}",
                confidence=0.0,
                alternatives_considered=[],
                model_used="error-handler",
            ),
            escalate_to_human=True,
            escalation_reason=f"Unhandled error in {self.name}: {error}",
        )
