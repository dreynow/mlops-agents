"""Pipeline orchestrator - DAG execution with event-driven agent handoffs.

Parses pipeline.yaml into a DAG, executes stages in topological order,
routes events between agents, handles success/failure branching,
and produces a full audit trail.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog

from mlops_agents.core.audit import AuditStore, SQLiteAuditStore
from mlops_agents.core.config import PipelineConfig, ReasoningConfig, StageConfig
from mlops_agents.core.decision import Decision, PipelineTrace
from mlops_agents.core.event import Event, EventBus, EventTypes, LocalAsyncEventBus
from mlops_agents.core.reasoning import (
    ClaudeReasoner,
    OllamaReasoner,
    OpenAIReasoner,
    ReasoningEngine,
    StaticReasoner,
)
from mlops_agents.providers.registry import ProviderRegistry, Providers

logger = structlog.get_logger()

# Agent name -> class mapping
AGENT_REGISTRY: dict[str, type] = {}


def _load_agent_registry() -> dict[str, type]:
    """Lazy-load agent classes to avoid circular imports."""
    if AGENT_REGISTRY:
        return AGENT_REGISTRY

    from mlops_agents.agents.cicd import CICDAgent
    from mlops_agents.agents.deployment import DeployAgent
    from mlops_agents.agents.evaluation import EvalAgent
    from mlops_agents.agents.feedback import FeedbackAgent
    from mlops_agents.agents.monitoring import MonitorAgent
    from mlops_agents.agents.retraining import RetrainAgent

    AGENT_REGISTRY.update(
        {
            "cicd": CICDAgent,
            "evaluation": EvalAgent,
            "deployment": DeployAgent,
            "monitoring": MonitorAgent,
            "retraining": RetrainAgent,
            "feedback": FeedbackAgent,
        }
    )
    return AGENT_REGISTRY


def _build_reasoner(config: ReasoningConfig) -> ReasoningEngine:
    """Create the right reasoning engine from config."""
    if config.engine == "claude":
        return ClaudeReasoner(model=config.model, api_key=config.api_key)
    elif config.engine == "openai":
        return OpenAIReasoner(model=config.model, api_key=config.api_key)
    elif config.engine == "ollama":
        return OllamaReasoner(model=config.model, host=config.host)
    elif config.engine == "static":
        return StaticReasoner()
    else:
        raise ValueError(f"Unknown reasoning engine: {config.engine}")


class Pipeline:
    """Executes a pipeline DAG by running agents stage-by-stage.

    The orchestrator:
      1. Parses stages from config
      2. Builds agent instances with shared event bus, audit store, providers
      3. Runs the entry stage
      4. Routes to next stages based on decision outcome (on_success / on_failure)
      5. Collects all decisions into a PipelineTrace
    """

    def __init__(
        self,
        config: PipelineConfig,
        event_bus: EventBus | None = None,
        audit_store: AuditStore | None = None,
        providers: Providers | None = None,
    ):
        self.config = config
        self.event_bus = event_bus or LocalAsyncEventBus()
        self.audit_store = audit_store or SQLiteAuditStore(db_path=config.audit.sqlite_path)

        # Build providers from config if not injected
        if providers is not None:
            self.providers = providers
        else:
            self.providers = ProviderRegistry.from_config(config.provider, event_bus=self.event_bus)

        self._reasoner = _build_reasoner(config.reasoning)
        self._agents: dict[str, Any] = {}
        self._build_agents()

    @classmethod
    def from_yaml(cls, path: str, **kwargs) -> Pipeline:
        config = PipelineConfig.from_yaml(path)
        return cls(config=config, **kwargs)

    def _build_agents(self) -> None:
        """Instantiate all agents referenced in the pipeline stages."""
        registry = _load_agent_registry()

        for stage_name, stage_config in self.config.stages.items():
            agent_name = stage_config.agent
            if agent_name in self._agents:
                continue

            agent_cls = registry.get(agent_name)
            if agent_cls is None:
                raise ValueError(
                    f"Unknown agent '{agent_name}' in stage '{stage_name}'. "
                    f"Available: {list(registry.keys())}"
                )

            # Pass stage-specific params to the agent constructor
            agent_kwargs = {
                "event_bus": self.event_bus,
                "audit_store": self.audit_store,
                "reasoning_engine": self._reasoner,
                "escalation_config": self.config.escalation,
            }
            # Merge stage params as constructor kwargs (agent ignores unknown ones via **kwargs)
            for k, v in stage_config.params.items():
                agent_kwargs[k] = v

            try:
                self._agents[agent_name] = agent_cls(**agent_kwargs)
            except TypeError as e:
                # If agent doesn't accept a param, fall back to no params
                logger.warning(
                    "pipeline.agent_init_fallback",
                    agent=agent_name,
                    error=str(e),
                )
                self._agents[agent_name] = agent_cls(
                    event_bus=self.event_bus,
                    audit_store=self.audit_store,
                    reasoning_engine=self._reasoner,
                    escalation_config=self.config.escalation,
                )

    async def run(
        self,
        entry_stage: str | None = None,
        initial_payload: dict[str, Any] | None = None,
        max_stages: int = 20,
    ) -> PipelineTrace:
        """Execute the pipeline starting from entry_stage.

        Args:
            entry_stage: First stage to run. Defaults to first stage in config.
            initial_payload: Payload for the initial event.
            max_stages: Safety limit to prevent infinite loops in cyclic DAGs.

        Returns:
            PipelineTrace with all decisions from this run.
        """
        trace_id = f"pipe-{uuid.uuid4().hex[:8]}"
        trace = PipelineTrace(
            trace_id=trace_id,
            pipeline_name=self.config.name,
        )

        # Save trace to audit store
        await self.audit_store.save_trace(trace)

        # Determine entry stage
        if entry_stage is None:
            stage_names = list(self.config.stages.keys())
            if not stage_names:
                trace.finalize("completed")
                return trace
            entry_stage = stage_names[0]

        log = logger.bind(pipeline=self.config.name, trace_id=trace_id)
        log.info("pipeline.start", entry_stage=entry_stage)

        # Publish pipeline started event
        await self.event_bus.publish(
            Event(
                type=EventTypes.PIPELINE_STARTED,
                source="orchestrator",
                trace_id=trace_id,
                payload={"pipeline": self.config.name, "entry_stage": entry_stage},
            )
        )

        # Execute stages
        current_stage = entry_stage
        payload = initial_payload or {}
        stages_executed = 0

        while current_stage and stages_executed < max_stages:
            stage_config = self.config.stages.get(current_stage)
            if stage_config is None:
                log.warning("pipeline.unknown_stage", stage=current_stage)
                break

            decision = await self._run_stage(
                stage_name=current_stage,
                stage_config=stage_config,
                trace_id=trace_id,
                payload=payload,
            )
            trace.add_decision(decision)
            stages_executed += 1

            # Route to next stage based on outcome
            if decision.escalate_to_human:
                log.info("pipeline.escalated", stage=current_stage)
                trace.finalize("escalated")
                await self.audit_store.save_trace(trace)
                return trace

            if decision.approved:
                next_stages = stage_config.on_success
            else:
                next_stages = stage_config.on_failure

            # Build payload for next stage: carry forward the original payload
            # merged with this decision's artifacts and metadata. Original keys
            # are preserved unless explicitly overwritten by the decision.
            payload = {
                **payload,  # Carry forward from previous stages
                **decision.artifacts,
                **decision.metadata,
                "previous_stage": current_stage,
                "previous_decision_id": decision.id,
            }

            if next_stages:
                # Execute first next stage (linear pipeline)
                # For parallel branching, would need async gather
                current_stage = next_stages[0]
                log.info(
                    "pipeline.next_stage",
                    from_stage=current_stage,
                    approved=decision.approved,
                    next=next_stages[0],
                )
            else:
                current_stage = None
                log.info("pipeline.stage_terminal", stage=current_stage)

        if stages_executed >= max_stages:
            log.warning("pipeline.max_stages_reached", max=max_stages)
            trace.finalize("failed")
        else:
            # Check if any decision was rejected
            any_failed = any(not d.approved for d in trace.decisions)
            trace.finalize("failed" if any_failed else "completed")

        log.info(
            "pipeline.complete",
            status=trace.status,
            stages_executed=stages_executed,
            decisions=len(trace.decisions),
        )

        await self.audit_store.save_trace(trace)

        # Publish pipeline completed event
        await self.event_bus.publish(
            Event(
                type=EventTypes.PIPELINE_COMPLETED
                if trace.status == "completed"
                else EventTypes.PIPELINE_FAILED,
                source="orchestrator",
                trace_id=trace_id,
                payload={
                    "status": trace.status,
                    "stages_executed": stages_executed,
                    "decisions": len(trace.decisions),
                },
            )
        )

        return trace

    async def _run_stage(
        self,
        stage_name: str,
        stage_config: StageConfig,
        trace_id: str,
        payload: dict[str, Any],
    ) -> Decision:
        """Execute a single pipeline stage."""
        agent = self._agents.get(stage_config.agent)
        if agent is None:
            raise ValueError(f"Agent '{stage_config.agent}' not found for stage '{stage_name}'")

        # Determine the event type from the agent's authority
        event_type = agent.authority[0] if agent.authority else stage_name

        event = Event(
            type=event_type,
            source="orchestrator",
            trace_id=trace_id,
            payload=payload,
        )

        logger.info("pipeline.stage.start", stage=stage_name, agent=stage_config.agent)

        # Build provider dict for the agent
        provider_dict = {
            "compute": self.providers.compute,
            "storage": self.providers.storage,
            "ml": self.providers.ml,
            "data": self.providers.data,
            "serving": self.providers.serving,
        }

        decision = await agent.run(event, providers=provider_dict)

        logger.info(
            "pipeline.stage.complete",
            stage=stage_name,
            approved=decision.approved,
            confidence=decision.reasoning.confidence,
        )

        return decision
