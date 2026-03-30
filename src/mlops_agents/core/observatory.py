"""Observatory integration - agent observability via auth.kanoniv.com.

Registers MLOps agents with the Kanoniv Trust Observatory, delegates
scoped authority, signs every decision as provenance, and tracks
reputation via feedback signals.

Every agent decision becomes:
  1. A signed provenance entry (who did what, when, with what result)
  2. A feedback signal (reward_signal = confidence * approved)
  3. A memory entry (decision reasoning persisted for future context)

All visible in the Observatory dashboard at trust.kanoniv.com:
  - Agent list with capabilities and DIDs
  - Delegation chain (who has what scopes)
  - Provenance timeline (every decision, signed)
  - Trust graph (agent relationships)
  - Reputation trends (7d/30d feedback)

Usage:
    observatory = Observatory(api_key="kt_live_...")
    await observatory.setup_pipeline(pipeline_config)
    # Then pass observatory to Pipeline constructor
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from mlops_agents.core.decision import Decision

logger = structlog.get_logger()

# Agent name -> scopes mapping for MLOps agents
AGENT_SCOPES = {
    "cicd": ["data.validate", "pipeline.trigger", "test.run"],
    "evaluation": ["model.evaluate", "model.compare", "model.register"],
    "deployment": ["model.deploy.canary", "model.deploy.promote", "model.rollback"],
    "monitoring": ["metrics.collect", "drift.detect", "alert.send"],
    "retraining": ["model.retrain", "data.select", "experiment.configure"],
    "feedback": ["feedback.collect", "feedback.analyze", "label.validate"],
}

AGENT_DESCRIPTIONS = {
    "cicd": "Validates data quality and triggers training pipelines",
    "evaluation": "Evaluates model quality, compares against champion",
    "deployment": "Manages canary deployments with automated rollback",
    "monitoring": "Detects drift and performance degradation",
    "retraining": "Decides when and how to retrain models",
    "feedback": "Analyzes feedback and curates data for retraining",
}


class Observatory:
    """Bridge between mlops-agents and the Kanoniv Trust Observatory.

    Handles agent registration, delegation, provenance logging,
    and reputation tracking. Gracefully degrades if the Observatory
    is unreachable (logs warning, continues without observability).
    """

    def __init__(
        self,
        api_key: str | None = None,
        url: str = "https://auth.kanoniv.com",
        orchestrator_name: str = "mlops-orchestrator",
    ):
        self._api_key = api_key
        self._url = url
        self._orchestrator_name = orchestrator_name
        self._client = None
        self._agent_dids: dict[str, str] = {}
        self._enabled = api_key is not None

    def _get_client(self):
        if self._client is None:
            try:
                from kanoniv_auth.cloud import TrustClient

                self._client = TrustClient(api_key=self._api_key, url=self._url)
            except ImportError:
                logger.warning(
                    "observatory.import_failed",
                    msg="kanoniv-auth not installed. pip install kanoniv-auth",
                )
                self._enabled = False
                return None
        return self._client

    async def setup_pipeline(self, agent_names: list[str] | None = None) -> None:
        """Register all agents and set up delegations.

        Call this once at pipeline startup. Registers the orchestrator
        and all agents, then delegates scoped authority from the
        orchestrator to each agent.
        """
        if not self._enabled:
            return

        client = self._get_client()
        if client is None:
            return

        agents_to_register = agent_names or list(AGENT_SCOPES.keys())

        try:
            # Register orchestrator
            result = client.register(
                self._orchestrator_name,
                capabilities=["orchestrate", "delegate"],
                description="MLOps pipeline orchestrator",
            )
            self._agent_dids[self._orchestrator_name] = result.get("did", "")
            logger.info(
                "observatory.orchestrator_registered",
                name=self._orchestrator_name,
                did=result.get("did"),
            )

            # Register each agent
            for name in agents_to_register:
                scopes = AGENT_SCOPES.get(name, [])
                desc = AGENT_DESCRIPTIONS.get(name, f"MLOps {name} agent")

                result = client.register(
                    f"mlops-{name}",
                    capabilities=scopes,
                    description=desc,
                )
                agent_did = result.get("did", "")
                self._agent_dids[name] = agent_did

                # Delegate scopes from orchestrator to agent
                # Pass X-Agent-Name header so the server records the correct grantor
                from datetime import timedelta

                expires_at = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
                client._http.post(
                    "/delegations",
                    json={
                        "grantor_name": self._orchestrator_name,
                        "agent_name": f"mlops-{name}",
                        "scopes": scopes,
                        "expires_at": expires_at,
                    },
                    headers={"X-Agent-Name": self._orchestrator_name},
                )

                logger.info(
                    "observatory.agent_registered",
                    name=name,
                    did=agent_did,
                    scopes=scopes,
                )

        except Exception as e:
            logger.warning("observatory.setup_failed", error=str(e))
            # Don't fail the pipeline because of Observatory issues

    async def log_decision(self, decision: Decision) -> None:
        """Log a decision as signed provenance + feedback + memory."""
        if not self._enabled:
            return

        client = self._get_client()
        if client is None:
            return

        agent_name = f"mlops-{decision.agent_name}"

        try:
            # 1. Log provenance (the signed action record)
            client.action(
                agent_name=agent_name,
                action=decision.action,
                metadata={
                    "trace_id": decision.trace_id,
                    "decision_id": decision.id,
                    "approved": decision.approved,
                    "confidence": decision.reasoning.confidence,
                    "conclusion": decision.reasoning.conclusion,
                    "escalated": decision.escalate_to_human,
                    "artifacts": decision.artifacts,
                },
            )

            # 2. Log feedback (for reputation tracking)
            agent_did = self._agent_dids.get(decision.agent_name, "")
            if agent_did:
                # Reward signal: confidence * success indicator
                reward = (
                    decision.reasoning.confidence
                    if decision.approved
                    else (decision.reasoning.confidence * 0.5)
                )
                client.feedback(
                    agent_did=agent_did,
                    action=decision.action,
                    result="success" if decision.approved else "blocked",
                    reward_signal=reward,
                    content=decision.reasoning.conclusion,
                )

            # 3. Log as memory (for future context)
            client.memorize(
                agent_name=agent_name,
                title=f"{decision.action}: "
                f"{'GO' if decision.approved else 'NO-GO'} "
                f"({decision.reasoning.confidence:.0%})",
                content=decision.reasoning.conclusion,
                entry_type="decision",
                metadata={
                    "trace_id": decision.trace_id,
                    "observations": decision.reasoning.observations,
                },
            )

            logger.debug(
                "observatory.decision_logged",
                agent=decision.agent_name,
                action=decision.action,
                approved=decision.approved,
            )

        except Exception as e:
            logger.warning(
                "observatory.log_failed",
                agent=decision.agent_name,
                error=str(e),
            )

    async def log_pipeline_event(
        self,
        event_type: str,
        trace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log pipeline-level events (start, complete, fail)."""
        if not self._enabled:
            return

        client = self._get_client()
        if client is None:
            return

        try:
            client.action(
                agent_name=self._orchestrator_name,
                action=event_type,
                metadata={
                    "trace_id": trace_id,
                    **(metadata or {}),
                },
            )
        except Exception as e:
            logger.warning("observatory.pipeline_event_failed", error=str(e))
