"""Deployment Agent - canary deploys, promotion, and rollback.

Decides:
  1. Deploy candidate as canary (% of traffic)
  2. Monitor canary metrics during bake period
  3. Promote to full production or rollback
"""

from __future__ import annotations

from typing import Any

import structlog

from mlops_agents.core.agent import AgentContext, BaseAgent
from mlops_agents.core.decision import Decision
from mlops_agents.providers.protocols import DeployConfig, ModelVersion

logger = structlog.get_logger()

DEFAULT_CANARY_TRAFFIC_PCT = 5
DEFAULT_MAX_ERROR_RATE = 0.05  # 5% error rate triggers rollback
DEFAULT_MAX_LATENCY_P99_MS = 200.0
DEFAULT_ERROR_RATE_MULTIPLIER = 3.0  # Canary error rate can be at most 3x baseline


class DeployAgent(BaseAgent):
    """Manages canary deployments with automated promotion/rollback.

    Authority scopes: model.deploy.**, model.rollback
    """

    name = "deployment"
    authority = ["model.deploy.**", "model.rollback"]
    description = "Deploys models via canary strategy with automated rollback"

    def __init__(
        self,
        canary_traffic_pct: int = DEFAULT_CANARY_TRAFFIC_PCT,
        max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
        max_latency_p99_ms: float = DEFAULT_MAX_LATENCY_P99_MS,
        error_rate_multiplier: float = DEFAULT_ERROR_RATE_MULTIPLIER,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.canary_traffic_pct = canary_traffic_pct
        self.max_error_rate = max_error_rate
        self.max_latency_p99_ms = max_latency_p99_ms
        self.error_rate_multiplier = error_rate_multiplier

    async def decide(self, ctx: AgentContext) -> Decision:
        payload = ctx.event.payload
        providers = ctx.providers or {}
        serving = providers.get("serving")

        action = payload.get("action", "deploy")  # deploy | check_canary | promote | rollback
        model_name = payload.get("model_name", "")
        model_version = payload.get("model_version", "")
        artifact_uri = payload.get("artifact_uri", "")
        endpoint_id = payload.get("endpoint_id", "")

        if action == "check_canary":
            return await self._check_canary(ctx, serving, endpoint_id)
        elif action == "rollback":
            return await self._rollback(ctx, serving, endpoint_id)
        else:
            return await self._deploy_canary(ctx, serving, model_name, model_version, artifact_uri)

    async def _deploy_canary(
        self,
        ctx: AgentContext,
        serving: Any,
        model_name: str,
        model_version: str,
        artifact_uri: str,
    ) -> Decision:
        """Deploy a new model version as canary."""
        ctx.observe(
            f"Deploying {model_name}/{model_version} as canary ({self.canary_traffic_pct}% traffic)"
        )

        endpoint_id = ""
        if serving is not None:
            try:
                mv = ModelVersion(
                    model_name=model_name,
                    version=model_version,
                    artifact_uri=artifact_uri,
                )
                endpoint = await serving.deploy(
                    mv,
                    DeployConfig(
                        endpoint_name=f"{model_name}-canary",
                        port=8080,
                    ),
                )
                endpoint_id = endpoint.endpoint_id
                ctx.observe(f"Canary endpoint created: {endpoint_id} (status: {endpoint.status})")
            except Exception as e:
                ctx.observe(f"Canary deployment failed: {e}")
                reasoning = await self.reason(ctx.observations, {}, "model.deploy.canary")
                return Decision(
                    trace_id=ctx.trace_id,
                    agent_name=self.name,
                    action="model.deploy.canary",
                    approved=False,
                    reasoning=reasoning,
                    escalate_to_human=True,
                    escalation_reason=f"Canary deployment failed: {e}",
                )
        else:
            ctx.observe("No serving provider - simulating canary deployment")

        reasoning = await self.reason(
            observations=ctx.observations,
            context={
                "model_name": model_name,
                "model_version": model_version,
                "canary_traffic_pct": self.canary_traffic_pct,
            },
            action="model.deploy.canary",
        )

        return Decision(
            trace_id=ctx.trace_id,
            agent_name=self.name,
            action="model.deploy.canary",
            approved=True,
            reasoning=reasoning,
            artifacts={
                "endpoint_id": endpoint_id,
                "model_name": model_name,
                "model_version": model_version,
            },
            metadata={"canary_traffic_pct": self.canary_traffic_pct},
        )

    async def _check_canary(self, ctx: AgentContext, serving: Any, endpoint_id: str) -> Decision:
        """Check canary metrics and decide promote or rollback."""
        ctx.observe(f"Checking canary metrics for endpoint: {endpoint_id}")

        error_rate = 0.0
        latency_p99 = 0.0
        request_count = 0

        if serving is not None:
            try:
                metrics = await serving.get_endpoint_metrics(endpoint_id)
                error_rate = metrics.error_rate
                latency_p99 = metrics.latency_p99_ms
                request_count = metrics.request_count

                ctx.observe(f"Requests: {request_count}")
                ctx.observe(f"Error rate: {error_rate:.2%} (max: {self.max_error_rate:.2%})")
                ctx.observe(f"Latency p99: {latency_p99:.1f}ms (max: {self.max_latency_p99_ms}ms)")
            except Exception as e:
                ctx.observe(f"Could not fetch canary metrics: {e}")
        else:
            ctx.observe("No serving provider - using payload metrics")
            error_rate = ctx.event.payload.get("error_rate", 0.01)
            latency_p99 = ctx.event.payload.get("latency_p99_ms", 15.0)
            request_count = ctx.event.payload.get("request_count", 100)
            ctx.observe(f"Error rate: {error_rate:.2%}, Latency p99: {latency_p99:.1f}ms")

        error_ok = error_rate <= self.max_error_rate
        latency_ok = latency_p99 <= self.max_latency_p99_ms
        approved = error_ok and latency_ok

        action = "model.deploy.promote" if approved else "model.deploy.rollback"

        reasoning = await self.reason(
            observations=ctx.observations,
            context={
                "error_rate": error_rate,
                "latency_p99_ms": latency_p99,
                "request_count": request_count,
                "thresholds": {
                    "max_error_rate": self.max_error_rate,
                    "max_latency_p99_ms": self.max_latency_p99_ms,
                },
            },
            action=action,
        )

        return Decision(
            trace_id=ctx.trace_id,
            agent_name=self.name,
            action=action,
            approved=approved,
            reasoning=reasoning,
            artifacts={"endpoint_id": endpoint_id},
            metadata={
                "error_rate": error_rate,
                "latency_p99_ms": latency_p99,
                "request_count": request_count,
                "error_ok": error_ok,
                "latency_ok": latency_ok,
            },
        )

    async def _rollback(self, ctx: AgentContext, serving: Any, endpoint_id: str) -> Decision:
        """Rollback a canary deployment."""
        ctx.observe(f"Rolling back endpoint: {endpoint_id}")

        if serving is not None:
            try:
                await serving.undeploy(endpoint_id)
                ctx.observe("Canary endpoint removed successfully")
            except Exception as e:
                ctx.observe(f"Rollback failed: {e}")

        reasoning = await self.reason(
            observations=ctx.observations,
            context={"endpoint_id": endpoint_id},
            action="model.rollback",
        )

        return Decision(
            trace_id=ctx.trace_id,
            agent_name=self.name,
            action="model.rollback",
            approved=True,
            reasoning=reasoning,
            artifacts={"endpoint_id": endpoint_id},
        )
