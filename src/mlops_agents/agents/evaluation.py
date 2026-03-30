"""Evaluation Agent - model quality gate.

The Evaluation Agent is the go/no-go decision point between training
and deployment. It:
  1. Loads the candidate model and current champion
  2. Runs the evaluation suite (accuracy, fairness, latency)
  3. Compares candidate vs champion
  4. Decides: promote, reject, or escalate

This is the most impactful agent for demos because it shows
the full reasoning chain: observations -> analysis -> decision.
"""

from __future__ import annotations

from typing import Any

import structlog

from mlops_agents.core.agent import AgentContext, BaseAgent
from mlops_agents.core.decision import Decision
from mlops_agents.providers.protocols import ModelVersion

logger = structlog.get_logger()

# Default thresholds - can be overridden via stage config
DEFAULT_MIN_IMPROVEMENT = 0.005  # 0.5% F1 improvement required
DEFAULT_MAX_FAIRNESS_DELTA = 0.05  # Max 5% demographic parity gap
DEFAULT_MAX_LATENCY_P99_MS = 100.0  # Max 100ms p99 latency


class EvalAgent(BaseAgent):
    """Evaluates model quality and decides go/no-go for deployment.

    Authority scopes: model.evaluate, model.compare, model.register
    """

    name = "evaluation"
    authority = ["model.evaluate", "model.compare", "model.register"]
    description = "Evaluates model quality, compares against champion, decides promotion"

    def __init__(
        self,
        min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
        max_fairness_delta: float = DEFAULT_MAX_FAIRNESS_DELTA,
        max_latency_p99_ms: float = DEFAULT_MAX_LATENCY_P99_MS,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_improvement = min_improvement
        self.max_fairness_delta = max_fairness_delta
        self.max_latency_p99_ms = max_latency_p99_ms

    async def decide(self, ctx: AgentContext) -> Decision:
        payload = ctx.event.payload
        providers = ctx.providers or {}
        ml = providers.get("ml")

        # Extract candidate info from event payload
        candidate_run_id = payload.get("run_id", "")
        candidate_metrics = payload.get("metrics", {})
        candidate_model_name = payload.get("model_name", "default-model")
        candidate_artifact_uri = payload.get("artifact_uri", "")

        # --- 1. Get champion model for comparison ---
        champion: ModelVersion | None = None
        champion_metrics: dict[str, float] = {}

        if ml is not None:
            try:
                champion = await ml.get_champion(candidate_model_name)
                if champion is not None:
                    champion_metrics = champion.metrics
                    ctx.observe(
                        f"Current champion: {champion.model_name}/{champion.version} "
                        f"(F1: {champion_metrics.get('f1', 0):.4f})"
                    )
                else:
                    ctx.observe("No existing champion - this is the first model version")
            except Exception as e:
                ctx.observe(f"Could not load champion model: {e}")

        # --- 2. Evaluate candidate metrics ---
        f1 = candidate_metrics.get("f1", candidate_metrics.get("f1_score", 0.0))
        precision = candidate_metrics.get("precision", 0.0)
        recall = candidate_metrics.get("recall", 0.0)
        candidate_metrics.get("accuracy", 0.0)
        auc = candidate_metrics.get("auc_roc", candidate_metrics.get("auc", 0.0))

        ctx.observe(
            f"Candidate metrics - F1: {f1:.4f}, Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, AUC: {auc:.4f}"
        )

        # --- 3. Compare against champion ---
        is_first_model = champion is None
        improvement = 0.0

        if not is_first_model:
            champion_f1 = champion_metrics.get("f1", champion_metrics.get("f1_score", 0.0))
            improvement = f1 - champion_f1
            pct_improvement = (improvement / max(champion_f1, 0.001)) * 100

            ctx.observe(f"F1 delta: {improvement:+.4f} ({pct_improvement:+.1f}%) vs champion")
        else:
            ctx.observe("First model - no champion comparison needed")

        # --- 4. Fairness check ---
        has_fairness = (
            "fairness_delta" in candidate_metrics or "demographic_parity_diff" in candidate_metrics
        )
        fairness_delta = candidate_metrics.get(
            "fairness_delta", candidate_metrics.get("demographic_parity_diff", 0.0)
        )
        fairness_ok = abs(fairness_delta) <= self.max_fairness_delta

        if has_fairness:
            ctx.observe(
                f"Fairness: demographic parity delta {fairness_delta:.3f} "
                f"({'OK' if fairness_ok else 'VIOLATION'}, threshold: {self.max_fairness_delta})"
            )
        else:
            ctx.observe("Fairness: no fairness metrics provided (skipping check)")
            fairness_ok = True

        # --- 5. Latency check ---
        latency_p99 = candidate_metrics.get("latency_p99_ms", 0.0)
        latency_ok = latency_p99 <= self.max_latency_p99_ms or latency_p99 == 0.0

        if latency_p99 > 0:
            ctx.observe(
                f"Latency p99: {latency_p99:.1f}ms "
                f"({'OK' if latency_ok else 'VIOLATION'}, SLA: {self.max_latency_p99_ms}ms)"
            )

        # --- 6. Make decision ---
        if is_first_model:
            approved = f1 > 0.0  # Accept any non-zero first model
            action_taken = "model.register" if approved else "model.evaluate"
        else:
            meets_improvement = improvement >= self.min_improvement
            approved = meets_improvement and fairness_ok and latency_ok
            action_taken = "model.evaluate"

        # --- 7. LLM reasoning ---
        reasoning = await self.reason(
            observations=ctx.observations,
            context={
                "candidate_metrics": candidate_metrics,
                "champion_metrics": champion_metrics,
                "improvement": improvement,
                "is_first_model": is_first_model,
                "fairness_ok": fairness_ok,
                "latency_ok": latency_ok,
                "thresholds": {
                    "min_improvement": self.min_improvement,
                    "max_fairness_delta": self.max_fairness_delta,
                    "max_latency_p99_ms": self.max_latency_p99_ms,
                },
            },
            action=action_taken,
        )

        # --- 8. Build decision ---
        artifacts = {"candidate_artifact_uri": candidate_artifact_uri}
        if candidate_run_id:
            artifacts["candidate_run_id"] = candidate_run_id
        if champion is not None:
            artifacts["champion_version"] = champion.version

        return Decision(
            trace_id=ctx.trace_id,
            agent_name=self.name,
            action=action_taken,
            approved=approved,
            reasoning=reasoning,
            artifacts=artifacts,
            metadata={
                "candidate_metrics": candidate_metrics,
                "champion_metrics": champion_metrics,
                "improvement": improvement,
                "is_first_model": is_first_model,
                "model_name": candidate_model_name,
            },
        )
