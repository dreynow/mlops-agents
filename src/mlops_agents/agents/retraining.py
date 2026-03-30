"""Retraining Agent - decides when, how, and on what data to retrain.

Decides:
  1. Should we retrain? (enough signal from drift/degradation/feedback)
  2. Full retrain or incremental fine-tune?
  3. What data window?
  4. What hyperparameter strategy?
"""

from __future__ import annotations

from typing import Any

import structlog

from mlops_agents.core.agent import AgentContext, BaseAgent
from mlops_agents.core.decision import Decision

logger = structlog.get_logger()

DEFAULT_MIN_FEEDBACK_SAMPLES = 50
DEFAULT_MIN_DRIFTED_FEATURES = 1
DEFAULT_FULL_RETRAIN_DRIFT_RATIO = 0.3  # >30% features drifted = full retrain


class RetrainAgent(BaseAgent):
    """Decides when and how to retrain models.

    Authority scopes: model.retrain, data.select, experiment.configure
    """

    name = "retraining"
    authority = ["model.retrain", "data.select", "experiment.configure"]
    description = "Evaluates retrain triggers and configures training strategy"

    def __init__(
        self,
        min_feedback_samples: int = DEFAULT_MIN_FEEDBACK_SAMPLES,
        min_drifted_features: int = DEFAULT_MIN_DRIFTED_FEATURES,
        full_retrain_drift_ratio: float = DEFAULT_FULL_RETRAIN_DRIFT_RATIO,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_feedback_samples = min_feedback_samples
        self.min_drifted_features = min_drifted_features
        self.full_retrain_drift_ratio = full_retrain_drift_ratio

    async def decide(self, ctx: AgentContext) -> Decision:
        payload = ctx.event.payload
        providers = ctx.providers or {}
        compute = providers.get("compute")

        trigger_source = payload.get("trigger_source", "manual")  # drift | feedback | degradation | manual
        model_name = payload.get("model_name", "")

        # Drift info (from monitoring agent)
        drifted_features = payload.get("drifted_features", [])
        total_features = payload.get("total_features", 0)
        psi_scores = payload.get("psi_scores", {})

        # Feedback info (from feedback agent)
        feedback_count = payload.get("feedback_count", 0)
        error_patterns = payload.get("error_patterns", [])

        # Performance info
        accuracy_drop = payload.get("accuracy_drop", 0.0)
        current_accuracy = payload.get("current_accuracy", 0.0)

        ctx.observe(f"Retrain trigger: {trigger_source} for model '{model_name}'")

        # --- 1. Assess retrain necessity ---
        should_retrain = False
        retrain_reasons = []

        if drifted_features:
            drift_ratio = len(drifted_features) / max(total_features, 1)
            ctx.observe(
                f"Drift: {len(drifted_features)}/{total_features} features drifted "
                f"({drift_ratio:.0%})"
            )
            if len(drifted_features) >= self.min_drifted_features:
                should_retrain = True
                retrain_reasons.append(f"{len(drifted_features)} features drifted")

        if feedback_count > 0:
            ctx.observe(f"Feedback: {feedback_count} new labeled samples")
            if feedback_count >= self.min_feedback_samples:
                should_retrain = True
                retrain_reasons.append(f"{feedback_count} feedback samples")

        if accuracy_drop > 0:
            ctx.observe(f"Performance: accuracy dropped by {accuracy_drop:.4f}")
            should_retrain = True
            retrain_reasons.append(f"accuracy dropped {accuracy_drop:.4f}")

        if error_patterns:
            ctx.observe(f"Error patterns: {len(error_patterns)} systematic failure patterns")
            should_retrain = True
            retrain_reasons.append(f"{len(error_patterns)} error patterns")

        if trigger_source == "manual":
            should_retrain = True
            retrain_reasons.append("manual trigger")

        if not should_retrain:
            ctx.observe("Insufficient signal for retraining")

        # --- 2. Determine strategy: full retrain vs fine-tune ---
        strategy = "skip"
        data_window = "none"

        if should_retrain:
            drift_ratio = len(drifted_features) / max(total_features, 1) if total_features > 0 else 0

            if drift_ratio > self.full_retrain_drift_ratio:
                strategy = "full_retrain"
                data_window = "all"
                ctx.observe(
                    f"Strategy: FULL RETRAIN (drift ratio {drift_ratio:.0%} > "
                    f"{self.full_retrain_drift_ratio:.0%} threshold)"
                )
            elif feedback_count > 0:
                strategy = "fine_tune"
                data_window = "recent_30d_plus_feedback"
                ctx.observe(
                    f"Strategy: FINE-TUNE on recent 30 days + {feedback_count} feedback samples"
                )
            else:
                strategy = "full_retrain"
                data_window = "recent_90d"
                ctx.observe("Strategy: FULL RETRAIN on recent 90 days")

        # --- 3. LLM reasoning ---
        reasoning = await self.reason(
            observations=ctx.observations,
            context={
                "trigger_source": trigger_source,
                "should_retrain": should_retrain,
                "retrain_reasons": retrain_reasons,
                "strategy": strategy,
                "data_window": data_window,
                "drifted_features": drifted_features,
                "feedback_count": feedback_count,
                "accuracy_drop": accuracy_drop,
            },
            action="model.retrain",
        )

        return Decision(
            trace_id=ctx.trace_id,
            agent_name=self.name,
            action="model.retrain",
            approved=should_retrain,
            reasoning=reasoning,
            artifacts={"model_name": model_name},
            metadata={
                "strategy": strategy,
                "data_window": data_window,
                "retrain_reasons": retrain_reasons,
                "trigger_source": trigger_source,
                "drifted_features": drifted_features,
                "feedback_count": feedback_count,
            },
        )
