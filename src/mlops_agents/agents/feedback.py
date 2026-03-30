"""Feedback Agent - collects, analyzes, and curates feedback for retraining.

Decides:
  1. Is this feedback actionable? (noise filtering)
  2. Are there systematic prediction failures? (error pattern detection)
  3. What's the label quality? (consistency checks)
  4. Is there enough feedback to trigger retraining?
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import structlog

from mlops_agents.core.agent import AgentContext, BaseAgent
from mlops_agents.core.decision import Decision

logger = structlog.get_logger()

DEFAULT_MIN_SAMPLES_FOR_RETRAIN = 50
DEFAULT_MIN_PATTERN_FREQUENCY = 5
DEFAULT_MIN_AGREEMENT_RATE = 0.7  # 70% inter-annotator agreement


class FeedbackAgent(BaseAgent):
    """Collects and analyzes human feedback for model improvement.

    Authority scopes: feedback.collect, feedback.analyze, label.validate
    """

    name = "feedback"
    authority = ["feedback.collect", "feedback.analyze", "label.validate"]
    description = "Analyzes feedback, detects error patterns, curates data for retraining"

    def __init__(
        self,
        min_samples_for_retrain: int = DEFAULT_MIN_SAMPLES_FOR_RETRAIN,
        min_pattern_frequency: int = DEFAULT_MIN_PATTERN_FREQUENCY,
        min_agreement_rate: float = DEFAULT_MIN_AGREEMENT_RATE,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_samples_for_retrain = min_samples_for_retrain
        self.min_pattern_frequency = min_pattern_frequency
        self.min_agreement_rate = min_agreement_rate

    async def decide(self, ctx: AgentContext) -> Decision:
        payload = ctx.event.payload

        model_name = payload.get("model_name", "")
        corrections = payload.get("corrections", [])
        flags = payload.get("flags", [])
        agreement_scores = payload.get("agreement_scores", [])
        prediction_segments = payload.get("prediction_segments", {})

        total_feedback = len(corrections) + len(flags)
        ctx.observe(f"Feedback received: {len(corrections)} corrections, {len(flags)} flags")

        # --- 1. Assess label quality ---
        agreement_ok = True
        avg_agreement = 0.0
        if agreement_scores:
            avg_agreement = sum(agreement_scores) / len(agreement_scores)
            agreement_ok = avg_agreement >= self.min_agreement_rate
            ctx.observe(
                f"Label quality: avg agreement {avg_agreement:.0%} "
                f"({'OK' if agreement_ok else 'LOW'}, threshold: {self.min_agreement_rate:.0%})"
            )
        else:
            ctx.observe("Label quality: no agreement scores provided (skipping check)")

        # --- 2. Detect error patterns ---
        error_patterns = []

        if corrections:
            # Group corrections by segment to find systematic failures
            segment_counts: Counter[str] = Counter()
            for correction in corrections:
                segment = correction.get("segment", "unknown")
                segment_counts[segment] += 1

            for segment, count in segment_counts.most_common():
                if count >= self.min_pattern_frequency:
                    error_patterns.append(
                        {
                            "segment": segment,
                            "count": count,
                            "percentage": count / len(corrections),
                        }
                    )

            if error_patterns:
                ctx.observe(f"Error patterns: {len(error_patterns)} systematic failure segments")
                for pattern in error_patterns[:3]:
                    ctx.observe(
                        f"  Segment '{pattern['segment']}': {pattern['count']} corrections "
                        f"({pattern['percentage']:.0%} of total)"
                    )
            else:
                ctx.observe("Error patterns: no systematic failure segments detected")

        # --- 3. Prediction segment analysis ---
        if prediction_segments:
            for segment_name, segment_data in prediction_segments.items():
                error_rate = segment_data.get("error_rate", 0)
                sample_count = segment_data.get("count", 0)
                if error_rate > 0.1:  # Segments with >10% error
                    ctx.observe(
                        f"High-error segment '{segment_name}': {error_rate:.1%} error rate "
                        f"({sample_count} samples)"
                    )

        # --- 4. Decide: enough feedback for retraining? ---
        enough_for_retrain = total_feedback >= self.min_samples_for_retrain

        if enough_for_retrain:
            ctx.observe(
                f"Sufficient feedback for retraining: {total_feedback} samples "
                f"(threshold: {self.min_samples_for_retrain})"
            )
        else:
            ctx.observe(
                f"Insufficient feedback: {total_feedback} samples "
                f"(need {self.min_samples_for_retrain})"
            )

        # Approved = feedback is actionable and sufficient for retraining
        approved = enough_for_retrain and agreement_ok
        action = "feedback.analyze"

        reasoning = await self.reason(
            observations=ctx.observations,
            context={
                "total_feedback": total_feedback,
                "corrections_count": len(corrections),
                "flags_count": len(flags),
                "error_patterns": error_patterns,
                "agreement_ok": agreement_ok,
                "avg_agreement": avg_agreement,
                "enough_for_retrain": enough_for_retrain,
                "thresholds": {
                    "min_samples": self.min_samples_for_retrain,
                    "min_pattern_frequency": self.min_pattern_frequency,
                    "min_agreement": self.min_agreement_rate,
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
            artifacts={"model_name": model_name},
            metadata={
                "total_feedback": total_feedback,
                "corrections_count": len(corrections),
                "flags_count": len(flags),
                "error_patterns": error_patterns,
                "agreement_ok": agreement_ok,
                "avg_agreement": avg_agreement,
                "enough_for_retrain": enough_for_retrain,
            },
        )
