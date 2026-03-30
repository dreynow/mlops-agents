"""Monitoring Agent - drift detection and performance monitoring.

Decides:
  1. Is feature drift statistically significant or noise?
  2. Is prediction performance degrading?
  3. Alert severity: info / warning / critical
  4. Should this trigger retraining?
"""

from __future__ import annotations

from typing import Any

import structlog

from mlops_agents.core.agent import AgentContext, BaseAgent
from mlops_agents.core.decision import Decision

logger = structlog.get_logger()

DEFAULT_PSI_THRESHOLD = 0.2  # Population Stability Index
DEFAULT_KS_P_VALUE_THRESHOLD = 0.01  # Kolmogorov-Smirnov test
DEFAULT_ACCURACY_DROP_THRESHOLD = 0.05  # 5% accuracy drop
DEFAULT_ERROR_RATE_THRESHOLD = 0.1  # 10% error rate


class MonitorAgent(BaseAgent):
    """Monitors production models for drift and degradation.

    Authority scopes: metrics.collect, drift.detect, alert.send
    """

    name = "monitoring"
    authority = ["metrics.collect", "drift.detect", "alert.send"]
    description = "Detects drift, monitors performance, decides alert severity and retrain triggers"

    def __init__(
        self,
        psi_threshold: float = DEFAULT_PSI_THRESHOLD,
        ks_p_value_threshold: float = DEFAULT_KS_P_VALUE_THRESHOLD,
        accuracy_drop_threshold: float = DEFAULT_ACCURACY_DROP_THRESHOLD,
        error_rate_threshold: float = DEFAULT_ERROR_RATE_THRESHOLD,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.psi_threshold = psi_threshold
        self.ks_p_value_threshold = ks_p_value_threshold
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.error_rate_threshold = error_rate_threshold

    async def decide(self, ctx: AgentContext) -> Decision:
        payload = ctx.event.payload
        providers = ctx.providers or {}
        serving = providers.get("serving")

        endpoint_id = payload.get("endpoint_id", "")
        model_name = payload.get("model_name", "")

        # Drift metrics (from monitoring system or payload)
        psi_scores = payload.get("psi_scores", {})
        ks_p_values = payload.get("ks_p_values", {})
        baseline_accuracy = payload.get("baseline_accuracy", 0.0)
        current_accuracy = payload.get("current_accuracy", 0.0)

        # Endpoint metrics (from serving provider)
        error_rate = payload.get("error_rate", 0.0)
        latency_p99 = payload.get("latency_p99_ms", 0.0)

        if serving and endpoint_id:
            try:
                metrics = await serving.get_endpoint_metrics(endpoint_id)
                error_rate = metrics.error_rate
                latency_p99 = metrics.latency_p99_ms
                ctx.observe(
                    f"Endpoint {endpoint_id}: error_rate={error_rate:.2%}, p99={latency_p99:.1f}ms"
                )
            except Exception as e:
                ctx.observe(f"Could not fetch endpoint metrics: {e}")

        # --- 1. Feature drift analysis ---
        drifted_features = []
        for feature, psi in psi_scores.items():
            if psi > self.psi_threshold:
                drifted_features.append((feature, psi))

        if psi_scores:
            max_psi = max(psi_scores.values())
            ctx.observe(
                f"Feature drift: {len(drifted_features)}/{len(psi_scores)} features drifted "
                f"(max PSI: {max_psi:.3f}, threshold: {self.psi_threshold})"
            )
            for feat, psi in drifted_features:
                ctx.observe(f"  Drifted: {feat} (PSI={psi:.3f})")
        else:
            ctx.observe("Feature drift: no PSI scores provided")

        # --- 2. Statistical significance (KS test) ---
        significant_drifts = []
        for feature, p_value in ks_p_values.items():
            if p_value < self.ks_p_value_threshold:
                significant_drifts.append((feature, p_value))

        if ks_p_values:
            ctx.observe(
                f"KS test: {len(significant_drifts)}/{len(ks_p_values)} features "
                f"statistically significant (p < {self.ks_p_value_threshold})"
            )

        # --- 3. Accuracy degradation ---
        accuracy_ok = True
        if baseline_accuracy > 0 and current_accuracy > 0:
            accuracy_drop = baseline_accuracy - current_accuracy
            accuracy_ok = accuracy_drop <= self.accuracy_drop_threshold
            ctx.observe(
                f"Accuracy: {current_accuracy:.4f} (baseline: {baseline_accuracy:.4f}, "
                f"drop: {accuracy_drop:+.4f}, {'OK' if accuracy_ok else 'DEGRADED'})"
            )

        # --- 4. Error rate ---
        error_ok = error_rate <= self.error_rate_threshold
        if error_rate > 0:
            ctx.observe(
                f"Error rate: {error_rate:.2%} ({'OK' if error_ok else 'HIGH'}, "
                f"threshold: {self.error_rate_threshold:.2%})"
            )

        # --- 5. Determine severity and action ---
        has_drift = len(drifted_features) > 0
        has_degradation = not accuracy_ok or not error_ok
        needs_retrain = has_drift and has_degradation

        if has_degradation:
            severity = "critical"
            action = "drift.detect"
        elif has_drift:
            severity = "warning"
            action = "drift.detect"
        else:
            severity = "info"
            action = "metrics.collect"

        # Approved means "system is healthy" - False means "intervention needed"
        approved = not has_drift and not has_degradation

        reasoning = await self.reason(
            observations=ctx.observations,
            context={
                "drifted_features": [f[0] for f in drifted_features],
                "significant_drifts": [f[0] for f in significant_drifts],
                "accuracy_ok": accuracy_ok,
                "error_ok": error_ok,
                "severity": severity,
                "needs_retrain": needs_retrain,
                "thresholds": {
                    "psi": self.psi_threshold,
                    "ks_p_value": self.ks_p_value_threshold,
                    "accuracy_drop": self.accuracy_drop_threshold,
                    "error_rate": self.error_rate_threshold,
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
            artifacts={"endpoint_id": endpoint_id, "model_name": model_name},
            metadata={
                "severity": severity,
                "drifted_features": [f[0] for f in drifted_features],
                "psi_scores": psi_scores,
                "accuracy_drop": baseline_accuracy - current_accuracy
                if baseline_accuracy > 0
                else 0,
                "error_rate": error_rate,
                "needs_retrain": needs_retrain,
            },
        )
