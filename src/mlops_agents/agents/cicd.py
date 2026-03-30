"""CI/CD Agent - data validation and pipeline triggering.

Decides:
  1. Is the incoming data valid? (schema, completeness, distribution)
  2. Has distribution shifted enough to warrant attention?
  3. Should training be triggered?
"""

from __future__ import annotations

from typing import Any

import structlog

from mlops_agents.core.agent import AgentContext, BaseAgent
from mlops_agents.core.decision import Decision

logger = structlog.get_logger()

# Default thresholds
DEFAULT_MIN_ROWS = 100
DEFAULT_MAX_NULL_RATE = 0.1  # 10% null rate max
DEFAULT_MAX_PSI = 0.2  # Population Stability Index threshold


class CICDAgent(BaseAgent):
    """Validates data quality and triggers training pipelines.

    Authority scopes: data.validate, pipeline.trigger, test.run
    """

    name = "cicd"
    authority = ["data.validate", "pipeline.trigger", "test.run", "artifact.version"]
    description = "Validates incoming data and triggers training when quality is acceptable"

    def __init__(
        self,
        min_rows: int = DEFAULT_MIN_ROWS,
        max_null_rate: float = DEFAULT_MAX_NULL_RATE,
        max_psi: float = DEFAULT_MAX_PSI,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_rows = min_rows
        self.max_null_rate = max_null_rate
        self.max_psi = max_psi

    async def decide(self, ctx: AgentContext) -> Decision:
        payload = ctx.event.payload
        providers = ctx.providers or {}
        data_provider = providers.get("data")

        dataset_name = payload.get("dataset_name", "")
        dataset_version = payload.get("dataset_version", "latest")
        validation_report = payload.get("validation_report", {})

        # If a pre-computed validation report is provided, use it directly
        if validation_report:
            return await self._decide_from_report(ctx, validation_report)

        # Otherwise, run our own validation checks
        num_rows = payload.get("num_rows", 0)
        num_columns = payload.get("num_columns", 0)
        null_rates = payload.get("null_rates", {})
        schema_changes = payload.get("schema_changes", [])
        psi_scores = payload.get("psi_scores", {})

        # Try to get dataset info from data provider
        if data_provider and dataset_name:
            try:
                dataset = await data_provider.get_dataset(dataset_name, dataset_version)
                num_rows = num_rows or dataset.num_rows
                num_columns = num_columns or dataset.num_columns
                ctx.observe(f"Dataset: {dataset_name}/{dataset.version} ({num_rows} rows, {num_columns} cols)")
            except Exception as e:
                ctx.observe(f"Could not load dataset: {e}")
        else:
            ctx.observe(f"Dataset: {dataset_name or 'unknown'} ({num_rows} rows)")

        # --- 1. Row count check ---
        rows_ok = num_rows >= self.min_rows
        if num_rows > 0:
            ctx.observe(
                f"Row count: {num_rows} ({'OK' if rows_ok else f'BELOW minimum {self.min_rows}'})"
            )
        else:
            ctx.observe("Row count: unknown (no row count provided)")
            rows_ok = True  # Can't validate without data

        # --- 2. Null rate check ---
        null_violations = []
        for field, rate in null_rates.items():
            if rate > self.max_null_rate:
                null_violations.append(f"{field}: {rate:.1%}")

        nulls_ok = len(null_violations) == 0
        if null_rates:
            max_rate = max(null_rates.values()) if null_rates else 0
            ctx.observe(
                f"Null rates: max {max_rate:.1%} across {len(null_rates)} fields "
                f"({'OK' if nulls_ok else f'{len(null_violations)} violations'})"
            )
            if not nulls_ok:
                ctx.observe(f"Null violations: {', '.join(null_violations)}")

        # --- 3. Schema change check ---
        schema_ok = len(schema_changes) == 0
        if schema_changes:
            ctx.observe(f"Schema changes detected: {', '.join(schema_changes)}")
        else:
            ctx.observe("Schema: no changes detected")

        # --- 4. Distribution drift check (PSI) ---
        psi_violations = []
        for field, psi in psi_scores.items():
            if psi > self.max_psi:
                psi_violations.append(f"{field}: PSI={psi:.3f}")

        drift_ok = len(psi_violations) == 0
        if psi_scores:
            max_psi_val = max(psi_scores.values())
            ctx.observe(
                f"Distribution drift: max PSI {max_psi_val:.3f} across {len(psi_scores)} fields "
                f"({'OK' if drift_ok else f'{len(psi_violations)} drifted fields'})"
            )
        else:
            ctx.observe("Distribution drift: no PSI scores provided (skipping)")
            drift_ok = True

        # --- 5. Overall decision ---
        approved = rows_ok and nulls_ok and schema_ok and drift_ok

        reasoning = await self.reason(
            observations=ctx.observations,
            context={
                "num_rows": num_rows,
                "null_violations": null_violations,
                "schema_changes": schema_changes,
                "psi_violations": psi_violations,
                "thresholds": {
                    "min_rows": self.min_rows,
                    "max_null_rate": self.max_null_rate,
                    "max_psi": self.max_psi,
                },
            },
            action="data.validate",
        )

        return Decision(
            trace_id=ctx.trace_id,
            agent_name=self.name,
            action="data.validate",
            approved=approved,
            reasoning=reasoning,
            artifacts={"dataset_name": dataset_name, "dataset_version": dataset_version},
            metadata={
                "num_rows": num_rows,
                "null_violations": null_violations,
                "schema_changes": schema_changes,
                "psi_violations": psi_violations,
                "rows_ok": rows_ok,
                "nulls_ok": nulls_ok,
                "schema_ok": schema_ok,
                "drift_ok": drift_ok,
            },
        )

    async def _decide_from_report(self, ctx: AgentContext, report: dict) -> Decision:
        """Use a pre-computed validation report (e.g. from Great Expectations)."""
        total_checks = report.get("total_checks", 0)
        passed_checks = report.get("passed_checks", 0)
        failed_checks = report.get("failed_checks", [])

        pass_rate = passed_checks / max(total_checks, 1)
        ctx.observe(f"Validation report: {passed_checks}/{total_checks} checks passed ({pass_rate:.0%})")

        if failed_checks:
            for check in failed_checks[:5]:  # Show first 5 failures
                ctx.observe(f"  Failed: {check}")

        approved = pass_rate >= 0.95  # 95% pass rate required

        reasoning = await self.reason(
            observations=ctx.observations,
            context={"pass_rate": pass_rate, "failed_checks": failed_checks},
            action="data.validate",
        )

        return Decision(
            trace_id=ctx.trace_id,
            agent_name=self.name,
            action="data.validate",
            approved=approved,
            reasoning=reasoning,
            metadata={"pass_rate": pass_rate, "total_checks": total_checks},
        )
