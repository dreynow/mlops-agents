"""Tests for CI/CD Agent."""

import pytest

from mlops_agents.agents.cicd import CICDAgent
from mlops_agents.core.audit import SQLiteAuditStore
from mlops_agents.core.event import Event, LocalAsyncEventBus
from mlops_agents.core.reasoning import StaticReasoner


@pytest.fixture
def agent(tmp_path):
    return CICDAgent(
        event_bus=LocalAsyncEventBus(),
        audit_store=SQLiteAuditStore(db_path=tmp_path / "test.db"),
        reasoning_engine=StaticReasoner(),
        min_rows=100,
        max_null_rate=0.1,
        max_psi=0.2,
    )


def _event(trace_id="pipe-test", **payload):
    return Event(type="data.validate", source="test", trace_id=trace_id, payload=payload)


class TestCICDDataValidation:
    @pytest.mark.asyncio
    async def test_clean_data_approved(self, agent):
        event = _event(
            dataset_name="transactions",
            num_rows=5000,
            num_columns=10,
            null_rates={"amount": 0.01, "merchant": 0.03},
            psi_scores={"amount": 0.05, "hour": 0.08},
        )
        decision = await agent.run(event)
        assert decision.approved is True
        assert decision.metadata["rows_ok"] is True
        assert decision.metadata["nulls_ok"] is True
        assert decision.metadata["drift_ok"] is True

    @pytest.mark.asyncio
    async def test_too_few_rows_rejected(self, agent):
        event = _event(num_rows=50)  # Below 100 minimum
        decision = await agent.run(event)
        assert decision.approved is False
        assert decision.metadata["rows_ok"] is False

    @pytest.mark.asyncio
    async def test_high_null_rate_rejected(self, agent):
        event = _event(
            num_rows=1000,
            null_rates={"email": 0.25, "phone": 0.02},  # email exceeds 10%
        )
        decision = await agent.run(event)
        assert decision.approved is False
        assert decision.metadata["nulls_ok"] is False
        assert len(decision.metadata["null_violations"]) == 1

    @pytest.mark.asyncio
    async def test_schema_change_rejected(self, agent):
        event = _event(
            num_rows=1000,
            schema_changes=["column 'amount' type changed from float to string"],
        )
        decision = await agent.run(event)
        assert decision.approved is False
        assert decision.metadata["schema_ok"] is False

    @pytest.mark.asyncio
    async def test_high_drift_rejected(self, agent):
        event = _event(
            num_rows=1000,
            psi_scores={"amount": 0.35, "hour": 0.05},  # amount PSI > 0.2
        )
        decision = await agent.run(event)
        assert decision.approved is False
        assert decision.metadata["drift_ok"] is False

    @pytest.mark.asyncio
    async def test_no_metrics_passes(self, agent):
        """No validation data provided - can't reject what you can't check."""
        event = _event(dataset_name="unknown")
        decision = await agent.run(event)
        assert decision.approved is True


class TestCICDValidationReport:
    @pytest.mark.asyncio
    async def test_high_pass_rate_approved(self, agent):
        event = _event(
            validation_report={
                "total_checks": 47,
                "passed_checks": 46,
                "failed_checks": ["column_x nullable check"],
            },
        )
        decision = await agent.run(event)
        assert decision.approved is True

    @pytest.mark.asyncio
    async def test_low_pass_rate_rejected(self, agent):
        event = _event(
            validation_report={
                "total_checks": 20,
                "passed_checks": 15,
                "failed_checks": ["check1", "check2", "check3", "check4", "check5"],
            },
        )
        decision = await agent.run(event)
        assert decision.approved is False
