"""Decision and ReasoningTrace - the audit trail primitives.

Every agent decision produces a Decision with a ReasoningTrace that captures
what the agent observed, how it reasoned, and what it concluded. These are
the atoms of the audit trail.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class ReasoningTrace(BaseModel):
    """Chain-of-thought trace from the reasoning engine.

    This is the differentiator: not just "model promoted" but WHY,
    with observations, analysis, and alternatives considered.
    """

    observations: list[str] = Field(
        description="What the agent observed (metrics, test results, drift scores)"
    )
    analysis: str = Field(
        description="LLM reasoning about the observations"
    )
    conclusion: str = Field(
        description="Final judgment in one sentence"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Agent confidence in this decision (0.0 - 1.0)"
    )
    alternatives_considered: list[str] = Field(
        default_factory=list,
        description="Other actions the agent considered and why it rejected them"
    )
    model_used: str = Field(
        default="",
        description="Which LLM produced this reasoning (e.g. claude-sonnet-4-20250514)"
    )

    model_config = {"frozen": True}


class Decision(BaseModel):
    """A single agent decision with full audit context.

    Immutable once created. Persisted to the audit store.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    trace_id: str = Field(
        description="Pipeline trace ID - correlates all decisions in one run"
    )
    agent_name: str = Field(
        description="Which agent made this decision"
    )
    action: str = Field(
        description="What action was taken (e.g. 'model.promote', 'data.validate', 'rollback')"
    )
    approved: bool = Field(
        description="Go / no-go"
    )
    reasoning: ReasoningTrace
    artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="References to artifacts (model URI, report path, dataset version)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs for agent-specific context"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    escalate_to_human: bool = Field(
        default=False,
        description="Agent flags this decision for human review"
    )
    escalation_reason: str | None = Field(
        default=None,
        description="Why the agent is escalating (low confidence, edge case, etc.)"
    )

    model_config = {"frozen": True}

    def is_escalated(self) -> bool:
        return self.escalate_to_human

    def summary(self) -> str:
        """One-line summary for CLI output."""
        status = "GO" if self.approved else "NO-GO"
        esc = " [ESCALATED]" if self.escalate_to_human else ""
        return (
            f"[{self.agent_name}] {self.action} -> {status} "
            f"(confidence: {self.reasoning.confidence:.0%}){esc}"
        )


class PipelineTrace(BaseModel):
    """Collection of decisions from a single pipeline run."""

    trace_id: str = Field(default_factory=lambda: f"pipe-{uuid.uuid4().hex[:8]}")
    pipeline_name: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    decisions: list[Decision] = Field(default_factory=list)
    status: str = "running"  # running | completed | failed | escalated

    def add_decision(self, decision: Decision) -> None:
        self.decisions.append(decision)

    def finalize(self, status: str = "completed") -> None:
        self.completed_at = datetime.now(timezone.utc)
        self.status = status

    def summary(self) -> str:
        lines = [f"Trace: {self.trace_id} ({self.status})"]
        for d in self.decisions:
            lines.append(f"  {d.summary()}")
        return "\n".join(lines)
