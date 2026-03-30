"""AuditStore - persistent decision trail.

Every Decision from every agent is persisted here. The audit trail is
the killer feature: queryable by trace_id, agent, time range, action.

Backends:
  - SQLite (local default)
  - BigQuery (GCP) - future
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import aiosqlite
import structlog

from mlops_agents.core.decision import Decision, PipelineTrace

logger = structlog.get_logger()

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    action TEXT NOT NULL,
    approved INTEGER NOT NULL,
    confidence REAL NOT NULL,
    escalated INTEGER NOT NULL DEFAULT 0,
    escalation_reason TEXT,
    reasoning_json TEXT NOT NULL,
    artifacts_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_decisions_trace_id ON decisions(trace_id);
CREATE INDEX IF NOT EXISTS idx_decisions_agent ON decisions(agent_name);
CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp);
CREATE INDEX IF NOT EXISTS idx_decisions_action ON decisions(action);

CREATE TABLE IF NOT EXISTS pipeline_traces (
    trace_id TEXT PRIMARY KEY,
    pipeline_name TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@runtime_checkable
class AuditStore(Protocol):
    """Protocol for audit trail backends."""

    async def log_decision(self, decision: Decision) -> None: ...

    async def get_trace(self, trace_id: str) -> list[Decision]: ...

    async def get_decisions_by_agent(self, agent_name: str, limit: int = 50) -> list[Decision]: ...

    async def get_recent(self, limit: int = 20) -> list[Decision]: ...

    async def save_trace(self, trace: PipelineTrace) -> None: ...

    async def get_pipeline_trace(self, trace_id: str) -> PipelineTrace | None: ...


class SQLiteAuditStore:
    """SQLite-backed audit store for local development."""

    def __init__(self, db_path: str | Path = "mlops_audit.db"):
        self.db_path = Path(db_path)
        self._initialized = False

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA_SQL)
            await db.commit()
        self._initialized = True

    async def log_decision(self, decision: Decision) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO decisions
                   (id, trace_id, agent_name, action, approved, confidence,
                    escalated, escalation_reason, reasoning_json, artifacts_json,
                    metadata_json, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    decision.id,
                    decision.trace_id,
                    decision.agent_name,
                    decision.action,
                    int(decision.approved),
                    decision.reasoning.confidence,
                    int(decision.escalate_to_human),
                    decision.escalation_reason,
                    decision.reasoning.model_dump_json(),
                    json.dumps(decision.artifacts),
                    json.dumps(decision.metadata, default=str),
                    decision.timestamp.isoformat(),
                ),
            )
            await db.commit()
        logger.info(
            "audit.decision_logged",
            id=decision.id,
            trace_id=decision.trace_id,
            agent=decision.agent_name,
            action=decision.action,
            approved=decision.approved,
        )

    async def get_trace(self, trace_id: str) -> list[Decision]:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM decisions WHERE trace_id = ? ORDER BY timestamp",
                (trace_id,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_decision(row) for row in rows]

    async def get_decisions_by_agent(self, agent_name: str, limit: int = 50) -> list[Decision]:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM decisions WHERE agent_name = ? ORDER BY timestamp DESC LIMIT ?",
                (agent_name, limit),
            )
            rows = await cursor.fetchall()
            return [self._row_to_decision(row) for row in rows]

    async def get_recent(self, limit: int = 20) -> list[Decision]:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_decision(row) for row in rows]

    async def save_trace(self, trace: PipelineTrace) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO pipeline_traces
                   (trace_id, pipeline_name, started_at, completed_at, status)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    trace.trace_id,
                    trace.pipeline_name,
                    trace.started_at.isoformat(),
                    trace.completed_at.isoformat() if trace.completed_at else None,
                    trace.status,
                ),
            )
            await db.commit()

    async def get_pipeline_trace(self, trace_id: str) -> PipelineTrace | None:
        await self._ensure_schema()
        decisions = await self.get_trace(trace_id)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM pipeline_traces WHERE trace_id = ?",
                (trace_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                if not decisions:
                    return None
                return PipelineTrace(trace_id=trace_id, decisions=decisions)

            return PipelineTrace(
                trace_id=row["trace_id"],
                pipeline_name=row["pipeline_name"],
                started_at=datetime.fromisoformat(row["started_at"]),
                completed_at=(
                    datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
                ),
                status=row["status"],
                decisions=decisions,
            )

    @staticmethod
    def _row_to_decision(row) -> Decision:
        from mlops_agents.core.decision import ReasoningTrace

        reasoning = ReasoningTrace.model_validate_json(row["reasoning_json"])
        return Decision(
            id=row["id"],
            trace_id=row["trace_id"],
            agent_name=row["agent_name"],
            action=row["action"],
            approved=bool(row["approved"]),
            reasoning=reasoning,
            artifacts=json.loads(row["artifacts_json"]),
            metadata=json.loads(row["metadata_json"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            escalate_to_human=bool(row["escalated"]),
            escalation_reason=row["escalation_reason"],
        )
