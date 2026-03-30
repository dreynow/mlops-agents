"""Configuration system for mlops-agents.

YAML-driven config with provider selection, agent settings, and
escalation thresholds. Supports per-stage override of confidence
thresholds (not hardcoded).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ReasoningConfig(BaseModel):
    """Which LLM backend to use for agent reasoning."""

    engine: str = Field(
        default="claude", description="Reasoning backend: 'claude', 'openai', 'ollama', 'static'"
    )
    model: str = Field(
        default="claude-sonnet-4-20250514", description="Model ID for the chosen engine"
    )
    api_key: str | None = Field(
        default=None, description="API key (falls back to env vars if not set)"
    )
    host: str = Field(default="http://localhost:11434", description="Host URL for Ollama")


class ProviderConfig(BaseModel):
    """Cloud provider selection and settings."""

    backend: str = Field(default="local", description="Provider backend: 'local' or 'gcp'")
    gcp: dict[str, Any] = Field(
        default_factory=dict, description="GCP-specific settings (project_id, region, etc.)"
    )
    local: dict[str, Any] = Field(
        default_factory=dict, description="Local provider settings (mlflow_uri, etc.)"
    )


class EscalationConfig(BaseModel):
    """Escalation thresholds - configurable per stage, not hardcoded."""

    default_confidence_threshold: float = Field(
        default=0.7, description="Default confidence below which agents escalate to human"
    )
    per_stage: dict[str, float] = Field(
        default_factory=dict,
        description="Per-stage overrides (e.g. {'evaluate': 0.8, 'deploy': 0.9})",
    )

    def threshold_for(self, stage: str) -> float:
        return self.per_stage.get(stage, self.default_confidence_threshold)


class AuditConfig(BaseModel):
    """Audit trail settings."""

    backend: str = Field(default="sqlite", description="'sqlite' or 'bigquery'")
    sqlite_path: str = Field(default="mlops_audit.db")
    bigquery_dataset: str = Field(default="")
    bigquery_table: str = Field(default="audit_decisions")


class StageConfig(BaseModel):
    """Configuration for a single pipeline stage."""

    agent: str
    on_success: list[str] = Field(default_factory=list)
    on_failure: list[str] = Field(default_factory=list)
    on_drift: list[str] = Field(default_factory=list)
    on_degradation: list[str] = Field(default_factory=list)
    on_sufficient_feedback: list[str] = Field(default_factory=list)
    mode: str = Field(default="oneshot", description="'oneshot' or 'continuous'")
    check_interval: str = Field(default="15m")
    collection_interval: str = Field(default="1h")
    strategy: str = Field(default="")
    canary_duration: str = Field(default="30m")
    canary_traffic: str = Field(default="5%")
    escalation: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)


class TriggerConfig(BaseModel):
    """Pipeline trigger configuration."""

    schedule: str = Field(default="", description="Cron expression")
    events: list[str] = Field(
        default_factory=list, description="Event types that trigger the pipeline"
    )


class PipelineConfig(BaseModel):
    """Full pipeline configuration loaded from YAML."""

    name: str = "default-pipeline"
    trigger: TriggerConfig = Field(default_factory=TriggerConfig)
    stages: dict[str, StageConfig] = Field(default_factory=dict)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        return cls.model_validate(data)
