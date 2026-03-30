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
    """Per-service provider selection.

    Each service can be configured independently. Users pick the backend
    for each service and provide settings for it. Default is all-local.

    Supports two formats:
      1. Per-service (new): each provider specified independently
         provider:
           compute: vertex_ai
           storage: gcs
           ml: local
           data: bigquery

      2. Bundle (legacy): one backend for everything
         provider:
           backend: local     # or gcp
    """

    # Per-service selection (new way)
    compute: str = Field(default="local", description="'local' or 'vertex_ai'")
    storage: str = Field(default="local", description="'local' or 'gcs'")
    ml: str = Field(default="local", description="'local' (MLflow-compatible)")
    data: str = Field(default="local", description="'local' (DuckDB) or 'bigquery'")
    event_bus: str = Field(default="local", description="'local' (AsyncIO)")
    serving: str = Field(default="local", description="'local' (in-memory)")

    # Bundle shortcut (legacy - sets all providers at once)
    backend: str = Field(
        default="",
        description="Legacy: 'local' or 'gcp'. Overridden by per-service settings.",
    )

    # Service-specific settings
    vertex_ai: dict[str, Any] = Field(
        default_factory=dict,
        description="Vertex AI settings: project, region, staging_bucket",
    )
    gcs: dict[str, Any] = Field(
        default_factory=dict,
        description="GCS settings: bucket, prefix",
    )
    bigquery: dict[str, Any] = Field(
        default_factory=dict,
        description="BigQuery settings: project, dataset, location",
    )
    mlflow: dict[str, Any] = Field(
        default_factory=dict,
        description="MLflow settings: tracking_uri, base_dir",
    )
    local: dict[str, Any] = Field(
        default_factory=dict,
        description="Local provider settings: base_dir",
    )

    # Legacy GCP bundle settings
    gcp: dict[str, Any] = Field(
        default_factory=dict,
        description="Legacy GCP bundle settings (use per-service instead)",
    )

    def model_post_init(self, __context: Any) -> None:
        """Apply bundle shortcut: backend=gcp sets GCP defaults for services
        that don't have their own settings dict configured."""
        if self.backend != "gcp" or not self.gcp:
            return

        # For each GCP service: set the provider type to GCP,
        # and populate settings from the gcp bundle if user didn't
        # provide service-specific settings.
        if self.compute == "local":
            self.compute = "vertex_ai"
        if not self.vertex_ai:
            self.vertex_ai = {
                "project": self.gcp.get("project_id", ""),
                "region": self.gcp.get("region", "us-central1"),
                "staging_bucket": self.gcp.get("staging_bucket", ""),
            }

        if self.storage == "local":
            self.storage = "gcs"
        if not self.gcs:
            bucket = self.gcp.get("staging_bucket", "")
            self.gcs = {
                "bucket": bucket.replace("gs://", "").strip("/"),
                "project": self.gcp.get("project_id", ""),
            }

        if self.data == "local":
            self.data = "bigquery"
        if not self.bigquery:
            self.bigquery = {
                "project": self.gcp.get("project_id", ""),
                "dataset": self.gcp.get("bigquery_dataset", "ml_features"),
                "location": self.gcp.get("bigquery_location", "US"),
            }


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
