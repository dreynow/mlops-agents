"""Provider protocols - the cloud abstraction layer.

Six interfaces that every backend (local, GCP, AWS) must implement.
Uses typing.Protocol for structural subtyping - no inheritance required,
just implement the methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


# --- Shared types ---

@dataclass(frozen=True)
class TrainConfig:
    """Configuration for a training job."""
    script_path: str
    args: dict[str, Any] = field(default_factory=dict)
    requirements: list[str] = field(default_factory=list)
    image: str = ""
    gpu: bool = False
    env: dict[str, str] = field(default_factory=dict)
    timeout_minutes: int = 120


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class JobHandle:
    """Reference to a submitted training job."""
    job_id: str
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Artifact:
    """A training artifact (model file, metrics, logs)."""
    name: str
    path: str
    artifact_type: str = "model"  # model, metrics, logs, checkpoint
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentRun:
    """A single experiment run with metrics and params."""
    run_id: str = ""
    experiment_name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelVersion:
    """A registered model version."""
    model_name: str
    version: str
    artifact_uri: str
    metrics: dict[str, float] = field(default_factory=dict)
    stage: str = "none"  # none, staging, production, archived
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(frozen=True)
class ModelArtifact:
    """Model artifact for registration."""
    model_name: str
    artifact_path: str
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparisonReport:
    """Comparison between experiment runs."""
    run_ids: list[str]
    metrics_comparison: dict[str, dict[str, float]] = field(default_factory=dict)
    best_run_id: str = ""
    summary: str = ""


@dataclass
class Dataset:
    """A versioned dataset."""
    name: str
    version: str
    path: str
    num_rows: int = 0
    num_columns: int = 0
    schema: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeployConfig:
    """Model deployment configuration."""
    endpoint_name: str = ""
    instance_type: str = "cpu"
    min_replicas: int = 1
    max_replicas: int = 1
    traffic_split: dict[str, int] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    health_check_path: str = "/health"
    port: int = 8080


@dataclass
class Endpoint:
    """A deployed model endpoint."""
    endpoint_id: str
    url: str
    model_name: str
    model_version: str
    status: str = "creating"  # creating, ready, failed, deleting
    traffic_split: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EndpointMetrics:
    """Metrics from a deployed endpoint."""
    endpoint_id: str
    request_count: int = 0
    error_count: int = 0
    error_rate: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    predictions: dict[str, int] = field(default_factory=dict)
    window_minutes: int = 15


# --- Provider protocols ---

@runtime_checkable
class ComputeProvider(Protocol):
    """Submit and manage training jobs.

    Local: subprocess / Docker container
    GCP: Vertex AI Custom Training Jobs
    """

    async def submit_job(self, config: TrainConfig) -> JobHandle: ...
    async def get_job_status(self, handle: JobHandle) -> JobStatus: ...
    async def get_artifacts(self, handle: JobHandle) -> list[Artifact]: ...
    async def cancel_job(self, handle: JobHandle) -> None: ...
    async def get_logs(self, handle: JobHandle) -> str: ...


@runtime_checkable
class StorageProvider(Protocol):
    """Store and retrieve artifacts (models, data, reports).

    Local: filesystem
    GCP: Google Cloud Storage
    """

    async def upload(self, local_path: Path, remote_key: str) -> str: ...
    async def download(self, remote_key: str, local_path: Path) -> None: ...
    async def list_artifacts(self, prefix: str) -> list[str]: ...
    async def exists(self, remote_key: str) -> bool: ...
    async def delete(self, remote_key: str) -> None: ...


@runtime_checkable
class MLPlatformProvider(Protocol):
    """Experiment tracking and model registry.

    Local: MLflow
    GCP: Vertex AI Experiments + Model Registry
    """

    async def log_experiment(self, run: ExperimentRun) -> str: ...
    async def get_run(self, run_id: str) -> ExperimentRun: ...
    async def register_model(self, model: ModelArtifact) -> ModelVersion: ...
    async def get_champion(self, model_name: str) -> ModelVersion | None: ...
    async def promote_model(self, model_name: str, version: str, stage: str) -> None: ...
    async def compare_runs(self, run_ids: list[str]) -> ComparisonReport: ...
    async def list_runs(self, experiment_name: str, limit: int = 20) -> list[ExperimentRun]: ...


@runtime_checkable
class DataProvider(Protocol):
    """Query and manage datasets.

    Local: DuckDB
    GCP: BigQuery
    """

    async def query(self, sql: str) -> list[dict[str, Any]]: ...
    async def get_dataset(self, name: str, version: str = "latest") -> Dataset: ...
    async def save_dataset(self, data: list[dict[str, Any]], name: str) -> Dataset: ...
    async def list_datasets(self) -> list[str]: ...


@runtime_checkable
class ServingProvider(Protocol):
    """Deploy and manage model endpoints.

    Local: FastAPI + Docker
    GCP: Cloud Run / Vertex Endpoints
    """

    async def deploy(self, model: ModelVersion, config: DeployConfig) -> Endpoint: ...
    async def get_endpoint(self, endpoint_id: str) -> Endpoint: ...
    async def get_endpoint_metrics(self, endpoint_id: str, window_minutes: int = 15) -> EndpointMetrics: ...
    async def set_traffic(self, endpoint_id: str, split: dict[str, int]) -> None: ...
    async def undeploy(self, endpoint_id: str) -> None: ...
