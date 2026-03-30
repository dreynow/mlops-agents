"""Vertex AI Experiments + Model Registry provider (Tier 2 - stub).

This provider will wrap Vertex AI Experiments for experiment tracking
and the Vertex AI Model Registry for model versioning and promotion.

Not yet implemented. The local MLflow provider covers this functionality
for development and demo use cases. Contribute this provider if you
need native Vertex AI experiment tracking.

API surface when implemented:
  - log_experiment() -> Vertex AI ExperimentRun
  - get_run() -> Fetch run by ID
  - register_model() -> Upload to Vertex AI Model Registry
  - get_champion() -> Model with 'default' alias
  - promote_model() -> Set model alias
  - compare_runs() -> Metrics comparison across runs
  - list_runs() -> Query experiment runs
"""

from __future__ import annotations

from mlops_agents.providers.protocols import (
    ComparisonReport,
    ExperimentRun,
    ModelArtifact,
    ModelVersion,
)


class VertexAIExperiments:
    """Vertex AI Experiments + Model Registry (not yet implemented).

    Use LocalMLPlatform for local development. This provider is planned
    for full Vertex AI integration.
    """

    def __init__(self, project: str, location: str = "us-central1"):
        self._project = project
        self._location = location

    async def log_experiment(self, run: ExperimentRun) -> str:
        raise NotImplementedError(
            "VertexAIExperiments is not yet implemented. Use LocalMLPlatform."
        )

    async def get_run(self, run_id: str) -> ExperimentRun:
        raise NotImplementedError(
            "VertexAIExperiments is not yet implemented. Use LocalMLPlatform."
        )

    async def register_model(self, model: ModelArtifact) -> ModelVersion:
        raise NotImplementedError(
            "VertexAIExperiments is not yet implemented. Use LocalMLPlatform."
        )

    async def get_champion(self, model_name: str) -> ModelVersion | None:
        raise NotImplementedError(
            "VertexAIExperiments is not yet implemented. Use LocalMLPlatform."
        )

    async def promote_model(self, model_name: str, version: str, stage: str) -> None:
        raise NotImplementedError(
            "VertexAIExperiments is not yet implemented. Use LocalMLPlatform."
        )

    async def compare_runs(self, run_ids: list[str]) -> ComparisonReport:
        raise NotImplementedError(
            "VertexAIExperiments is not yet implemented. Use LocalMLPlatform."
        )

    async def list_runs(self, experiment_name: str, limit: int = 20) -> list[ExperimentRun]:
        raise NotImplementedError(
            "VertexAIExperiments is not yet implemented. Use LocalMLPlatform."
        )
