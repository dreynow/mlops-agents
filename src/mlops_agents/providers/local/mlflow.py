"""Local MLflow-compatible experiment tracking.

Lightweight in-memory implementation that follows the MLPlatformProvider
protocol. Does NOT require a running MLflow server - stores everything
in local JSON files for zero-dependency local development.

When users want real MLflow, they can swap in the full MLflow provider.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog

from mlops_agents.providers.protocols import (
    ComparisonReport,
    ExperimentRun,
    ModelArtifact,
    ModelVersion,
)

logger = structlog.get_logger()


class LocalMLPlatform:
    """File-backed experiment tracking and model registry.

    Stores experiments and models as JSON files. No external
    dependencies required.
    """

    def __init__(self, base_dir: str = ".mlops/mlplatform"):
        self.base_dir = Path(base_dir)
        self.experiments_dir = self.base_dir / "experiments"
        self.models_dir = self.base_dir / "models"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    async def log_experiment(self, run: ExperimentRun) -> str:
        run_id = run.run_id or uuid.uuid4().hex[:12]
        exp_name = run.experiment_name or "default"

        exp_dir = self.experiments_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        run_data = {
            "run_id": run_id,
            "experiment_name": exp_name,
            "params": run.params,
            "metrics": run.metrics,
            "tags": run.tags,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        run_file = exp_dir / f"{run_id}.json"
        run_file.write_text(json.dumps(run_data, indent=2))

        logger.info("mlplatform.local.log_experiment", run_id=run_id, experiment=exp_name)
        return run_id

    async def get_run(self, run_id: str) -> ExperimentRun:
        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir():
                run_file = exp_dir / f"{run_id}.json"
                if run_file.exists():
                    data = json.loads(run_file.read_text())
                    return ExperimentRun(
                        run_id=data["run_id"],
                        experiment_name=data["experiment_name"],
                        params=data.get("params", {}),
                        metrics=data.get("metrics", {}),
                        tags=data.get("tags", {}),
                    )
        raise ValueError(f"Run not found: {run_id}")

    async def register_model(self, model: ModelArtifact) -> ModelVersion:
        model_dir = self.models_dir / model.model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Determine next version
        existing = sorted(model_dir.glob("v*.json"))
        next_version = f"v{len(existing) + 1}"

        version_data = {
            "model_name": model.model_name,
            "version": next_version,
            "artifact_path": model.artifact_path,
            "metrics": model.metrics,
            "params": model.params,
            "tags": model.tags,
            "stage": "none",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        version_file = model_dir / f"{next_version}.json"
        version_file.write_text(json.dumps(version_data, indent=2))

        logger.info(
            "mlplatform.local.register_model",
            model=model.model_name,
            version=next_version,
        )

        return ModelVersion(
            model_name=model.model_name,
            version=next_version,
            artifact_uri=model.artifact_path,
            metrics=model.metrics,
            stage="none",
            created_at=datetime.now(timezone.utc),
        )

    async def get_champion(self, model_name: str) -> ModelVersion | None:
        model_dir = self.models_dir / model_name
        if not model_dir.exists():
            return None

        # Find model in "production" stage, or latest version
        versions = sorted(model_dir.glob("v*.json"))
        champion = None

        for v_file in versions:
            data = json.loads(v_file.read_text())
            if data.get("stage") == "production":
                champion = data
                break

        if champion is None and versions:
            champion = json.loads(versions[-1].read_text())

        if champion is None:
            return None

        return ModelVersion(
            model_name=champion["model_name"],
            version=champion["version"],
            artifact_uri=champion["artifact_path"],
            metrics=champion.get("metrics", {}),
            stage=champion.get("stage", "none"),
            created_at=datetime.fromisoformat(champion["created_at"]) if "created_at" in champion else None,
        )

    async def promote_model(self, model_name: str, version: str, stage: str) -> None:
        model_dir = self.models_dir / model_name
        version_file = model_dir / f"{version}.json"
        if not version_file.exists():
            raise ValueError(f"Model version not found: {model_name}/{version}")

        # Demote current champion if promoting to production
        if stage == "production":
            for v_file in model_dir.glob("v*.json"):
                data = json.loads(v_file.read_text())
                if data.get("stage") == "production":
                    data["stage"] = "archived"
                    v_file.write_text(json.dumps(data, indent=2))

        data = json.loads(version_file.read_text())
        data["stage"] = stage
        version_file.write_text(json.dumps(data, indent=2))

        logger.info("mlplatform.local.promote", model=model_name, version=version, stage=stage)

    async def compare_runs(self, run_ids: list[str]) -> ComparisonReport:
        runs = []
        for rid in run_ids:
            try:
                run = await self.get_run(rid)
                runs.append(run)
            except ValueError:
                continue

        if not runs:
            return ComparisonReport(run_ids=run_ids, summary="No runs found")

        # Build metrics comparison
        all_metric_keys = set()
        for r in runs:
            all_metric_keys.update(r.metrics.keys())

        metrics_comparison: dict[str, dict[str, float]] = {}
        for key in sorted(all_metric_keys):
            metrics_comparison[key] = {
                r.run_id: r.metrics.get(key, 0.0) for r in runs
            }

        # Find best run by F1 (preferred), then AUC, then first available metric
        best_run_id = ""
        if runs and runs[0].metrics:
            preferred = ["f1", "f1_score", "auc_roc", "auc", "accuracy"]
            primary_metric = next(
                (m for m in preferred if m in all_metric_keys),
                sorted(all_metric_keys)[0],
            )
            best_run_id = max(runs, key=lambda r: r.metrics.get(primary_metric, 0.0)).run_id

        return ComparisonReport(
            run_ids=run_ids,
            metrics_comparison=metrics_comparison,
            best_run_id=best_run_id,
            summary=f"Compared {len(runs)} runs across {len(all_metric_keys)} metrics",
        )

    async def list_runs(self, experiment_name: str, limit: int = 20) -> list[ExperimentRun]:
        exp_dir = self.experiments_dir / experiment_name
        if not exp_dir.exists():
            return []

        runs = []
        for run_file in sorted(exp_dir.glob("*.json"), reverse=True)[:limit]:
            data = json.loads(run_file.read_text())
            runs.append(ExperimentRun(
                run_id=data["run_id"],
                experiment_name=data["experiment_name"],
                params=data.get("params", {}),
                metrics=data.get("metrics", {}),
                tags=data.get("tags", {}),
            ))
        return runs
