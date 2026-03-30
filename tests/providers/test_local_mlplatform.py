"""Tests for local ML platform provider."""

import pytest

from mlops_agents.providers.local.mlflow import LocalMLPlatform
from mlops_agents.providers.protocols import ExperimentRun, ModelArtifact


@pytest.fixture
def ml(tmp_path):
    return LocalMLPlatform(base_dir=str(tmp_path / "mlplatform"))


class TestLocalMLPlatform:
    @pytest.mark.asyncio
    async def test_log_and_get_experiment(self, ml):
        run_id = await ml.log_experiment(ExperimentRun(
            run_id="run-001",
            experiment_name="fraud-detection",
            params={"n_estimators": 100, "max_depth": 10},
            metrics={"f1": 0.834, "auc": 0.967},
        ))
        assert run_id == "run-001"

        run = await ml.get_run("run-001")
        assert run.metrics["f1"] == 0.834
        assert run.params["n_estimators"] == 100

    @pytest.mark.asyncio
    async def test_register_model(self, ml):
        version = await ml.register_model(ModelArtifact(
            model_name="fraud-detector",
            artifact_path="/models/model.pkl",
            metrics={"f1": 0.834},
        ))
        assert version.version == "v1"
        assert version.model_name == "fraud-detector"

    @pytest.mark.asyncio
    async def test_sequential_versions(self, ml):
        v1 = await ml.register_model(ModelArtifact(
            model_name="fraud-detector",
            artifact_path="/models/v1.pkl",
            metrics={"f1": 0.80},
        ))
        v2 = await ml.register_model(ModelArtifact(
            model_name="fraud-detector",
            artifact_path="/models/v2.pkl",
            metrics={"f1": 0.85},
        ))
        assert v1.version == "v1"
        assert v2.version == "v2"

    @pytest.mark.asyncio
    async def test_get_champion_returns_latest(self, ml):
        await ml.register_model(ModelArtifact(
            model_name="fraud-detector",
            artifact_path="/models/v1.pkl",
            metrics={"f1": 0.80},
        ))
        await ml.register_model(ModelArtifact(
            model_name="fraud-detector",
            artifact_path="/models/v2.pkl",
            metrics={"f1": 0.85},
        ))

        champion = await ml.get_champion("fraud-detector")
        assert champion is not None
        assert champion.version == "v2"

    @pytest.mark.asyncio
    async def test_get_champion_returns_production_stage(self, ml):
        await ml.register_model(ModelArtifact(
            model_name="fraud-detector",
            artifact_path="/models/v1.pkl",
            metrics={"f1": 0.80},
        ))
        v2 = await ml.register_model(ModelArtifact(
            model_name="fraud-detector",
            artifact_path="/models/v2.pkl",
            metrics={"f1": 0.85},
        ))
        await ml.register_model(ModelArtifact(
            model_name="fraud-detector",
            artifact_path="/models/v3.pkl",
            metrics={"f1": 0.83},
        ))

        # Promote v2 to production
        await ml.promote_model("fraud-detector", "v2", "production")

        champion = await ml.get_champion("fraud-detector")
        assert champion.version == "v2"

    @pytest.mark.asyncio
    async def test_promote_archives_previous(self, ml):
        v1 = await ml.register_model(ModelArtifact(
            model_name="m", artifact_path="/v1.pkl", metrics={"f1": 0.8},
        ))
        v2 = await ml.register_model(ModelArtifact(
            model_name="m", artifact_path="/v2.pkl", metrics={"f1": 0.85},
        ))

        await ml.promote_model("m", "v1", "production")
        await ml.promote_model("m", "v2", "production")

        champion = await ml.get_champion("m")
        assert champion.version == "v2"

    @pytest.mark.asyncio
    async def test_get_champion_nonexistent(self, ml):
        result = await ml.get_champion("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_compare_runs(self, ml):
        await ml.log_experiment(ExperimentRun(
            run_id="r1", experiment_name="exp", metrics={"f1": 0.80, "auc": 0.90},
        ))
        await ml.log_experiment(ExperimentRun(
            run_id="r2", experiment_name="exp", metrics={"f1": 0.85, "auc": 0.95},
        ))

        report = await ml.compare_runs(["r1", "r2"])
        assert report.best_run_id == "r2"
        assert "f1" in report.metrics_comparison
        assert report.metrics_comparison["f1"]["r2"] == 0.85

    @pytest.mark.asyncio
    async def test_list_runs(self, ml):
        for i in range(5):
            await ml.log_experiment(ExperimentRun(
                run_id=f"r{i}", experiment_name="exp", metrics={"f1": 0.8 + i * 0.01},
            ))

        runs = await ml.list_runs("exp", limit=3)
        assert len(runs) == 3

    @pytest.mark.asyncio
    async def test_list_runs_empty_experiment(self, ml):
        runs = await ml.list_runs("nonexistent")
        assert runs == []
