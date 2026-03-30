"""Tests for Pipeline orchestrator."""

import pytest

from mlops_agents.core.audit import SQLiteAuditStore
from mlops_agents.core.config import PipelineConfig
from mlops_agents.core.event import LocalAsyncEventBus
from mlops_agents.core.pipeline import Pipeline
from mlops_agents.providers.local.mlflow import LocalMLPlatform
from mlops_agents.providers.protocols import ModelArtifact
from mlops_agents.providers.registry import ProviderRegistry, Providers
from mlops_agents.core.config import ProviderConfig


@pytest.fixture
def bus():
    return LocalAsyncEventBus()


@pytest.fixture
def store(tmp_path):
    return SQLiteAuditStore(db_path=tmp_path / "test.db")


@pytest.fixture
def providers(bus, tmp_path):
    return ProviderRegistry.from_config(
        ProviderConfig(backend="local", local={"base_dir": str(tmp_path / "mlops")}),
        event_bus=bus,
    )


def _simple_config(stages: dict | None = None) -> PipelineConfig:
    if stages is None:
        stages = {
            "validate": {
                "agent": "cicd",
                "on_success": ["evaluate"],
                "on_failure": [],
                "params": {"min_rows": 10},
            },
            "evaluate": {
                "agent": "evaluation",
                "on_success": [],
                "on_failure": [],
                "params": {"min_improvement": 0.001},
            },
        }
    return PipelineConfig.from_dict({
        "name": "test-pipeline",
        "reasoning": {"engine": "static"},
        "provider": {"backend": "local"},
        "stages": stages,
    })


class TestPipelineExecution:
    @pytest.mark.asyncio
    async def test_simple_two_stage_pipeline(self, bus, store, providers):
        config = _simple_config()
        pipeline = Pipeline(config=config, event_bus=bus, audit_store=store, providers=providers)

        trace = await pipeline.run(
            initial_payload={
                "num_rows": 1000,
                "model_name": "test-model",
                "metrics": {"f1": 0.85},
            },
        )

        assert len(trace.decisions) == 2
        assert trace.decisions[0].agent_name == "cicd"
        assert trace.decisions[1].agent_name == "evaluation"
        assert trace.status in ("completed", "failed")

    @pytest.mark.asyncio
    async def test_single_stage_pipeline(self, bus, store, providers):
        config = PipelineConfig.from_dict({
            "name": "single",
            "reasoning": {"engine": "static"},
            "stages": {"validate": {"agent": "cicd", "params": {"min_rows": 10}}},
        })
        pipeline = Pipeline(config=config, event_bus=bus, audit_store=store, providers=providers)

        trace = await pipeline.run(initial_payload={"num_rows": 500})

        assert len(trace.decisions) == 1
        assert trace.decisions[0].agent_name == "cicd"
        assert trace.status == "completed"

    @pytest.mark.asyncio
    async def test_failure_routes_to_on_failure(self, bus, store, providers):
        config = PipelineConfig.from_dict({
            "name": "failure-test",
            "reasoning": {"engine": "static"},
            "stages": {
                "validate": {
                    "agent": "cicd",
                    "on_success": ["evaluate"],
                    "on_failure": ["feedback"],
                    "params": {"min_rows": 10000},  # High threshold = will fail
                },
                "evaluate": {"agent": "evaluation", "params": {}},
                "feedback": {"agent": "feedback", "params": {}},
            },
        })
        pipeline = Pipeline(config=config, event_bus=bus, audit_store=store, providers=providers)

        trace = await pipeline.run(initial_payload={"num_rows": 50})

        assert trace.decisions[0].agent_name == "cicd"
        assert trace.decisions[0].approved is False
        # Should route to feedback (on_failure), not evaluate
        assert trace.decisions[1].agent_name == "feedback"

    @pytest.mark.asyncio
    async def test_empty_pipeline(self, bus, store, providers):
        config = PipelineConfig.from_dict({
            "name": "empty",
            "reasoning": {"engine": "static"},
            "stages": {},
        })
        pipeline = Pipeline(config=config, event_bus=bus, audit_store=store, providers=providers)

        trace = await pipeline.run()
        assert len(trace.decisions) == 0
        assert trace.status == "completed"

    @pytest.mark.asyncio
    async def test_max_stages_safety_limit(self, bus, store, providers):
        # Create a cycle: a -> b -> a (would loop forever)
        config = PipelineConfig.from_dict({
            "name": "cycle-test",
            "reasoning": {"engine": "static"},
            "stages": {
                "validate": {
                    "agent": "cicd",
                    "on_success": ["retrain"],
                    "params": {"min_rows": 10},
                },
                "retrain": {
                    "agent": "retraining",
                    "on_success": ["validate"],
                    "params": {},
                },
            },
        })
        pipeline = Pipeline(config=config, event_bus=bus, audit_store=store, providers=providers)

        trace = await pipeline.run(initial_payload={"num_rows": 500}, max_stages=5)

        assert len(trace.decisions) == 5
        assert trace.status == "failed"  # Hit max stages


class TestPipelineTraceId:
    @pytest.mark.asyncio
    async def test_trace_id_generated(self, bus, store, providers):
        config = _simple_config({"validate": {"agent": "cicd", "params": {"min_rows": 10}}})
        pipeline = Pipeline(config=config, event_bus=bus, audit_store=store, providers=providers)

        trace = await pipeline.run(initial_payload={"num_rows": 500})
        assert trace.trace_id.startswith("pipe-")

    @pytest.mark.asyncio
    async def test_all_decisions_share_trace_id(self, bus, store, providers):
        config = _simple_config()
        pipeline = Pipeline(config=config, event_bus=bus, audit_store=store, providers=providers)

        trace = await pipeline.run(initial_payload={"num_rows": 500, "metrics": {"f1": 0.9}})

        trace_ids = {d.trace_id for d in trace.decisions}
        assert len(trace_ids) == 1  # All same trace_id


class TestPipelineAuditPersistence:
    @pytest.mark.asyncio
    async def test_decisions_persisted(self, bus, store, providers):
        config = _simple_config()
        pipeline = Pipeline(config=config, event_bus=bus, audit_store=store, providers=providers)

        trace = await pipeline.run(initial_payload={"num_rows": 500, "metrics": {"f1": 0.9}})

        stored = await store.get_trace(trace.trace_id)
        assert len(stored) == len(trace.decisions)

    @pytest.mark.asyncio
    async def test_pipeline_trace_persisted(self, bus, store, providers):
        config = _simple_config({"validate": {"agent": "cicd", "params": {"min_rows": 10}}})
        pipeline = Pipeline(config=config, event_bus=bus, audit_store=store, providers=providers)

        trace = await pipeline.run(initial_payload={"num_rows": 500})

        stored_trace = await store.get_pipeline_trace(trace.trace_id)
        assert stored_trace is not None
        assert stored_trace.pipeline_name == "test-pipeline"
        assert stored_trace.status == trace.status


class TestPipelineEvents:
    @pytest.mark.asyncio
    async def test_pipeline_start_event_emitted(self, bus, store, providers):
        received = []

        async def handler(event):
            received.append(event)

        await bus.subscribe("pipeline.*", handler)

        config = _simple_config({"validate": {"agent": "cicd", "params": {"min_rows": 10}}})
        pipeline = Pipeline(config=config, event_bus=bus, audit_store=store, providers=providers)

        await pipeline.run(initial_payload={"num_rows": 500})

        event_types = [e.type for e in received]
        assert "pipeline.started" in event_types
        assert "pipeline.completed" in event_types


class TestPipelineFromYaml:
    @pytest.mark.asyncio
    async def test_load_from_yaml(self, tmp_path):
        yaml_content = """
name: yaml-test
reasoning:
  engine: static
provider:
  backend: local
  local:
    base_dir: {base_dir}
stages:
  validate:
    agent: cicd
    params:
      min_rows: 10
""".format(base_dir=str(tmp_path / "mlops"))

        yaml_path = tmp_path / "test_pipeline.yaml"
        yaml_path.write_text(yaml_content)

        pipeline = Pipeline.from_yaml(str(yaml_path))
        assert pipeline.config.name == "yaml-test"

        trace = await pipeline.run(initial_payload={"num_rows": 500})
        assert len(trace.decisions) == 1
