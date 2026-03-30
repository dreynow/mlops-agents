"""Tests for config system."""

import pytest
import tempfile
from pathlib import Path

from mlops_agents.core.config import (
    AuditConfig,
    EscalationConfig,
    PipelineConfig,
    ProviderConfig,
    ReasoningConfig,
    StageConfig,
)


class TestEscalationConfig:
    def test_default_threshold(self):
        config = EscalationConfig()
        assert config.threshold_for("any_stage") == 0.7

    def test_per_stage_override(self):
        config = EscalationConfig(
            default_confidence_threshold=0.7,
            per_stage={"deploy": 0.9, "evaluate": 0.8},
        )
        assert config.threshold_for("deploy") == 0.9
        assert config.threshold_for("evaluate") == 0.8
        assert config.threshold_for("cicd") == 0.7  # Falls back to default


class TestPipelineConfig:
    def test_from_dict(self):
        config = PipelineConfig.from_dict({
            "name": "test-pipeline",
            "reasoning": {"engine": "claude", "model": "claude-sonnet-4-20250514"},
            "provider": {"backend": "local"},
            "stages": {
                "validate": {"agent": "cicd", "on_success": ["train"]},
                "train": {"agent": "retraining", "on_success": ["evaluate"]},
            },
        })
        assert config.name == "test-pipeline"
        assert len(config.stages) == 2
        assert config.stages["validate"].agent == "cicd"
        assert config.stages["validate"].on_success == ["train"]

    def test_from_yaml(self, tmp_path):
        yaml_content = """
name: fraud-detection
trigger:
  schedule: "0 2 * * *"
  events:
    - data.new_batch

reasoning:
  engine: claude
  model: claude-sonnet-4-20250514

provider:
  backend: local

escalation:
  default_confidence_threshold: 0.7
  per_stage:
    deploy: 0.9

stages:
  validate:
    agent: cicd
    on_success: [train]
    on_failure: [alert_human]
  train:
    agent: retraining
    on_success: [evaluate]
  evaluate:
    agent: evaluation
    on_success: [deploy]
    escalation:
      when: confidence < 0.7
      to: human
  deploy:
    agent: deployment
    strategy: canary
    canary_duration: 30m
    canary_traffic: "5%"
    on_success: [monitor]
"""
        yaml_path = tmp_path / "pipeline.yaml"
        yaml_path.write_text(yaml_content)

        config = PipelineConfig.from_yaml(yaml_path)
        assert config.name == "fraud-detection"
        assert config.trigger.schedule == "0 2 * * *"
        assert config.trigger.events == ["data.new_batch"]
        assert config.reasoning.engine == "claude"
        assert config.escalation.threshold_for("deploy") == 0.9
        assert config.escalation.threshold_for("evaluate") == 0.7
        assert len(config.stages) == 4
        assert config.stages["deploy"].strategy == "canary"
        assert config.stages["deploy"].canary_traffic == "5%"

    def test_defaults(self):
        config = PipelineConfig()
        assert config.name == "default-pipeline"
        assert config.reasoning.engine == "claude"
        assert config.provider.backend == "local"
        assert config.escalation.default_confidence_threshold == 0.7
        assert config.audit.backend == "sqlite"

    def test_gcp_provider_config(self):
        config = PipelineConfig.from_dict({
            "provider": {
                "backend": "gcp",
                "gcp": {
                    "project_id": "my-project",
                    "region": "us-central1",
                },
            },
        })
        assert config.provider.backend == "gcp"
        assert config.provider.gcp["project_id"] == "my-project"

    def test_stage_continuous_mode(self):
        config = PipelineConfig.from_dict({
            "stages": {
                "monitor": {
                    "agent": "monitoring",
                    "mode": "continuous",
                    "check_interval": "15m",
                    "on_drift": ["retrain"],
                },
            },
        })
        stage = config.stages["monitor"]
        assert stage.mode == "continuous"
        assert stage.check_interval == "15m"
        assert stage.on_drift == ["retrain"]
