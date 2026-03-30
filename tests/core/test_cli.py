"""Tests for CLI commands."""

import pytest
from typer.testing import CliRunner
from pathlib import Path

from mlops_agents.cli.main import app

runner = CliRunner()


class TestCLIRun:
    def test_run_nonexistent_file(self):
        result = runner.invoke(app, ["run", "nonexistent.yaml"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_run_valid_pipeline(self, tmp_path):
        yaml_content = """
name: cli-test
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

        yaml_path = tmp_path / "pipeline.yaml"
        yaml_path.write_text(yaml_content)

        result = runner.invoke(app, ["run", str(yaml_path)])
        assert result.exit_code == 0
        assert "cli-test" in result.output

    def test_run_with_stage_flag(self, tmp_path):
        yaml_content = """
name: stage-test
reasoning:
  engine: static
provider:
  backend: local
  local:
    base_dir: {base_dir}
stages:
  validate:
    agent: cicd
    on_success: [evaluate]
    params:
      min_rows: 10
  evaluate:
    agent: evaluation
    params: {{}}
""".format(base_dir=str(tmp_path / "mlops"))

        yaml_path = tmp_path / "pipeline.yaml"
        yaml_path.write_text(yaml_content)

        result = runner.invoke(app, ["run", str(yaml_path), "--stage", "validate"])
        assert result.exit_code == 0


class TestCLIAudit:
    def test_audit_empty_db(self, tmp_path):
        db_path = tmp_path / "empty.db"
        result = runner.invoke(app, ["audit", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "No decisions" in result.output

    def test_audit_nonexistent_trace(self, tmp_path):
        db_path = tmp_path / "empty.db"
        result = runner.invoke(app, ["audit", "--trace", "pipe-nonexistent", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "No decisions" in result.output


class TestCLIStatus:
    def test_status_empty(self, tmp_path):
        db_path = tmp_path / "empty.db"
        result = runner.invoke(app, ["status", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "No pipeline activity" in result.output


class TestCLIHelp:
    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Agentic MLOps" in result.output

    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "pipeline" in result.output.lower()

    def test_audit_help(self):
        result = runner.invoke(app, ["audit", "--help"])
        assert result.exit_code == 0

    def test_status_help(self):
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0
