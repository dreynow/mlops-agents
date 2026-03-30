"""Tests for code generator."""

import json
from pathlib import Path

from mlops_agents.ingest.generator import (
    generate_all,
    generate_pipeline_yaml,
    generate_requirements,
    generate_train_script,
)
from mlops_agents.ingest.parser import analyze_notebook


def _make_notebook(cells: list[dict], path: Path) -> Path:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": c.get("type", "code"),
                "source": [line + "\n" for line in c["source"].split("\n")],
                "metadata": {},
                "outputs": [],
            }
            for c in cells
        ],
    }
    nb_path = path / "test.ipynb"
    nb_path.write_text(json.dumps(nb))
    return nb_path


class TestGenerateTrainScript:
    def test_blueprint_notebook(self, tmp_path):
        nb = _make_notebook(
            [
                {
                    "source": (
                        "# mlops: imports\nimport pandas as pd\n"
                        "from sklearn.ensemble import RandomForestClassifier"
                    )
                },
                {
                    "source": (
                        "# mlops: data-loading\ndf = pd.read_csv('data.csv')\n"
                        "X_train, X_test, y_train, y_test = train_test_split(df)"
                    )
                },
                {
                    "source": (
                        "# mlops: training\n"
                        "model = RandomForestClassifier()\nmodel.fit(X_train, y_train)"
                    )
                },
                {
                    "source": (
                        "# mlops: metrics\nfrom sklearn.metrics import f1_score\n"
                        "metrics = {'f1': f1_score(y_test, model.predict(X_test))}"
                    )
                },
            ],
            tmp_path,
        )
        structure = analyze_notebook(nb)
        script = generate_train_script(structure)

        assert "import pandas" in script
        assert "def train_model" in script
        assert "def main():" in script
        assert "pickle.dump" in script
        assert "metrics.json" in script

    def test_inferred_notebook(self, tmp_path):
        nb = _make_notebook(
            [
                {"source": "import pandas as pd\nimport numpy as np"},
                {"source": "model = RandomForestClassifier()\nmodel.fit(X_train, y_train)"},
                {"source": "acc = accuracy_score(y_test, y_pred)"},
            ],
            tmp_path,
        )
        structure = analyze_notebook(nb)
        script = generate_train_script(structure)

        assert "def main():" in script
        assert "__name__" in script


class TestGeneratePipelineYaml:
    def test_classification_thresholds(self, tmp_path):
        nb = _make_notebook(
            [
                {"source": "# mlops: training\nmodel.fit(X, y)"},
                {"source": "# mlops: metrics\nf1 = f1_score(y, p)"},
            ],
            tmp_path,
        )
        structure = analyze_notebook(nb)
        yaml = generate_pipeline_yaml(structure)

        assert "min_improvement: 0.005" in yaml
        assert "max_fairness_delta: 0.05" in yaml
        assert "agent: evaluation" in yaml
        assert "agent: deployment" in yaml

    def test_includes_model_type_comment(self, tmp_path):
        nb = _make_notebook(
            [{"source": "model = RandomForestClassifier()\nmodel.fit(X, y)"}],
            tmp_path,
        )
        structure = analyze_notebook(nb)
        yaml = generate_pipeline_yaml(structure)
        assert "random_forest" in yaml


class TestGenerateRequirements:
    def test_maps_sklearn_to_scikit_learn(self, tmp_path):
        nb = _make_notebook(
            [{"source": "import sklearn\nimport pandas\nimport numpy"}],
            tmp_path,
        )
        structure = analyze_notebook(nb)
        reqs = generate_requirements(structure)
        assert "scikit-learn" in reqs
        assert "pandas" in reqs
        assert "numpy" in reqs
        assert "mlops-agents" in reqs

    def test_excludes_stdlib(self, tmp_path):
        nb = _make_notebook(
            [{"source": "import os\nimport json\nimport pandas"}],
            tmp_path,
        )
        structure = analyze_notebook(nb)
        reqs = generate_requirements(structure)
        assert "os" not in reqs.split("\n")
        assert "json" not in reqs.split("\n")


class TestGenerateAll:
    def test_creates_output_files(self, tmp_path):
        nb = _make_notebook(
            [
                {"source": "# mlops: imports\nimport pandas as pd"},
                {"source": "# mlops: training\nmodel.fit(X, y)"},
                {"source": "# mlops: metrics\nmetrics = {'f1': 0.9}"},
            ],
            tmp_path,
        )
        structure = analyze_notebook(nb)
        output = tmp_path / "output"
        files = generate_all(structure, output_dir=output)

        assert "train.py" in files
        assert "pipeline.yaml" in files
        assert "requirements.txt" in files
        assert (output / "train.py").exists()
        assert (output / "pipeline.yaml").exists()
        assert (output / "requirements.txt").exists()
