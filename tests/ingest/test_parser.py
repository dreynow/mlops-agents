"""Tests for notebook parser."""

import json
from pathlib import Path

from mlops_agents.ingest.parser import (
    SectionType,
    analyze_notebook,
    detect_blueprint_tags,
    detect_model_type,
    extract_imports,
    extract_metrics,
    infer_sections,
    parse_notebook,
)


def _make_notebook(cells: list[dict], path: Path) -> Path:
    """Create a minimal .ipynb file."""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"language": "python"}},
        "cells": [
            {
                "cell_type": c.get("type", "code"),
                "source": [line + "\n" for line in c["source"].split("\n")],
                "metadata": c.get("metadata", {}),
                "outputs": c.get("outputs", []),
            }
            for c in cells
        ],
    }
    nb_path = path / "test.ipynb"
    nb_path.write_text(json.dumps(nb))
    return nb_path


class TestBlueprintMode:
    def test_detects_all_tags(self, tmp_path):
        nb = _make_notebook(
            [
                {"source": "# mlops: imports\nimport pandas as pd\nimport sklearn"},
                {"source": "# mlops: data-loading\ndf = pd.read_csv('data.csv')"},
                {"source": "# mlops: training\nmodel.fit(X, y)"},
                {"source": "# mlops: metrics\nf1 = f1_score(y, pred)"},
            ],
            tmp_path,
        )
        cells = parse_notebook(nb)
        sections = detect_blueprint_tags(cells)

        assert SectionType.IMPORTS in sections
        assert SectionType.DATA_LOADING in sections
        assert SectionType.TRAINING in sections
        assert SectionType.METRICS in sections

    def test_cells_tagged_with_confidence_1(self, tmp_path):
        nb = _make_notebook(
            [{"source": "# mlops: training\nmodel.fit(X, y)"}],
            tmp_path,
        )
        cells = parse_notebook(nb)
        sections = detect_blueprint_tags(cells)
        assert sections[SectionType.TRAINING][0].confidence == 1.0

    def test_analyze_returns_blueprint_mode(self, tmp_path):
        nb = _make_notebook(
            [
                {"source": "# mlops: training\nmodel.fit(X, y)"},
                {"source": "# mlops: metrics\nmetrics = {'f1': 0.9}"},
            ],
            tmp_path,
        )
        structure = analyze_notebook(nb)
        assert structure.mode == "blueprint"
        assert len(structure.warnings) == 0


class TestManifestMode:
    def test_manifest_maps_cells(self, tmp_path):
        nb = _make_notebook(
            [
                {"source": "import pandas as pd"},  # cell 0
                {"source": "df = pd.read_csv('data.csv')"},  # cell 1
                {"source": "model.fit(X, y)"},  # cell 2
                {"source": "f1 = f1_score(y, p)"},  # cell 3
                {
                    "source": (
                        "# mlops: manifest\n"
                        "# cell 0: imports\n"
                        "# cell 1: data-loading\n"
                        "# cell 2: training\n"
                        "# cell 3: metrics"
                    )
                },  # cell 4
            ],
            tmp_path,
        )
        structure = analyze_notebook(nb)
        assert structure.mode == "manifest"
        assert structure.has_section(SectionType.IMPORTS)
        assert structure.has_section(SectionType.TRAINING)
        assert structure.has_section(SectionType.METRICS)

    def test_manifest_range(self, tmp_path):
        nb = _make_notebook(
            [
                {"source": "x = 1"},  # cell 0
                {"source": "model = RF()"},  # cell 1
                {"source": "model.fit(X, y)"},  # cell 2
                {"source": "pred = model.predict(X)"},  # cell 3
                {"source": ("# mlops: manifest\n# cell 1-2: training\n# cell 3: evaluation")},
            ],
            tmp_path,
        )
        structure = analyze_notebook(nb)
        assert structure.mode == "manifest"
        training_cells = structure.sections[SectionType.TRAINING]
        assert len(training_cells) == 2
        assert training_cells[0].index == 1
        assert training_cells[1].index == 2

    def test_manifest_confidence_is_1(self, tmp_path):
        nb = _make_notebook(
            [
                {"source": "model.fit(X, y)"},
                {"source": ("# mlops: manifest\n# cell 0: training")},
            ],
            tmp_path,
        )
        structure = analyze_notebook(nb)
        assert structure.sections[SectionType.TRAINING][0].confidence == 1.0

    def test_manifest_takes_priority_over_inline(self, tmp_path):
        nb = _make_notebook(
            [
                {"source": "# mlops: imports\nimport pandas"},  # inline tag
                {"source": "model.fit(X, y)"},  # cell 1
                {"source": ("# mlops: manifest\n# cell 1: training")},
            ],
            tmp_path,
        )
        structure = analyze_notebook(nb)
        # Manifest wins over inline
        assert structure.mode == "manifest"

    def test_manifest_warns_on_missing_sections(self, tmp_path):
        nb = _make_notebook(
            [
                {"source": "import pandas"},
                {"source": ("# mlops: manifest\n# cell 0: imports")},
            ],
            tmp_path,
        )
        structure = analyze_notebook(nb)
        assert SectionType.TRAINING in structure.missing_sections


class TestInferenceMode:
    def test_detects_imports(self, tmp_path):
        nb = _make_notebook(
            [
                {
                    "source": (
                        "import pandas as pd\nimport numpy as np\nfrom sklearn.ensemble import RF"
                    )
                }
            ],
            tmp_path,
        )
        cells = parse_notebook(nb)
        sections = infer_sections(cells)
        assert SectionType.IMPORTS in sections

    def test_detects_training(self, tmp_path):
        nb = _make_notebook(
            [{"source": "model = RandomForestClassifier()\nmodel.fit(X_train, y_train)"}],
            tmp_path,
        )
        cells = parse_notebook(nb)
        sections = infer_sections(cells)
        assert SectionType.TRAINING in sections
        assert sections[SectionType.TRAINING][0].confidence >= 0.9

    def test_detects_data_loading(self, tmp_path):
        nb = _make_notebook(
            [
                {
                    "source": (
                        "df = pd.read_csv('transactions.csv')\n"
                        "X_train, X_test = train_test_split(X)"
                    )
                }
            ],
            tmp_path,
        )
        cells = parse_notebook(nb)
        sections = infer_sections(cells)
        assert SectionType.DATA_LOADING in sections

    def test_detects_metrics(self, tmp_path):
        nb = _make_notebook(
            [{"source": "acc = accuracy_score(y_test, y_pred)\nf1 = f1_score(y_test, y_pred)"}],
            tmp_path,
        )
        cells = parse_notebook(nb)
        sections = infer_sections(cells)
        assert SectionType.METRICS in sections

    def test_analyze_returns_inferred_mode(self, tmp_path):
        nb = _make_notebook(
            [
                {"source": "import pandas as pd"},
                {"source": "df = pd.read_csv('data.csv')"},
                {"source": "model.fit(X, y)"},
            ],
            tmp_path,
        )
        structure = analyze_notebook(nb)
        assert structure.mode == "inferred"
        assert any("No # mlops: tags" in w for w in structure.warnings)

    def test_warns_about_missing_required_sections(self, tmp_path):
        nb = _make_notebook(
            [{"source": "import pandas as pd"}],
            tmp_path,
        )
        structure = analyze_notebook(nb)
        assert len(structure.missing_sections) > 0
        assert SectionType.TRAINING in structure.missing_sections


class TestMetricDetection:
    def test_detects_classification_metrics(self, tmp_path):
        nb = _make_notebook(
            [
                {
                    "source": (
                        "from sklearn.metrics import f1_score, accuracy_score\n"
                        "acc = accuracy_score(y, p)"
                    )
                }
            ],
            tmp_path,
        )
        cells = parse_notebook(nb)
        metrics = extract_metrics(cells)
        assert "f1" in metrics
        assert "accuracy" in metrics

    def test_detects_regression_metrics(self, tmp_path):
        nb = _make_notebook(
            [{"source": "mse = mean_squared_error(y, p)\nr2 = r2_score(y, p)"}],
            tmp_path,
        )
        cells = parse_notebook(nb)
        metrics = extract_metrics(cells)
        assert "mse" in metrics
        assert "r2" in metrics


class TestModelDetection:
    def test_detects_random_forest(self, tmp_path):
        nb = _make_notebook(
            [{"source": "model = RandomForestClassifier(n_estimators=100)"}],
            tmp_path,
        )
        cells = parse_notebook(nb)
        assert detect_model_type(cells) == "random_forest"

    def test_detects_xgboost(self, tmp_path):
        nb = _make_notebook(
            [{"source": "model = XGBClassifier()"}],
            tmp_path,
        )
        cells = parse_notebook(nb)
        assert detect_model_type(cells) == "xgboost"

    def test_detects_neural_network(self, tmp_path):
        nb = _make_notebook(
            [{"source": "model = Sequential([Dense(64), Dense(1)])"}],
            tmp_path,
        )
        cells = parse_notebook(nb)
        assert detect_model_type(cells) == "neural_network"

    def test_unknown_model(self, tmp_path):
        nb = _make_notebook(
            [{"source": "x = 1 + 2"}],
            tmp_path,
        )
        cells = parse_notebook(nb)
        assert detect_model_type(cells) == "unknown"


class TestImportExtraction:
    def test_extracts_imports(self, tmp_path):
        nb = _make_notebook(
            [
                {
                    "source": (
                        "import pandas as pd\n"
                        "from sklearn.ensemble import RandomForestClassifier\n"
                        "x = 1"
                    )
                }
            ],
            tmp_path,
        )
        cells = parse_notebook(nb)
        imports = extract_imports(cells)
        assert "import pandas as pd" in imports
        assert any("sklearn" in i for i in imports)

    def test_skips_notebook_imports(self, tmp_path):
        nb = _make_notebook(
            [{"source": "from IPython.display import display\nimport matplotlib.pyplot as plt"}],
            tmp_path,
        )
        cells = parse_notebook(nb)
        imports = extract_imports(cells)
        assert len(imports) == 0
