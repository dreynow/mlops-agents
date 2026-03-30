"""Notebook parser - extracts structure from .ipynb files.

Two modes:
  1. Blueprint mode: notebook has `# mlops:` tags. Extraction is deterministic.
  2. Inference mode: no tags. LLM analyzes cells and infers structure with
     confidence levels. Honest about what it can and can't detect.

Blueprint tags (add to first line of a cell):
  # mlops: imports
  # mlops: data-loading
  # mlops: feature-engineering
  # mlops: training
  # mlops: evaluation
  # mlops: metrics
  # mlops: config
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SectionType(str, Enum):
    IMPORTS = "imports"
    DATA_LOADING = "data-loading"
    FEATURE_ENGINEERING = "feature-engineering"
    TRAINING = "training"
    EVALUATION = "evaluation"
    METRICS = "metrics"
    CONFIG = "config"
    UNKNOWN = "unknown"


# Regex to match blueprint tags: # mlops: section-name
BLUEPRINT_TAG_RE = re.compile(r"^\s*#\s*mlops:\s*(\S+)", re.MULTILINE)

# Regex to match manifest entries: # cell 5: training  or  # cell 7-9: data-loading
MANIFEST_ENTRY_RE = re.compile(r"^\s*#\s*cell\s+(\d+)(?:\s*-\s*(\d+))?\s*:\s*(\S+)", re.MULTILINE)

# All recognized blueprint tag values
VALID_TAGS = {s.value for s in SectionType if s != SectionType.UNKNOWN}


@dataclass
class NotebookCell:
    """A single cell from a Jupyter notebook."""

    index: int
    cell_type: str  # "code" or "markdown"
    source: str
    outputs: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    section: SectionType = SectionType.UNKNOWN
    confidence: float = 0.0


@dataclass
class NotebookStructure:
    """Parsed notebook with sections identified."""

    path: str
    cells: list[NotebookCell]
    mode: str  # "blueprint" or "inferred"
    sections: dict[SectionType, list[NotebookCell]]
    warnings: list[str] = field(default_factory=list)
    detected_imports: list[str] = field(default_factory=list)
    detected_metrics: list[str] = field(default_factory=list)
    detected_model_type: str = ""

    def get_section_code(self, section: SectionType) -> str:
        """Get concatenated code for a section."""
        cells = self.sections.get(section, [])
        return "\n\n".join(c.source for c in cells if c.cell_type == "code")

    def has_section(self, section: SectionType) -> bool:
        return section in self.sections and len(self.sections[section]) > 0

    @property
    def missing_sections(self) -> list[SectionType]:
        """Required sections that are missing."""
        required = [
            SectionType.TRAINING,
            SectionType.METRICS,
        ]
        return [s for s in required if not self.has_section(s)]


def parse_notebook(path: str | Path) -> list[NotebookCell]:
    """Parse a .ipynb file into a list of NotebookCells."""
    path = Path(path)
    with open(path) as f:
        nb = json.load(f)

    cells = []
    for i, cell in enumerate(nb.get("cells", [])):
        source = "".join(cell.get("source", []))
        cell_type = cell.get("cell_type", "code")
        outputs = cell.get("outputs", [])
        metadata = cell.get("metadata", {})
        tags = metadata.get("tags", [])

        cells.append(
            NotebookCell(
                index=i,
                cell_type=cell_type,
                source=source,
                outputs=outputs,
                tags=tags,
            )
        )

    return cells


def detect_blueprint_tags(cells: list[NotebookCell]) -> dict[SectionType, list[NotebookCell]]:
    """Scan cells for `# mlops:` blueprint tags.

    Returns a dict of section -> cells mapping. Only returns sections
    that have at least one tagged cell.
    """
    sections: dict[SectionType, list[NotebookCell]] = {}

    for cell in cells:
        if cell.cell_type != "code":
            continue

        match = BLUEPRINT_TAG_RE.search(cell.source)
        if match:
            tag_value = match.group(1).lower().strip()
            if tag_value in VALID_TAGS:
                section = SectionType(tag_value)
                cell.section = section
                cell.confidence = 1.0
                sections.setdefault(section, []).append(cell)

    return sections


def detect_manifest(cells: list[NotebookCell]) -> dict[SectionType, list[NotebookCell]]:
    """Scan for a manifest cell that maps cell numbers to sections.

    The manifest is a single cell (usually the last) tagged with
    `# mlops: manifest` followed by lines like:
        # cell 1: imports
        # cell 4: data-loading
        # cell 7-9: training
        # cell 12: metrics

    This lets data scientists declare structure without touching
    their working cells. One summary cell at the bottom.
    """
    # Find the manifest cell
    manifest_source = ""
    for cell in cells:
        if cell.cell_type != "code":
            continue
        if re.search(r"^\s*#\s*mlops:\s*manifest", cell.source, re.MULTILINE):
            manifest_source = cell.source
            break

    if not manifest_source:
        return {}

    # Parse manifest entries
    sections: dict[SectionType, list[NotebookCell]] = {}
    cell_count = len(cells)

    for match in MANIFEST_ENTRY_RE.finditer(manifest_source):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        tag_value = match.group(3).lower().strip()

        if tag_value not in VALID_TAGS:
            continue

        section = SectionType(tag_value)

        for idx in range(start, end + 1):
            if 0 <= idx < cell_count:
                target = cells[idx]
                if target.cell_type == "code":
                    target.section = section
                    target.confidence = 1.0
                    sections.setdefault(section, []).append(target)

    return sections


def infer_sections(cells: list[NotebookCell]) -> dict[SectionType, list[NotebookCell]]:
    """Heuristic-based section inference when no blueprint tags exist.

    Uses import patterns, function calls, and variable names to guess
    what each cell does. Returns confidence levels for each assignment.
    """
    sections: dict[SectionType, list[NotebookCell]] = {}

    for cell in cells:
        if cell.cell_type != "code":
            continue

        source = cell.source.strip()
        if not source:
            continue

        section, confidence = _classify_cell(source)
        cell.section = section
        cell.confidence = confidence

        if section != SectionType.UNKNOWN:
            sections.setdefault(section, []).append(cell)

    return sections


def _classify_cell(source: str) -> tuple[SectionType, float]:
    """Classify a code cell by analyzing its content.

    Returns (section_type, confidence).
    """
    lines = source.strip().split("\n")

    # --- Imports ---
    import_lines = sum(1 for line in lines if re.match(r"^\s*(import |from \S+ import )", line))
    if import_lines > 0 and import_lines / max(len(lines), 1) > 0.5:
        return SectionType.IMPORTS, 0.95

    # --- Metrics computation ---
    metric_patterns = [
        r"(accuracy_score|f1_score|precision_score|recall_score|roc_auc_score)",
        r"(classification_report|confusion_matrix)",
        r"metrics\s*=\s*\{",
        r"(mean_squared_error|r2_score|mean_absolute_error)",
    ]
    for pattern in metric_patterns:
        if re.search(pattern, source):
            return SectionType.METRICS, 0.9

    # --- Evaluation ---
    eval_patterns = [
        r"\.(predict|predict_proba|score)\(",
        r"(y_pred|y_hat|predictions)\s*=",
        r"model\.evaluate\(",
    ]
    for pattern in eval_patterns:
        if re.search(pattern, source):
            return SectionType.EVALUATION, 0.85

    # --- Training ---
    training_patterns = [
        r"\.fit\(",
        r"model\.train\(",
        r"trainer\.train\(",
        r"\.compile\(.*optimizer",
    ]
    for pattern in training_patterns:
        if re.search(pattern, source):
            return SectionType.TRAINING, 0.9

    # --- Data loading ---
    data_patterns = [
        r"(pd\.read_csv|pd\.read_parquet|pd\.read_json|pd\.read_sql)",
        r"(load_dataset|fetch_\w+|from_csv)",
        r"(train_test_split|StratifiedKFold)",
        r"(X_train|X_test|y_train|y_test)\s*=",
    ]
    for pattern in data_patterns:
        if re.search(pattern, source):
            return SectionType.DATA_LOADING, 0.85

    # --- Feature engineering ---
    feature_patterns = [
        r"(StandardScaler|MinMaxScaler|LabelEncoder|OneHotEncoder)",
        r"\.transform\(",
        r"\.fit_transform\(",
        r"(feature_\w+|encode_|normalize_)",
    ]
    for pattern in feature_patterns:
        if re.search(pattern, source):
            return SectionType.FEATURE_ENGINEERING, 0.8

    # --- Config ---
    config_patterns = [
        r"(EPOCHS|BATCH_SIZE|LEARNING_RATE|N_ESTIMATORS)\s*=",
        r"(hyperparams|config|params)\s*=\s*\{",
    ]
    for pattern in config_patterns:
        if re.search(pattern, source):
            return SectionType.CONFIG, 0.8

    return SectionType.UNKNOWN, 0.0


def extract_imports(cells: list[NotebookCell]) -> list[str]:
    """Extract all import statements from the notebook."""
    imports = []
    for cell in cells:
        if cell.cell_type != "code":
            continue
        for line in cell.source.split("\n"):
            line = line.strip()
            if line.startswith("import ") or line.startswith("from "):
                # Skip notebook-specific imports
                if any(skip in line for skip in ["IPython", "display", "widgets", "matplotlib"]):
                    continue
                imports.append(line)
    return sorted(set(imports))


def extract_metrics(cells: list[NotebookCell]) -> list[str]:
    """Detect which metrics are computed in the notebook."""
    metric_names = []
    patterns = {
        "accuracy": r"accuracy_score|\.accuracy",
        "f1": r"f1_score",
        "precision": r"precision_score",
        "recall": r"recall_score",
        "auc_roc": r"roc_auc_score|auc",
        "mse": r"mean_squared_error",
        "mae": r"mean_absolute_error",
        "r2": r"r2_score",
    }

    all_code = "\n".join(c.source for c in cells if c.cell_type == "code")
    for name, pattern in patterns.items():
        if re.search(pattern, all_code):
            metric_names.append(name)

    return metric_names


def detect_model_type(cells: list[NotebookCell]) -> str:
    """Detect what kind of model is being trained."""
    all_code = "\n".join(c.source for c in cells if c.cell_type == "code")

    patterns = {
        "random_forest": r"RandomForest(Classifier|Regressor)",
        "xgboost": r"(XGB|xgb)\w*(Classifier|Regressor|\.train)",
        "lightgbm": r"(LGB|lgb)\w*(Classifier|Regressor|\.train)",
        "logistic_regression": r"LogisticRegression",
        "svm": r"(SVC|SVR|LinearSVC)",
        "neural_network": r"(Sequential|nn\.Module|keras\.Model|torch\.nn)",
        "gradient_boosting": r"GradientBoosting(Classifier|Regressor)",
        "linear_regression": r"LinearRegression",
        "decision_tree": r"DecisionTree(Classifier|Regressor)",
    }

    for name, pattern in patterns.items():
        if re.search(pattern, all_code):
            return name

    return "unknown"


def analyze_notebook(path: str | Path) -> NotebookStructure:
    """Full notebook analysis - blueprint tags or heuristic inference.

    This is the main entry point. Returns a NotebookStructure with
    sections identified, imports extracted, metrics detected, and
    warnings for anything that couldn't be determined.
    """
    path = Path(path)
    cells = parse_notebook(path)

    # Priority: 1) manifest cell, 2) inline tags, 3) heuristic inference
    manifest_sections = detect_manifest(cells)
    blueprint_sections = detect_blueprint_tags(cells)

    if manifest_sections:
        mode = "manifest"
        sections = manifest_sections
        warnings = []

        missing = []
        for required in [SectionType.TRAINING, SectionType.METRICS]:
            if required not in sections:
                missing.append(required.value)
        if missing:
            warnings.append(f"Missing required sections: {', '.join(missing)}")
    elif blueprint_sections:
        mode = "blueprint"
        sections = blueprint_sections
        warnings = []

        missing = []
        for required in [SectionType.TRAINING, SectionType.METRICS]:
            if required not in sections:
                missing.append(required.value)
        if missing:
            warnings.append(f"Missing required sections: {', '.join(missing)}")
    else:
        mode = "inferred"
        sections = infer_sections(cells)
        warnings = [
            "No # mlops: tags found. Structure inferred from code patterns.",
        ]

        # Report confidence for each detected section
        for section_type, section_cells in sections.items():
            avg_conf = sum(c.confidence for c in section_cells) / len(section_cells)
            level = "high" if avg_conf >= 0.9 else "medium" if avg_conf >= 0.8 else "low"
            warnings.append(
                f"  {section_type.value}: cells "
                f"{','.join(str(c.index) for c in section_cells)} "
                f"(confidence: {level})"
            )

        # Warn about sections we couldn't detect
        if SectionType.TRAINING not in sections:
            warnings.append("  Could not detect training logic - add # mlops: training tag")
        if SectionType.METRICS not in sections:
            warnings.append("  Could not detect metrics - add # mlops: metrics tag")

    return NotebookStructure(
        path=str(path),
        cells=cells,
        mode=mode,
        sections=sections,
        warnings=warnings,
        detected_imports=extract_imports(cells),
        detected_metrics=extract_metrics(cells),
        detected_model_type=detect_model_type(cells),
    )
