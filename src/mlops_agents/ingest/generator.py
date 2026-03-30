"""Code generator - turns parsed notebook structure into production files.

Generates:
  1. train.py - Clean training script from notebook cells
  2. pipeline.yaml - Pipeline config with detected thresholds
  3. requirements.txt - Dependencies from notebook imports

Two paths:
  - Deterministic: when blueprint tags are present, assembles sections directly
  - LLM-assisted: when inferred, uses Claude to clean up and refactor
"""

from __future__ import annotations

import re
from pathlib import Path

import structlog

from mlops_agents.ingest.parser import NotebookStructure, SectionType

logger = structlog.get_logger()

# Map pip package names from import names
IMPORT_TO_PACKAGE = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "attr": "attrs",
    "dotenv": "python-dotenv",
}


def generate_train_script(structure: NotebookStructure) -> str:
    """Generate a clean train.py from the parsed notebook."""
    parts = []

    # Header
    parts.append('"""Training script generated from notebook by mlops-agents ingest."""\n')
    parts.append("import json")
    parts.append("import os")
    parts.append("import pickle")
    parts.append("from pathlib import Path")
    parts.append("")

    # Imports from notebook
    seen_imports = {"json", "os", "pickle", "pathlib"}
    for imp in structure.detected_imports:
        # Extract module name to check for duplicates
        match = re.match(r"(?:from\s+(\S+)|import\s+(\S+))", imp)
        if match:
            module = match.group(1) or match.group(2)
            root_module = module.split(".")[0]
            if root_module not in seen_imports:
                parts.append(imp)
                seen_imports.add(root_module)
    parts.append("")

    # Config section
    config_code = structure.get_section_code(SectionType.CONFIG)
    if config_code:
        parts.append("# --- Configuration ---")
        parts.append(config_code)
        parts.append("")

    # Data loading
    data_code = structure.get_section_code(SectionType.DATA_LOADING)
    if data_code:
        parts.append("")
        parts.append("def load_data():")
        # Indent the code
        for line in data_code.split("\n"):
            if line.strip():
                parts.append(f"    {line}")
            else:
                parts.append("")
        # Try to detect the return variables
        returns = _detect_return_vars(
            data_code, ["X_train", "X_test", "y_train", "y_test", "X", "y", "df", "data"]
        )
        if returns:
            parts.append(f"    return {', '.join(returns)}")
        parts.append("")

    # Feature engineering
    feat_code = structure.get_section_code(SectionType.FEATURE_ENGINEERING)
    if feat_code:
        parts.append("")
        parts.append("def engineer_features(data):")
        for line in feat_code.split("\n"):
            if line.strip():
                parts.append(f"    {line}")
            else:
                parts.append("")
        parts.append("    return data")
        parts.append("")

    # Training
    train_code = structure.get_section_code(SectionType.TRAINING)
    if train_code:
        parts.append("")
        parts.append("def train_model(X_train, y_train):")
        for line in train_code.split("\n"):
            if line.strip():
                parts.append(f"    {line}")
            else:
                parts.append("")
        # Detect model variable name
        model_var = _detect_model_var(train_code)
        if model_var:
            parts.append(f"    return {model_var}")
        else:
            parts.append("    return model")
        parts.append("")

    # Evaluation + Metrics
    eval_code = structure.get_section_code(SectionType.EVALUATION)
    metrics_code = structure.get_section_code(SectionType.METRICS)
    combined_eval = "\n".join(filter(None, [eval_code, metrics_code]))

    if combined_eval:
        parts.append("")
        parts.append("def evaluate_model(model, X_test, y_test):")
        for line in combined_eval.split("\n"):
            if line.strip():
                parts.append(f"    {line}")
            else:
                parts.append("")
        parts.append("    return metrics")
        parts.append("")

    # Main function
    parts.append("")
    parts.append("def main():")
    parts.append('    output_dir = Path(os.environ.get("MLOPS_OUTPUT_DIR", "."))')
    parts.append('    job_id = os.environ.get("MLOPS_JOB_ID", "local")')
    parts.append("")
    parts.append('    print(f"Job ID: {job_id}")')
    parts.append("")

    if data_code:
        returns = _detect_return_vars(data_code, ["X_train", "X_test", "y_train", "y_test"])
        if returns:
            parts.append(f"    {', '.join(returns)} = load_data()")
        else:
            parts.append("    data = load_data()")
    else:
        parts.append("    # TODO: Add data loading")
        parts.append("    X_train = X_test = y_train = y_test = None")

    parts.append("")

    if train_code:
        parts.append("    model = train_model(X_train, y_train)")
    else:
        parts.append("    # TODO: Add training logic")
        parts.append("    model = None")

    parts.append("")

    if combined_eval:
        parts.append("    metrics = evaluate_model(model, X_test, y_test)")
    else:
        parts.append("    # TODO: Add evaluation")
        parts.append("    metrics = {}")

    parts.append("")
    parts.append("    # Save model")
    parts.append('    model_path = output_dir / "model.pkl"')
    parts.append("    if model is not None:")
    parts.append('        with open(model_path, "wb") as f:')
    parts.append("            pickle.dump(model, f)")
    parts.append("")
    parts.append("    # Save metrics")
    parts.append('    metrics_path = output_dir / "metrics.json"')
    parts.append("    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))")
    parts.append("")
    parts.append('    print(f"Model saved to {model_path}")')
    parts.append('    print(f"Metrics: {metrics}")')
    parts.append("")
    parts.append("")
    parts.append('if __name__ == "__main__":')
    parts.append("    main()")
    parts.append("")

    return "\n".join(parts)


def generate_pipeline_yaml(structure: NotebookStructure, name: str = "") -> str:
    """Generate a pipeline.yaml from detected notebook structure."""
    if not name:
        name = Path(structure.path).stem.replace(" ", "-").lower()

    # Determine thresholds from detected metrics
    has_classification = any(
        m in structure.detected_metrics for m in ["accuracy", "f1", "precision", "recall"]
    )

    lines = []
    lines.append(f"name: {name}")
    lines.append("")
    lines.append("reasoning:")
    lines.append("  engine: static  # Change to 'claude' with ANTHROPIC_API_KEY")
    lines.append("")
    lines.append("provider:")
    lines.append("  backend: local")
    lines.append("")
    lines.append("escalation:")
    lines.append("  default_confidence_threshold: 0.7")
    lines.append("  per_stage:")
    lines.append("    deployment: 0.9")
    lines.append("")
    lines.append("stages:")
    lines.append("  validate:")
    lines.append("    agent: cicd")
    lines.append("    on_success: [evaluate]")
    lines.append("    on_failure: []")
    lines.append("    params:")
    lines.append("      min_rows: 100")
    lines.append("")
    lines.append("  evaluate:")
    lines.append("    agent: evaluation")
    lines.append("    on_success: [deploy]")
    lines.append("    on_failure: []")
    lines.append("    params:")

    if has_classification:
        lines.append("      min_improvement: 0.005")
        lines.append("      max_fairness_delta: 0.05")
    else:
        lines.append("      min_improvement: 0.01")

    lines.append("      max_latency_p99_ms: 100")
    lines.append("")
    lines.append("  deploy:")
    lines.append("    agent: deployment")
    lines.append("    strategy: canary")
    lines.append('    canary_traffic: "5%"')
    lines.append("    on_success: [monitor]")
    lines.append("    on_failure: []")
    lines.append("")
    lines.append("  monitor:")
    lines.append("    agent: monitoring")
    lines.append("    mode: continuous")
    lines.append("    check_interval: 15m")
    lines.append("    on_drift: [retrain]")
    lines.append("")
    lines.append("  retrain:")
    lines.append("    agent: retraining")
    lines.append("    on_success: [evaluate]")
    lines.append("")

    # Add comments about detected model
    if structure.detected_model_type != "unknown":
        lines.append(f"# Detected model type: {structure.detected_model_type}")
    if structure.detected_metrics:
        lines.append(f"# Detected metrics: {', '.join(structure.detected_metrics)}")

    lines.append("")
    return "\n".join(lines)


def generate_requirements(structure: NotebookStructure) -> str:
    """Generate requirements.txt from detected imports."""
    packages = set()

    for imp in structure.detected_imports:
        match = re.match(r"(?:from\s+(\S+)|import\s+(\S+))", imp)
        if match:
            module = match.group(1) or match.group(2)
            root = module.split(".")[0]

            # Skip stdlib
            if root in _STDLIB_MODULES:
                continue

            # Map to pip package name
            pkg = IMPORT_TO_PACKAGE.get(root, root)
            packages.add(pkg)

    # Always include mlops-agents
    packages.add("mlops-agents")

    return "\n".join(sorted(packages)) + "\n"


def generate_all(
    structure: NotebookStructure,
    output_dir: str | Path = "pipeline",
) -> dict[str, str]:
    """Generate all files and write to output directory.

    Returns dict of filename -> content.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    # train.py
    train_content = generate_train_script(structure)
    files["train.py"] = train_content
    (output_dir / "train.py").write_text(train_content)

    # pipeline.yaml
    pipeline_content = generate_pipeline_yaml(structure)
    files["pipeline.yaml"] = pipeline_content
    (output_dir / "pipeline.yaml").write_text(pipeline_content)

    # requirements.txt
    req_content = generate_requirements(structure)
    files["requirements.txt"] = req_content
    (output_dir / "requirements.txt").write_text(req_content)

    logger.info(
        "ingest.generate.complete",
        output_dir=str(output_dir),
        files=list(files.keys()),
    )

    return files


def _detect_return_vars(code: str, candidates: list[str]) -> list[str]:
    """Detect which variables from candidates are assigned in the code."""
    found = []
    for var in candidates:
        if re.search(rf"\b{var}\s*=", code):
            found.append(var)
    return found


def _detect_model_var(code: str) -> str:
    """Detect the model variable name from training code."""
    patterns = [
        r"(\w+)\s*=\s*\w+(?:Classifier|Regressor|Model|Sequential)\(",
        r"(\w+)\.fit\(",
        r"(\w+)\s*=\s*train\(",
    ]
    for pattern in patterns:
        match = re.search(pattern, code)
        if match:
            return match.group(1)
    return ""


# Common Python stdlib modules (to exclude from requirements)
_STDLIB_MODULES = {
    "os",
    "sys",
    "json",
    "csv",
    "math",
    "re",
    "time",
    "datetime",
    "pathlib",
    "collections",
    "itertools",
    "functools",
    "typing",
    "dataclasses",
    "abc",
    "io",
    "copy",
    "pickle",
    "hashlib",
    "random",
    "statistics",
    "logging",
    "warnings",
    "argparse",
    "tempfile",
    "shutil",
    "glob",
    "unittest",
    "contextlib",
    "enum",
    "uuid",
    "struct",
    "base64",
    "string",
}
