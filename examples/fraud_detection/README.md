# Fraud Detection Example

End-to-end fraud detection pipeline using mlops-agents. Demonstrates notebook ingestion, all 6 agents, Claude reasoning, and Observatory integration.

## The Workflow

```
fraud_model.ipynb  -->  mlops-agents ingest  -->  train.py + pipeline.yaml  -->  mlops-agents run
```

Start from a data scientist's notebook, ingest it into a production pipeline, run it with autonomous agents.

## Files

| File | Purpose |
|---|---|
| `fraud_model.ipynb` | Original notebook with manifest cell - the starting point |
| `pipeline.yaml` | Full 7-stage pipeline config (hand-written, for advanced use) |
| `train.py` | Hand-written training script (alternative to notebook ingestion) |
| `demo.py` | Quick demo - runs the Evaluation Agent standalone |
| `demo_pipeline.py` | Full pipeline demo - 3 stages with local providers |
| `demo_live.py` | Interview demo - real Claude reasoning + Observatory |

## Quick Start

### Option A: Start from a notebook (recommended)

```bash
cd examples/fraud_detection

# 1. Ingest the notebook - generates train.py + pipeline.yaml
mlops-agents ingest fraud_model.ipynb -o my_pipeline

  Manifest cell detected - deterministic extraction
  Model type: random_forest
  Metrics: accuracy, f1, precision, recall, auc_roc

  Generated 3 files in my_pipeline/

# 2. Run the generated pipeline
mlops-agents run my_pipeline/pipeline.yaml
```

The notebook has a manifest cell at the bottom that tells the parser which cells map to which pipeline stages:

```python
# mlops: manifest
# cell 1: imports
# cell 3-4: data-loading
# cell 6: training
# cell 8: evaluation
# cell 9: metrics
```

No inline tags needed. The data scientist's working code is untouched.

### Option B: Run demos directly

```bash
# Simple demo (no API keys needed)
python demo.py

# Full pipeline demo (no API keys needed)
python demo_pipeline.py

# Live demo with Claude + Observatory
export ANTHROPIC_API_KEY=sk-ant-...
export KANONIV_AUTH_KEY=kt_live_...    # optional
python demo_live.py
```

## What Each Demo Shows

### `demo.py` - Evaluation Agent standalone

Registers a champion model (F1: 0.821), runs a candidate (F1: 0.834) through the Evaluation Agent, prints the full decision with reasoning trace and audit trail.

```
[evaluation] model.evaluate -> GO (confidence: 92%)
  F1 delta: +0.0130 (+1.6%) vs champion
  Fairness: demographic parity delta 0.020 (OK)
  Latency p99: 12.0ms (OK, SLA: 50ms)
```

### `demo_pipeline.py` - Full orchestrator

Runs 3 stages through the Pipeline orchestrator:
1. **CI/CD Agent** validates data (row count, null rates, PSI drift)
2. **Evaluation Agent** compares candidate vs champion
3. **Deployment Agent** deploys canary at 5% traffic

All decisions logged to SQLite audit trail.

### `demo_live.py` - Interview demo

Same pipeline but with:
- **Claude Sonnet** generating real chain-of-thought reasoning
- **Observatory** (auth.kanoniv.com) logging every decision as signed provenance
- Agents registered with DIDs, scoped delegations, reputation tracking

## The Notebook

`fraud_model.ipynb` is a realistic data science notebook:

- Generates synthetic fraud data (5K transactions, 3% fraud rate)
- 6 features: amount, hour, merchant category, international flag, velocity, distance
- Trains a Random Forest with class balancing
- Evaluates with F1, precision, recall, AUC-ROC
- Manifest cell at the bottom declares structure for ingestion

The notebook is designed to show how `mlops-agents ingest` works with a real ML workflow. Data scientists add one cell at the bottom and the framework takes ownership of the lifecycle.

## Pipeline Config

`pipeline.yaml` defines the full lifecycle with feedback loop:

```
validate -> train -> evaluate -> deploy -> monitor
                ^                              |
                |                    rollback <-+
                |                    |
                +--- retrain <--- feedback
```

Key settings:
- Reasoning: `static` by default (change to `claude` for real LLM)
- Escalation: 70% default, 90% for deployment
- Canary: 5% traffic, 30m bake time
- Monitoring: 15m check interval
- Feedback: 1h collection interval
