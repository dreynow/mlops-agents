# Fraud Detection Example

End-to-end fraud detection pipeline using mlops-agents. Demonstrates all 6 agents, Claude reasoning, and Observatory integration.

## Files

| File | Purpose |
|---|---|
| `pipeline.yaml` | Full 7-stage pipeline config (validate, train, evaluate, deploy, monitor, feedback, retrain) |
| `train.py` | Training script - Random Forest on synthetic fraud data (5K transactions, 3% fraud rate) |
| `demo.py` | Quick demo - runs the Evaluation Agent standalone against a champion model |
| `demo_pipeline.py` | Full pipeline demo - runs 3 stages (validate -> evaluate -> deploy) with local providers |
| `demo_live.py` | Interview demo - real Claude reasoning + Observatory observability |

## Quick Start

```bash
# From repo root
cd examples/fraud_detection

# 1. Simple demo (no API keys needed)
python demo.py

# 2. Full pipeline demo (no API keys needed)
python demo_pipeline.py

# 3. Live demo with Claude + Observatory
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

This is the one you demo in the interview.

## Training Script

`train.py` generates synthetic fraud data and trains a Random Forest:

```bash
# Run standalone
python train.py

# Or let the ComputeProvider run it
# The agent sets MLOPS_JOB_ID and MLOPS_OUTPUT_DIR env vars
```

Output: `model.pkl` + `metrics.json` in `MLOPS_OUTPUT_DIR`.

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
