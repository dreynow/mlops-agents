"""E2E Demo: Run the full fraud detection pipeline through the orchestrator.

This demo shows:
  1. Pipeline loads from YAML
  2. Orchestrator creates all agents
  3. Stages execute in DAG order: validate -> evaluate -> deploy
  4. Each agent makes a decision with reasoning
  5. Full audit trail produced

Run:
  cd examples/fraud_detection
  python demo_pipeline.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mlops_agents.core.audit import SQLiteAuditStore
from mlops_agents.core.event import LocalAsyncEventBus, Event
from mlops_agents.core.pipeline import Pipeline
from mlops_agents.providers.local.mlflow import LocalMLPlatform
from mlops_agents.providers.protocols import ModelArtifact
from mlops_agents.providers.registry import ProviderRegistry, Providers
from mlops_agents.core.config import PipelineConfig, ProviderConfig


async def main():
    print("=" * 70)
    print("  Agentic MLOps - Full Pipeline Demo")
    print("=" * 70)

    # --- Setup ---
    bus = LocalAsyncEventBus()
    audit = SQLiteAuditStore(db_path=".mlops/pipeline_audit.db")

    # Build providers
    providers = ProviderRegistry.from_config(
        ProviderConfig(backend="local", local={"base_dir": ".mlops/demo"}),
        event_bus=bus,
    )

    # Register a champion model so the eval agent has something to compare against
    print("\n[Setup] Registering champion model...")
    champion = await providers.ml.register_model(ModelArtifact(
        model_name="fraud-detector",
        artifact_path="/models/fraud-v11/model.pkl",
        metrics={"f1": 0.821, "precision": 0.856, "recall": 0.789, "auc_roc": 0.945},
    ))
    await providers.ml.promote_model("fraud-detector", champion.version, "production")
    print(f"  Champion: {champion.model_name}/{champion.version} (F1: 0.821)")

    # --- Load pipeline config ---
    # We use a simplified 3-stage pipeline for the demo
    config = PipelineConfig.from_dict({
        "name": "fraud-detection-demo",
        "reasoning": {"engine": "static"},
        "provider": {"backend": "local", "local": {"base_dir": ".mlops/demo"}},
        "audit": {"sqlite_path": ".mlops/pipeline_audit.db"},
        "escalation": {
            "default_confidence_threshold": 0.7,
            "per_stage": {"deployment": 0.9},
        },
        "stages": {
            "validate": {
                "agent": "cicd",
                "on_success": ["evaluate"],
                "on_failure": [],
                "params": {"min_rows": 100},
            },
            "evaluate": {
                "agent": "evaluation",
                "on_success": ["deploy"],
                "on_failure": [],
                "params": {"min_improvement": 0.005},
            },
            "deploy": {
                "agent": "deployment",
                "on_success": [],
                "on_failure": [],
                "params": {},
            },
        },
    })

    # --- Create pipeline ---
    pipeline = Pipeline(
        config=config,
        event_bus=bus,
        audit_store=audit,
        providers=providers,
    )

    # --- Run pipeline ---
    print("\n[Pipeline] Starting fraud-detection-demo pipeline...")
    print(f"  Stages: {' -> '.join(config.stages.keys())}")
    print()

    # The initial payload simulates data arriving + a new trained model
    initial_payload = {
        # For CI/CD agent (data validation)
        "dataset_name": "transactions_march",
        "num_rows": 5000,
        "num_columns": 10,
        "null_rates": {"amount": 0.01, "merchant": 0.02},
        "psi_scores": {"amount": 0.08, "hour": 0.05},
        # For Eval agent (carried forward)
        "model_name": "fraud-detector",
        "run_id": "run-demo-001",
        "artifact_uri": "/models/fraud-v12/model.pkl",
        "metrics": {
            "f1": 0.834,
            "precision": 0.891,
            "recall": 0.783,
            "auc_roc": 0.967,
            "fairness_delta": 0.02,
            "latency_p99_ms": 12.0,
        },
    }

    trace = await pipeline.run(initial_payload=initial_payload)

    # --- Print results ---
    print("\n" + "=" * 70)
    print("  PIPELINE RESULTS")
    print("=" * 70)

    status_icon = {"completed": "OK", "failed": "FAILED", "escalated": "ESCALATED"}
    print(f"\n  Trace: {trace.trace_id}")
    print(f"  Status: {status_icon.get(trace.status, trace.status)}")
    print(f"  Decisions: {len(trace.decisions)}")

    for i, d in enumerate(trace.decisions, 1):
        status = "GO" if d.approved else "NO-GO"
        if d.escalate_to_human:
            status = "ESCALATED"

        print(f"\n  --- Stage {i}: {d.agent_name} ---")
        print(f"  Action: {d.action}")
        print(f"  Decision: {status} (confidence: {d.reasoning.confidence:.0%})")
        print(f"  Conclusion: {d.reasoning.conclusion}")
        if d.reasoning.observations:
            print(f"  Observations:")
            for obs in d.reasoning.observations:
                print(f"    - {obs}")
        if d.artifacts:
            print(f"  Artifacts: {d.artifacts}")

    # --- Audit trail ---
    print("\n" + "=" * 70)
    print("  AUDIT TRAIL")
    print("=" * 70)
    stored = await audit.get_trace(trace.trace_id)
    print(f"\n  {len(stored)} decisions persisted to SQLite")
    for d in stored:
        print(f"  [{d.timestamp.strftime('%H:%M:%S')}] {d.summary()}")

    print("\n" + "=" * 70)
    print("  Pipeline demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
