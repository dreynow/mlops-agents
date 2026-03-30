"""Live Demo: Full pipeline with real Claude reasoning + Observatory observability.

This is the interview demo. It shows:
  1. Real LLM chain-of-thought reasoning on every decision
  2. All decisions logged to auth.kanoniv.com Observatory
  3. Full audit trail with reasoning traces
  4. 3-stage pipeline: validate -> evaluate -> deploy

Prerequisites:
  pip install mlops-agents[claude,observatory]
  export ANTHROPIC_API_KEY=sk-ant-...
  # Optional: export KANONIV_AUTH_KEY=kt_live_... (for Observatory)

Run:
  cd examples/fraud_detection
  python demo_live.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mlops_agents.core.audit import SQLiteAuditStore
from mlops_agents.core.config import PipelineConfig, ProviderConfig
from mlops_agents.core.event import LocalAsyncEventBus
from mlops_agents.core.pipeline import Pipeline
from mlops_agents.providers.local.mlflow import LocalMLPlatform
from mlops_agents.providers.protocols import ModelArtifact
from mlops_agents.providers.registry import ProviderRegistry


async def main():
    print("=" * 70)
    print("  Agentic MLOps - Live Demo (Claude Reasoning + Observatory)")
    print("=" * 70)

    # --- Determine reasoning engine ---
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        reasoning_config = {
            "engine": "claude",
            "model": "claude-sonnet-4-20250514",
            "api_key": anthropic_key,
        }
        print("\n  Reasoning: Claude (claude-sonnet-4-20250514)")
    else:
        reasoning_config = {"engine": "static"}
        print("\n  Reasoning: Static (set ANTHROPIC_API_KEY for real LLM reasoning)")

    # --- Setup Observatory ---
    observatory = None
    kanoniv_key = os.environ.get("KANONIV_AUTH_KEY")
    if kanoniv_key:
        try:
            from mlops_agents.core.observatory import Observatory

            observatory = Observatory(api_key=kanoniv_key)
            print("  Observatory: auth.kanoniv.com (live)")
        except ImportError:
            print("  Observatory: disabled (pip install kanoniv-auth)")
    else:
        print("  Observatory: disabled (set KANONIV_AUTH_KEY for live observability)")

    # --- Setup providers ---
    bus = LocalAsyncEventBus()
    audit = SQLiteAuditStore(db_path=".mlops/live_audit.db")
    providers = ProviderRegistry.from_config(
        ProviderConfig(backend="local", local={"base_dir": ".mlops/live"}),
        event_bus=bus,
    )

    # Register champion model
    print("\n  Registering champion model...")
    champion = await providers.ml.register_model(
        ModelArtifact(
            model_name="fraud-detector",
            artifact_path="/models/fraud-v11/model.pkl",
            metrics={
                "f1": 0.821,
                "precision": 0.856,
                "recall": 0.789,
                "auc_roc": 0.945,
            },
        )
    )
    await providers.ml.promote_model("fraud-detector", champion.version, "production")
    print(f"  Champion: fraud-detector/{champion.version} (F1: 0.821)")

    # --- Build pipeline ---
    config = PipelineConfig.from_dict({
        "name": "fraud-detection-live",
        "reasoning": reasoning_config,
        "provider": {"backend": "local", "local": {"base_dir": ".mlops/live"}},
        "audit": {"sqlite_path": ".mlops/live_audit.db"},
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

    pipeline = Pipeline(
        config=config,
        event_bus=bus,
        audit_store=audit,
        providers=providers,
        observatory=observatory,
    )

    # --- Run pipeline ---
    print("\n" + "-" * 70)
    print("  Running pipeline: validate -> evaluate -> deploy")
    print("-" * 70)

    trace = await pipeline.run(
        initial_payload={
            "dataset_name": "transactions_march",
            "num_rows": 5000,
            "num_columns": 10,
            "null_rates": {"amount": 0.01, "merchant": 0.02},
            "psi_scores": {"amount": 0.08, "hour": 0.05},
            "model_name": "fraud-detector",
            "run_id": "run-live-001",
            "artifact_uri": "/models/fraud-v12/model.pkl",
            "metrics": {
                "f1": 0.834,
                "precision": 0.891,
                "recall": 0.783,
                "auc_roc": 0.967,
                "fairness_delta": 0.02,
                "latency_p99_ms": 12.0,
            },
        },
    )

    # --- Print results ---
    print("\n" + "=" * 70)
    print("  PIPELINE RESULTS")
    print("=" * 70)
    print(f"\n  Trace: {trace.trace_id}")
    print(f"  Status: {trace.status.upper()}")

    for i, d in enumerate(trace.decisions, 1):
        status = "GO" if d.approved else "NO-GO"
        if d.escalate_to_human:
            status = "ESCALATED"

        print(f"\n  {'=' * 60}")
        print(f"  Stage {i}: {d.agent_name}")
        print(f"  {'=' * 60}")
        print(f"  Action:     {d.action}")
        print(f"  Decision:   {status}")
        print(f"  Confidence: {d.reasoning.confidence:.0%}")
        print(f"  Model:      {d.reasoning.model_used}")

        print(f"\n  Observations:")
        for obs in d.reasoning.observations:
            print(f"    - {obs}")

        print(f"\n  Analysis:")
        # Wrap long analysis text
        analysis = d.reasoning.analysis
        words = analysis.split()
        line = "    "
        for word in words:
            if len(line) + len(word) > 72:
                print(line)
                line = "    " + word
            else:
                line += " " + word if line.strip() else word
        if line.strip():
            print(line)

        print(f"\n  Conclusion: {d.reasoning.conclusion}")

        if d.reasoning.alternatives_considered:
            print(f"\n  Alternatives considered:")
            for alt in d.reasoning.alternatives_considered:
                print(f"    - {alt}")

    if observatory:
        print(f"\n  Observatory: decisions visible at trust.kanoniv.com")

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
