"""Demo: Run the Evaluation Agent on a trained fraud model.

This demo shows the full agent lifecycle:
  1. Register a champion model (simulating an existing production model)
  2. Submit a new candidate with better metrics
  3. Run the Evaluation Agent to get a go/no-go decision
  4. Print the full audit trail with reasoning

Run:
  cd examples/fraud_detection
  python demo.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mlops_agents.agents.evaluation import EvalAgent
from mlops_agents.core.audit import SQLiteAuditStore
from mlops_agents.core.config import EscalationConfig
from mlops_agents.core.event import Event, EventTypes, LocalAsyncEventBus
from mlops_agents.core.reasoning import StaticReasoner
from mlops_agents.providers.local.mlflow import LocalMLPlatform
from mlops_agents.providers.protocols import ModelArtifact


async def main():
    print("=" * 60)
    print("  Agentic MLOps - Evaluation Agent Demo")
    print("=" * 60)

    # --- Setup ---
    bus = LocalAsyncEventBus()
    audit = SQLiteAuditStore(db_path=".mlops/demo_audit.db")
    ml = LocalMLPlatform(base_dir=".mlops/demo_mlplatform")
    reasoner = StaticReasoner(default_confidence=0.92)

    # Track events emitted by the agent
    events_received = []

    async def event_logger(event: Event):
        events_received.append(event)
        print(f"\n  >> Event: {event.type} (from {event.source})")

    await bus.subscribe("*", event_logger)

    # --- 1. Register existing champion model ---
    print("\n[Step 1] Registering champion model...")
    champion = await ml.register_model(ModelArtifact(
        model_name="fraud-detector",
        artifact_path="/models/fraud-v11/model.pkl",
        metrics={"f1": 0.821, "precision": 0.856, "recall": 0.789, "auc_roc": 0.945},
    ))
    await ml.promote_model("fraud-detector", champion.version, "production")
    print(f"  Champion: {champion.model_name}/{champion.version} (F1: 0.821)")

    # --- 2. Create evaluation agent ---
    print("\n[Step 2] Creating Evaluation Agent...")
    eval_agent = EvalAgent(
        event_bus=bus,
        audit_store=audit,
        reasoning_engine=reasoner,
        escalation_config=EscalationConfig(
            default_confidence_threshold=0.7,
            per_stage={"evaluation": 0.8},
        ),
        min_improvement=0.005,
        max_fairness_delta=0.05,
        max_latency_p99_ms=50.0,
    )

    # --- 3. Simulate a new candidate model ---
    print("\n[Step 3] New candidate model trained - running evaluation...")
    candidate_event = Event(
        type="model.evaluate",
        source="retraining-agent",
        trace_id="pipe-demo-001",
        payload={
            "model_name": "fraud-detector",
            "run_id": "run-abc123",
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

    decision = await eval_agent.run(candidate_event, providers={"ml": ml})

    # --- 4. Print results ---
    print("\n" + "=" * 60)
    print("  EVALUATION DECISION")
    print("=" * 60)
    print(f"\n  {decision.summary()}")
    print(f"\n  Reasoning:")
    print(f"    Observations:")
    for obs in decision.reasoning.observations:
        print(f"      - {obs}")
    print(f"    Analysis: {decision.reasoning.analysis}")
    print(f"    Conclusion: {decision.reasoning.conclusion}")
    print(f"    Confidence: {decision.reasoning.confidence:.0%}")
    if decision.escalate_to_human:
        print(f"    ESCALATED: {decision.escalation_reason}")

    print(f"\n  Artifacts:")
    for k, v in decision.artifacts.items():
        print(f"    {k}: {v}")

    # --- 5. Show audit trail ---
    print("\n" + "=" * 60)
    print("  AUDIT TRAIL")
    print("=" * 60)
    trace_decisions = await audit.get_trace("pipe-demo-001")
    for d in trace_decisions:
        print(f"\n  [{d.timestamp.strftime('%H:%M:%S')}] {d.summary()}")

    # --- 6. Show events ---
    print(f"\n  Events emitted: {len(events_received)}")
    for e in events_received:
        print(f"    {e.type} (trace: {e.trace_id})")

    print("\n" + "=" * 60)
    print("  Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
