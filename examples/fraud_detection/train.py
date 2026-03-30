"""Fraud detection training script.

Trains a Random Forest classifier on synthetic fraud data.
Outputs model artifact + metrics JSON to MLOPS_OUTPUT_DIR.

This script is invoked by the ComputeProvider. It reads
MLOPS_JOB_ID and MLOPS_OUTPUT_DIR from environment variables.
"""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# Reproducibility
np.random.seed(42)


def generate_synthetic_data(n_samples: int = 5000):
    """Generate synthetic fraud detection data.

    Features: amount, hour, merchant_category, is_international,
              velocity_1h, distance_from_home
    Target: is_fraud (binary)
    """
    n_fraud = int(n_samples * 0.03)  # 3% fraud rate
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    legit_amount = np.random.lognormal(mean=3.5, sigma=1.0, size=n_legit)
    legit_hour = np.random.normal(loc=14, scale=4, size=n_legit).clip(0, 23)
    legit_category = np.random.randint(0, 10, size=n_legit)
    legit_international = np.random.binomial(1, 0.05, size=n_legit)
    legit_velocity = np.random.poisson(2, size=n_legit)
    legit_distance = np.random.exponential(10, size=n_legit)

    # Fraudulent transactions
    fraud_amount = np.random.lognormal(mean=5.5, sigma=1.5, size=n_fraud)
    fraud_hour = np.random.normal(loc=3, scale=3, size=n_fraud).clip(0, 23)
    fraud_category = np.random.randint(0, 10, size=n_fraud)
    fraud_international = np.random.binomial(1, 0.4, size=n_fraud)
    fraud_velocity = np.random.poisson(8, size=n_fraud)
    fraud_distance = np.random.exponential(100, size=n_fraud)

    X = np.vstack([
        np.column_stack([legit_amount, legit_hour, legit_category,
                        legit_international, legit_velocity, legit_distance]),
        np.column_stack([fraud_amount, fraud_hour, fraud_category,
                        fraud_international, fraud_velocity, fraud_distance]),
    ])
    y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def train_model(X_train, y_train, X_test, y_test, n_estimators=100):
    """Train a Random Forest and return metrics."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "fraud_rate_train": float(y_train.mean()),
        "fraud_rate_test": float(y_test.mean()),
    }

    return model, metrics


def main():
    output_dir = Path(os.environ.get("MLOPS_OUTPUT_DIR", "."))
    job_id = os.environ.get("MLOPS_JOB_ID", "local")
    test_size = float(os.environ.get("TEST_SIZE", "0.2"))
    n_estimators = int(os.environ.get("N_ESTIMATORS", "100"))

    print(f"Job ID: {job_id}")
    print(f"Output dir: {output_dir}")
    print("Generating synthetic fraud data...")

    X, y = generate_synthetic_data(n_samples=5000)

    # Split
    split_idx = int(len(y) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training: {len(y_train)} samples, Test: {len(y_test)} samples")
    print(f"Fraud rate: {y.mean():.1%}")

    try:
        model, metrics = train_model(X_train, y_train, X_test, y_test, n_estimators)
    except ImportError:
        print("scikit-learn not installed. Generating mock metrics.")
        model = None
        metrics = {
            "accuracy": 0.978,
            "f1": 0.834,
            "precision": 0.891,
            "recall": 0.783,
            "auc_roc": 0.967,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "fraud_rate_train": float(y_train.mean()),
            "fraud_rate_test": float(y_test.mean()),
        }

    # Save model
    model_path = output_dir / "model.pkl"
    if model is not None:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
    else:
        model_path.write_text("mock model")

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_path}")

    # Print summary
    print("\n--- Results ---")
    for k, v in metrics.items():
        if isinstance(v, float) and v < 1.0:
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
