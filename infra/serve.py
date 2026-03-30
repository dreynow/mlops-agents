"""Lightweight model serving endpoint.

Loads a scikit-learn model and serves predictions via FastAPI.
Used by the Deployment Agent for canary deployments.
"""

import os
import pickle
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="MLOps Model Server")

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/current/model.pkl")
model = None
model_load_time = None


class PredictRequest(BaseModel):
    features: list[list[float]]


class PredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[list[float]]
    model_version: str
    latency_ms: float


@app.on_event("startup")
def load_model():
    global model, model_load_time
    path = Path(MODEL_PATH)
    if path.exists():
        with open(path, "rb") as f:
            model = pickle.load(f)
        model_load_time = time.time()
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"No model at {MODEL_PATH} - server running without model")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "uptime_s": time.time() - model_load_time if model_load_time else 0,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    start = time.perf_counter()
    features = np.array(request.features)
    predictions = model.predict(features).tolist()
    probabilities = model.predict_proba(features).tolist()
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        predictions=predictions,
        probabilities=probabilities,
        model_version=os.environ.get("MODEL_VERSION", "unknown"),
        latency_ms=round(latency_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
