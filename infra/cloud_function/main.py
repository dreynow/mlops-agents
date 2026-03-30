"""Cloud Function trigger for mlops-agents pipeline.

Triggers when new data lands in a GCS bucket. Extracts dataset info
from the upload event and calls the mlops-agents pipeline API or
runs it directly.

Deploy:
  gcloud functions deploy mlops-pipeline-trigger \
    --runtime python312 \
    --trigger-bucket $DATA_BUCKET \
    --entry-point handle_gcs_upload \
    --set-env-vars PIPELINE_CONFIG=gs://config-bucket/pipeline.yaml
"""

from __future__ import annotations

import json
import os
import logging

import functions_framework
from google.cloud import storage

logger = logging.getLogger(__name__)


@functions_framework.cloud_event
def handle_gcs_upload(cloud_event):
    """Triggered by a GCS object upload.

    Extracts the uploaded file info and triggers the mlops-agents
    pipeline with the dataset details as initial payload.
    """
    data = cloud_event.data
    bucket_name = data["bucket"]
    file_name = data["name"]
    content_type = data.get("contentType", "")
    size = int(data.get("size", 0))

    logger.info(f"New file: gs://{bucket_name}/{file_name} ({size} bytes)")

    # Only trigger on data files
    valid_extensions = (".csv", ".parquet", ".json", ".jsonl")
    if not any(file_name.endswith(ext) for ext in valid_extensions):
        logger.info(f"Skipping non-data file: {file_name}")
        return {"status": "skipped", "reason": "not a data file"}

    # Build pipeline payload
    dataset_name = file_name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    payload = {
        "dataset_name": dataset_name,
        "dataset_uri": f"gs://{bucket_name}/{file_name}",
        "content_type": content_type,
        "size_bytes": size,
        "trigger": "gcs_upload",
    }

    # Option 1: Call pipeline API (if running as a service)
    api_url = os.environ.get("PIPELINE_API_URL")
    if api_url:
        return _trigger_via_api(api_url, payload)

    # Option 2: Publish to Pub/Sub (for async processing)
    pubsub_topic = os.environ.get("PIPELINE_PUBSUB_TOPIC")
    if pubsub_topic:
        return _trigger_via_pubsub(pubsub_topic, payload)

    # Option 3: Run pipeline directly (for simple setups)
    return _trigger_direct(payload)


@functions_framework.cloud_event
def handle_pubsub_message(cloud_event):
    """Triggered by a Pub/Sub message.

    Expects a JSON payload with pipeline trigger info.
    """
    import base64

    data = cloud_event.data
    message_data = base64.b64decode(data["message"]["data"]).decode()
    payload = json.loads(message_data)

    logger.info(f"Pub/Sub trigger: {payload.get('trigger', 'unknown')}")

    return _trigger_direct(payload)


def _trigger_via_api(api_url: str, payload: dict) -> dict:
    """Call the pipeline API to trigger a run."""
    import requests

    response = requests.post(
        f"{api_url}/v1/pipeline/run",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    logger.info(f"Pipeline triggered via API: {result.get('trace_id')}")
    return {"status": "triggered", "method": "api", **result}


def _trigger_via_pubsub(topic: str, payload: dict) -> dict:
    """Publish to Pub/Sub for async pipeline execution."""
    from google.cloud import pubsub_v1

    publisher = pubsub_v1.PublisherClient()
    future = publisher.publish(
        topic,
        json.dumps(payload).encode(),
        trigger="gcs_upload",
    )
    message_id = future.result()
    logger.info(f"Published to {topic}: {message_id}")
    return {"status": "published", "method": "pubsub", "message_id": message_id}


def _trigger_direct(payload: dict) -> dict:
    """Run the pipeline directly in the Cloud Function."""
    import asyncio

    async def run():
        from mlops_agents.core.pipeline import Pipeline

        config_path = os.environ.get("PIPELINE_CONFIG", "pipeline.yaml")

        # Download config from GCS if needed
        if config_path.startswith("gs://"):
            config_path = _download_gcs_file(config_path)

        pipeline = Pipeline.from_yaml(config_path)
        trace = await pipeline.run(initial_payload=payload)
        return {
            "status": "completed",
            "trace_id": trace.trace_id,
            "pipeline_status": trace.status,
            "decisions": len(trace.decisions),
        }

    return asyncio.run(run())


def _download_gcs_file(gs_uri: str) -> str:
    """Download a GCS file to /tmp and return local path."""
    bucket_name, blob_name = gs_uri.replace("gs://", "").split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    local_path = f"/tmp/{blob_name.replace('/', '_')}"
    blob.download_to_filename(local_path)
    return local_path
