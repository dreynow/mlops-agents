"""Vertex AI compute provider for training job management.

Handles custom training job submission, status polling with exponential
backoff, cancellation, and artifact path resolution after completion.

Vertex AI jobs can run for hours. The polling loop uses exponential
backoff (5s -> 10s -> 20s -> ... -> 300s max) to avoid hammering
the API while still detecting completion promptly.

Requires: pip install mlops-agents[gcp]
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import structlog

from mlops_agents.providers.protocols import (
    Artifact,
    JobHandle,
    JobStatus,
    TrainConfig,
)

logger = structlog.get_logger()

# Polling backoff parameters
INITIAL_POLL_INTERVAL_S = 5
MAX_POLL_INTERVAL_S = 300  # 5 minutes
BACKOFF_MULTIPLIER = 2.0

# Map Vertex AI job states to our JobStatus
_VERTEX_STATE_MAP = {
    "JOB_STATE_QUEUED": JobStatus.PENDING,
    "JOB_STATE_PENDING": JobStatus.PENDING,
    "JOB_STATE_RUNNING": JobStatus.RUNNING,
    "JOB_STATE_SUCCEEDED": JobStatus.SUCCEEDED,
    "JOB_STATE_FAILED": JobStatus.FAILED,
    "JOB_STATE_CANCELLED": JobStatus.CANCELLED,
    "JOB_STATE_CANCELLING": JobStatus.RUNNING,
    "JOB_STATE_PAUSED": JobStatus.RUNNING,
    "JOB_STATE_EXPIRED": JobStatus.FAILED,
}


class VertexAICompute:
    """Vertex AI Custom Training job management.

    Submits Python training scripts as custom jobs on Vertex AI.
    Artifacts are written to a GCS staging bucket and resolved
    after job completion.

    Usage:
        compute = VertexAICompute(
            project="my-project",
            location="us-central1",
            staging_bucket="gs://my-mlops-staging",
        )
        handle = await compute.submit_job(TrainConfig(
            script_path="train.py",
            args={"epochs": 10},
        ))
        status = await compute.get_job_status(handle)
    """

    def __init__(
        self,
        project: str,
        location: str = "us-central1",
        staging_bucket: str = "",
        container_image: str = "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12.py310:latest",
        machine_type: str = "n1-standard-4",
        gpu_machine_type: str = "n1-standard-8",
        gpu_accelerator: str = "NVIDIA_TESLA_T4",
        gpu_count: int = 1,
        service_account: str | None = None,
    ):
        self._project = project
        self._location = location
        self._staging_bucket = staging_bucket.rstrip("/")
        self._container_image = container_image
        self._machine_type = machine_type
        self._gpu_machine_type = gpu_machine_type
        self._gpu_accelerator = gpu_accelerator
        self._gpu_count = gpu_count
        self._service_account = service_account
        self._client = None
        self._jobs: dict[str, dict[str, Any]] = {}

    def _get_aiplatform(self):
        try:
            from google.cloud import aiplatform

            return aiplatform
        except ImportError:
            raise RuntimeError(
                "google-cloud-aiplatform required for VertexAICompute. "
                "Install with: pip install mlops-agents[gcp]"
            )

    def _init_vertex(self):
        if self._client is None:
            aiplatform = self._get_aiplatform()
            aiplatform.init(
                project=self._project,
                location=self._location,
                staging_bucket=self._staging_bucket or None,
            )
            self._client = aiplatform
        return self._client

    async def submit_job(self, config: TrainConfig) -> JobHandle:
        """Submit a custom training job to Vertex AI.

        The training script is uploaded to the staging bucket and
        executed in a managed container.
        """
        aiplatform = self._init_vertex()
        job_id = f"mlops-{uuid.uuid4().hex[:8]}"

        # Build args list
        args_list = []
        for k, v in config.args.items():
            args_list.extend([f"--{k}", str(v)])

        # Determine machine config
        machine_type = self._machine_type
        accelerator_type = None
        accelerator_count = 0

        if config.gpu:
            machine_type = self._gpu_machine_type
            accelerator_type = self._gpu_accelerator
            accelerator_count = self._gpu_count

        # GCS output directory for this job
        output_dir = f"{self._staging_bucket}/jobs/{job_id}/output"

        # Add output dir to environment
        env_vars = {**config.env, "MLOPS_JOB_ID": job_id, "MLOPS_OUTPUT_DIR": output_dir}

        logger.info(
            "vertex.submit.start",
            job_id=job_id,
            script=config.script_path,
            machine=machine_type,
            gpu=config.gpu,
        )

        loop = asyncio.get_event_loop()

        # Use CustomJob for script-based training
        custom_job = await loop.run_in_executor(
            None,
            lambda: aiplatform.CustomJob(
                display_name=job_id,
                worker_pool_specs=[
                    {
                        "machine_spec": {
                            "machine_type": machine_type,
                            **(
                                {
                                    "accelerator_type": accelerator_type,
                                    "accelerator_count": accelerator_count,
                                }
                                if accelerator_type
                                else {}
                            ),
                        },
                        "replica_count": 1,
                        "container_spec": {
                            "image_uri": config.image or self._container_image,
                            "command": ["python", config.script_path],
                            "args": args_list,
                            "env": [{"name": k, "value": v} for k, v in env_vars.items()],
                        },
                    }
                ],
                staging_bucket=self._staging_bucket or None,
            ),
        )

        # Submit asynchronously (non-blocking)
        await loop.run_in_executor(
            None,
            lambda: custom_job.run(
                service_account=self._service_account,
                timeout=config.timeout_minutes * 60,
                sync=False,  # Don't block - we poll ourselves
            ),
        )

        handle = JobHandle(
            job_id=job_id,
            backend="vertex_ai",
            metadata={
                "resource_name": custom_job.resource_name,
                "output_dir": output_dir,
            },
        )

        self._jobs[job_id] = {
            "custom_job": custom_job,
            "output_dir": output_dir,
            "config": config,
        }

        logger.info(
            "vertex.submit.complete",
            job_id=job_id,
            resource_name=custom_job.resource_name,
        )

        return handle

    async def get_job_status(self, handle: JobHandle) -> JobStatus:
        """Poll Vertex AI for current job status.

        Maps Vertex AI job states to our JobStatus enum.
        """
        job_info = self._jobs.get(handle.job_id)
        if job_info is None:
            # Try to fetch from Vertex AI by resource name
            resource_name = handle.metadata.get("resource_name", "")
            if resource_name:
                return await self._fetch_status_by_resource(resource_name)
            return JobStatus.FAILED

        custom_job = job_info["custom_job"]

        loop = asyncio.get_event_loop()
        try:
            # Refresh the job state
            await loop.run_in_executor(None, custom_job._sync_gca_resource)
            state = custom_job.state.name
        except Exception as e:
            logger.warning("vertex.status.error", job_id=handle.job_id, error=str(e))
            return JobStatus.FAILED

        status = _VERTEX_STATE_MAP.get(state, JobStatus.RUNNING)
        logger.debug("vertex.status", job_id=handle.job_id, state=state, status=status.value)
        return status

    async def get_artifacts(self, handle: JobHandle) -> list[Artifact]:
        """List output artifacts from the job's GCS output directory."""
        job_info = self._jobs.get(handle.job_id)
        if job_info is None:
            return []

        output_dir = job_info["output_dir"]
        if not output_dir:
            return []

        # List GCS blobs in the output directory
        try:
            from google.cloud import storage
        except ImportError:
            return []

        loop = asyncio.get_event_loop()
        client = storage.Client(project=self._project)

        # Parse gs:// URI
        bucket_name, prefix = self._parse_gs_uri(output_dir)

        blobs = await loop.run_in_executor(
            None,
            lambda: list(client.list_blobs(bucket_name, prefix=prefix)),
        )

        artifacts = []
        for blob in blobs:
            name = blob.name.split("/")[-1]
            suffix = Path(name).suffix
            artifact_type = (
                "model"
                if suffix in (".pkl", ".joblib", ".pt", ".onnx", ".h5")
                else "metrics"
                if suffix == ".json"
                else "logs"
            )
            artifacts.append(
                Artifact(
                    name=name,
                    path=f"gs://{bucket_name}/{blob.name}",
                    artifact_type=artifact_type,
                    size_bytes=blob.size or 0,
                )
            )

        return artifacts

    async def cancel_job(self, handle: JobHandle) -> None:
        """Cancel a running Vertex AI training job."""
        job_info = self._jobs.get(handle.job_id)
        if job_info is None:
            logger.warning("vertex.cancel.not_found", job_id=handle.job_id)
            return

        custom_job = job_info["custom_job"]

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, custom_job.cancel)
            logger.info("vertex.cancel.complete", job_id=handle.job_id)
        except Exception as e:
            logger.error("vertex.cancel.error", job_id=handle.job_id, error=str(e))

    async def get_logs(self, handle: JobHandle) -> str:
        """Get training job logs from Vertex AI.

        Note: Full logs are in Cloud Logging. This returns a summary.
        """
        job_info = self._jobs.get(handle.job_id)
        if job_info is None:
            return f"Job {handle.job_id} not found in local cache."

        custom_job = job_info["custom_job"]
        resource_name = getattr(custom_job, "resource_name", "unknown")
        state = "unknown"
        try:
            state = custom_job.state.name
        except Exception:
            pass

        return (
            f"Vertex AI Job: {handle.job_id}\n"
            f"Resource: {resource_name}\n"
            f"State: {state}\n"
            f"Output: {job_info['output_dir']}\n"
            f"Full logs: https://console.cloud.google.com/vertex-ai/training/custom-jobs"
        )

    async def wait_for_completion(
        self,
        handle: JobHandle,
        timeout_minutes: int = 120,
    ) -> JobStatus:
        """Poll until the job completes with exponential backoff.

        Args:
            handle: Job handle from submit_job.
            timeout_minutes: Max time to wait before giving up.

        Returns:
            Final job status.
        """
        interval = INITIAL_POLL_INTERVAL_S
        elapsed = 0
        timeout_s = timeout_minutes * 60

        logger.info("vertex.wait.start", job_id=handle.job_id, timeout_m=timeout_minutes)

        while elapsed < timeout_s:
            status = await self.get_job_status(handle)

            if status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
                logger.info(
                    "vertex.wait.done", job_id=handle.job_id, status=status.value, elapsed_s=elapsed
                )
                return status

            logger.debug(
                "vertex.wait.poll",
                job_id=handle.job_id,
                status=status.value,
                next_poll_s=interval,
                elapsed_s=elapsed,
            )

            await asyncio.sleep(interval)
            elapsed += interval
            interval = min(interval * BACKOFF_MULTIPLIER, MAX_POLL_INTERVAL_S)

        logger.warning("vertex.wait.timeout", job_id=handle.job_id, timeout_m=timeout_minutes)
        return JobStatus.FAILED

    async def _fetch_status_by_resource(self, resource_name: str) -> JobStatus:
        """Fetch job status directly from Vertex AI by resource name."""
        aiplatform = self._init_vertex()
        loop = asyncio.get_event_loop()

        try:
            job = await loop.run_in_executor(
                None,
                lambda: aiplatform.CustomJob.get(resource_name),
            )
            state = job.state.name
            return _VERTEX_STATE_MAP.get(state, JobStatus.FAILED)
        except Exception as e:
            logger.warning("vertex.fetch_status.error", resource=resource_name, error=str(e))
            return JobStatus.FAILED

    @staticmethod
    def _parse_gs_uri(uri: str) -> tuple[str, str]:
        """Parse gs://bucket/prefix into (bucket, prefix)."""
        path = uri.replace("gs://", "")
        parts = path.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return bucket, prefix
