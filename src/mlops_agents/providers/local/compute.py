"""Local compute provider - runs training as subprocesses."""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import structlog

from mlops_agents.providers.protocols import (
    Artifact,
    JobHandle,
    JobStatus,
    TrainConfig,
)

logger = structlog.get_logger()


class LocalDockerCompute:
    """Runs training jobs as local Python subprocesses.

    Artifacts are written to a local directory. No Docker required
    for basic usage - just runs the training script directly.
    """

    def __init__(self, artifacts_dir: str = ".mlops/artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, dict] = {}
        self._tasks: set[asyncio.Task] = set()  # prevent GC of background tasks

    async def submit_job(self, config: TrainConfig) -> JobHandle:
        job_id = f"local-{uuid.uuid4().hex[:8]}"
        job_dir = self.artifacts_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        env = {**os.environ, **config.env, "MLOPS_JOB_ID": job_id, "MLOPS_OUTPUT_DIR": str(job_dir)}

        args_list = []
        for k, v in config.args.items():
            args_list.extend([f"--{k}", str(v)])

        cmd = ["python", config.script_path] + args_list

        logger.info("compute.local.submit", job_id=job_id, cmd=" ".join(cmd))

        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path(config.script_path).parent) if "/" in config.script_path else None,
        )

        self._jobs[job_id] = {
            "process": process,
            "config": config,
            "dir": job_dir,
            "status": JobStatus.RUNNING,
            "stdout": b"",
            "stderr": b"",
        }

        # Wait in background - store reference to prevent GC
        task = asyncio.create_task(self._wait_for_job(job_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        return JobHandle(job_id=job_id, backend="local")

    async def get_job_status(self, handle: JobHandle) -> JobStatus:
        job = self._jobs.get(handle.job_id)
        if job is None:
            return JobStatus.FAILED
        return job["status"]

    async def get_artifacts(self, handle: JobHandle) -> list[Artifact]:
        job = self._jobs.get(handle.job_id)
        if job is None:
            return []

        job_dir = job["dir"]
        artifacts = []
        if job_dir.exists():
            for f in job_dir.iterdir():
                if f.is_file():
                    artifact_type = "model" if f.suffix in (".pkl", ".joblib", ".pt", ".onnx") else "metrics" if f.suffix == ".json" else "logs"
                    artifacts.append(Artifact(
                        name=f.name,
                        path=str(f),
                        artifact_type=artifact_type,
                        size_bytes=f.stat().st_size,
                    ))
        return artifacts

    async def cancel_job(self, handle: JobHandle) -> None:
        job = self._jobs.get(handle.job_id)
        if job and job["status"] == JobStatus.RUNNING:
            job["process"].terminate()
            job["status"] = JobStatus.CANCELLED

    async def get_logs(self, handle: JobHandle) -> str:
        job = self._jobs.get(handle.job_id)
        if job is None:
            return ""
        stdout = job["stdout"].decode(errors="replace") if job["stdout"] else ""
        stderr = job["stderr"].decode(errors="replace") if job["stderr"] else ""
        return f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"

    async def _wait_for_job(self, job_id: str) -> None:
        job = self._jobs[job_id]
        process = job["process"]
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=job["config"].timeout_minutes * 60,
            )
            job["stdout"] = stdout
            job["stderr"] = stderr
            job["status"] = JobStatus.SUCCEEDED if process.returncode == 0 else JobStatus.FAILED
            logger.info(
                "compute.local.completed",
                job_id=job_id,
                status=job["status"].value,
                return_code=process.returncode,
            )
        except asyncio.TimeoutError:
            process.terminate()
            job["status"] = JobStatus.FAILED
            logger.error("compute.local.timeout", job_id=job_id)
        except Exception as e:
            job["status"] = JobStatus.FAILED
            logger.error("compute.local.error", job_id=job_id, error=str(e))
