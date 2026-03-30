"""Google Cloud Storage provider for artifact management.

Handles model files, evaluation reports, training outputs, and
any binary artifacts that need to persist between pipeline stages.

Requires: pip install mlops-agents[gcp]
"""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger()


class GCSStorage:
    """GCS-backed artifact storage.

    Maps the StorageProvider protocol to GCS operations.
    All keys are relative to a configured bucket + prefix.

    Usage:
        storage = GCSStorage(bucket="my-mlops-bucket", prefix="artifacts/")
        uri = await storage.upload(Path("model.pkl"), "models/v1/model.pkl")
        # -> "gs://my-mlops-bucket/artifacts/models/v1/model.pkl"
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        project: str | None = None,
    ):
        self.bucket_name = bucket
        self.prefix = prefix.strip("/")
        self._project = project
        self._client = None
        self._bucket = None

    def _get_client(self):
        if self._client is None:
            try:
                from google.cloud import storage
            except ImportError:
                raise RuntimeError(
                    "google-cloud-storage required for GCSStorage. "
                    "Install with: pip install mlops-agents[gcp]"
                )
            self._client = storage.Client(project=self._project)
            self._bucket = self._client.bucket(self.bucket_name)
        return self._client

    def _blob_name(self, key: str) -> str:
        """Build the full blob name from prefix + key."""
        if self.prefix:
            return f"{self.prefix}/{key}"
        return key

    def _gs_uri(self, blob_name: str) -> str:
        return f"gs://{self.bucket_name}/{blob_name}"

    async def upload(self, local_path: Path, remote_key: str) -> str:
        """Upload a local file to GCS.

        Returns the gs:// URI of the uploaded blob.
        """
        self._get_client()
        blob_name = self._blob_name(remote_key)
        blob = self._bucket.blob(blob_name)

        logger.info(
            "gcs.upload.start",
            local=str(local_path),
            blob=blob_name,
            bucket=self.bucket_name,
        )

        # Run blocking upload in executor to keep async
        import asyncio

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: blob.upload_from_filename(str(local_path)),
        )

        uri = self._gs_uri(blob_name)
        logger.info("gcs.upload.complete", uri=uri, size=local_path.stat().st_size)
        return uri

    async def download(self, remote_key: str, local_path: Path) -> None:
        """Download a blob from GCS to a local file."""
        self._get_client()
        blob_name = self._blob_name(remote_key)
        blob = self._bucket.blob(blob_name)

        local_path.parent.mkdir(parents=True, exist_ok=True)

        import asyncio

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: blob.download_to_filename(str(local_path)),
        )

        logger.info("gcs.download.complete", blob=blob_name, local=str(local_path))

    async def list_artifacts(self, prefix: str) -> list[str]:
        """List blobs under a prefix, returning keys relative to the base prefix."""
        self._get_client()
        full_prefix = self._blob_name(prefix)

        import asyncio

        loop = asyncio.get_event_loop()
        blobs = await loop.run_in_executor(
            None,
            lambda: list(self._client.list_blobs(self.bucket_name, prefix=full_prefix)),
        )

        # Strip base prefix to return relative keys
        base = f"{self.prefix}/" if self.prefix else ""
        return [
            blob.name[len(base) :] if blob.name.startswith(base) else blob.name for blob in blobs
        ]

    async def exists(self, remote_key: str) -> bool:
        """Check if a blob exists in GCS."""
        self._get_client()
        blob_name = self._blob_name(remote_key)
        blob = self._bucket.blob(blob_name)

        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, blob.exists)

    async def delete(self, remote_key: str) -> None:
        """Delete a blob from GCS."""
        self._get_client()
        blob_name = self._blob_name(remote_key)
        blob = self._bucket.blob(blob_name)

        import asyncio

        loop = asyncio.get_event_loop()

        try:
            await loop.run_in_executor(None, blob.delete)
            logger.info("gcs.delete.complete", blob=blob_name)
        except Exception as e:
            # google.api_core.exceptions.NotFound
            if "NotFound" in type(e).__name__ or "404" in str(e):
                logger.debug("gcs.delete.not_found", blob=blob_name)
            else:
                raise
