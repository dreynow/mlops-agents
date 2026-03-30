"""Local file storage provider."""

from __future__ import annotations

import shutil
from pathlib import Path

import structlog

logger = structlog.get_logger()


class LocalFileStorage:
    """Local filesystem storage for artifacts.

    Mimics cloud object storage semantics with a base directory
    acting as the "bucket" and keys as relative paths.
    """

    def __init__(self, base_dir: str = ".mlops/storage"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def upload(self, local_path: Path, remote_key: str) -> str:
        dest = self.base_dir / remote_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        uri = f"file://{dest.resolve()}"
        logger.debug("storage.local.upload", key=remote_key, uri=uri)
        return uri

    async def download(self, remote_key: str, local_path: Path) -> None:
        src = self.base_dir / remote_key
        if not src.exists():
            raise FileNotFoundError(f"Artifact not found: {remote_key}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)

    async def list_artifacts(self, prefix: str) -> list[str]:
        prefix_path = self.base_dir / prefix
        if not prefix_path.exists():
            # Try prefix as a glob pattern
            parent = self.base_dir
            return [
                str(p.relative_to(self.base_dir)) for p in parent.rglob(f"{prefix}*") if p.is_file()
            ]
        if prefix_path.is_dir():
            return [
                str(p.relative_to(self.base_dir)) for p in prefix_path.rglob("*") if p.is_file()
            ]
        return [prefix]

    async def exists(self, remote_key: str) -> bool:
        return (self.base_dir / remote_key).exists()

    async def delete(self, remote_key: str) -> None:
        path = self.base_dir / remote_key
        if path.exists():
            path.unlink()
            logger.debug("storage.local.delete", key=remote_key)
