"""Tests for local file storage provider."""

import pytest

from mlops_agents.providers.local.storage import LocalFileStorage


@pytest.fixture
def storage(tmp_path):
    return LocalFileStorage(base_dir=str(tmp_path / "storage"))


@pytest.fixture
def sample_file(tmp_path):
    f = tmp_path / "sample.pkl"
    f.write_text("model data")
    return f


class TestLocalFileStorage:
    @pytest.mark.asyncio
    async def test_upload_and_download(self, storage, sample_file, tmp_path):
        uri = await storage.upload(sample_file, "models/v1/model.pkl")
        assert "file://" in uri

        dest = tmp_path / "downloaded.pkl"
        await storage.download("models/v1/model.pkl", dest)
        assert dest.read_text() == "model data"

    @pytest.mark.asyncio
    async def test_exists(self, storage, sample_file):
        assert not await storage.exists("models/v1/model.pkl")
        await storage.upload(sample_file, "models/v1/model.pkl")
        assert await storage.exists("models/v1/model.pkl")

    @pytest.mark.asyncio
    async def test_delete(self, storage, sample_file):
        await storage.upload(sample_file, "models/v1/model.pkl")
        assert await storage.exists("models/v1/model.pkl")

        await storage.delete("models/v1/model.pkl")
        assert not await storage.exists("models/v1/model.pkl")

    @pytest.mark.asyncio
    async def test_list_artifacts(self, storage, sample_file):
        await storage.upload(sample_file, "models/v1/model.pkl")
        await storage.upload(sample_file, "models/v1/metrics.json")
        await storage.upload(sample_file, "models/v2/model.pkl")

        v1_artifacts = await storage.list_artifacts("models/v1")
        assert len(v1_artifacts) == 2

        all_artifacts = await storage.list_artifacts("models")
        assert len(all_artifacts) == 3

    @pytest.mark.asyncio
    async def test_download_nonexistent(self, storage, tmp_path):
        with pytest.raises(FileNotFoundError):
            await storage.download("nonexistent", tmp_path / "out.pkl")
