"""Tests for local DuckDB data provider."""

import pytest

from mlops_agents.providers.local.duckdb import DuckDBData


@pytest.fixture
def data_provider(tmp_path):
    return DuckDBData(base_dir=str(tmp_path / "data"))


class TestDuckDBData:
    @pytest.mark.asyncio
    async def test_save_and_get_dataset(self, data_provider):
        records = [
            {"amount": 100.0, "is_fraud": 0},
            {"amount": 5000.0, "is_fraud": 1},
            {"amount": 50.0, "is_fraud": 0},
        ]

        dataset = await data_provider.save_dataset(records, "transactions")
        assert dataset.name == "transactions"
        assert dataset.version == "v1"
        assert dataset.num_rows == 3

        retrieved = await data_provider.get_dataset("transactions")
        assert retrieved.num_rows == 3
        assert retrieved.version == "v1"

    @pytest.mark.asyncio
    async def test_sequential_versions(self, data_provider):
        await data_provider.save_dataset([{"a": 1}], "test")
        await data_provider.save_dataset([{"a": 2}], "test")

        v1 = await data_provider.get_dataset("test", "v1")
        v2 = await data_provider.get_dataset("test", "v2")
        assert v1.version == "v1"
        assert v2.version == "v2"

    @pytest.mark.asyncio
    async def test_get_latest_version(self, data_provider):
        await data_provider.save_dataset([{"a": 1}], "test")
        await data_provider.save_dataset([{"a": 2}], "test")

        latest = await data_provider.get_dataset("test", "latest")
        assert latest.version == "v2"

    @pytest.mark.asyncio
    async def test_list_datasets(self, data_provider):
        await data_provider.save_dataset([{"a": 1}], "dataset_a")
        await data_provider.save_dataset([{"b": 2}], "dataset_b")

        datasets = await data_provider.list_datasets()
        assert "dataset_a" in datasets
        assert "dataset_b" in datasets

    @pytest.mark.asyncio
    async def test_get_nonexistent_dataset(self, data_provider):
        with pytest.raises(ValueError, match="not found"):
            await data_provider.get_dataset("nonexistent")

    @pytest.mark.asyncio
    async def test_schema_detection(self, data_provider):
        records = [{"name": "alice", "age": 30, "score": 0.95}]
        dataset = await data_provider.save_dataset(records, "users")
        assert dataset.schema["name"] == "str"
        assert dataset.schema["age"] == "int"
        assert dataset.schema["score"] == "float"
