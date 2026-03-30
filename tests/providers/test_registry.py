"""Tests for provider registry."""

import pytest

from mlops_agents.core.config import ProviderConfig
from mlops_agents.providers.registry import ProviderRegistry, Providers


class TestProviderRegistry:
    def test_all_local_defaults(self):
        config = ProviderConfig()
        providers = ProviderRegistry.from_config(config)
        assert isinstance(providers, Providers)
        assert providers.compute is not None
        assert providers.storage is not None
        assert providers.ml is not None
        assert providers.data is not None
        assert providers.event_bus is not None
        assert providers.serving is not None

    def test_local_with_base_dir(self):
        config = ProviderConfig(local={"base_dir": ".test_mlops"})
        providers = ProviderRegistry.from_config(config)
        assert providers.compute is not None

    def test_per_service_mix_and_match(self):
        """Vertex AI compute + GCS storage + local everything else."""
        config = ProviderConfig(
            compute="vertex_ai",
            storage="gcs",
            ml="local",
            data="local",
            vertex_ai={
                "project": "test-project",
                "region": "us-central1",
                "staging_bucket": "gs://test",
            },
            gcs={"bucket": "test-bucket", "project": "test-project"},
        )
        providers = ProviderRegistry.from_config(config)
        assert providers.compute is not None
        assert providers.storage is not None
        assert providers.ml is not None

    def test_bigquery_data(self):
        config = ProviderConfig(
            data="bigquery",
            bigquery={
                "project": "test-project",
                "dataset": "ml_features",
                "location": "US",
            },
        )
        providers = ProviderRegistry.from_config(config)
        assert providers.data is not None

    def test_bigquery_requires_project(self):
        config = ProviderConfig(data="bigquery", bigquery={})
        with pytest.raises(ValueError, match="project"):
            ProviderRegistry.from_config(config)

    def test_unknown_compute_raises(self):
        config = ProviderConfig(compute="sagemaker")
        with pytest.raises(ValueError, match="Unknown compute"):
            ProviderRegistry.from_config(config)

    def test_unknown_storage_raises(self):
        config = ProviderConfig(storage="s3")
        with pytest.raises(ValueError, match="Unknown storage"):
            ProviderRegistry.from_config(config)

    def test_unknown_data_raises(self):
        config = ProviderConfig(data="redshift")
        with pytest.raises(ValueError, match="Unknown data"):
            ProviderRegistry.from_config(config)


class TestLegacyBackendShortcut:
    def test_backend_local(self):
        config = ProviderConfig(backend="local")
        providers = ProviderRegistry.from_config(config)
        assert providers.compute is not None

    def test_backend_gcp_expands(self):
        """backend: gcp should set compute=vertex_ai, storage=gcs, data=bigquery."""
        config = ProviderConfig(
            backend="gcp",
            gcp={
                "project_id": "test-project",
                "region": "us-central1",
                "staging_bucket": "gs://test-bucket",
                "bigquery_dataset": "ml_features",
            },
        )
        # The model_post_init should have expanded the bundle
        assert config.compute == "vertex_ai"
        assert config.storage == "gcs"
        assert config.data == "bigquery"
        assert config.ml == "local"  # Tier 2 stays local

        providers = ProviderRegistry.from_config(config)
        assert providers.compute is not None
        assert providers.storage is not None
        assert providers.data is not None

    def test_service_settings_preserved_over_bundle(self):
        """If vertex_ai settings are pre-provided, bundle doesn't overwrite them."""
        config = ProviderConfig(
            backend="gcp",
            vertex_ai={"project": "custom", "region": "europe-west1"},
            gcp={"project_id": "bundle-project"},
        )
        # compute should be vertex_ai (bundle sets it)
        assert config.compute == "vertex_ai"
        # But vertex_ai settings should be the user's, not from gcp bundle
        assert config.vertex_ai["project"] == "custom"
        assert config.vertex_ai["region"] == "europe-west1"
