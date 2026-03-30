"""Tests for provider registry."""

import pytest

from mlops_agents.core.config import ProviderConfig
from mlops_agents.providers.registry import ProviderRegistry, Providers


class TestProviderRegistry:
    def test_build_local(self):
        config = ProviderConfig(backend="local", local={"base_dir": ".mlops_test"})
        providers = ProviderRegistry.from_config(config)
        assert isinstance(providers, Providers)
        assert providers.compute is not None
        assert providers.storage is not None
        assert providers.ml is not None
        assert providers.data is not None
        assert providers.event_bus is not None
        assert providers.serving is not None

    def test_build_gcp(self):
        config = ProviderConfig(
            backend="gcp",
            gcp={
                "project_id": "test-project",
                "region": "us-central1",
                "staging_bucket": "gs://test-bucket",
                "bigquery_dataset": "ml_features",
            },
        )
        providers = ProviderRegistry.from_config(config)
        assert isinstance(providers, Providers)
        assert providers.compute is not None
        assert providers.storage is not None
        assert providers.data is not None

    def test_gcp_requires_project_id(self):
        config = ProviderConfig(backend="gcp", gcp={})
        with pytest.raises(ValueError, match="project_id"):
            ProviderRegistry.from_config(config)

    def test_unknown_backend(self):
        config = ProviderConfig(backend="azure")
        with pytest.raises(ValueError, match="Unknown"):
            ProviderRegistry.from_config(config)
