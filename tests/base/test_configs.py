import os
import pytest
from pathlib import Path

from nl2graph.base.configs import ConfigService

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestConfigService:
    @pytest.fixture
    def config_service(self):
        config_files = [FIXTURES_DIR / "test_config.yaml"]
        env_path = FIXTURES_DIR / ".env.test"
        return ConfigService(config_dir=config_files, env_path=env_path)

    def test_get_yaml_config(self, config_service):
        assert config_service.get("app.name") == "test-app"
        assert config_service.get("app.debug") is True

    def test_get_nested_config(self, config_service):
        assert config_service.get("database.host") == "localhost"
        assert config_service.get("database.port") == 5432

    def test_get_default_value(self, config_service):
        assert config_service.get("nonexistent.key", "default") == "default"

    def test_get_env(self, config_service):
        assert config_service.get_env("TEST_API_KEY") == "test-key-123"
        assert config_service.get_env("DATABASE_URL") == "postgres://localhost/test"

    def test_get_env_missing(self, config_service):
        assert config_service.get_env("NONEXISTENT_KEY") is None
