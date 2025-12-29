import os
import pytest

from nl2graph.base.configs import ConfigService


class TestConfigService:

    def test_load_single_config(self, temp_config_dir):
        config_files = list(temp_config_dir.glob("*.yaml"))
        service = ConfigService(config_dir=config_files, env_path=".env.nonexistent")
        assert service.get("log.log_dir") == "outputs/logs"
        assert service.get("log.console_level") == "INFO"

    def test_get_nested_value(self, temp_config_dir):
        config_files = list(temp_config_dir.glob("*.yaml"))
        service = ConfigService(config_dir=config_files, env_path=".env.nonexistent")
        assert service.get("llm.openai.gpt-4o.timeout") == 60

    def test_get_nonexistent_key(self, temp_config_dir):
        config_files = list(temp_config_dir.glob("*.yaml"))
        service = ConfigService(config_dir=config_files, env_path=".env.nonexistent")
        assert service.get("nonexistent.key") is None
        assert service.get("nonexistent.key", default="default") == "default"

    def test_get_partial_path(self, temp_config_dir):
        config_files = list(temp_config_dir.glob("*.yaml"))
        service = ConfigService(config_dir=config_files, env_path=".env.nonexistent")
        result = service.get("llm.openai")
        assert isinstance(result, dict)
        assert "gpt-4o" in result

    def test_get_top_level(self, temp_config_dir):
        config_files = list(temp_config_dir.glob("*.yaml"))
        service = ConfigService(config_dir=config_files, env_path=".env.nonexistent")
        result = service.get("log")
        assert isinstance(result, dict)
        assert result["log_dir"] == "outputs/logs"

    def test_load_env_file(self, temp_config_dir, temp_env_file):
        config_files = list(temp_config_dir.glob("*.yaml"))
        service = ConfigService(config_dir=config_files, env_path=str(temp_env_file))
        assert service.get_env("OPENAI_API_KEY") == "test-key"
        assert service.get_env("DEEPSEEK_API_KEY") == "test-key-2"

    def test_get_env_nonexistent(self, temp_config_dir):
        config_files = list(temp_config_dir.glob("*.yaml"))
        service = ConfigService(config_dir=config_files, env_path=".env.nonexistent")
        assert service.get_env("NONEXISTENT_KEY") is None

    def test_get_env_with_hyphen(self, temp_config_dir, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MY_API_KEY=hyphen-test\n")
        config_files = list(temp_config_dir.glob("*.yaml"))
        service = ConfigService(config_dir=config_files, env_path=str(env_file))
        assert service.get_env("my-api-key") == "hyphen-test"

    def test_multiple_config_files(self, tmp_path):
        config1 = tmp_path / "config1.yaml"
        config1.write_text("service1:\n  key1: value1\n")
        config2 = tmp_path / "config2.yaml"
        config2.write_text("service2:\n  key2: value2\n")

        config_files = list(tmp_path.glob("*.yaml"))
        service = ConfigService(config_dir=config_files, env_path=".env.nonexistent")

        assert service.get("service1.key1") == "value1"
        assert service.get("service2.key2") == "value2"

    def test_empty_config_file(self, tmp_path):
        config = tmp_path / "empty.yaml"
        config.write_text("")
        config_files = [config]
        service = ConfigService(config_dir=config_files, env_path=".env.nonexistent")
        assert service.get("any.key") is None

    def test_get_seq2seq_models(self, temp_config_dir):
        config_files = list(temp_config_dir.glob("*.yaml"))
        service = ConfigService(config_dir=config_files, env_path=".env.nonexistent")
        models = service.get("seq2seq.models")
        assert isinstance(models, dict)
        assert "bart-base" in models
        assert models["bart-base"]["max_length"] == 512
