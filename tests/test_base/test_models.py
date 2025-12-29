import pytest
from pathlib import Path

from nl2graph.base.models.entity import ModelConfig
from nl2graph.base.models.service import ModelService
from nl2graph.base.configs import ConfigService


class TestModelConfig:

    def test_create_minimal(self):
        config = ModelConfig(
            name="bart-base",
            path=Path("models/pretrained/bart-base"),
        )
        assert config.name == "bart-base"
        assert config.path == Path("models/pretrained/bart-base")
        assert config.tokenizer_path is None
        assert config.max_length == 512
        assert config.special_tokens == []

    def test_create_full(self):
        config = ModelConfig(
            name="custom-model",
            path=Path("models/custom"),
            tokenizer_path=Path("models/tokenizer"),
            max_length=1024,
            special_tokens=["<special1>", "<special2>"],
        )
        assert config.name == "custom-model"
        assert config.tokenizer_path == Path("models/tokenizer")
        assert config.max_length == 1024
        assert config.special_tokens == ["<special1>", "<special2>"]


class TestModelService:

    @pytest.fixture
    def config_service(self, temp_config_dir):
        config_files = list(temp_config_dir.glob("*.yaml"))
        return ConfigService(config_dir=config_files, env_path=".env.nonexistent")

    def test_load_models_from_config(self, config_service):
        service = ModelService(config=config_service)
        models = service.ls_models()
        assert "bart-base" in models

    def test_get_model_config(self, config_service):
        service = ModelService(config=config_service)
        config = service.get_model_config("bart-base")
        assert config is not None
        assert config.name == "bart-base"
        assert config.max_length == 512

    def test_get_nonexistent_model(self, config_service):
        service = ModelService(config=config_service)
        config = service.get_model_config("nonexistent")
        assert config is None

    def test_get_checkpoint_path(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
seq2seq:
  checkpoints_dir: "models/checkpoints"
  models:
    bart-base:
      path: "models/pretrained/bart-base"
""")
        config = ConfigService(config_dir=[config_file], env_path=".env.nonexistent")
        service = ModelService(config=config)
        path = service.get_checkpoint_path("kqapro", "bart-base")
        assert path == Path("models/checkpoints/kqapro/bart-base/checkpoint-best")

    def test_register_model(self, config_service):
        service = ModelService(config=config_service)
        new_config = ModelConfig(
            name="new-model",
            path=Path("models/new"),
            max_length=256,
        )
        service.register_model(new_config)
        assert "new-model" in service.ls_models()
        assert service.get_model_config("new-model") == new_config

    def test_empty_config(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("other_key: value\n")
        config_service = ConfigService(
            config_dir=[config_file],
            env_path=".env.nonexistent",
        )
        service = ModelService(config=config_service)
        assert service.ls_models() == []
