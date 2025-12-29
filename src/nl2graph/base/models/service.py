from typing import Dict, Optional
from pathlib import Path

from ..configs import ConfigService
from .entity import ModelConfig


class ModelService:

    def __init__(self, config: ConfigService):
        self._config = config
        self._models: Dict[str, ModelConfig] = {}
        self._load_config()

    def _load_config(self):
        models_config = self._config.get("seq2seq.models", {})
        if not models_config:
            return

        for model_name, model_cfg in models_config.items():
            path = model_cfg.get("path", f"models/pretrained/{model_name}")
            self._models[model_name] = ModelConfig(
                name=model_name,
                path=Path(path),
                tokenizer_path=Path(model_cfg.get("tokenizer_path")) if model_cfg.get("tokenizer_path") else None,
                max_length=model_cfg.get("max_length", 512),
                special_tokens=model_cfg.get("special_tokens", []),
            )

    def ls_models(self):
        return list(self._models.keys())

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        return self._models.get(model_name)

    def get_checkpoint_path(self, dataset: str, model_name: str) -> Path:
        checkpoints_base = self._config.get("seq2seq.checkpoints_dir", "models/checkpoints")
        return Path(checkpoints_base) / dataset / model_name / "checkpoint-best"

    def register_model(self, model_config: ModelConfig):
        self._models[model_config.name] = model_config
