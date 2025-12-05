import os
from typing import Any
from dotenv import load_dotenv
from yaml import safe_load
from pathlib import Path


class ConfigService:
    def __init__(self, config_dir, env_path):
        self.config_dir = config_dir
        self.env_path = env_path
        self.configs = {}
        for path in config_dir:
            with open(path) as f:
                data = safe_load(f) or {}
                self.configs.update(data)
        load_dotenv(self.env_path, override=True)

    def get_env(self, key: str, default: Any = None) -> Any:
        env_key = key.upper().replace("-", "_")
        if env_key in os.environ:
            return os.environ[env_key]

    def get(self, keys: str, default: Any = None) -> Any:
        keys = keys.split(".")
        value = self.configs
        for key in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(key)
        return value
