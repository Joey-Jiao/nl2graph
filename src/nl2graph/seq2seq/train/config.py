import importlib.util
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    special_tokens: List[str]
    load_data: Callable
    evaluate: Optional[Callable] = None
    translate: Optional[Callable] = None


class ConfigLoader:

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._module = None

    def load(self) -> DatasetConfig:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        spec = importlib.util.spec_from_file_location("config", self.config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._module = module

        return DatasetConfig(
            special_tokens=getattr(module, 'special_tokens', []),
            load_data=getattr(module, 'load_data'),
            evaluate=getattr(module, 'evaluate', None),
            translate=getattr(module, 'translate', None),
        )

    @property
    def module(self):
        return self._module
