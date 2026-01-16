import time
from abc import ABC, abstractmethod
from typing import List

from ..entity import LLMMessage, LLMResponse


class BaseClient(ABC):
    @abstractmethod
    def chat(self, messages: List[LLMMessage]) -> LLMResponse:
        pass

    def embed(self, text: str) -> List[float]:
        pass
