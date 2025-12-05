from abc import ABC, abstractmethod
from typing import List

from ..entity import LLMMessage


class BaseClient(ABC):
    @abstractmethod
    def chat(self, messages: List[LLMMessage]) -> LLMMessage:
        pass

    def embed(self, text: str) -> List[float]:
        pass
