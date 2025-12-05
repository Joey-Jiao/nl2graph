from abc import ABC, abstractmethod
from typing import List, Optional, Any

from ..entity import LLMMessage


class BaseAdapter(ABC):
    @abstractmethod
    def to_chat_messages(self, messages: List[LLMMessage]):
        raise NotImplementedError()

    @abstractmethod
    def extract_chat_message(self, resp) -> LLMMessage:
        raise NotImplementedError()

    def to_embedding(self, text: str) -> Any:
        return text

    def extract_embedding(self, raw) -> List[float]:
        return raw.data[0].embedding

