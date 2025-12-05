from typing import List, Any, Optional

from .base import BaseAdapter
from ..entity import LLMMessage


class OpenAIAdapter(BaseAdapter):
    @classmethod
    def to_chat_messages(cls, messages: List[LLMMessage]):
        """
        [{"role": "...", "content": "..."}]
        """
        return [
            {"role": message.role, "content": message.content}
            for message in messages
        ]

    @classmethod
    def extract_chat_message(cls, resp) -> LLMMessage:
        content = resp.output_text
        return LLMMessage.assistant(text=content)
