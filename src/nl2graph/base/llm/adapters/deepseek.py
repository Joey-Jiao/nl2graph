from abc import ABC
from typing import List, Any, Optional

from .base import BaseAdapter
from ..entity import LLMMessage


class DeepSeekAdapter(BaseAdapter):
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
        content = resp.choices[0].message.content
        return LLMMessage.assistant(text=content)
