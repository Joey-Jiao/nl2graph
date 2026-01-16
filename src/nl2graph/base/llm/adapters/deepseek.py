from typing import List

from .base import BaseAdapter
from ..entity import LLMMessage, LLMUsage


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

    @classmethod
    def extract_usage(cls, resp) -> LLMUsage:
        usage = resp.usage
        return LLMUsage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cached_tokens=usage.prompt_cache_hit_tokens or 0,
        )
