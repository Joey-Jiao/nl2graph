from typing import List

from .base import BaseAdapter
from ..entity import LLMMessage, LLMUsage


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

    @classmethod
    def extract_usage(cls, resp) -> LLMUsage:
        usage = resp.usage
        cached = 0
        if usage.input_tokens_details:
            cached = usage.input_tokens_details.cached_tokens or 0
        return LLMUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=cached,
        )
