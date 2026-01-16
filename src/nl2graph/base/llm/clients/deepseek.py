import time
from typing import List

from openai import OpenAI

from .base import BaseClient
from ..entity import ClientConfig, LLMMessage, LLMResponse
from ..adapters import DeepSeekAdapter


class DeepSeekClient(BaseClient):
    def __init__(self, config: ClientConfig):
        super().__init__()
        self.config = config
        self.adapter = DeepSeekAdapter
        self.client = OpenAI(api_key=config.api_key, base_url=config.endpoint)

    def chat(self, messages: List[LLMMessage]) -> LLMResponse:
        chat_messages = self.adapter.to_chat_messages(messages)

        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.config.model,
            messages=chat_messages,
        )
        duration = time.perf_counter() - start

        return LLMResponse(
            message=self.adapter.extract_chat_message(resp),
            usage=self.adapter.extract_usage(resp),
            duration=duration,
        )
