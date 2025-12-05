from typing import List, Any

from openai import OpenAI

from .base import BaseClient
from ..entity import ClientConfig, LLMMessage
from ..adapters import DeepSeekAdapter


class DeepSeekClient(BaseClient):
    def __init__(self, config: ClientConfig):
        super().__init__()
        self.config = config
        self.adapter = DeepSeekAdapter
        self.client = OpenAI(api_key=config.api_key, base_url=config.endpoint)

    def chat(self, messages: List[LLMMessage]) -> LLMMessage:
        chat_messages = self.adapter.to_chat_messages(messages)

        resp = self.client.chat.completions.create(
            model=self.config.model,
            messages=chat_messages,
        )

        return self.adapter.extract_chat_message(resp)
