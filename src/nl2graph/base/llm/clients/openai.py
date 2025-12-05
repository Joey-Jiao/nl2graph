from typing import List, Any

from openai import OpenAI

from .base import BaseClient
from ..entity import ClientConfig, LLMMessage
from ..adapters import OpenAIAdapter


class OpenAIClient(BaseClient):
    def __init__(self, config: ClientConfig):
        self.config = config
        self.adapter = OpenAIAdapter
        self.client = OpenAI(api_key=config.api_key)

    def chat(self, messages: List[LLMMessage]) -> LLMMessage:
        chat_messages = self.adapter.to_chat_messages(messages)

        resp = self.client.responses.create(
            model=self.config.model,
            input=chat_messages,
        )

        return self.adapter.extract_chat_message(resp)
