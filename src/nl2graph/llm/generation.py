from typing import List

from ..base.llm.service import LLMService
from ..base.llm.entity import LLMMessage


class Generation:

    def __init__(
        self,
        llm_service: LLMService,
        provider: str,
        model: str,
    ):
        self.llm_service = llm_service
        self.provider = provider
        self.model = model

    def generate(self, prompt: str) -> str:
        client = self.llm_service.get_client(self.provider, self.model)
        messages = [LLMMessage.user(prompt)]
        response = client.chat(messages)
        return response.content

    def generate_batch(self, prompts: List[str]) -> List[str]:
        return [self.generate(prompt) for prompt in prompts]
