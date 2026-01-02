from ...base.llm.service import LLMService
from ...base.llm.entity import LLMMessage


class Generation:

    def __init__(self, llm_service: LLMService, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.client = llm_service.get_client(provider, model)

    def generate(self, prompt: str) -> str:
        messages = [LLMMessage.user(prompt)]
        response = self.client.chat(messages)
        return response.content
