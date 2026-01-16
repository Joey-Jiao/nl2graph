from ...base import LLMService, LLMMessage
from ...data import GenerationOutput


class Generation:

    def __init__(self, llm_service: LLMService, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.client = llm_service.get_client(provider, model)

    def generate(self, prompt: str) -> GenerationOutput:
        messages = [LLMMessage.user(prompt)]
        response = self.client.chat(messages)
        stats = {
            "duration": response.duration,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cached_tokens": response.usage.cached_tokens,
        }
        return GenerationOutput(content=response.message.content, stats=stats)
