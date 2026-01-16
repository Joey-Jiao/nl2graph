import re
from typing import Optional

from ...base import LLMService, LLMMessage, TemplateService
from ...data import GenerationOutput
from ...execution.schema.base import BaseSchema


class Generation:

    def __init__(
        self,
        llm_service: LLMService,
        provider: str,
        model: str,
        template_service: Optional[TemplateService] = None,
        template_name: Optional[str] = None,
        extract_query: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.client = llm_service.get_client(provider, model)
        self.template_service = template_service
        self.template_name = template_name
        self.extract_query = extract_query

    def generate(self, question: str, schema: Optional[BaseSchema] = None) -> GenerationOutput:
        if self.template_service and self.template_name and schema:
            prompt = self._build_prompt(question, schema)
        else:
            prompt = question

        messages = [LLMMessage.user(prompt)]
        response = self.client.chat(messages)

        content = response.message.content
        if self.extract_query:
            content = self._extract_query(content)

        stats = {
            "duration": response.duration,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cached_tokens": response.usage.cached_tokens,
        }
        return GenerationOutput(content=content, stats=stats)

    def _build_prompt(self, question: str, schema: BaseSchema) -> str:
        return self.template_service.render(
            "prompts",
            self.template_name,
            question=question,
            schema=schema.to_prompt_string(),
        )

    def _extract_query(self, raw: str) -> str:
        patterns = [
            r"```(?:cypher|sparql|gremlin)?\s*\n?(.*?)```",
            r"`([^`]+)`",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return raw.strip()
