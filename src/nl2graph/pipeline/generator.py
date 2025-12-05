import re

from ..base.llm.service import LLMService
from ..base.templates.service import TemplateService
from ..graph.schema.base import BaseSchema
from .entity import Record, GenerationResult


class QueryGenerator:
    def __init__(
        self,
        llm_service: LLMService,
        template_service: TemplateService,
        provider: str,
        model: str,
        lang: str,
        prompt_template: str,
    ):
        self.llm_service = llm_service
        self.template_service = template_service
        self.provider = provider
        self.model = model
        self.lang = lang
        self.prompt_template = prompt_template

    def generate(self, record: Record, schema: BaseSchema) -> GenerationResult:
        client = self.llm_service.get_client(self.provider, self.model)

        prompt = self.template_service.render(
            "prompts",
            self.prompt_template,
            question=record.question,
            schema=schema.to_prompt_string(),
            lang=self.lang,
        )

        from ..base.llm.entity import LLMMessage
        messages = [LLMMessage.user(prompt)]
        response = client.chat(messages)

        query_raw = response.content
        query_processed = self._extract_query(query_raw)

        return GenerationResult(
            query_raw=query_raw,
            query_processed=query_processed,
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
