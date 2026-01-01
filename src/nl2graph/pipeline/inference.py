import re
from typing import List, Optional, Any, Protocol, runtime_checkable, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ..data import Record, Result, GenerationResult, ExecutionResult, ResultRepository
from ..execution import Execution
from ..evaluation import Scoring
from ..base.templates.service import TemplateService
from ..execution.schema.base import BaseSchema


@runtime_checkable
class Generator(Protocol):

    def generate(self, text: str) -> str: ...

    def generate_batch(self, texts: List[str]) -> List[str]: ...


class InferencePipeline:

    def __init__(
        self,
        generator: Generator,
        dst: ResultRepository,
        template_service: Optional[TemplateService] = None,
        template_name: Optional[str] = None,
        translator: Optional[Any] = None,
        execution: Optional[Execution] = None,
        scoring: Optional[Scoring] = None,
        method: Literal["llm", "seq2seq"] = "llm",
        lang: str = "cypher",
        model: str = "unknown",
        ir_mode: bool = False,
        extract_query: bool = False,
        workers: int = 1,
    ):
        self.generator = generator
        self.dst = dst
        self.template_service = template_service
        self.template_name = template_name
        self.translator = translator
        self.execution = execution
        self.scoring = scoring or Scoring()
        self.method = method
        self.lang = lang
        self.model = model
        self.ir_mode = ir_mode
        self.extract_query = extract_query
        self.workers = workers

    def _build_prompt(self, question: str, schema: BaseSchema) -> str:
        return self.template_service.render(
            "prompts",
            self.template_name,
            question=question,
            schema=schema.to_prompt_string(),
            lang=self.lang,
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

    def _translate_ir(self, ir: str) -> str:
        try:
            if self.lang == "cypher":
                return self.translator.to_cypher(ir)
            elif self.lang == "sparql":
                return self.translator.to_sparql(ir)
            elif self.lang == "kopl":
                return self.translator.to_kopl(ir)
        except Exception:
            return ir
        return ir

    def generate(self, records: List[Record], schema: Optional[BaseSchema] = None) -> List[Record]:
        questions = [r.question for r in records]

        if self.template_service and self.template_name and schema:
            inputs = [self._build_prompt(q, schema) for q in questions]
        else:
            inputs = questions

        if self.workers > 1:
            outputs = self._generate_parallel(inputs)
        else:
            outputs = self.generator.generate_batch(inputs)

        if self.translator and self.ir_mode:
            queries = [self._translate_ir(o) for o in outputs]
        elif self.extract_query:
            queries = [self._extract_query(o) for o in outputs]
        else:
            queries = outputs

        for record, raw, query in zip(records, outputs, queries):
            gen = GenerationResult(
                query_raw=raw,
                query=query,
                ir=raw if self.ir_mode else None,
            )
            self.dst.save_generation(record.id, self.method, self.lang, self.model, gen)

        return records

    def _generate_parallel(self, inputs: List[str]) -> List[str]:
        results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_idx = {
                executor.submit(self.generator.generate, inp): i
                for i, inp in enumerate(inputs)
            }
            for future in tqdm(as_completed(future_to_idx), total=len(inputs), desc="Generating"):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return [results[i] for i in range(len(inputs))]

    def execute(self, records: List[Record]) -> List[Record]:
        valid_records = [r for r in records if self.dst.exists(r.id, self.method, self.lang, self.model)]

        if self.workers > 1:
            self._execute_parallel(valid_records)
        else:
            for record in tqdm(valid_records, desc="Executing"):
                result = self.dst.get(record.id, self.method, self.lang, self.model)
                exec_result = self.execution.execute(result)
                self.dst.save_execution(record.id, self.method, self.lang, self.model, exec_result)

        return records

    def _execute_parallel(self, records: List[Record]) -> None:
        def _execute_one(record: Record) -> tuple[Record, ExecutionResult]:
            result = self.dst.get(record.id, self.method, self.lang, self.model)
            return record, self.execution.execute(result)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_record = {
                executor.submit(_execute_one, r): r for r in records
            }
            for future in tqdm(as_completed(future_to_record), total=len(records), desc="Executing"):
                record, exec_result = future.result()
                self.dst.save_execution(record.id, self.method, self.lang, self.model, exec_result)

    def evaluate(self, records: List[Record]) -> List[Record]:
        valid_records = [r for r in records if self.dst.exists(r.id, self.method, self.lang, self.model)]

        for record in tqdm(valid_records, desc="Evaluating"):
            result = self.dst.get(record.id, self.method, self.lang, self.model)
            eval_result = self.scoring.evaluate(record, result)
            self.dst.save_evaluation(record.id, self.method, self.lang, self.model, eval_result)

        return records
