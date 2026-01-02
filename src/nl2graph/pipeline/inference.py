import re
from typing import List, Optional, Any, Protocol, runtime_checkable, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ..data import Record, GenerationResult, ExecutionResult, ResultRepository
from ..execution import Execution
from ..evaluation import Scoring
from ..base.templates.service import TemplateService
from ..execution.schema.base import BaseSchema


@runtime_checkable
class Generator(Protocol):

    def generate(self, text: str) -> str: ...


IfExists = Literal["skip", "override"]


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
        if_exists: IfExists = "skip",
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
        self.if_exists = if_exists

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
        if self.if_exists == "skip":
            pending = [r for r in records if not self.dst.exists(r.id, self.method, self.lang, self.model)]
            if len(pending) < len(records):
                tqdm.write(f"Skipping {len(records) - len(pending)} existing records")
        else:
            pending = records

        if not pending:
            return records

        if self.template_service and self.template_name and schema:
            inputs = [(r, self._build_prompt(r.question, schema)) for r in pending]
        else:
            inputs = [(r, r.question) for r in pending]

        if self.workers > 1:
            self._generate_parallel(inputs)
        else:
            for record, prompt in tqdm(inputs, desc="Generating"):
                raw = self.generator.generate(prompt)
                self._save_one(record, raw)

        return records

    def _save_one(self, record: Record, raw: str) -> None:
        if self.translator and self.ir_mode:
            query = self._translate_ir(raw)
        elif self.extract_query:
            query = self._extract_query(raw)
        else:
            query = raw

        gen = GenerationResult(query=query)
        self.dst.save_generation(record.id, self.method, self.lang, self.model, gen)

    def _generate_parallel(self, inputs: List[tuple[Record, str]]) -> None:
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_record = {
                executor.submit(self.generator.generate, prompt): record
                for record, prompt in inputs
            }
            for future in tqdm(as_completed(future_to_record), total=len(inputs), desc="Generating"):
                record = future_to_record[future]
                raw = future.result()
                self._save_one(record, raw)

    def execute(self, records: List[Record]) -> List[Record]:
        has_gen = [r for r in records if self.dst.exists(r.id, self.method, self.lang, self.model)]
        if len(has_gen) < len(records):
            tqdm.write(f"Skipping {len(records) - len(has_gen)} records without generation")

        if self.if_exists == "skip":
            pending = []
            for r in has_gen:
                result = self.dst.get(r.id, self.method, self.lang, self.model)
                if result.exec is None:
                    pending.append(r)
            if len(pending) < len(has_gen):
                tqdm.write(f"Skipping {len(has_gen) - len(pending)} existing records")
        else:
            pending = has_gen

        if not pending:
            return records

        if self.workers > 1:
            self._execute_parallel(pending)
        else:
            for record in tqdm(pending, desc="Executing"):
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
        has_exec = []
        for r in records:
            if not self.dst.exists(r.id, self.method, self.lang, self.model):
                continue
            result = self.dst.get(r.id, self.method, self.lang, self.model)
            if result.exec is not None:
                has_exec.append(r)
        if len(has_exec) < len(records):
            tqdm.write(f"Skipping {len(records) - len(has_exec)} records without execution")

        if self.if_exists == "skip":
            pending = []
            for r in has_exec:
                result = self.dst.get(r.id, self.method, self.lang, self.model)
                if result.eval is None:
                    pending.append(r)
            if len(pending) < len(has_exec):
                tqdm.write(f"Skipping {len(has_exec) - len(pending)} existing records")
        else:
            pending = has_exec

        if not pending:
            return records

        for record in tqdm(pending, desc="Evaluating"):
            result = self.dst.get(record.id, self.method, self.lang, self.model)
            eval_result = self.scoring.evaluate(record, result)
            self.dst.save_evaluation(record.id, self.method, self.lang, self.model, eval_result)

        return records
