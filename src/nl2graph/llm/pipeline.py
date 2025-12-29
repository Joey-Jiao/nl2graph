from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ..graph.schema.base import BaseSchema
from ..eval import Record, RunResult, GenerationResult, ExecutionResult, EvaluationResult, Scoring, Execution
from .generation import Generation


class LLMPipeline:

    def __init__(
        self,
        generation: Optional[Generation] = None,
        execution: Optional[Execution] = None,
        scoring: Optional[Scoring] = None,
        lang: str = "cypher",
        model: str = "unknown",
        workers: int = 1,
    ):
        self.generation = generation
        self.execution = execution
        self.scoring = scoring or Scoring()
        self.lang = lang
        self.model = model
        self.run_id = f"{lang}--{model}"
        self.workers = workers

    def _ensure_run(self, record: Record):
        if self.run_id not in record.runs:
            record.runs[self.run_id] = RunResult(
                method="llm",
                lang=self.lang,
                model=self.model,
            )

    def _generate_one(self, record: Record, schema: BaseSchema) -> GenerationResult:
        return self.generation.generate(record, schema)

    def _execute_one(self, record: Record) -> ExecutionResult:
        return self.execution.execute(record, self.run_id)

    def _evaluate_one(self, record: Record) -> EvaluationResult:
        return self.scoring.evaluate_record(record, self.run_id)

    def generate(self, records: List[Record], schema: BaseSchema) -> List[Record]:
        for record in records:
            self._ensure_run(record)

        if self.workers == 1:
            for record in tqdm(records, desc="Generating"):
                record.runs[self.run_id].gen = self._generate_one(record, schema)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_record = {
                    executor.submit(self._generate_one, record, schema): record
                    for record in records
                }
                for future in tqdm(as_completed(future_to_record), total=len(records), desc="Generating"):
                    record = future_to_record[future]
                    record.runs[self.run_id].gen = future.result()

        return records

    def execute(self, records: List[Record]) -> List[Record]:
        valid_records = [r for r in records if self.run_id in r.runs]

        if self.workers == 1:
            for record in tqdm(valid_records, desc="Executing"):
                record.runs[self.run_id].exec = self._execute_one(record)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_record = {
                    executor.submit(self._execute_one, record): record
                    for record in valid_records
                }
                for future in tqdm(as_completed(future_to_record), total=len(valid_records), desc="Executing"):
                    record = future_to_record[future]
                    record.runs[self.run_id].exec = future.result()

        return records

    def evaluate(self, records: List[Record]) -> List[Record]:
        valid_records = [r for r in records if self.run_id in r.runs]

        for record in tqdm(valid_records, desc="Evaluating"):
            record.runs[self.run_id].eval = self._evaluate_one(record)

        return records
