from typing import List, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ..data import Record, ExecutionResult
from ..data.repository import ResultRepository
from ..execution import Execution


IfExists = Literal["skip", "override"]


class ExecutePipeline:

    def __init__(
        self,
        execution: Execution,
        dst: ResultRepository,
        method: Literal["llm", "seq2seq"],
        lang: str,
        model: str,
        workers: int = 1,
        if_exists: IfExists = "skip",
    ):
        self.execution = execution
        self.dst = dst
        self.method = method
        self.lang = lang
        self.model = model
        self.workers = workers
        self.if_exists = if_exists

    def run(self, records: List[Record]) -> List[Record]:
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
            self._run_parallel(pending)
        else:
            for record in tqdm(pending, desc="Executing"):
                result = self.dst.get(record.id, self.method, self.lang, self.model)
                exec_result = self.execution.execute(result)
                self.dst.save_execution(record.id, self.method, self.lang, self.model, exec_result)

        return records

    def _run_parallel(self, records: List[Record]) -> None:
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
