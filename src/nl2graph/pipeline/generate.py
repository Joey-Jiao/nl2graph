from typing import List, Optional, Protocol, runtime_checkable, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ..data import Record, GenerationOutput, GenerationResult
from ..data.repository import ResultRepository
from ..data.schema.base import BaseSchema


@runtime_checkable
class Generator(Protocol):

    def generate(self, question: str, schema: Optional[BaseSchema] = None) -> GenerationOutput: ...


IfExists = Literal["skip", "override"]


class GeneratePipeline:

    def __init__(
        self,
        generator: Generator,
        dst: ResultRepository,
        method: Literal["llm", "seq2seq"],
        lang: str,
        model: str,
        workers: int = 1,
        if_exists: IfExists = "skip",
    ):
        self.generator = generator
        self.dst = dst
        self.method = method
        self.lang = lang
        self.model = model
        self.workers = workers
        self.if_exists = if_exists

    def run(self, records: List[Record], schema: Optional[BaseSchema] = None) -> List[Record]:
        if self.if_exists == "skip":
            pending = [r for r in records if not self.dst.exists(r.id, self.method, self.lang, self.model)]
            if len(pending) < len(records):
                tqdm.write(f"Skipping {len(records) - len(pending)} existing records")
        else:
            pending = records

        if not pending:
            return records

        if self.workers > 1:
            self._run_parallel(pending, schema)
        else:
            for record in tqdm(pending, desc="Generating"):
                output = self.generator.generate(record.question, schema)
                self._save(record, output)

        return records

    def _save(self, record: Record, output: GenerationOutput) -> None:
        gen = GenerationResult(query=output.content, stats=output.stats)
        self.dst.save_generation(record.id, self.method, self.lang, self.model, gen)

    def _run_parallel(self, records: List[Record], schema: Optional[BaseSchema]) -> None:
        def _generate_one(record: Record) -> tuple[Record, GenerationOutput]:
            return record, self.generator.generate(record.question, schema)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_record = {
                executor.submit(_generate_one, r): r for r in records
            }
            for future in tqdm(as_completed(future_to_record), total=len(records), desc="Generating"):
                record, output = future.result()
                self._save(record, output)
