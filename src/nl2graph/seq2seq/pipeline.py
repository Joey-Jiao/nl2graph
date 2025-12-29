from typing import List, Optional

from tqdm import tqdm

from ..eval import Record, RunResult, GenerationResult, Scoring, Execution
from .generation import Generation


class Seq2SeqPipeline:

    def __init__(
        self,
        generation: Optional[Generation] = None,
        execution: Optional[Execution] = None,
        scoring: Optional[Scoring] = None,
        lang: str = "cypher",
        model: str = "bart-base",
        ir_mode: Optional[str] = None,
        ir_translator=None,
    ):
        self.generation = generation
        self.execution = execution
        self.scoring = scoring or Scoring()
        self.lang = lang
        self.model = model
        self.ir_mode = ir_mode
        self.ir_translator = ir_translator
        self.run_id = f"{lang}--{model}"

    def _ensure_run(self, record: Record):
        if self.run_id not in record.runs:
            record.runs[self.run_id] = RunResult(
                method="seq2seq",
                lang=self.lang,
                model=self.model,
            )

    def _translate_ir(self, ir_output: str) -> str:
        if not self.ir_translator or not self.ir_mode:
            return ir_output

        try:
            if self.lang == "cypher":
                return self.ir_translator.to_cypher(ir_output)
            elif self.lang == "sparql":
                return self.ir_translator.to_sparql(ir_output)
            elif self.lang == "kopl":
                return self.ir_translator.to_kopl(ir_output)
        except Exception:
            pass

        return ir_output

    def generate(self, records: List[Record]) -> List[Record]:
        for record in records:
            self._ensure_run(record)

        questions = [r.question for r in records]
        predictions = self.generation.generate_batch(questions)

        for record, pred in zip(records, predictions):
            ir_output = pred if self.ir_mode else None
            query = self._translate_ir(pred) if self.ir_mode else pred

            record.runs[self.run_id].gen = GenerationResult(
                query_raw=pred,
                query=query,
                ir=ir_output,
            )

        return records

    def execute(self, records: List[Record]) -> List[Record]:
        valid_records = [r for r in records if self.run_id in r.runs]

        for record in tqdm(valid_records, desc="Executing"):
            record.runs[self.run_id].exec = self.execution.execute(record, self.run_id)

        return records

    def evaluate(self, records: List[Record]) -> List[Record]:
        valid_records = [r for r in records if self.run_id in r.runs]

        for record in tqdm(valid_records, desc="Evaluating"):
            record.runs[self.run_id].eval = self.scoring.evaluate_record(record, self.run_id)

        return records
