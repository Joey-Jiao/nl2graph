from typing import List, Literal

from tqdm import tqdm

from ..data import Record
from ..data.repository import ResultRepository
from ..evaluation import Scoring


IfExists = Literal["skip", "override"]


class EvaluatePipeline:

    def __init__(
        self,
        dst: ResultRepository,
        method: Literal["llm", "seq2seq"],
        lang: str,
        model: str,
        scoring: Scoring = None,
        if_exists: IfExists = "skip",
    ):
        self.dst = dst
        self.method = method
        self.lang = lang
        self.model = model
        self.scoring = scoring or Scoring()
        self.if_exists = if_exists

    def run(self, records: List[Record]) -> List[Record]:
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
