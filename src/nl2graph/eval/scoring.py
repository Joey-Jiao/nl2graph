from typing import List, Dict, Any

from .entity import Record, EvaluationResult
from .metrics import Metrics


class Scoring:

    def __init__(self):
        self.metrics = Metrics()

    def evaluate_record(self, record: Record, run_id: str) -> EvaluationResult:
        run = record.runs.get(run_id)
        if not run or not run.exec or not run.exec.success:
            return EvaluationResult()

        gold = self.metrics.normalize_answers(record.answer)
        pred = self.metrics.normalize_answers(run.exec.result or [])

        exact_match = self.metrics.exact_match(gold, pred)
        precision, recall, f1 = self.metrics.precision_recall_f1(gold, pred)

        return EvaluationResult(
            exact_match=exact_match,
            f1=f1,
            precision=precision,
            recall=recall,
        )

    def evaluate_batch(
        self, records: List[Record], run_id: str
    ) -> Dict[str, Any]:
        total = 0
        correct = 0
        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        error_count = 0

        for record in records:
            run = record.runs.get(run_id)
            if not run:
                continue

            total += 1

            if not run.exec or not run.exec.success:
                error_count += 1
                continue

            eval_result = self.evaluate_record(record, run_id)
            if eval_result.exact_match == 1.0:
                correct += 1
            total_f1 += eval_result.f1 or 0.0
            total_precision += eval_result.precision or 0.0
            total_recall += eval_result.recall or 0.0

        valid_count = total - error_count
        return {
            "total": total,
            "correct": correct,
            "error_count": error_count,
            "accuracy": self.metrics.accuracy(correct, valid_count),
            "avg_f1": total_f1 / valid_count if valid_count > 0 else 0.0,
            "avg_precision": total_precision / valid_count if valid_count > 0 else 0.0,
            "avg_recall": total_recall / valid_count if valid_count > 0 else 0.0,
        }
