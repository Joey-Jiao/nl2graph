from typing import List, Dict, Any, Tuple

from .entity import Record, Result, EvaluationResult
from . import metrics


class Scoring:

    def evaluate(self, record: Record, result: Result) -> EvaluationResult:
        if not result.exec or not result.exec.success:
            return EvaluationResult()

        gold = metrics.normalize_answers(record.answer)
        pred = metrics.normalize_answers(result.exec.result or [])

        em = metrics.exact_match(gold, pred)
        precision, recall, f1 = metrics.precision_recall_f1(gold, pred)

        return EvaluationResult(
            exact_match=em,
            f1=f1,
            precision=precision,
            recall=recall,
        )

    def evaluate_batch(
        self, pairs: List[Tuple[Record, Result]]
    ) -> Dict[str, Any]:
        total = 0
        correct = 0
        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        error_count = 0

        for record, result in pairs:
            total += 1

            if not result.exec or not result.exec.success:
                error_count += 1
                continue

            eval_result = self.evaluate(record, result)
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
            "accuracy": metrics.accuracy(correct, valid_count),
            "avg_f1": total_f1 / valid_count if valid_count > 0 else 0.0,
            "avg_precision": total_precision / valid_count if valid_count > 0 else 0.0,
            "avg_recall": total_recall / valid_count if valid_count > 0 else 0.0,
        }
