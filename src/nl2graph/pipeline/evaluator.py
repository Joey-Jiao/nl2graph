from typing import List, Any, Set

from .entity import Record, EvaluationResult


class Evaluator:
    def evaluate(self, record: Record, run_id: str) -> EvaluationResult:
        run = record.runs.get(run_id)
        if not run or not run.exec or not run.exec.success:
            return EvaluationResult()

        gold = self._normalize_answers(record.answer)
        pred = self._normalize_answers(run.exec.answer or [])

        exact_match = self._exact_match(gold, pred)
        precision, recall, f1 = self._precision_recall_f1(gold, pred)
        hits_at_1 = self._hits_at_1(gold, pred)

        return EvaluationResult(
            exact_match=exact_match,
            f1=f1,
            precision=precision,
            recall=recall,
            hits_at_1=hits_at_1,
        )

    def _normalize_answers(self, answers: List[Any]) -> Set[str]:
        normalized = set()
        for ans in answers:
            if ans is None:
                continue
            normalized.add(str(ans).strip().lower())
        return normalized

    def _exact_match(self, gold: Set[str], pred: Set[str]) -> float:
        return 1.0 if gold == pred else 0.0

    def _precision_recall_f1(self, gold: Set[str], pred: Set[str]) -> tuple:
        if not pred:
            return (0.0, 0.0, 0.0)
        if not gold:
            return (0.0, 0.0, 0.0)

        tp = len(gold & pred)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(gold) if gold else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return (precision, recall, f1)

    def _hits_at_1(self, gold: Set[str], pred: Set[str]) -> float:
        if not pred:
            return 0.0
        first_pred = next(iter(pred))
        return 1.0 if first_pred in gold else 0.0
