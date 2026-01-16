from ..data.entity import Record, Result, EvaluationResult
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
