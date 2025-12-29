from typing import List, Any, Set


class Metrics:

    @staticmethod
    def normalize_answers(answers: List[Any]) -> Set[str]:
        normalized = set()
        for ans in answers:
            if ans is None:
                continue
            normalized.add(str(ans).strip().lower())
        return normalized

    @staticmethod
    def exact_match(gold: Set[str], pred: Set[str]) -> float:
        return 1.0 if gold == pred else 0.0

    @staticmethod
    def precision_recall_f1(gold: Set[str], pred: Set[str]) -> tuple:
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

    @staticmethod
    def accuracy(correct: int, total: int) -> float:
        return correct / total if total > 0 else 0.0

    @staticmethod
    def string_match(pred: str, target: str) -> bool:
        return pred.strip().lower() == target.strip().lower()
