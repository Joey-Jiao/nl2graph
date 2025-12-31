from typing import List, Dict, Tuple
from collections import defaultdict

from ..eval.entity import Record, Result
from .entity import Report, GroupStats, ErrorAnalysis
from .analysis import Analysis


class Reporting:

    def __init__(self):
        self.analysis = Analysis()

    def generate(self, pairs: List[Tuple[Record, Result]], config_id: str) -> Report:
        summary = self._compute_stats(pairs)
        by_hop = self._compute_by_hop(pairs)
        results = [r for _, r in pairs]
        errors = self._compute_errors(results)

        return Report(
            run_id=config_id,
            total=len(pairs),
            summary=summary,
            by_hop=by_hop,
            errors=errors,
        )

    def _compute_stats(self, pairs: List[Tuple[Record, Result]]) -> GroupStats:
        total = 0
        error_count = 0
        correct = 0
        sum_f1 = 0.0
        sum_precision = 0.0
        sum_recall = 0.0

        for _, result in pairs:
            total += 1

            if not result.exec or not result.exec.success:
                error_count += 1
                continue

            if result.eval:
                if result.eval.exact_match == 1.0:
                    correct += 1
                sum_f1 += result.eval.f1 or 0.0
                sum_precision += result.eval.precision or 0.0
                sum_recall += result.eval.recall or 0.0

        valid = total - error_count
        return GroupStats(
            count=total,
            error_count=error_count,
            accuracy=correct / valid if valid > 0 else 0.0,
            avg_f1=sum_f1 / valid if valid > 0 else 0.0,
            avg_precision=sum_precision / valid if valid > 0 else 0.0,
            avg_recall=sum_recall / valid if valid > 0 else 0.0,
        )

    def _compute_by_hop(self, pairs: List[Tuple[Record, Result]]) -> Dict[int, GroupStats]:
        grouped: Dict[int, List[Tuple[Record, Result]]] = defaultdict(list)
        for record, result in pairs:
            if record.hop is not None:
                grouped[record.hop].append((record, result))

        out = {}
        for hop, group in sorted(grouped.items()):
            out[hop] = self._compute_stats(group)
        return out

    def _compute_errors(self, results: List[Result]) -> ErrorAnalysis:
        error_count = 0
        for result in results:
            if result.exec and not result.exec.success:
                error_count += 1

        missing_rels = self.analysis.extract_missing_relations(results)
        error_types = self.analysis.categorize_errors(results)

        return ErrorAnalysis(
            total_errors=error_count,
            missing_relations=missing_rels,
            error_types=error_types,
        )
