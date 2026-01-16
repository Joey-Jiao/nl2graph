from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

from ..data.entity import Record, Result
from .entity import Report, GroupStats, ErrorAnalysis
from .analysis import Analysis


class Reporting:

    def __init__(self):
        self.analysis = Analysis()

    def generate(
        self,
        pairs: List[Tuple[Record, Result]],
        config_id: str,
        group_by: Optional[List[str]] = None,
    ) -> Report:
        summary = self._compute_stats(pairs)
        by_field = {}
        for field in (group_by or []):
            by_field[field] = self._compute_by_field(pairs, field)
        results = [r for _, r in pairs]
        errors = self._compute_errors(results)

        return Report(
            run_id=config_id,
            total=len(pairs),
            summary=summary,
            by_field=by_field,
            errors=errors,
        )

    def _compute_stats(self, pairs: List[Tuple[Record, Result]]) -> GroupStats:
        total = 0
        error_count = 0
        correct = 0
        sum_f1 = 0.0
        sum_precision = 0.0
        sum_recall = 0.0

        total_duration = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cached_tokens = 0
        gen_count = 0

        for _, result in pairs:
            total += 1

            if result.gen and result.gen.stats:
                stats = result.gen.stats
                total_duration += stats.get("duration", 0.0)
                total_input_tokens += stats.get("input_tokens", 0)
                total_output_tokens += stats.get("output_tokens", 0)
                total_cached_tokens += stats.get("cached_tokens", 0)
                gen_count += 1

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
            total_duration=total_duration,
            avg_duration=total_duration / gen_count if gen_count > 0 else 0.0,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cached_tokens=total_cached_tokens,
            avg_input_tokens=total_input_tokens / gen_count if gen_count > 0 else 0.0,
            avg_output_tokens=total_output_tokens / gen_count if gen_count > 0 else 0.0,
            avg_cached_tokens=total_cached_tokens / gen_count if gen_count > 0 else 0.0,
        )

    def _compute_by_field(self, pairs: List[Tuple[Record, Result]], field: str) -> Dict[str, GroupStats]:
        grouped: Dict[Any, List[Tuple[Record, Result]]] = defaultdict(list)
        for record, result in pairs:
            value = record.get_field(field)
            if value is not None:
                grouped[value].append((record, result))

        out = {}
        for key, group in sorted(grouped.items(), key=lambda x: str(x[0])):
            out[str(key)] = self._compute_stats(group)
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
