import re
from typing import List, Dict, Set
from collections import Counter

from ..data.entity import Result


class Analysis:

    MISSING_REL_PATTERNS = [
        r"Unknown relationship type '([^']+)'",
        r"Relationship type `([^`]+)` not found",
        r"relationship type (\w+) does not exist",
        r"\[:(\w+)\].*not found",
    ]

    def extract_missing_relations(self, results: List[Result]) -> List[str]:
        missing: Set[str] = set()
        for result in results:
            if not result.exec or result.exec.success:
                continue
            error = result.exec.error or ""
            for pattern in self.MISSING_REL_PATTERNS:
                match = re.search(pattern, error, re.IGNORECASE)
                if match:
                    missing.add(match.group(1))
        return sorted(missing)

    def categorize_errors(self, results: List[Result]) -> Dict[str, int]:
        categories: Counter = Counter()
        for result in results:
            if not result.exec or result.exec.success:
                continue
            error = result.exec.error or ""
            category = self._classify_error(error)
            categories[category] += 1
        return dict(categories)

    def _classify_error(self, error: str) -> str:
        error_lower = error.lower()
        if "timeout" in error_lower:
            return "timeout"
        if "connection" in error_lower:
            return "connection"
        if "syntax" in error_lower:
            return "syntax"
        if "relationship" in error_lower or "rel_type" in error_lower:
            return "missing_relationship"
        if "label" in error_lower or "node" in error_lower:
            return "missing_label"
        if "property" in error_lower:
            return "missing_property"
        if "no query" in error_lower:
            return "no_query"
        return "other"
