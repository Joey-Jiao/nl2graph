from typing import List, Dict, Any, Optional

from pydantic import BaseModel


class GroupStats(BaseModel):
    count: int = 0
    error_count: int = 0
    accuracy: float = 0.0
    avg_f1: float = 0.0
    avg_precision: float = 0.0
    avg_recall: float = 0.0

    total_duration: float = 0.0
    avg_duration: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    avg_cached_tokens: float = 0.0


class ErrorAnalysis(BaseModel):
    total_errors: int = 0
    missing_relations: List[str] = []
    error_types: Dict[str, int] = {}


class Report(BaseModel):
    run_id: str
    total: int = 0
    summary: GroupStats = GroupStats()
    by_field: Dict[str, Dict[str, GroupStats]] = {}
    errors: ErrorAnalysis = ErrorAnalysis()
    metadata: Dict[str, Any] = {}
