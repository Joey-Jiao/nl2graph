from typing import List, Dict, Any, Optional

from pydantic import BaseModel


class GroupStats(BaseModel):
    count: int = 0
    error_count: int = 0
    accuracy: float = 0.0
    avg_f1: float = 0.0
    avg_precision: float = 0.0
    avg_recall: float = 0.0


class ErrorAnalysis(BaseModel):
    total_errors: int = 0
    missing_relations: List[str] = []
    error_types: Dict[str, int] = {}


class Report(BaseModel):
    run_id: str
    total: int = 0
    summary: GroupStats = GroupStats()
    by_hop: Dict[int, GroupStats] = {}
    errors: ErrorAnalysis = ErrorAnalysis()
    metadata: Dict[str, Any] = {}
