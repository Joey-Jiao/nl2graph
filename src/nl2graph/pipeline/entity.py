from typing import List, Dict, Optional, Any

from pydantic import BaseModel


class GenerationResult(BaseModel):
    query_raw: Optional[str] = None
    query_processed: Optional[str] = None


class ExecutionResult(BaseModel):
    answer: Optional[List[Any]] = None
    success: bool = False
    error: Optional[str] = None


class EvaluationResult(BaseModel):
    exact_match: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    hits_at_1: Optional[float] = None


class RunResult(BaseModel):
    lang: str
    model: str
    gen: Optional[GenerationResult] = None
    exec: Optional[ExecutionResult] = None
    eval: Optional[EvaluationResult] = None


class Record(BaseModel):
    question: str
    answer: List[Any]
    hop: Optional[int] = None
    runs: Dict[str, RunResult] = {}

    class Config:
        extra = "allow"
