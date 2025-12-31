from typing import List, Optional, Any, Literal

from pydantic import BaseModel


class GenerationResult(BaseModel):
    query_raw: Optional[str] = None
    query: Optional[str] = None
    ir: Optional[str] = None


class ExecutionResult(BaseModel):
    result: Optional[List[Any]] = None
    success: bool = False
    error: Optional[str] = None


class EvaluationResult(BaseModel):
    exact_match: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None


class Record(BaseModel):
    id: str
    question: str
    answer: List[Any]
    hop: Optional[int] = None

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "Record":
        return cls.model_validate(data)


class Result(BaseModel):
    record_id: str
    method: Literal["llm", "seq2seq"]
    lang: str
    model: str
    gen: Optional[GenerationResult] = None
    exec: Optional[ExecutionResult] = None
    eval: Optional[EvaluationResult] = None

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "Result":
        return cls.model_validate(data)
