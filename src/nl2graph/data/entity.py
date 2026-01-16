from dataclasses import dataclass
from typing import List, Optional, Any, Literal, Dict

from pydantic import BaseModel, ConfigDict


@dataclass
class GenerationOutput:
    content: str
    stats: Optional[Dict[str, Any]] = None


class GenerationResult(BaseModel):
    query: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None


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
    model_config = ConfigDict(extra='allow')

    id: str
    question: str
    answer: List[Any]

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "Record":
        return cls.model_validate(data)

    def get_field(self, field: str, default: Any = None) -> Any:
        if hasattr(self, field):
            return getattr(self, field)
        return self.model_extra.get(field, default)


class Result(BaseModel):
    question_id: str
    method: Literal["llm", "seq2seq"]
    lang: str
    model: str
    gen: Optional[GenerationResult] = None
    exec: Optional[ExecutionResult] = None
    eval: Optional[EvaluationResult] = None
