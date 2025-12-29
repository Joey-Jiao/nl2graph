from typing import List, Dict, Optional, Any, Literal

from pydantic import BaseModel, ConfigDict


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


class RunResult(BaseModel):
    method: Literal["llm", "seq2seq"]
    lang: str
    model: str
    gen: Optional[GenerationResult] = None
    exec: Optional[ExecutionResult] = None
    eval: Optional[EvaluationResult] = None


class Record(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: str
    answer: List[Any]
    hop: Optional[int] = None
    runs: Dict[str, RunResult] = {}

    def get_run_id(self, lang: str, model: str) -> str:
        return f"{lang}--{model}"

    def add_run(self, run: RunResult) -> str:
        run_id = self.get_run_id(run.lang, run.model)
        self.runs[run_id] = run
        return run_id
