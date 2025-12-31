import pytest

from nl2graph.eval.entity import (
    GenerationResult,
    ExecutionResult,
    EvaluationResult,
    Record,
    Result,
)


class TestGenerationResult:

    def test_create_empty(self):
        result = GenerationResult()
        assert result.query_raw is None
        assert result.query is None
        assert result.ir is None

    def test_create_with_values(self):
        result = GenerationResult(
            query_raw="MATCH (n) RETURN n",
            query="MATCH (n) RETURN n",
            ir="some_ir_repr",
        )
        assert result.query_raw == "MATCH (n) RETURN n"
        assert result.query == "MATCH (n) RETURN n"
        assert result.ir == "some_ir_repr"


class TestExecutionResult:

    def test_create_default(self):
        result = ExecutionResult()
        assert result.result is None
        assert result.success is False
        assert result.error is None

    def test_create_success(self):
        result = ExecutionResult(
            result=["Paris", "London"],
            success=True,
        )
        assert result.result == ["Paris", "London"]
        assert result.success is True
        assert result.error is None

    def test_create_failure(self):
        result = ExecutionResult(
            success=False,
            error="Connection timeout",
        )
        assert result.success is False
        assert result.error == "Connection timeout"


class TestEvaluationResult:

    def test_create_empty(self):
        result = EvaluationResult()
        assert result.exact_match is None
        assert result.f1 is None
        assert result.precision is None
        assert result.recall is None

    def test_create_with_metrics(self):
        result = EvaluationResult(
            exact_match=1.0,
            f1=0.8,
            precision=0.75,
            recall=0.85,
        )
        assert result.exact_match == 1.0
        assert result.f1 == 0.8
        assert result.precision == 0.75
        assert result.recall == 0.85


class TestRecord:

    def test_create(self):
        record = Record(id="q001", question="What is X?", answer=["Y"])
        assert record.id == "q001"
        assert record.question == "What is X?"
        assert record.answer == ["Y"]
        assert record.hop is None

    def test_create_with_hop(self):
        record = Record(id="q001", question="Q", answer=["A"], hop=2)
        assert record.hop == 2

    def test_to_dict(self):
        record = Record(id="q001", question="Q", answer=["A"], hop=1)
        d = record.to_dict()
        assert d["id"] == "q001"
        assert d["question"] == "Q"
        assert d["answer"] == ["A"]
        assert d["hop"] == 1

    def test_from_dict(self):
        d = {"id": "q001", "question": "Q", "answer": ["A"], "hop": 1}
        record = Record.from_dict(d)
        assert record.id == "q001"
        assert record.question == "Q"
        assert record.answer == ["A"]
        assert record.hop == 1

    def test_id_required(self):
        with pytest.raises(ValueError):
            Record(question="Q", answer=["A"])


class TestResult:

    def test_create_minimal(self):
        result = Result(
            record_id="q001",
            method="llm",
            lang="cypher",
            model="gpt-4o",
        )
        assert result.record_id == "q001"
        assert result.method == "llm"
        assert result.lang == "cypher"
        assert result.model == "gpt-4o"
        assert result.gen is None
        assert result.exec is None
        assert result.eval is None

    def test_create_with_nested(self):
        result = Result(
            record_id="q001",
            method="seq2seq",
            lang="sparql",
            model="bart",
            gen=GenerationResult(query="SELECT ?x WHERE { ?x a :Person }"),
            exec=ExecutionResult(result=["Alice"], success=True),
            eval=EvaluationResult(exact_match=1.0),
        )
        assert result.method == "seq2seq"
        assert result.gen.query == "SELECT ?x WHERE { ?x a :Person }"
        assert result.exec.success is True
        assert result.eval.exact_match == 1.0

    def test_method_literal_validation(self):
        with pytest.raises(ValueError):
            Result(record_id="q001", method="invalid", lang="cypher", model="test")

    def test_to_dict(self):
        result = Result(
            record_id="q001",
            method="llm",
            lang="cypher",
            model="gpt-4o",
            gen=GenerationResult(query="MATCH..."),
        )
        d = result.to_dict()
        assert d["record_id"] == "q001"
        assert d["method"] == "llm"
        assert d["gen"]["query"] == "MATCH..."

    def test_from_dict(self):
        d = {
            "record_id": "q001",
            "method": "llm",
            "lang": "cypher",
            "model": "gpt-4o",
            "gen": {"query": "MATCH..."},
            "exec": None,
            "eval": None,
        }
        result = Result.from_dict(d)
        assert result.record_id == "q001"
        assert result.gen.query == "MATCH..."
