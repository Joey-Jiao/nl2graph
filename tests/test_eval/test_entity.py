import pytest

from nl2graph.eval.entity import (
    GenerationResult,
    ExecutionResult,
    EvaluationResult,
    RunResult,
    Record,
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


class TestRunResult:

    def test_create_minimal(self):
        result = RunResult(
            method="llm",
            lang="cypher",
            model="gpt-4o",
        )
        assert result.method == "llm"
        assert result.lang == "cypher"
        assert result.model == "gpt-4o"
        assert result.gen is None
        assert result.exec is None
        assert result.eval is None

    def test_create_with_nested(self):
        result = RunResult(
            method="seq2seq",
            lang="sparql",
            model="bart-base",
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
            RunResult(method="invalid", lang="cypher", model="test")


class TestRecord:

    def test_create_minimal(self):
        record = Record(question="What is X?", answer=["Y"])
        assert record.question == "What is X?"
        assert record.answer == ["Y"]
        assert record.hop is None
        assert record.runs == {}

    def test_create_with_hop(self):
        record = Record(question="Q", answer=["A"], hop=2)
        assert record.hop == 2

    def test_get_run_id(self):
        record = Record(question="Q", answer=["A"])
        run_id = record.get_run_id("cypher", "gpt-4o")
        assert run_id == "cypher--gpt-4o"

    def test_add_run(self):
        record = Record(question="Q", answer=["A"])
        run = RunResult(method="llm", lang="cypher", model="gpt-4o")
        run_id = record.add_run(run)
        assert run_id == "cypher--gpt-4o"
        assert "cypher--gpt-4o" in record.runs
        assert record.runs["cypher--gpt-4o"] == run

    def test_multiple_runs(self):
        record = Record(question="Q", answer=["A"])
        run1 = RunResult(method="llm", lang="cypher", model="gpt-4o")
        run2 = RunResult(method="seq2seq", lang="sparql", model="bart")
        record.add_run(run1)
        record.add_run(run2)
        assert len(record.runs) == 2
        assert "cypher--gpt-4o" in record.runs
        assert "sparql--bart" in record.runs

    def test_extra_fields_allowed(self):
        record = Record(
            question="Q",
            answer=["A"],
            extra_field="extra_value",
        )
        assert record.question == "Q"
