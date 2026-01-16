import pytest

from nl2graph.data.entity import (
    Record,
    Result,
    GenerationResult,
    ExecutionResult,
    EvaluationResult,
)
from nl2graph.evaluation.scoring import Scoring


class TestScoringEvaluate:

    @pytest.fixture
    def scoring(self):
        return Scoring()

    def test_evaluate_exact_match(self, scoring):
        record = Record(id="q001", question="Q", answer=["Paris"])
        result = Result(
            question_id="q001",
            method="llm",
            lang="cypher",
            model="gpt-4o",
            gen=GenerationResult(query="MATCH..."),
            exec=ExecutionResult(result=["Paris"], success=True),
        )

        eval_result = scoring.evaluate(record, result)
        assert eval_result.exact_match == 1.0
        assert eval_result.f1 == 1.0
        assert eval_result.precision == 1.0
        assert eval_result.recall == 1.0

    def test_evaluate_partial_match(self, scoring):
        record = Record(id="q001", question="Q", answer=["Paris", "London", "Berlin"])
        result = Result(
            question_id="q001",
            method="llm",
            lang="cypher",
            model="gpt-4o",
            gen=GenerationResult(query="MATCH..."),
            exec=ExecutionResult(result=["Paris", "London"], success=True),
        )

        eval_result = scoring.evaluate(record, result)
        assert eval_result.exact_match == 0.0
        assert eval_result.precision == 1.0
        assert eval_result.recall == pytest.approx(2/3)

    def test_evaluate_no_match(self, scoring):
        record = Record(id="q001", question="Q", answer=["Paris"])
        result = Result(
            question_id="q001",
            method="llm",
            lang="cypher",
            model="gpt-4o",
            gen=GenerationResult(query="MATCH..."),
            exec=ExecutionResult(result=["London"], success=True),
        )

        eval_result = scoring.evaluate(record, result)
        assert eval_result.exact_match == 0.0
        assert eval_result.f1 == 0.0

    def test_evaluate_failed_execution(self, scoring):
        record = Record(id="q001", question="Q", answer=["Paris"])
        result = Result(
            question_id="q001",
            method="llm",
            lang="cypher",
            model="gpt-4o",
            exec=ExecutionResult(success=False, error="timeout"),
        )

        eval_result = scoring.evaluate(record, result)
        assert eval_result.exact_match is None

    def test_evaluate_case_insensitive(self, scoring):
        record = Record(id="q001", question="Q", answer=["PARIS"])
        result = Result(
            question_id="q001",
            method="llm",
            lang="cypher",
            model="gpt-4o",
            gen=GenerationResult(query="MATCH..."),
            exec=ExecutionResult(result=["paris"], success=True),
        )

        eval_result = scoring.evaluate(record, result)
        assert eval_result.exact_match == 1.0
