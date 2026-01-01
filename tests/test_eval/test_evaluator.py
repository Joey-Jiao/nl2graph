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
            record_id="q001",
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
            record_id="q001",
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
            record_id="q001",
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
            record_id="q001",
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
            record_id="q001",
            method="llm",
            lang="cypher",
            model="gpt-4o",
            gen=GenerationResult(query="MATCH..."),
            exec=ExecutionResult(result=["paris"], success=True),
        )

        eval_result = scoring.evaluate(record, result)
        assert eval_result.exact_match == 1.0


class TestScoringEvaluateBatch:

    @pytest.fixture
    def scoring(self):
        return Scoring()

    def test_evaluate_batch_all_correct(self, scoring):
        pairs = []
        for i in range(5):
            record = Record(id=f"q{i:03d}", question=f"Q{i}", answer=[f"A{i}"])
            result = Result(
                record_id=f"q{i:03d}",
                method="llm",
                lang="cypher",
                model="gpt-4o",
                exec=ExecutionResult(result=[f"A{i}"], success=True),
            )
            pairs.append((record, result))

        batch_result = scoring.evaluate_batch(pairs)
        assert batch_result["total"] == 5
        assert batch_result["correct"] == 5
        assert batch_result["error_count"] == 0
        assert batch_result["accuracy"] == 1.0
        assert batch_result["avg_f1"] == 1.0

    def test_evaluate_batch_partial(self, scoring):
        pairs = []
        for i in range(4):
            record = Record(id=f"q{i:03d}", question=f"Q{i}", answer=["correct"])
            if i < 2:
                result = Result(
                    record_id=f"q{i:03d}",
                    method="llm",
                    lang="cypher",
                    model="gpt-4o",
                    exec=ExecutionResult(result=["correct"], success=True),
                )
            else:
                result = Result(
                    record_id=f"q{i:03d}",
                    method="llm",
                    lang="cypher",
                    model="gpt-4o",
                    exec=ExecutionResult(result=["wrong"], success=True),
                )
            pairs.append((record, result))

        batch_result = scoring.evaluate_batch(pairs)
        assert batch_result["total"] == 4
        assert batch_result["correct"] == 2
        assert batch_result["accuracy"] == 0.5

    def test_evaluate_batch_with_errors(self, scoring):
        pairs = []
        for i in range(4):
            record = Record(id=f"q{i:03d}", question=f"Q{i}", answer=["A"])
            if i < 2:
                result = Result(
                    record_id=f"q{i:03d}",
                    method="llm",
                    lang="cypher",
                    model="gpt-4o",
                    exec=ExecutionResult(result=["A"], success=True),
                )
            else:
                result = Result(
                    record_id=f"q{i:03d}",
                    method="llm",
                    lang="cypher",
                    model="gpt-4o",
                    exec=ExecutionResult(success=False, error="error"),
                )
            pairs.append((record, result))

        batch_result = scoring.evaluate_batch(pairs)
        assert batch_result["total"] == 4
        assert batch_result["error_count"] == 2
        assert batch_result["correct"] == 2
        assert batch_result["accuracy"] == 1.0

    def test_evaluate_batch_empty(self, scoring):
        batch_result = scoring.evaluate_batch([])
        assert batch_result["total"] == 0
        assert batch_result["accuracy"] == 0.0
