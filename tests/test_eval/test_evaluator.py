import pytest

from nl2graph.eval.entity import (
    Record,
    RunResult,
    GenerationResult,
    ExecutionResult,
    EvaluationResult,
)
from nl2graph.eval.scoring import Scoring


class TestScoringEvaluateRecord:

    @pytest.fixture
    def scoring(self):
        return Scoring()

    def test_evaluate_exact_match(self, scoring):
        record = Record(question="Q", answer=["Paris"])
        run = RunResult(
            method="llm",
            lang="cypher",
            model="gpt-4o",
            gen=GenerationResult(query="MATCH..."),
            exec=ExecutionResult(result=["Paris"], success=True),
        )
        record.add_run(run)

        result = scoring.evaluate_record(record, "cypher--gpt-4o")
        assert result.exact_match == 1.0
        assert result.f1 == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_evaluate_partial_match(self, scoring):
        record = Record(question="Q", answer=["Paris", "London", "Berlin"])
        run = RunResult(
            method="llm",
            lang="cypher",
            model="gpt-4o",
            gen=GenerationResult(query="MATCH..."),
            exec=ExecutionResult(result=["Paris", "London"], success=True),
        )
        record.add_run(run)

        result = scoring.evaluate_record(record, "cypher--gpt-4o")
        assert result.exact_match == 0.0
        assert result.precision == 1.0
        assert result.recall == pytest.approx(2/3)

    def test_evaluate_no_match(self, scoring):
        record = Record(question="Q", answer=["Paris"])
        run = RunResult(
            method="llm",
            lang="cypher",
            model="gpt-4o",
            gen=GenerationResult(query="MATCH..."),
            exec=ExecutionResult(result=["London"], success=True),
        )
        record.add_run(run)

        result = scoring.evaluate_record(record, "cypher--gpt-4o")
        assert result.exact_match == 0.0
        assert result.f1 == 0.0

    def test_evaluate_missing_run(self, scoring):
        record = Record(question="Q", answer=["Paris"])
        result = scoring.evaluate_record(record, "nonexistent--run")
        assert result.exact_match is None

    def test_evaluate_failed_execution(self, scoring):
        record = Record(question="Q", answer=["Paris"])
        run = RunResult(
            method="llm",
            lang="cypher",
            model="gpt-4o",
            exec=ExecutionResult(success=False, error="timeout"),
        )
        record.add_run(run)

        result = scoring.evaluate_record(record, "cypher--gpt-4o")
        assert result.exact_match is None

    def test_evaluate_case_insensitive(self, scoring):
        record = Record(question="Q", answer=["PARIS"])
        run = RunResult(
            method="llm",
            lang="cypher",
            model="gpt-4o",
            gen=GenerationResult(query="MATCH..."),
            exec=ExecutionResult(result=["paris"], success=True),
        )
        record.add_run(run)

        result = scoring.evaluate_record(record, "cypher--gpt-4o")
        assert result.exact_match == 1.0


class TestScoringEvaluateBatch:

    @pytest.fixture
    def scoring(self):
        return Scoring()

    def test_evaluate_batch_all_correct(self, scoring):
        records = []
        for i in range(5):
            record = Record(question=f"Q{i}", answer=[f"A{i}"])
            run = RunResult(
                method="llm",
                lang="cypher",
                model="gpt-4o",
                exec=ExecutionResult(result=[f"A{i}"], success=True),
            )
            record.add_run(run)
            records.append(record)

        result = scoring.evaluate_batch(records, "cypher--gpt-4o")
        assert result["total"] == 5
        assert result["correct"] == 5
        assert result["error_count"] == 0
        assert result["accuracy"] == 1.0
        assert result["avg_f1"] == 1.0

    def test_evaluate_batch_partial(self, scoring):
        records = []
        for i in range(4):
            record = Record(question=f"Q{i}", answer=["correct"])
            if i < 2:
                run = RunResult(
                    method="llm",
                    lang="cypher",
                    model="gpt-4o",
                    exec=ExecutionResult(result=["correct"], success=True),
                )
            else:
                run = RunResult(
                    method="llm",
                    lang="cypher",
                    model="gpt-4o",
                    exec=ExecutionResult(result=["wrong"], success=True),
                )
            record.add_run(run)
            records.append(record)

        result = scoring.evaluate_batch(records, "cypher--gpt-4o")
        assert result["total"] == 4
        assert result["correct"] == 2
        assert result["accuracy"] == 0.5

    def test_evaluate_batch_with_errors(self, scoring):
        records = []
        for i in range(4):
            record = Record(question=f"Q{i}", answer=["A"])
            if i < 2:
                run = RunResult(
                    method="llm",
                    lang="cypher",
                    model="gpt-4o",
                    exec=ExecutionResult(result=["A"], success=True),
                )
            else:
                run = RunResult(
                    method="llm",
                    lang="cypher",
                    model="gpt-4o",
                    exec=ExecutionResult(success=False, error="error"),
                )
            record.add_run(run)
            records.append(record)

        result = scoring.evaluate_batch(records, "cypher--gpt-4o")
        assert result["total"] == 4
        assert result["error_count"] == 2
        assert result["correct"] == 2
        assert result["accuracy"] == 1.0

    def test_evaluate_batch_empty(self, scoring):
        result = scoring.evaluate_batch([], "cypher--gpt-4o")
        assert result["total"] == 0
        assert result["accuracy"] == 0.0

    def test_evaluate_batch_no_matching_runs(self, scoring):
        records = [Record(question="Q", answer=["A"])]
        result = scoring.evaluate_batch(records, "nonexistent--run")
        assert result["total"] == 0
