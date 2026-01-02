import pytest

from nl2graph.data.entity import Record, Result, GenerationResult, ExecutionResult, EvaluationResult
from nl2graph.analysis import Reporting, Report, GroupStats


class TestReporting:

    @pytest.fixture
    def reporting(self):
        return Reporting()

    @pytest.fixture
    def sample_pairs(self):
        pairs = []

        r1 = Record(id="q001", question="Q1", answer=["A1"], hop=1)
        res1 = Result(
            question_id="q001",
            method="llm",
            lang="cypher",
            model="gpt-4o",
            gen=GenerationResult(query="MATCH..."),
            exec=ExecutionResult(result=["A1"], success=True),
            eval=EvaluationResult(exact_match=1.0, f1=1.0, precision=1.0, recall=1.0),
        )
        pairs.append((r1, res1))

        r2 = Record(id="q002", question="Q2", answer=["A2"], hop=1)
        res2 = Result(
            question_id="q002",
            method="llm",
            lang="cypher",
            model="gpt-4o",
            gen=GenerationResult(query="MATCH..."),
            exec=ExecutionResult(result=["wrong"], success=True),
            eval=EvaluationResult(exact_match=0.0, f1=0.0, precision=0.0, recall=0.0),
        )
        pairs.append((r2, res2))

        r3 = Record(id="q003", question="Q3", answer=["A3"], hop=2)
        res3 = Result(
            question_id="q003",
            method="llm",
            lang="cypher",
            model="gpt-4o",
            gen=GenerationResult(query="MATCH..."),
            exec=ExecutionResult(success=False, error="Unknown relationship type 'DIRECTED_BY'"),
        )
        pairs.append((r3, res3))

        r4 = Record(id="q004", question="Q4", answer=["A4"], hop=2)
        res4 = Result(
            question_id="q004",
            method="llm",
            lang="cypher",
            model="gpt-4o",
            gen=GenerationResult(query="MATCH..."),
            exec=ExecutionResult(result=["A4"], success=True),
            eval=EvaluationResult(exact_match=1.0, f1=1.0, precision=1.0, recall=1.0),
        )
        pairs.append((r4, res4))

        return pairs

    def test_generate_summary(self, reporting, sample_pairs):
        report = reporting.generate(sample_pairs, "cypher--gpt-4o")

        assert report.total == 4
        assert report.summary.count == 4
        assert report.summary.error_count == 1
        assert report.summary.accuracy == 2 / 3

    def test_generate_by_hop(self, reporting, sample_pairs):
        report = reporting.generate(sample_pairs, "cypher--gpt-4o", group_by=["hop"])

        assert "hop" in report.by_field
        assert "1" in report.by_field["hop"]
        assert "2" in report.by_field["hop"]

        hop1 = report.by_field["hop"]["1"]
        assert hop1.count == 2
        assert hop1.error_count == 0
        assert hop1.accuracy == 0.5

        hop2 = report.by_field["hop"]["2"]
        assert hop2.count == 2
        assert hop2.error_count == 1
        assert hop2.accuracy == 1.0

    def test_generate_errors(self, reporting, sample_pairs):
        report = reporting.generate(sample_pairs, "cypher--gpt-4o")

        assert report.errors.total_errors == 1
        assert "DIRECTED_BY" in report.errors.missing_relations
        assert report.errors.error_types.get("missing_relationship", 0) == 1

    def test_empty_records(self, reporting):
        report = reporting.generate([], "cypher--gpt-4o")

        assert report.total == 0
        assert report.summary.count == 0
        assert report.summary.accuracy == 0.0
        assert report.by_field == {}
        assert report.errors.total_errors == 0
