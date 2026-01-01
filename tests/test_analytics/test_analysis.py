import pytest

from nl2graph.data.entity import Result, GenerationResult, ExecutionResult
from nl2graph.analysis import Analysis


class TestAnalysis:

    @pytest.fixture
    def analysis(self):
        return Analysis()

    def test_extract_missing_relations_single(self, analysis):
        results = [
            Result(
                record_id="q001",
                method="llm",
                lang="cypher",
                model="test",
                gen=GenerationResult(query="MATCH..."),
                exec=ExecutionResult(
                    success=False,
                    error="Unknown relationship type 'DIRECTED_BY'"
                ),
            )
        ]

        missing = analysis.extract_missing_relations(results)
        assert missing == ["DIRECTED_BY"]

    def test_extract_missing_relations_multiple(self, analysis):
        results = [
            Result(
                record_id="q001",
                method="llm",
                lang="cypher",
                model="test",
                gen=GenerationResult(query="..."),
                exec=ExecutionResult(success=False, error="Unknown relationship type 'ACTED_IN'"),
            ),
            Result(
                record_id="q002",
                method="llm",
                lang="cypher",
                model="test",
                gen=GenerationResult(query="..."),
                exec=ExecutionResult(success=False, error="Unknown relationship type 'DIRECTED_BY'"),
            ),
            Result(
                record_id="q003",
                method="llm",
                lang="cypher",
                model="test",
                gen=GenerationResult(query="..."),
                exec=ExecutionResult(success=False, error="Unknown relationship type 'ACTED_IN'"),
            ),
        ]

        missing = analysis.extract_missing_relations(results)
        assert missing == ["ACTED_IN", "DIRECTED_BY"]

    def test_extract_missing_relations_skip_success(self, analysis):
        results = [
            Result(
                record_id="q001",
                method="llm",
                lang="cypher",
                model="test",
                gen=GenerationResult(query="..."),
                exec=ExecutionResult(result=["A"], success=True),
            )
        ]

        missing = analysis.extract_missing_relations(results)
        assert missing == []

    def test_categorize_errors(self, analysis):
        results = [
            Result(
                record_id="q001",
                method="llm",
                lang="cypher",
                model="test",
                exec=ExecutionResult(success=False, error="Connection refused"),
            ),
            Result(
                record_id="q002",
                method="llm",
                lang="cypher",
                model="test",
                exec=ExecutionResult(success=False, error="Syntax error at position 10"),
            ),
            Result(
                record_id="q003",
                method="llm",
                lang="cypher",
                model="test",
                exec=ExecutionResult(success=False, error="Query timeout after 30s"),
            ),
            Result(
                record_id="q004",
                method="llm",
                lang="cypher",
                model="test",
                exec=ExecutionResult(success=False, error="Unknown relationship type 'X'"),
            ),
        ]

        categories = analysis.categorize_errors(results)
        assert categories["connection"] == 1
        assert categories["syntax"] == 1
        assert categories["timeout"] == 1
        assert categories["missing_relationship"] == 1

    def test_classify_error_no_query(self, analysis):
        category = analysis._classify_error("no query to execute")
        assert category == "no_query"

    def test_classify_error_missing_label(self, analysis):
        assert analysis._classify_error("Label 'Person' not found") == "missing_label"
        assert analysis._classify_error("Unknown node type") == "missing_label"

    def test_classify_error_missing_property(self, analysis):
        assert analysis._classify_error("Property 'name' does not exist") == "missing_property"

    def test_classify_error_other(self, analysis):
        category = analysis._classify_error("Some weird error message")
        assert category == "other"
