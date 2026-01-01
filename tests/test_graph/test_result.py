import pytest

from nl2graph.execution.result.entity import QueryResult
from nl2graph.execution.entity import QueryLanguage


class TestQueryResult:

    def test_create_empty(self):
        result = QueryResult()
        assert result.columns == []
        assert result.rows == []
        assert result.raw is None
        assert result.is_empty is True
        assert result.row_count == 0

    def test_create_with_data(self):
        result = QueryResult(
            columns=["name", "age"],
            rows=[
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ],
        )
        assert result.columns == ["name", "age"]
        assert len(result.rows) == 2
        assert result.is_empty is False
        assert result.row_count == 2

    def test_to_list(self):
        result = QueryResult(
            columns=["name"],
            rows=[{"name": "Alice"}, {"name": "Bob"}],
        )
        assert result.to_list() == [{"name": "Alice"}, {"name": "Bob"}]

    def test_to_values(self):
        result = QueryResult(
            columns=["name", "age"],
            rows=[
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ],
        )
        values = result.to_values()
        assert values == [["Alice", 30], ["Bob", 25]]

    def test_to_values_missing_column(self):
        result = QueryResult(
            columns=["name", "age"],
            rows=[
                {"name": "Alice"},
            ],
        )
        values = result.to_values()
        assert values == [["Alice", None]]

    def test_with_raw(self):
        raw_data = {"neo4j_result": "some_object"}
        result = QueryResult(
            columns=["x"],
            rows=[{"x": 1}],
            raw=raw_data,
        )
        assert result.raw == raw_data


class TestQueryLanguage:

    def test_cypher(self):
        assert QueryLanguage.CYPHER.value == "cypher"

    def test_sparql(self):
        assert QueryLanguage.SPARQL.value == "sparql"

    def test_gremlin(self):
        assert QueryLanguage.GREMLIN.value == "gremlin"


