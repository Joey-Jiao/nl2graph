from typing import Optional, TYPE_CHECKING
from pathlib import Path

from ..entity import QueryLanguage
from ..result.entity import QueryResult
from ..result.converter import convert_rdf_value
from .base import BaseConnector

if TYPE_CHECKING:
    from rdflib import Graph


class RDFLibConnector(BaseConnector):
    query_language = QueryLanguage.SPARQL

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._graph: Optional["Graph"] = None
        self.data_path = kwargs.get('data_path')
        self.data_format = kwargs.get('data_format', 'turtle')

    def connect(self) -> None:
        from rdflib import Graph

        self._graph = Graph()
        if self.data_path:
            path = Path(self.data_path)
            if path.exists():
                self._graph.parse(str(path), format=self.data_format)

    def close(self) -> None:
        self._graph = None

    def load_data(self, data: str, format: str = "turtle") -> None:
        from rdflib import Graph

        if self._graph is None:
            self._graph = Graph()
        self._graph.parse(data=data, format=format)

    def load_file(self, path: str, format: Optional[str] = None) -> None:
        from rdflib import Graph

        if self._graph is None:
            self._graph = Graph()
        self._graph.parse(path, format=format)

    def execute(self, query: str, timeout: Optional[int] = None) -> QueryResult:
        result = self._graph.query(query)

        if hasattr(result, "bindings"):
            if not result.bindings:
                return QueryResult(columns=[], rows=[], raw=result)

            columns = [str(v) for v in result.vars]
            rows = []
            for binding in result.bindings:
                row = {}
                for var in result.vars:
                    val = binding.get(var)
                    row[str(var)] = convert_rdf_value(val)
                rows.append(row)
            return QueryResult(columns=columns, rows=rows, raw=result)

        elif hasattr(result, "askAnswer"):
            return QueryResult(
                columns=["result"],
                rows=[{"result": result.askAnswer}],
                raw=result,
            )

        return QueryResult(columns=[], rows=[], raw=result)
