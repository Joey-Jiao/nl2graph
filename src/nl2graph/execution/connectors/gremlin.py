from typing import Optional, TYPE_CHECKING

from ..entity import QueryLanguage
from ..result.entity import QueryResult
from ..result.converter import convert_gremlin_value
from .base import BaseConnector

if TYPE_CHECKING:
    from gremlin_python.process.graph_traversal import GraphTraversalSource


class GremlinConnector(BaseConnector):
    query_language = QueryLanguage.GREMLIN

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._connection = None
        self._g: Optional["GraphTraversalSource"] = None

    def connect(self) -> None:
        from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
        from gremlin_python.process.anonymous_traversal import traversal

        url = f"ws://{self.host}:{self.port}/gremlin"
        self._connection = DriverRemoteConnection(url, "g")
        self._g = traversal().withRemote(self._connection)

    def close(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None
            self._g = None

    @property
    def g(self) -> "GraphTraversalSource":
        return self._g

    def execute(self, query: str, timeout: Optional[int] = None) -> QueryResult:
        traversal = eval(query, {"g": self._g})

        if hasattr(traversal, "toList"):
            raw_results = traversal.toList()
        elif hasattr(traversal, "next"):
            raw_results = [traversal.next()]
        else:
            raw_results = [traversal]

        rows = []
        for item in raw_results:
            converted = convert_gremlin_value(item)
            if isinstance(converted, dict):
                rows.append(converted)
            else:
                rows.append({"value": converted})

        columns = list(rows[0].keys()) if rows else []
        return QueryResult(columns=columns, rows=rows, raw=raw_results)

    def execute_traversal(self, traversal_func) -> QueryResult:
        result = traversal_func(self._g)

        if hasattr(result, "toList"):
            raw_results = result.toList()
        else:
            raw_results = [result]

        rows = []
        for item in raw_results:
            converted = convert_gremlin_value(item)
            if isinstance(converted, dict):
                rows.append(converted)
            else:
                rows.append({"value": converted})

        columns = list(rows[0].keys()) if rows else []
        return QueryResult(columns=columns, rows=rows, raw=raw_results)
