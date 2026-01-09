from typing import Optional, TYPE_CHECKING

from ..entity import QueryLanguage
from ..schema.gremlin import GremlinSchema, NodeSchema, EdgeSchema, PropertySchema
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
        result = eval(query, {"g": self._g})

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

    def get_schema(self) -> GremlinSchema:
        from gremlin_python.process.graph_traversal import __

        vertex_labels = self._g.V().label().dedup().toList()

        nodes = []
        for label in vertex_labels:
            props = self._g.V().hasLabel(label).properties().key().dedup().toList()
            properties = [PropertySchema(name=p, data_type="any") for p in props]
            nodes.append(NodeSchema(label=label, properties=properties))

        edge_labels = self._g.E().label().dedup().toList()

        edges = []
        seen = set()
        for label in edge_labels:
            edge_info = (
                self._g.E()
                .hasLabel(label)
                .project("src", "tgt")
                .by(__.outV().label())
                .by(__.inV().label())
                .dedup()
                .toList()
            )
            for info in edge_info:
                key = (label, info["src"], info["tgt"])
                if key not in seen:
                    seen.add(key)
                    edges.append(EdgeSchema(
                        label=label,
                        source_label=info["src"],
                        target_label=info["tgt"],
                    ))

        return GremlinSchema(name=self.name, nodes=nodes, edges=edges)
