from typing import Optional, List, Dict, Any

from ..entity import QueryLanguage, ConnectionConfig
from ..schema.base import BaseSchema
from ..schema.property_graph import PropertyGraphSchema, NodeSchema, EdgeSchema, PropertySchema
from ..result.entity import QueryResult
from ..result.converter import convert_neo4j_value
from .base import BaseConnector


NODE_PROPERTIES_QUERY = """
CALL apoc.meta.data()
YIELD label, property, type, elementType
WHERE elementType = 'node'
RETURN label, property, type
ORDER BY label, property
"""

REL_QUERY = """
CALL apoc.meta.data()
YIELD label, other, elementType
WHERE elementType = 'relationship' AND other IS NOT NULL
UNWIND other AS target
RETURN label AS rel_type, target AS source_label, label AS target_label
"""

DIRECT_NODE_PROPERTIES_QUERY = """
MATCH (n)
WITH labels(n) AS lbls, keys(n) AS ks, n
UNWIND lbls AS label
UNWIND ks AS key
WITH label, key, n[key] AS val
RETURN DISTINCT label, key AS property, apoc.meta.cypher.type(val) AS type
ORDER BY label, property
"""

DIRECT_REL_QUERY = """
MATCH (a)-[r]->(b)
WITH type(r) AS rel_type, labels(a) AS src_labels, labels(b) AS tgt_labels
UNWIND src_labels AS src
UNWIND tgt_labels AS tgt
RETURN DISTINCT rel_type, src AS source_label, tgt AS target_label
ORDER BY rel_type
"""


class Neo4jConnector(BaseConnector):
    query_language = QueryLanguage.CYPHER

    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self._driver = None

    def connect(self) -> None:
        from neo4j import GraphDatabase

        uri = f"bolt://{self.config.host}:{self.config.port}"
        self._driver = GraphDatabase.driver(
            uri,
            auth=(self.config.username, self.config.password),
        )

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def execute(self, query: str, timeout: Optional[int] = None) -> QueryResult:
        timeout_ms = (timeout or self.config.timeout) * 1000
        database = self.config.database or "neo4j"

        with self._driver.session(database=database) as session:
            result = session.run(query, timeout=timeout_ms)
            records = list(result)
            columns = result.keys() if records else []

            rows = []
            for record in records:
                row = {}
                for key in columns:
                    row[key] = convert_neo4j_value(record[key])
                rows.append(row)

            return QueryResult(columns=list(columns), rows=rows, raw=records)

    def get_schema(self, mode: str = "direct") -> PropertyGraphSchema:
        if mode == "apoc":
            return self._get_schema_apoc()
        return self._get_schema_direct()

    def _get_schema_apoc(self) -> PropertyGraphSchema:
        node_result = self.execute(NODE_PROPERTIES_QUERY)
        rel_result = self.execute(REL_QUERY)

        nodes_map: Dict[str, List[PropertySchema]] = {}
        for row in node_result.rows:
            label = row["label"]
            if label not in nodes_map:
                nodes_map[label] = []
            nodes_map[label].append(PropertySchema(
                name=row["property"],
                data_type=row["type"],
            ))

        nodes = [NodeSchema(label=k, properties=v) for k, v in nodes_map.items()]

        edges = []
        for row in rel_result.rows:
            edges.append(EdgeSchema(
                label=row["rel_type"],
                source_label=row["source_label"],
                target_label=row["target_label"],
            ))

        return PropertyGraphSchema(name=self.config.name, nodes=nodes, edges=edges)

    def _get_schema_direct(self) -> PropertyGraphSchema:
        node_result = self.execute(DIRECT_NODE_PROPERTIES_QUERY)
        rel_result = self.execute(DIRECT_REL_QUERY)

        nodes_map: Dict[str, List[PropertySchema]] = {}
        for row in node_result.rows:
            label = row["label"]
            if label not in nodes_map:
                nodes_map[label] = []
            nodes_map[label].append(PropertySchema(
                name=row["property"],
                data_type=row["type"],
            ))

        nodes = [NodeSchema(label=k, properties=v) for k, v in nodes_map.items()]

        edges_set = set()
        edges = []
        for row in rel_result.rows:
            key = (row["rel_type"], row["source_label"], row["target_label"])
            if key not in edges_set:
                edges_set.add(key)
                edges.append(EdgeSchema(
                    label=row["rel_type"],
                    source_label=row["source_label"],
                    target_label=row["target_label"],
                ))

        return PropertyGraphSchema(name=self.config.name, nodes=nodes, edges=edges)
