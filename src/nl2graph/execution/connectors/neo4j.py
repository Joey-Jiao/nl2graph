import re
from typing import Optional

from ..entity import QueryLanguage
from ..result.entity import QueryResult
from ..result.converter import convert_neo4j_value
from .base import BaseConnector


class Neo4jConnector(BaseConnector):
    query_language = QueryLanguage.CYPHER

    SANITY_HANDLERS = {
        "lowercase_relationships": "_lowercase_relationships",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._driver = None
        self.sanity = kwargs.get('sanity', [])

    def connect(self) -> None:
        import logging
        from neo4j import GraphDatabase

        logging.getLogger("neo4j").setLevel(logging.ERROR)

        uri = f"bolt://{self.host}:{self.port}"
        self._driver = GraphDatabase.driver(
            uri,
            auth=(self.username, self.password),
        )

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def execute(self, query: str, timeout: Optional[int] = None) -> QueryResult:
        query = self._apply_sanity(query)
        timeout_ms = (timeout or self.timeout) * 1000
        database = self.database or "neo4j"

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

    def _apply_sanity(self, query: str) -> str:
        for name in self.sanity:
            if name in self.SANITY_HANDLERS:
                handler = getattr(self, self.SANITY_HANDLERS[name])
                query = handler(query)
        return query

    def _lowercase_relationships(self, query: str) -> str:
        pattern = r':\s*([A-Z_]+)(?=\s*[\]\{])'
        return re.sub(pattern, lambda m: ':' + m.group(1).lower(), query)
