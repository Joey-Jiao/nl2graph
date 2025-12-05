from typing import Dict

from ..base.configs import ConfigService
from .entity import QueryLanguage, ConnectionConfig
from .connectors.base import BaseConnector
from .connectors.neo4j import Neo4jConnector
from .connectors.rdflib import RDFLibConnector
from .connectors.gremlin import GremlinConnector


class GraphService:
    def __init__(self, config: ConfigService):
        self._config = config
        self._connectors: Dict[str, BaseConnector] = {}

    def get_connector(self, dataset: str, lang: str) -> BaseConnector:
        key = f"{dataset}/{lang}"
        if key in self._connectors:
            return self._connectors[key]

        conn_config = self._config.get(f"data.{dataset}.connections.{lang}")
        if not conn_config:
            raise KeyError(f"connection not found: data.{dataset}.connections.{lang}")

        connector = self._create_connector(lang, conn_config)
        connector.connect()
        self._connectors[key] = connector
        return connector

    def _create_connector(self, lang: str, config: dict) -> BaseConnector:
        conn_config = ConnectionConfig(
            name=lang,
            query_language=QueryLanguage(lang),
            host=config.get("host"),
            port=config.get("port"),
            username=config.get("username"),
            password=config.get("password"),
            database=config.get("database"),
            timeout=config.get("timeout", 30),
        )

        if lang == "cypher":
            return Neo4jConnector(conn_config)
        elif lang == "sparql":
            return RDFLibConnector(conn_config)
        elif lang == "gremlin":
            return GremlinConnector(conn_config)
        else:
            raise ValueError(f"unsupported query language: {lang}")

    def close_all(self):
        for connector in self._connectors.values():
            connector.close()
        self._connectors.clear()
