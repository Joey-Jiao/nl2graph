from typing import Dict

from ..base import ConfigService
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

        config = self._config.get(f"data.{dataset}.connection.{lang}")
        if not config:
            raise KeyError(f"connection not found: data.{dataset}.connection.{lang}")

        connector = self._create_connector(lang, config)
        connector.connect()
        self._connectors[key] = connector
        return connector

    def _create_connector(self, lang: str, config: dict) -> BaseConnector:
        config = {**config, "name": lang}

        if lang == "cypher":
            return Neo4jConnector(**config)
        elif lang == "sparql":
            return RDFLibConnector(**config)
        elif lang == "gremlin":
            return GremlinConnector(**config)
        else:
            raise ValueError(f"unsupported query language: {lang}")

    def close_all(self):
        for connector in self._connectors.values():
            connector.close()
        self._connectors.clear()
