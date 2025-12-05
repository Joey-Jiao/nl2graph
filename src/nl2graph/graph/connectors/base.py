from abc import ABC, abstractmethod
from typing import Optional

from ..entity import QueryLanguage, ConnectionConfig
from ..schema.base import BaseSchema
from ..result.entity import QueryResult


class BaseConnector(ABC):
    query_language: QueryLanguage

    def __init__(self, config: ConnectionConfig):
        self.config = config

    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def execute(self, query: str, timeout: Optional[int] = None) -> QueryResult:
        pass

    @abstractmethod
    def get_schema(self) -> BaseSchema:
        pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
