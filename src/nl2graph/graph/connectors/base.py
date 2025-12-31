from abc import ABC, abstractmethod
from typing import Optional

from ..entity import QueryLanguage
from ..schema.base import BaseSchema
from ..result.entity import QueryResult


class BaseConnector(ABC):
    query_language: QueryLanguage

    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        self.host = kwargs.get('host')
        self.port = kwargs.get('port')
        self.username = kwargs.get('username')
        self.password = kwargs.get('password')
        self.database = kwargs.get('database')
        self.timeout = kwargs.get('timeout', 30)

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
