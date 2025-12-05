from enum import Enum
from typing import Optional

from pydantic import BaseModel


class QueryLanguage(str, Enum):
    CYPHER = "cypher"
    SPARQL = "sparql"
    GREMLIN = "gremlin"


class ConnectionConfig(BaseModel):
    name: str
    query_language: QueryLanguage
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    timeout: int = 30
