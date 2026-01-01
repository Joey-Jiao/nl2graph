from enum import Enum


class QueryLanguage(str, Enum):
    CYPHER = "cypher"
    SPARQL = "sparql"
    GREMLIN = "gremlin"
