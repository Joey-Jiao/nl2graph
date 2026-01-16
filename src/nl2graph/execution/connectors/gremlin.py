from typing import Optional, TYPE_CHECKING

from ..entity import QueryLanguage
from ..result.entity import QueryResult
from ..result.converter import convert_gremlin_value
from .base import BaseConnector

if TYPE_CHECKING:
    from gremlin_python.driver.client import Client


class GremlinConnector(BaseConnector):
    query_language = QueryLanguage.GREMLIN

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client: Optional["Client"] = None

    def connect(self) -> None:
        from gremlin_python.driver.client import Client

        url = f"ws://{self.host}:{self.port}/gremlin"
        self._client = Client(url, "g")

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def execute(self, query: str, timeout: Optional[int] = None) -> QueryResult:
        result_set = self._client.submit(query)
        raw_results = result_set.all().result()

        rows = []
        for item in raw_results:
            converted = convert_gremlin_value(item)
            if isinstance(converted, dict):
                rows.append(converted)
            else:
                rows.append({"value": converted})

        columns = list(rows[0].keys()) if rows else []
        return QueryResult(columns=columns, rows=rows, raw=raw_results)
