from typing import List, Dict, Any, Optional

from pydantic import BaseModel


class QueryResult(BaseModel):
    columns: List[str] = []
    rows: List[Dict[str, Any]] = []
    raw: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def is_empty(self) -> bool:
        return len(self.rows) == 0

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_list(self) -> List[Dict[str, Any]]:
        return self.rows

    def to_values(self) -> List[List[Any]]:
        return [[row.get(col) for col in self.columns] for row in self.rows]
