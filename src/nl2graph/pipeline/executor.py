from typing import List, Any

from ..graph.connectors.base import BaseConnector
from .entity import Record, ExecutionResult


class QueryExecutor:
    def __init__(self, connector: BaseConnector):
        self.connector = connector

    def execute(self, record: Record, run_id: str) -> ExecutionResult:
        run = record.runs.get(run_id)
        if not run or not run.gen or not run.gen.query_processed:
            return ExecutionResult(
                success=False,
                error="no query to execute",
            )

        query = run.gen.query_processed

        try:
            result = self.connector.execute(query)
            answer = self._extract_answer(result.rows)
            return ExecutionResult(
                answer=answer,
                success=True,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
            )

    def _extract_answer(self, rows: List[dict]) -> List[Any]:
        if not rows:
            return []

        answers = []
        for row in rows:
            values = list(row.values())
            if len(values) == 1:
                answers.append(values[0])
            else:
                answers.append(values)

        return answers
