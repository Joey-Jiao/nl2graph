from typing import List, Any

from .connectors.base import BaseConnector
from ..data.entity import Result, ExecutionResult


class Execution:

    def __init__(self, connector: BaseConnector):
        self.connector = connector

    def execute(self, result: Result) -> ExecutionResult:
        if not result.gen or not result.gen.query:
            return ExecutionResult(
                success=False,
                error="no query to execute",
            )

        query = result.gen.query

        try:
            exec_result = self.connector.execute(query)
            answer = self._extract_answer(exec_result.rows)
            return ExecutionResult(
                result=answer,
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
