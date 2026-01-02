import json
import sqlite3
import threading
from typing import Optional, Iterator, List, Any
from pathlib import Path

from .entity import Record, Result, GenerationResult, ExecutionResult, EvaluationResult


class SourceRepository:
    TABLE = "data"

    def __init__(self, db_path: str):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._ensure_table()

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn'):
            conn = sqlite3.connect(str(self._db_path), timeout=30, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return self._local.conn

    def _ensure_table(self) -> None:
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE} (
                id TEXT PRIMARY KEY,
                question TEXT,
                answer TEXT,
                extra TEXT
            )
        """)

    def init_from_json(self, json_path: str) -> int:
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            id_ = record.pop("id")
            question = record.pop("question")
            answer = record.pop("answer")
            extra = record if record else None

            self._conn.execute(
                f"INSERT OR REPLACE INTO {self.TABLE} (id, question, answer, extra) VALUES (?, ?, ?, ?)",
                (id_, question, json.dumps(answer, ensure_ascii=False), json.dumps(extra, ensure_ascii=False) if extra else None)
            )
        return len(records)

    def _row_to_record(self, row: sqlite3.Row) -> Record:
        data = {
            "id": row["id"],
            "question": row["question"],
            "answer": json.loads(row["answer"]),
        }
        if row["extra"]:
            extra = json.loads(row["extra"])
            data.update(extra)
        return Record.from_dict(data)

    def get(self, id: str) -> Optional[Record]:
        cursor = self._conn.execute(
            f"SELECT * FROM {self.TABLE} WHERE id = ?", (id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def exists(self, id: str) -> bool:
        cursor = self._conn.execute(
            f"SELECT 1 FROM {self.TABLE} WHERE id = ? LIMIT 1", (id,)
        )
        return cursor.fetchone() is not None

    def count(self) -> int:
        cursor = self._conn.execute(f"SELECT COUNT(*) FROM {self.TABLE}")
        return cursor.fetchone()[0]

    def iter_all(self) -> Iterator[Record]:
        cursor = self._conn.execute(f"SELECT * FROM {self.TABLE}")
        for row in cursor:
            yield self._row_to_record(row)

    def iter_by_filter(self, **filters: Any) -> Iterator[Record]:
        for record in self.iter_all():
            if all(record.get_field(k) == v for k, v in filters.items() if v is not None):
                yield record

    def close(self) -> None:
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            del self._local.conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ResultRepository:
    TABLE = "data"

    def __init__(self, db_path: str):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._ensure_table()

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn'):
            conn = sqlite3.connect(str(self._db_path), timeout=30, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return self._local.conn

    def _ensure_table(self) -> None:
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE} (
                question_id TEXT,
                method TEXT,
                lang TEXT,
                model TEXT,
                gen TEXT,
                exec TEXT,
                eval TEXT,
                PRIMARY KEY (question_id, method, lang, model)
            )
        """)

    def _row_to_result(self, row: sqlite3.Row) -> Result:
        gen = GenerationResult.model_validate(json.loads(row["gen"])) if row["gen"] else None
        exec_ = ExecutionResult.model_validate(json.loads(row["exec"])) if row["exec"] else None
        eval_ = EvaluationResult.model_validate(json.loads(row["eval"])) if row["eval"] else None

        return Result(
            question_id=row["question_id"],
            method=row["method"],
            lang=row["lang"],
            model=row["model"],
            gen=gen,
            exec=exec_,
            eval=eval_,
        )

    def get(self, question_id: str, method: str, lang: str, model: str) -> Optional[Result]:
        cursor = self._conn.execute(
            f"SELECT * FROM {self.TABLE} WHERE question_id = ? AND method = ? AND lang = ? AND model = ?",
            (question_id, method, lang, model)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_result(row)

    def exists(self, question_id: str, method: str, lang: str, model: str) -> bool:
        cursor = self._conn.execute(
            f"SELECT 1 FROM {self.TABLE} WHERE question_id = ? AND method = ? AND lang = ? AND model = ? LIMIT 1",
            (question_id, method, lang, model)
        )
        return cursor.fetchone() is not None

    def save_generation(
        self,
        question_id: str,
        method: str,
        lang: str,
        model: str,
        gen: GenerationResult,
    ) -> None:
        gen_json = json.dumps(gen.model_dump(), ensure_ascii=False)
        self._conn.execute(f"""
            INSERT INTO {self.TABLE} (question_id, method, lang, model, gen, exec, eval)
            VALUES (?, ?, ?, ?, ?, NULL, NULL)
            ON CONFLICT (question_id, method, lang, model)
            DO UPDATE SET gen = excluded.gen, exec = NULL, eval = NULL
        """, (question_id, method, lang, model, gen_json))

    def save_execution(
        self,
        question_id: str,
        method: str,
        lang: str,
        model: str,
        exec_: ExecutionResult,
    ) -> None:
        exec_json = json.dumps(exec_.model_dump(), ensure_ascii=False)
        self._conn.execute(f"""
            UPDATE {self.TABLE} SET exec = ?, eval = NULL
            WHERE question_id = ? AND method = ? AND lang = ? AND model = ?
        """, (exec_json, question_id, method, lang, model))

    def save_evaluation(
        self,
        question_id: str,
        method: str,
        lang: str,
        model: str,
        eval_: EvaluationResult,
    ) -> None:
        eval_json = json.dumps(eval_.model_dump(), ensure_ascii=False)
        self._conn.execute(f"""
            UPDATE {self.TABLE} SET eval = ?
            WHERE question_id = ? AND method = ? AND lang = ? AND model = ?
        """, (eval_json, question_id, method, lang, model))

    def count(self) -> int:
        cursor = self._conn.execute(f"SELECT COUNT(*) FROM {self.TABLE}")
        return cursor.fetchone()[0]

    def iter_all(self) -> Iterator[Result]:
        cursor = self._conn.execute(f"SELECT * FROM {self.TABLE}")
        for row in cursor:
            yield self._row_to_result(row)

    def iter_by_question(self, question_id: str) -> Iterator[Result]:
        cursor = self._conn.execute(
            f"SELECT * FROM {self.TABLE} WHERE question_id = ?", (question_id,)
        )
        for row in cursor:
            yield self._row_to_result(row)

    def iter_by_config(self, method: str, lang: str, model: str) -> Iterator[Result]:
        cursor = self._conn.execute(
            f"SELECT * FROM {self.TABLE} WHERE method = ? AND lang = ? AND model = ?",
            (method, lang, model)
        )
        for row in cursor:
            yield self._row_to_result(row)

    def export_json(self, path: str) -> List[dict]:
        results = []
        for result in self.iter_all():
            results.append(result.model_dump())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return results

    def clear_stage(self, method: str, lang: str, model: str, stage: str) -> int:
        cascade = {
            "gen": "gen = NULL, exec = NULL, eval = NULL",
            "exec": "exec = NULL, eval = NULL",
            "eval": "eval = NULL",
        }
        if stage not in cascade:
            raise ValueError(f"Invalid stage: {stage}")

        cursor = self._conn.execute(f"""
            UPDATE {self.TABLE} SET {cascade[stage]}
            WHERE method = ? AND lang = ? AND model = ?
        """, (method, lang, model))
        return cursor.rowcount

    def close(self) -> None:
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            del self._local.conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
