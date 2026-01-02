import json
import sqlite3
from typing import Optional, Iterator, List, Any
from pathlib import Path


class StorageService:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

    def _ensure_table(self, table: str) -> None:
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                key TEXT PRIMARY KEY,
                data TEXT
            )
        """)
        self._conn.commit()

    def get(self, table: str, key: str) -> Optional[dict]:
        self._ensure_table(table)
        cursor = self._conn.execute(
            f"SELECT data FROM {table} WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return json.loads(row["data"])

    def put(self, table: str, key: str, data: dict) -> None:
        self._ensure_table(table)
        self._conn.execute(
            f"INSERT OR REPLACE INTO {table} (key, data) VALUES (?, ?)",
            (key, json.dumps(data, ensure_ascii=False))
        )
        self._conn.commit()

    def update(self, table: str, key: str, partial: dict) -> None:
        existing = self.get(table, key)
        if existing is None:
            self.put(table, key, partial)
        else:
            self._merge_dict(existing, partial)
            self.put(table, key, existing)

    def _merge_dict(self, base: dict, update: dict) -> None:
        for k, v in update.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._merge_dict(base[k], v)
            else:
                base[k] = v

    def delete(self, table: str, key: str) -> None:
        self._ensure_table(table)
        self._conn.execute(f"DELETE FROM {table} WHERE key = ?", (key,))
        self._conn.commit()

    def exists(self, table: str, key: str) -> bool:
        self._ensure_table(table)
        cursor = self._conn.execute(
            f"SELECT 1 FROM {table} WHERE key = ? LIMIT 1", (key,)
        )
        return cursor.fetchone() is not None

    def count(self, table: str) -> int:
        self._ensure_table(table)
        cursor = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
        return cursor.fetchone()[0]

    def keys(self, table: str) -> List[str]:
        self._ensure_table(table)
        cursor = self._conn.execute(f"SELECT key FROM {table}")
        return [row["key"] for row in cursor.fetchall()]

    def iter_all(self, table: str) -> Iterator[tuple[str, dict]]:
        self._ensure_table(table)
        cursor = self._conn.execute(f"SELECT key, data FROM {table}")
        for row in cursor:
            yield row["key"], json.loads(row["data"])

    def init_from_json(self, table: str, json_path: str, key_field: str) -> None:
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        self._ensure_table(table)
        for record in records:
            key = record[key_field]
            self.put(table, key, record)

    def export_json(self, table: str, path: str) -> List[dict]:
        records = [data for _, data in self.iter_all(table)]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return records

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

