from typing import Optional, Iterator, List, Any

from ..base.storage import StorageService
from .entity import Record, Result, GenerationResult, ExecutionResult, EvaluationResult


class SourceRepository:
    def __init__(self, db_path: str, table: str = "records"):
        self._storage = StorageService(db_path)
        self._table = table

    def init_from_json(self, json_path: str) -> int:
        self._storage.init_from_json(self._table, json_path, key_field="id")
        return self._storage.count(self._table)

    def get(self, id: str) -> Optional[Record]:
        data = self._storage.get(self._table, id)
        if data is None:
            return None
        return Record.from_dict(data)

    def exists(self, id: str) -> bool:
        return self._storage.exists(self._table, id)

    def count(self) -> int:
        return self._storage.count(self._table)

    def iter_all(self) -> Iterator[Record]:
        for _, data in self._storage.iter_all(self._table):
            yield Record.from_dict(data)

    def iter_by_filter(self, **filters: Any) -> Iterator[Record]:
        for _, data in self._storage.iter_all(self._table):
            if all(data.get(k) == v for k, v in filters.items() if v is not None):
                yield Record.from_dict(data)

    def close(self) -> None:
        self._storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ResultRepository:
    TABLE = "results"
    KEY_FIELDS = ["record_id", "method", "lang", "model"]

    def __init__(self, db_path: str):
        self._storage = StorageService(db_path)
        self._storage._ensure_table_composite(self.TABLE, self.KEY_FIELDS)

    def _make_keys(self, record_id: str, method: str, lang: str, model: str) -> dict:
        return {
            "record_id": record_id,
            "method": method,
            "lang": lang,
            "model": model,
        }

    def get(self, record_id: str, method: str, lang: str, model: str) -> Optional[Result]:
        keys = self._make_keys(record_id, method, lang, model)
        data = self._storage.get_composite(self.TABLE, keys)
        if data is None:
            return None
        return Result.from_dict(data)

    def exists(self, record_id: str, method: str, lang: str, model: str) -> bool:
        keys = self._make_keys(record_id, method, lang, model)
        return self._storage.exists_composite(self.TABLE, keys)

    def save_generation(
        self,
        record_id: str,
        method: str,
        lang: str,
        model: str,
        gen: GenerationResult,
    ) -> None:
        keys = self._make_keys(record_id, method, lang, model)
        data = {
            **keys,
            "gen": gen.model_dump(),
            "exec": None,
            "eval": None,
        }
        self._storage.put_composite(self.TABLE, keys, data)

    def save_execution(
        self,
        record_id: str,
        method: str,
        lang: str,
        model: str,
        exec: ExecutionResult,
    ) -> None:
        keys = self._make_keys(record_id, method, lang, model)
        self._storage.update_composite(self.TABLE, keys, {
            "exec": exec.model_dump(),
            "eval": None,
        })

    def save_evaluation(
        self,
        record_id: str,
        method: str,
        lang: str,
        model: str,
        eval: EvaluationResult,
    ) -> None:
        keys = self._make_keys(record_id, method, lang, model)
        self._storage.update_composite(self.TABLE, keys, {
            "eval": eval.model_dump(),
        })

    def count(self) -> int:
        return self._storage.count_composite(self.TABLE)

    def iter_all(self) -> Iterator[Result]:
        for data in self._storage.iter_all_composite(self.TABLE):
            yield Result.from_dict(data)

    def iter_by_record(self, record_id: str) -> Iterator[Result]:
        for data in self._storage.iter_by_field(self.TABLE, "record_id", record_id):
            yield Result.from_dict(data)

    def iter_by_config(self, method: str, lang: str, model: str) -> Iterator[Result]:
        for data in self._storage.iter_all_composite(self.TABLE):
            if data["method"] == method and data["lang"] == lang and data["model"] == model:
                yield Result.from_dict(data)

    def iter_pending(
        self,
        src: SourceRepository,
        method: str,
        lang: str,
        model: str,
    ) -> Iterator[Record]:
        for record in src.iter_all():
            if not self.exists(record.id, method, lang, model):
                yield record
            else:
                result = self.get(record.id, method, lang, model)
                if result.eval is None:
                    yield record

    def export_json(self, path: str) -> List[dict]:
        return self._storage.export_json_composite(self.TABLE, path)

    def close(self) -> None:
        self._storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
