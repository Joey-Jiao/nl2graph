import pytest
import tempfile
import json
from pathlib import Path

from nl2graph.base.storage import StorageService


class TestStorageService:

    @pytest.fixture
    def storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            svc = StorageService(str(db_path))
            yield svc
            svc.close()

    def test_put_and_get(self, storage):
        storage.put("records", "q1", {"question": "What is 1+1?", "answer": 2})
        result = storage.get("records", "q1")
        assert result == {"question": "What is 1+1?", "answer": 2}

    def test_get_nonexistent(self, storage):
        result = storage.get("records", "nonexistent")
        assert result is None

    def test_update_existing(self, storage):
        storage.put("records", "q1", {"question": "test", "runs": {}})
        storage.update("records", "q1", {"runs": {"run1": {"score": 0.9}}})
        result = storage.get("records", "q1")
        assert result["question"] == "test"
        assert result["runs"]["run1"]["score"] == 0.9

    def test_update_nonexistent(self, storage):
        storage.update("records", "q1", {"question": "new"})
        result = storage.get("records", "q1")
        assert result == {"question": "new"}

    def test_update_deep_merge(self, storage):
        storage.put("records", "q1", {
            "question": "test",
            "runs": {"run1": {"gen": "a", "exec": "b"}}
        })
        storage.update("records", "q1", {
            "runs": {"run1": {"eval": "c"}, "run2": {"gen": "d"}}
        })
        result = storage.get("records", "q1")
        assert result["runs"]["run1"] == {"gen": "a", "exec": "b", "eval": "c"}
        assert result["runs"]["run2"] == {"gen": "d"}

    def test_delete(self, storage):
        storage.put("records", "q1", {"data": 1})
        storage.delete("records", "q1")
        assert storage.get("records", "q1") is None

    def test_exists(self, storage):
        assert storage.exists("records", "q1") is False
        storage.put("records", "q1", {"data": 1})
        assert storage.exists("records", "q1") is True

    def test_count(self, storage):
        assert storage.count("records") == 0
        storage.put("records", "q1", {"data": 1})
        storage.put("records", "q2", {"data": 2})
        assert storage.count("records") == 2

    def test_keys(self, storage):
        storage.put("records", "q1", {"data": 1})
        storage.put("records", "q2", {"data": 2})
        keys = storage.keys("records")
        assert set(keys) == {"q1", "q2"}

    def test_iter_all(self, storage):
        storage.put("records", "q1", {"data": 1})
        storage.put("records", "q2", {"data": 2})
        items = list(storage.iter_all("records"))
        assert len(items) == 2
        keys = {k for k, v in items}
        assert keys == {"q1", "q2"}

    def test_init_from_json(self, storage):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([
                {"id": "a", "value": 1},
                {"id": "b", "value": 2},
            ], f)
            f.flush()

            storage.init_from_json("records", f.name, key_field="id")

        assert storage.get("records", "a") == {"id": "a", "value": 1}
        assert storage.get("records", "b") == {"id": "b", "value": 2}

    def test_export_json(self, storage):
        storage.put("records", "q1", {"data": 1})
        storage.put("records", "q2", {"data": 2})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            storage.export_json("records", f.name)

            with open(f.name, "r") as rf:
                exported = json.load(rf)

        assert len(exported) == 2

    def test_multiple_tables(self, storage):
        storage.put("table1", "k1", {"a": 1})
        storage.put("table2", "k1", {"b": 2})
        assert storage.get("table1", "k1") == {"a": 1}
        assert storage.get("table2", "k1") == {"b": 2}

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with StorageService(str(db_path)) as storage:
                storage.put("records", "q1", {"data": 1})
                assert storage.get("records", "q1") == {"data": 1}


