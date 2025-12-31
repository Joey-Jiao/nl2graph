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


class TestStorageServiceComposite:

    @pytest.fixture
    def storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            svc = StorageService(str(db_path))
            svc._ensure_table_composite("results", ["record_id", "method", "lang", "model"])
            yield svc
            svc.close()

    def test_put_and_get_composite(self, storage):
        keys = {"record_id": "q1", "method": "llm", "lang": "cypher", "model": "gpt-4o"}
        data = {**keys, "gen": {"query": "MATCH..."}, "exec": None, "eval": None}
        storage.put_composite("results", keys, data)
        result = storage.get_composite("results", keys)
        assert result["record_id"] == "q1"
        assert result["gen"]["query"] == "MATCH..."

    def test_get_composite_nonexistent(self, storage):
        keys = {"record_id": "q1", "method": "llm", "lang": "cypher", "model": "gpt-4o"}
        result = storage.get_composite("results", keys)
        assert result is None

    def test_put_composite_override(self, storage):
        keys = {"record_id": "q1", "method": "llm", "lang": "cypher", "model": "gpt-4o"}
        data1 = {**keys, "gen": {"query": "v1"}}
        data2 = {**keys, "gen": {"query": "v2"}}
        storage.put_composite("results", keys, data1)
        storage.put_composite("results", keys, data2)
        result = storage.get_composite("results", keys)
        assert result["gen"]["query"] == "v2"

    def test_update_composite_existing(self, storage):
        keys = {"record_id": "q1", "method": "llm", "lang": "cypher", "model": "gpt-4o"}
        data = {**keys, "gen": {"query": "MATCH..."}, "exec": None}
        storage.put_composite("results", keys, data)
        storage.update_composite("results", keys, {"exec": {"result": [1, 2, 3]}})
        result = storage.get_composite("results", keys)
        assert result["gen"]["query"] == "MATCH..."
        assert result["exec"]["result"] == [1, 2, 3]

    def test_update_composite_nonexistent(self, storage):
        keys = {"record_id": "q1", "method": "llm", "lang": "cypher", "model": "gpt-4o"}
        storage.update_composite("results", keys, {"gen": {"query": "new"}})
        result = storage.get_composite("results", keys)
        assert result["gen"]["query"] == "new"
        assert result["record_id"] == "q1"

    def test_delete_composite(self, storage):
        keys = {"record_id": "q1", "method": "llm", "lang": "cypher", "model": "gpt-4o"}
        storage.put_composite("results", keys, {**keys, "gen": {}})
        storage.delete_composite("results", keys)
        assert storage.get_composite("results", keys) is None

    def test_exists_composite(self, storage):
        keys = {"record_id": "q1", "method": "llm", "lang": "cypher", "model": "gpt-4o"}
        assert storage.exists_composite("results", keys) is False
        storage.put_composite("results", keys, {**keys})
        assert storage.exists_composite("results", keys) is True

    def test_iter_all_composite(self, storage):
        keys1 = {"record_id": "q1", "method": "llm", "lang": "cypher", "model": "gpt-4o"}
        keys2 = {"record_id": "q1", "method": "seq2seq", "lang": "cypher", "model": "bart"}
        storage.put_composite("results", keys1, {**keys1, "value": 1})
        storage.put_composite("results", keys2, {**keys2, "value": 2})
        items = list(storage.iter_all_composite("results"))
        assert len(items) == 2

    def test_iter_by_field(self, storage):
        keys1 = {"record_id": "q1", "method": "llm", "lang": "cypher", "model": "gpt-4o"}
        keys2 = {"record_id": "q2", "method": "llm", "lang": "cypher", "model": "gpt-4o"}
        keys3 = {"record_id": "q1", "method": "seq2seq", "lang": "cypher", "model": "bart"}
        storage.put_composite("results", keys1, {**keys1})
        storage.put_composite("results", keys2, {**keys2})
        storage.put_composite("results", keys3, {**keys3})
        items = list(storage.iter_by_field("results", "record_id", "q1"))
        assert len(items) == 2

    def test_count_composite(self, storage):
        assert storage.count_composite("results") == 0
        keys1 = {"record_id": "q1", "method": "llm", "lang": "cypher", "model": "gpt-4o"}
        keys2 = {"record_id": "q2", "method": "llm", "lang": "cypher", "model": "gpt-4o"}
        storage.put_composite("results", keys1, {**keys1})
        storage.put_composite("results", keys2, {**keys2})
        assert storage.count_composite("results") == 2

    def test_multiple_keys_same_record(self, storage):
        keys_llm = {"record_id": "q1", "method": "llm", "lang": "cypher", "model": "gpt-4o"}
        keys_seq = {"record_id": "q1", "method": "seq2seq", "lang": "cypher", "model": "bart"}
        storage.put_composite("results", keys_llm, {**keys_llm, "score": 0.9})
        storage.put_composite("results", keys_seq, {**keys_seq, "score": 0.8})
        llm_result = storage.get_composite("results", keys_llm)
        seq_result = storage.get_composite("results", keys_seq)
        assert llm_result["score"] == 0.9
        assert seq_result["score"] == 0.8
