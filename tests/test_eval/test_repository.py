import pytest
import tempfile
import json
from pathlib import Path

from nl2graph.eval.repository import SourceRepository, ResultRepository
from nl2graph.eval.entity import (
    Record,
    Result,
    GenerationResult,
    ExecutionResult,
    EvaluationResult,
)


class TestSourceRepository:

    @pytest.fixture
    def src(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "src.db"
            repo = SourceRepository(str(db_path))
            yield repo
            repo.close()

    @pytest.fixture
    def sample_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([
                {"id": "q001", "question": "What is 1+1?", "answer": [2], "hop": 1},
                {"id": "q002", "question": "Who directed Inception?", "answer": ["Christopher Nolan"], "hop": 1},
                {"id": "q003", "question": "What is the capital of France?", "answer": ["Paris"], "hop": 2},
            ], f, ensure_ascii=False)
            return f.name

    def test_init_from_json(self, src, sample_json):
        count = src.init_from_json(sample_json)
        assert count == 3

    def test_get(self, src, sample_json):
        src.init_from_json(sample_json)
        record = src.get("q001")
        assert record is not None
        assert record.id == "q001"
        assert record.question == "What is 1+1?"
        assert record.answer == [2]

    def test_get_nonexistent(self, src):
        result = src.get("nonexistent")
        assert result is None

    def test_exists(self, src, sample_json):
        src.init_from_json(sample_json)
        assert src.exists("q001") is True
        assert src.exists("nonexistent") is False

    def test_count(self, src, sample_json):
        assert src.count() == 0
        src.init_from_json(sample_json)
        assert src.count() == 3

    def test_iter_all(self, src, sample_json):
        src.init_from_json(sample_json)
        records = list(src.iter_all())
        assert len(records) == 3
        ids = {r.id for r in records}
        assert ids == {"q001", "q002", "q003"}

    def test_iter_by_hop(self, src, sample_json):
        src.init_from_json(sample_json)
        hop1 = list(src.iter_by_hop(1))
        hop2 = list(src.iter_by_hop(2))
        assert len(hop1) == 2
        assert len(hop2) == 1

    def test_context_manager(self, sample_json):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "src.db"
            with SourceRepository(str(db_path)) as src:
                src.init_from_json(sample_json)
                assert src.count() == 3


class TestResultRepository:

    @pytest.fixture
    def dst(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "dst.db"
            repo = ResultRepository(str(db_path))
            yield repo
            repo.close()

    def test_save_and_get(self, dst):
        gen = GenerationResult(query="MATCH (n) RETURN n")
        dst.save_generation("q001", "llm", "cypher", "gpt-4o", gen)
        result = dst.get("q001", "llm", "cypher", "gpt-4o")
        assert result is not None
        assert result.record_id == "q001"
        assert result.gen.query == "MATCH (n) RETURN n"
        assert result.exec is None
        assert result.eval is None

    def test_get_nonexistent(self, dst):
        result = dst.get("q001", "llm", "cypher", "gpt-4o")
        assert result is None

    def test_exists(self, dst):
        assert dst.exists("q001", "llm", "cypher", "gpt-4o") is False
        gen = GenerationResult(query="MATCH...")
        dst.save_generation("q001", "llm", "cypher", "gpt-4o", gen)
        assert dst.exists("q001", "llm", "cypher", "gpt-4o") is True

    def test_save_execution(self, dst):
        gen = GenerationResult(query="MATCH...")
        dst.save_generation("q001", "llm", "cypher", "gpt-4o", gen)
        exec = ExecutionResult(result=[1, 2, 3], success=True)
        dst.save_execution("q001", "llm", "cypher", "gpt-4o", exec)
        result = dst.get("q001", "llm", "cypher", "gpt-4o")
        assert result.gen.query == "MATCH..."
        assert result.exec.result == [1, 2, 3]
        assert result.exec.success is True
        assert result.eval is None

    def test_save_evaluation(self, dst):
        gen = GenerationResult(query="MATCH...")
        dst.save_generation("q001", "llm", "cypher", "gpt-4o", gen)
        exec = ExecutionResult(result=[1], success=True)
        dst.save_execution("q001", "llm", "cypher", "gpt-4o", exec)
        eval = EvaluationResult(exact_match=1.0, f1=1.0)
        dst.save_evaluation("q001", "llm", "cypher", "gpt-4o", eval)
        result = dst.get("q001", "llm", "cypher", "gpt-4o")
        assert result.eval.exact_match == 1.0
        assert result.eval.f1 == 1.0

    def test_override_generation_clears_downstream(self, dst):
        gen1 = GenerationResult(query="v1")
        dst.save_generation("q001", "llm", "cypher", "gpt-4o", gen1)
        exec = ExecutionResult(result=[1], success=True)
        dst.save_execution("q001", "llm", "cypher", "gpt-4o", exec)
        eval = EvaluationResult(exact_match=1.0)
        dst.save_evaluation("q001", "llm", "cypher", "gpt-4o", eval)
        gen2 = GenerationResult(query="v2")
        dst.save_generation("q001", "llm", "cypher", "gpt-4o", gen2)
        result = dst.get("q001", "llm", "cypher", "gpt-4o")
        assert result.gen.query == "v2"
        assert result.exec is None
        assert result.eval is None

    def test_override_execution_clears_eval(self, dst):
        gen = GenerationResult(query="MATCH...")
        dst.save_generation("q001", "llm", "cypher", "gpt-4o", gen)
        exec1 = ExecutionResult(result=[1], success=True)
        dst.save_execution("q001", "llm", "cypher", "gpt-4o", exec1)
        eval = EvaluationResult(exact_match=1.0)
        dst.save_evaluation("q001", "llm", "cypher", "gpt-4o", eval)
        exec2 = ExecutionResult(result=[2], success=True)
        dst.save_execution("q001", "llm", "cypher", "gpt-4o", exec2)
        result = dst.get("q001", "llm", "cypher", "gpt-4o")
        assert result.exec.result == [2]
        assert result.eval is None

    def test_count(self, dst):
        assert dst.count() == 0
        dst.save_generation("q001", "llm", "cypher", "gpt-4o", GenerationResult())
        dst.save_generation("q002", "llm", "cypher", "gpt-4o", GenerationResult())
        assert dst.count() == 2

    def test_iter_all(self, dst):
        dst.save_generation("q001", "llm", "cypher", "gpt-4o", GenerationResult())
        dst.save_generation("q002", "llm", "cypher", "gpt-4o", GenerationResult())
        results = list(dst.iter_all())
        assert len(results) == 2

    def test_iter_by_record(self, dst):
        dst.save_generation("q001", "llm", "cypher", "gpt-4o", GenerationResult())
        dst.save_generation("q001", "seq2seq", "cypher", "bart", GenerationResult())
        dst.save_generation("q002", "llm", "cypher", "gpt-4o", GenerationResult())
        results = list(dst.iter_by_record("q001"))
        assert len(results) == 2

    def test_iter_by_config(self, dst):
        dst.save_generation("q001", "llm", "cypher", "gpt-4o", GenerationResult())
        dst.save_generation("q002", "llm", "cypher", "gpt-4o", GenerationResult())
        dst.save_generation("q001", "seq2seq", "cypher", "bart", GenerationResult())
        results = list(dst.iter_by_config("llm", "cypher", "gpt-4o"))
        assert len(results) == 2

    def test_iter_pending(self, dst):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = Path(tmpdir) / "src.db"
            json_path = Path(tmpdir) / "records.json"
            with open(json_path, "w") as f:
                json.dump([
                    {"id": "q001", "question": "Q1", "answer": [1]},
                    {"id": "q002", "question": "Q2", "answer": [2]},
                    {"id": "q003", "question": "Q3", "answer": [3]},
                ], f)
            src = SourceRepository(str(src_path))
            src.init_from_json(str(json_path))
            dst.save_generation("q001", "llm", "cypher", "gpt-4o", GenerationResult())
            dst.save_execution("q001", "llm", "cypher", "gpt-4o", ExecutionResult(success=True))
            dst.save_evaluation("q001", "llm", "cypher", "gpt-4o", EvaluationResult(exact_match=1.0))
            dst.save_generation("q002", "llm", "cypher", "gpt-4o", GenerationResult())
            pending = list(dst.iter_pending(src, "llm", "cypher", "gpt-4o"))
            pending_ids = {r.id for r in pending}
            assert pending_ids == {"q002", "q003"}
            src.close()

    def test_export_json(self, dst):
        dst.save_generation("q001", "llm", "cypher", "gpt-4o", GenerationResult(query="Q1"))
        dst.save_generation("q002", "llm", "cypher", "gpt-4o", GenerationResult(query="Q2"))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            dst.export_json(f.name)
            with open(f.name, "r") as rf:
                exported = json.load(rf)
        assert len(exported) == 2

    def test_multiple_configs_same_record(self, dst):
        dst.save_generation("q001", "llm", "cypher", "gpt-4o", GenerationResult(query="llm-cypher"))
        dst.save_generation("q001", "seq2seq", "cypher", "bart", GenerationResult(query="seq2seq-cypher"))
        dst.save_generation("q001", "llm", "sparql", "gpt-4o", GenerationResult(query="llm-sparql"))
        assert dst.count() == 3
        r1 = dst.get("q001", "llm", "cypher", "gpt-4o")
        r2 = dst.get("q001", "seq2seq", "cypher", "bart")
        r3 = dst.get("q001", "llm", "sparql", "gpt-4o")
        assert r1.gen.query == "llm-cypher"
        assert r2.gen.query == "seq2seq-cypher"
        assert r3.gen.query == "llm-sparql"

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "dst.db"
            with ResultRepository(str(db_path)) as dst:
                dst.save_generation("q001", "llm", "cypher", "gpt-4o", GenerationResult())
                assert dst.count() == 1
