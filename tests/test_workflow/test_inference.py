import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from nl2graph.pipeline.generate import GeneratePipeline
from nl2graph.pipeline.execute import ExecutePipeline
from nl2graph.pipeline.evaluate import EvaluatePipeline
from nl2graph.data import Record, GenerationResult, ExecutionResult, GenerationOutput
from nl2graph.data.repository import ResultRepository
from nl2graph.evaluation import Scoring
from nl2graph.execution import Execution


class TestGeneratePipeline:

    @pytest.fixture
    def mock_generator(self):
        generator = Mock()
        generator.generate.side_effect = [
            GenerationOutput(content="MATCH (n) RETURN n"),
            GenerationOutput(content="MATCH (m) RETURN m"),
        ]
        return generator

    @pytest.fixture
    def dst(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "dst.db"
            repo = ResultRepository(str(db_path))
            yield repo
            repo.close()

    def test_init(self, mock_generator, dst):
        pipeline = GeneratePipeline(
            generator=mock_generator,
            dst=dst,
            method="llm",
            lang="cypher",
            model="gpt-4o",
        )
        assert pipeline.method == "llm"
        assert pipeline.lang == "cypher"
        assert pipeline.model == "gpt-4o"

    def test_run_generates_queries(self, mock_generator, dst):
        pipeline = GeneratePipeline(
            generator=mock_generator,
            dst=dst,
            method="seq2seq",
            lang="cypher",
            model="bart-base",
        )

        records = [
            Record(id="q001", question="Q1", answer=["A1"]),
            Record(id="q002", question="Q2", answer=["A2"]),
        ]

        result = pipeline.run(records)

        assert len(result) == 2
        assert mock_generator.generate.call_count == 2

        res1 = dst.get("q001", "seq2seq", "cypher", "bart-base")
        res2 = dst.get("q002", "seq2seq", "cypher", "bart-base")
        assert res1.gen.query == "MATCH (n) RETURN n"
        assert res2.gen.query == "MATCH (m) RETURN m"

    def test_run_with_schema(self, dst):
        mock_gen = Mock()
        mock_gen.generate.return_value = GenerationOutput(content="MATCH (n:Person) RETURN n")

        mock_schema = Mock()
        mock_schema.to_prompt_string.return_value = "schema"

        pipeline = GeneratePipeline(
            generator=mock_gen,
            dst=dst,
            method="llm",
            lang="cypher",
            model="gpt-4o",
        )

        records = [Record(id="q001", question="Q1", answer=["A1"])]
        result = pipeline.run(records, schema=mock_schema)

        mock_gen.generate.assert_called_once_with("Q1", mock_schema)

    def test_run_skips_existing(self, mock_generator, dst):
        dst.save_generation(
            "q001", "seq2seq", "cypher", "bart-base",
            GenerationResult(query="EXISTING")
        )

        pipeline = GeneratePipeline(
            generator=mock_generator,
            dst=dst,
            method="seq2seq",
            lang="cypher",
            model="bart-base",
            if_exists="skip",
        )

        records = [
            Record(id="q001", question="Q1", answer=["A1"]),
            Record(id="q002", question="Q2", answer=["A2"]),
        ]

        pipeline.run(records)

        assert mock_generator.generate.call_count == 1
        res1 = dst.get("q001", "seq2seq", "cypher", "bart-base")
        assert res1.gen.query == "EXISTING"


class TestExecutePipeline:

    @pytest.fixture
    def mock_execution(self):
        execution = Mock(spec=Execution)
        execution.execute.return_value = ExecutionResult(
            result=["result"],
            success=True,
        )
        return execution

    @pytest.fixture
    def dst(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "dst.db"
            repo = ResultRepository(str(db_path))
            yield repo
            repo.close()

    def test_init(self, mock_execution, dst):
        pipeline = ExecutePipeline(
            execution=mock_execution,
            dst=dst,
            method="llm",
            lang="cypher",
            model="gpt-4o",
        )
        assert pipeline.method == "llm"
        assert pipeline.lang == "cypher"
        assert pipeline.model == "gpt-4o"

    def test_run_executes_queries(self, mock_execution, dst):
        dst.save_generation(
            "q001", "seq2seq", "cypher", "bart-base",
            GenerationResult(query="MATCH (n) RETURN n")
        )

        pipeline = ExecutePipeline(
            execution=mock_execution,
            dst=dst,
            method="seq2seq",
            lang="cypher",
            model="bart-base",
        )

        records = [Record(id="q001", question="Q1", answer=["A1"])]
        result = pipeline.run(records)

        mock_execution.execute.assert_called_once()
        res = dst.get("q001", "seq2seq", "cypher", "bart-base")
        assert res.exec.success is True

    def test_run_skips_without_generation(self, mock_execution, dst):
        pipeline = ExecutePipeline(
            execution=mock_execution,
            dst=dst,
            method="seq2seq",
            lang="cypher",
            model="bart-base",
        )

        records = [Record(id="q001", question="Q1", answer=["A1"])]
        pipeline.run(records)

        mock_execution.execute.assert_not_called()


class TestEvaluatePipeline:

    @pytest.fixture
    def dst(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "dst.db"
            repo = ResultRepository(str(db_path))
            yield repo
            repo.close()

    def test_init(self, dst):
        pipeline = EvaluatePipeline(
            dst=dst,
            method="llm",
            lang="cypher",
            model="gpt-4o",
        )
        assert pipeline.method == "llm"
        assert pipeline.lang == "cypher"
        assert pipeline.model == "gpt-4o"

    def test_run_evaluates_results(self, dst):
        dst.save_generation(
            "q001", "seq2seq", "cypher", "bart-base",
            GenerationResult(query="MATCH...")
        )
        dst.save_execution(
            "q001", "seq2seq", "cypher", "bart-base",
            ExecutionResult(result=["result"], success=True)
        )

        pipeline = EvaluatePipeline(
            dst=dst,
            method="seq2seq",
            lang="cypher",
            model="bart-base",
            scoring=Scoring(),
        )

        records = [Record(id="q001", question="Q1", answer=["result"])]
        result = pipeline.run(records)

        res = dst.get("q001", "seq2seq", "cypher", "bart-base")
        assert res.eval is not None
        assert res.eval.exact_match == 1.0

    def test_run_skips_without_execution(self, dst):
        dst.save_generation(
            "q001", "seq2seq", "cypher", "bart-base",
            GenerationResult(query="MATCH...")
        )

        pipeline = EvaluatePipeline(
            dst=dst,
            method="seq2seq",
            lang="cypher",
            model="bart-base",
            scoring=Scoring(),
        )

        records = [Record(id="q001", question="Q1", answer=["result"])]
        pipeline.run(records)

        res = dst.get("q001", "seq2seq", "cypher", "bart-base")
        assert res.eval is None


class TestTranslateIR:

    def test_translate_cypher(self):
        mock_translator = Mock()
        mock_translator.to_cypher.return_value = "MATCH (n) RETURN n"

        from nl2graph.generation.seq2seq.generation import Generation
        gen = Generation.__new__(Generation)
        gen.translator = mock_translator
        gen.lang = "cypher"

        result = gen._translate_ir("IR_representation")

        assert result == "MATCH (n) RETURN n"
        mock_translator.to_cypher.assert_called_once_with("IR_representation")

    def test_translate_sparql(self):
        mock_translator = Mock()
        mock_translator.to_sparql.return_value = "SELECT ?x"

        from nl2graph.generation.seq2seq.generation import Generation
        gen = Generation.__new__(Generation)
        gen.translator = mock_translator
        gen.lang = "sparql"

        result = gen._translate_ir("IR_representation")
        assert result == "SELECT ?x"

    def test_translate_exception_fallback(self):
        mock_translator = Mock()
        mock_translator.to_cypher.side_effect = Exception("parse error")

        from nl2graph.generation.seq2seq.generation import Generation
        gen = Generation.__new__(Generation)
        gen.translator = mock_translator
        gen.lang = "cypher"

        result = gen._translate_ir("invalid_ir")
        assert result == "invalid_ir"
