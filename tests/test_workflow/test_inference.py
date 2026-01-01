import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from nl2graph.pipeline.inference import InferencePipeline
from nl2graph.data import Record, Result, GenerationResult, ExecutionResult
from nl2graph.data.repository import ResultRepository
from nl2graph.evaluation import Scoring
from nl2graph.execution import Execution


class TestInferencePipeline:

    @pytest.fixture
    def mock_generator(self):
        generator = Mock()
        generator.generate.return_value = "MATCH (n) RETURN n"
        generator.generate_batch.return_value = [
            "MATCH (n) RETURN n",
            "MATCH (m) RETURN m",
        ]
        return generator

    @pytest.fixture
    def mock_execution(self):
        execution = Mock(spec=Execution)
        execution.execute.return_value = ExecutionResult(
            result=["result"],
            success=True,
        )
        return execution

    @pytest.fixture
    def mock_scoring(self):
        return Scoring()

    @pytest.fixture
    def dst(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "dst.db"
            repo = ResultRepository(str(db_path))
            yield repo
            repo.close()

    def test_init(self, mock_generator, dst):
        pipeline = InferencePipeline(
            generator=mock_generator,
            dst=dst,
            method="llm",
            lang="cypher",
            model="gpt-4o",
        )
        assert pipeline.method == "llm"
        assert pipeline.lang == "cypher"
        assert pipeline.model == "gpt-4o"

    def test_generate_without_template(self, mock_generator, dst):
        pipeline = InferencePipeline(
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

        result = pipeline.generate(records)

        assert len(result) == 2
        mock_generator.generate_batch.assert_called_once_with(["Q1", "Q2"])

        res1 = dst.get("q001", "seq2seq", "cypher", "bart-base")
        res2 = dst.get("q002", "seq2seq", "cypher", "bart-base")
        assert res1.gen.query == "MATCH (n) RETURN n"
        assert res2.gen.query == "MATCH (m) RETURN m"

    def test_generate_with_template(self, mock_generator, dst):
        mock_template_service = Mock()
        mock_template_service.render.side_effect = ["prompt1", "prompt2"]

        mock_schema = Mock()
        mock_schema.to_prompt_string.return_value = "schema"

        pipeline = InferencePipeline(
            generator=mock_generator,
            dst=dst,
            template_service=mock_template_service,
            template_name="cypher.jinja2",
            method="llm",
            lang="cypher",
            model="gpt-4o",
        )

        records = [
            Record(id="q001", question="Q1", answer=["A1"]),
            Record(id="q002", question="Q2", answer=["A2"]),
        ]

        result = pipeline.generate(records, schema=mock_schema)

        mock_generator.generate_batch.assert_called_once_with(["prompt1", "prompt2"])

    def test_generate_with_extract_query(self, mock_generator, dst):
        mock_generator.generate_batch.return_value = [
            "```cypher\nMATCH (n) RETURN n\n```",
        ]

        pipeline = InferencePipeline(
            generator=mock_generator,
            dst=dst,
            extract_query=True,
            method="llm",
            lang="cypher",
            model="gpt-4o",
        )

        records = [Record(id="q001", question="Q1", answer=["A1"])]
        result = pipeline.generate(records)

        res = dst.get("q001", "llm", "cypher", "gpt-4o")
        assert res.gen.query_raw == "```cypher\nMATCH (n) RETURN n\n```"
        assert res.gen.query == "MATCH (n) RETURN n"

    def test_generate_with_ir_mode(self, mock_generator, dst):
        mock_translator = Mock()
        mock_translator.to_cypher.return_value = "MATCH (translated) RETURN translated"

        mock_generator.generate_batch.return_value = ["IR_representation"]

        pipeline = InferencePipeline(
            generator=mock_generator,
            dst=dst,
            translator=mock_translator,
            method="seq2seq",
            lang="cypher",
            model="bart-base",
            ir_mode=True,
        )

        records = [Record(id="q001", question="Q1", answer=["A1"])]
        result = pipeline.generate(records)

        res = dst.get("q001", "seq2seq", "cypher", "bart-base")
        assert res.gen.ir == "IR_representation"
        assert res.gen.query == "MATCH (translated) RETURN translated"

    def test_execute(self, mock_generator, mock_execution, dst):
        pipeline = InferencePipeline(
            generator=mock_generator,
            dst=dst,
            execution=mock_execution,
            method="seq2seq",
            lang="cypher",
            model="bart-base",
        )

        records = [Record(id="q001", question="Q1", answer=["A1"])]
        dst.save_generation(
            "q001", "seq2seq", "cypher", "bart-base",
            GenerationResult(query="MATCH (n) RETURN n")
        )

        result = pipeline.execute(records)

        mock_execution.execute.assert_called_once()
        res = dst.get("q001", "seq2seq", "cypher", "bart-base")
        assert res.exec.success is True

    def test_evaluate(self, mock_generator, mock_execution, mock_scoring, dst):
        pipeline = InferencePipeline(
            generator=mock_generator,
            dst=dst,
            execution=mock_execution,
            scoring=mock_scoring,
            method="seq2seq",
            lang="cypher",
            model="bart-base",
        )

        records = [Record(id="q001", question="Q1", answer=["result"])]
        dst.save_generation(
            "q001", "seq2seq", "cypher", "bart-base",
            GenerationResult(query="MATCH...")
        )
        dst.save_execution(
            "q001", "seq2seq", "cypher", "bart-base",
            ExecutionResult(result=["result"], success=True)
        )

        result = pipeline.evaluate(records)

        res = dst.get("q001", "seq2seq", "cypher", "bart-base")
        assert res.eval is not None
        assert res.eval.exact_match == 1.0


class TestExtractQuery:

    @pytest.fixture
    def pipeline(self):
        mock_generator = Mock()
        mock_dst = Mock()
        return InferencePipeline(generator=mock_generator, dst=mock_dst)

    def test_extract_from_code_block(self, pipeline):
        raw = "Here's the query:\n```cypher\nMATCH (n) RETURN n\n```"
        assert pipeline._extract_query(raw) == "MATCH (n) RETURN n"

    def test_extract_from_code_block_no_lang(self, pipeline):
        raw = "```\nSELECT * FROM t\n```"
        assert pipeline._extract_query(raw) == "SELECT * FROM t"

    def test_extract_from_sparql_block(self, pipeline):
        raw = "```sparql\nSELECT ?x WHERE { ?x a :Person }\n```"
        assert pipeline._extract_query(raw) == "SELECT ?x WHERE { ?x a :Person }"

    def test_extract_from_inline_code(self, pipeline):
        raw = "Use this query: `MATCH (n) RETURN n`"
        assert pipeline._extract_query(raw) == "MATCH (n) RETURN n"

    def test_extract_plain_text(self, pipeline):
        raw = "MATCH (n) RETURN n"
        assert pipeline._extract_query(raw) == "MATCH (n) RETURN n"


class TestTranslateIR:

    def test_translate_cypher(self):
        mock_generator = Mock()
        mock_dst = Mock()
        mock_translator = Mock()
        mock_translator.to_cypher.return_value = "MATCH (n) RETURN n"

        pipeline = InferencePipeline(
            generator=mock_generator,
            dst=mock_dst,
            translator=mock_translator,
            lang="cypher",
        )

        result = pipeline._translate_ir("IR_representation")

        assert result == "MATCH (n) RETURN n"
        mock_translator.to_cypher.assert_called_once_with("IR_representation")

    def test_translate_sparql(self):
        mock_generator = Mock()
        mock_dst = Mock()
        mock_translator = Mock()
        mock_translator.to_sparql.return_value = "SELECT ?x"

        pipeline = InferencePipeline(
            generator=mock_generator,
            dst=mock_dst,
            translator=mock_translator,
            lang="sparql",
        )

        result = pipeline._translate_ir("IR_representation")
        assert result == "SELECT ?x"

    def test_translate_exception_fallback(self):
        mock_generator = Mock()
        mock_dst = Mock()
        mock_translator = Mock()
        mock_translator.to_cypher.side_effect = Exception("parse error")

        pipeline = InferencePipeline(
            generator=mock_generator,
            dst=mock_dst,
            translator=mock_translator,
            lang="cypher",
        )

        result = pipeline._translate_ir("invalid_ir")
        assert result == "invalid_ir"
