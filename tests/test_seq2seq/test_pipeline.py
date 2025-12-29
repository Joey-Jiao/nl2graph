import pytest
from unittest.mock import Mock

from nl2graph.seq2seq.pipeline import Seq2SeqPipeline
from nl2graph.seq2seq.generation import Generation
from nl2graph.eval import Record, RunResult, GenerationResult, ExecutionResult, Scoring, Execution


class TestSeq2SeqPipeline:

    @pytest.fixture
    def mock_generation(self):
        generation = Mock(spec=Generation)
        generation.generate_batch.return_value = [
            "MATCH (n) RETURN n",
            "MATCH (m) RETURN m",
        ]
        return generation

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

    def test_init(self, mock_generation, mock_execution):
        pipeline = Seq2SeqPipeline(
            generation=mock_generation,
            execution=mock_execution,
            lang="cypher",
            model="bart-base",
        )
        assert pipeline.run_id == "cypher--bart-base"
        assert pipeline.ir_mode is None

    def test_init_with_ir_mode(self, mock_generation, mock_execution):
        pipeline = Seq2SeqPipeline(
            generation=mock_generation,
            execution=mock_execution,
            lang="cypher",
            model="bart-base",
            ir_mode="graphq",
        )
        assert pipeline.ir_mode == "graphq"

    def test_generate(self, mock_generation, mock_execution):
        pipeline = Seq2SeqPipeline(
            generation=mock_generation,
            execution=mock_execution,
            lang="cypher",
            model="bart-base",
        )

        records = [
            Record(question="Q1", answer=["A1"]),
            Record(question="Q2", answer=["A2"]),
        ]

        result = pipeline.generate(records)

        assert len(result) == 2
        mock_generation.generate_batch.assert_called_once_with(["Q1", "Q2"])
        assert result[0].runs["cypher--bart-base"].gen.query == "MATCH (n) RETURN n"
        assert result[1].runs["cypher--bart-base"].gen.query == "MATCH (m) RETURN m"

    def test_generate_with_ir_translator(self, mock_generation, mock_execution):
        mock_translator = Mock()
        mock_translator.to_cypher.return_value = "MATCH (translated) RETURN translated"

        mock_generation.generate_batch.return_value = ["GraphQ_IR_representation"]

        pipeline = Seq2SeqPipeline(
            generation=mock_generation,
            execution=mock_execution,
            lang="cypher",
            model="bart-base",
            ir_mode="graphq",
            ir_translator=mock_translator,
        )

        records = [Record(question="Q1", answer=["A1"])]
        result = pipeline.generate(records)

        assert result[0].runs["cypher--bart-base"].gen.ir == "GraphQ_IR_representation"
        assert result[0].runs["cypher--bart-base"].gen.query == "MATCH (translated) RETURN translated"

    def test_execute(self, mock_generation, mock_execution):
        pipeline = Seq2SeqPipeline(
            generation=mock_generation,
            execution=mock_execution,
            lang="cypher",
            model="bart-base",
        )

        records = [Record(question="Q1", answer=["A1"])]
        records[0].runs["cypher--bart-base"] = RunResult(
            method="seq2seq",
            lang="cypher",
            model="bart-base",
            gen=GenerationResult(query="MATCH (n) RETURN n"),
        )

        result = pipeline.execute(records)

        mock_execution.execute.assert_called_once()
        assert result[0].runs["cypher--bart-base"].exec.success is True

    def test_evaluate(self, mock_generation, mock_execution, mock_scoring):
        pipeline = Seq2SeqPipeline(
            generation=mock_generation,
            execution=mock_execution,
            scoring=mock_scoring,
            lang="cypher",
            model="bart-base",
        )

        records = [Record(question="Q1", answer=["result"])]
        records[0].runs["cypher--bart-base"] = RunResult(
            method="seq2seq",
            lang="cypher",
            model="bart-base",
            gen=GenerationResult(query="MATCH..."),
            exec=ExecutionResult(result=["result"], success=True),
        )

        result = pipeline.evaluate(records)

        assert result[0].runs["cypher--bart-base"].eval is not None
        assert result[0].runs["cypher--bart-base"].eval.exact_match == 1.0
