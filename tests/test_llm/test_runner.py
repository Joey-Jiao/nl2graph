import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from nl2graph.pipeline import InferencePipeline
from nl2graph.generation.llm.generation import Generation
from nl2graph.execution import Execution
from nl2graph.base.llm.service import LLMService
from nl2graph.base.templates.service import TemplateService
from nl2graph.base.configs import ConfigService
from nl2graph.data import Record, Result, GenerationResult, ExecutionResult
from nl2graph.evaluation import Scoring
from nl2graph.data.repository import ResultRepository
from nl2graph.execution.schema.property_graph import PropertyGraphSchema, NodeSchema, EdgeSchema, PropertySchema


@pytest.fixture
def config_service():
    config_file = Path(__file__).parent.parent.parent / "configs" / "configs.yaml"
    env_file = Path(__file__).parent.parent.parent / ".env"
    return ConfigService(config_dir=[config_file], env_path=str(env_file))


@pytest.fixture
def llm_service(config_service):
    return LLMService(config=config_service)


@pytest.fixture
def template_service(tmp_path):
    prompts_dir = tmp_path / "templates" / "prompts"
    prompts_dir.mkdir(parents=True)

    query_template = prompts_dir / "cypher.jinja2"
    query_template.write_text("""You are a Cypher query expert.

Schema:
{{ schema }}

Question: {{ question }}

Generate only the Cypher query in a ```cypher``` code block.""")

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"templates:\n  prompts: {prompts_dir}\n")

    config = ConfigService(config_dir=[config_file], env_path=".env.nonexistent")
    return TemplateService(config)


@pytest.fixture
def movie_schema():
    return PropertyGraphSchema(
        name="MovieDB",
        nodes=[
            NodeSchema(label="Person", properties=[PropertySchema(name="name", data_type="string")]),
            NodeSchema(label="Movie", properties=[PropertySchema(name="title", data_type="string")]),
        ],
        edges=[
            EdgeSchema(label="ACTED_IN", source_label="Person", target_label="Movie"),
            EdgeSchema(label="DIRECTED", source_label="Person", target_label="Movie"),
        ],
    )


@pytest.fixture
def dst():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "dst.db"
        repo = ResultRepository(str(db_path))
        yield repo
        repo.close()


class TestLLMPipelineReal:

    def test_full_pipeline_openai(self, config_service, llm_service, template_service, movie_schema, dst):
        api_key = config_service.get_env("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        generation = Generation(
            llm_service=llm_service,
            provider="openai",
            model="gpt-4o-mini",
        )

        mock_connector = Mock()
        mock_result = Mock()
        mock_result.rows = [{"name": "Christopher Nolan"}]
        mock_connector.execute.return_value = mock_result

        execution = Execution(mock_connector)
        scoring = Scoring()

        pipeline = InferencePipeline(
            generator=generation,
            dst=dst,
            template_service=template_service,
            template_name="cypher",
            extract_query=True,
            execution=execution,
            scoring=scoring,
            method="llm",
            lang="cypher",
            model="gpt-4o-mini",
            workers=1,
        )

        records = [
            Record(id="q001", question="Who directed Inception?", answer=["Christopher Nolan"]),
        ]

        print("\n=== OpenAI Full Pipeline Test ===")

        records = pipeline.generate(records, movie_schema)
        res = dst.get("q001", "llm", "cypher", "gpt-4o-mini")
        print(f"[Generated Query]: {res.gen.query}")
        assert res.gen.query is not None

        records = pipeline.execute(records)
        res = dst.get("q001", "llm", "cypher", "gpt-4o-mini")
        print(f"[Execution Success]: {res.exec.success}")
        print(f"[Execution Result]: {res.exec.result}")
        assert res.exec.success is True

        records = pipeline.evaluate(records)
        res = dst.get("q001", "llm", "cypher", "gpt-4o-mini")
        print(f"[Exact Match]: {res.eval.exact_match}")
        print(f"[F1]: {res.eval.f1}")
        assert res.eval is not None

    def test_full_pipeline_deepseek(self, config_service, llm_service, template_service, movie_schema, dst):
        api_key = config_service.get_env("DEEPSEEK_API_KEY")
        if not api_key:
            pytest.skip("DEEPSEEK_API_KEY not set")

        generation = Generation(
            llm_service=llm_service,
            provider="deepseek",
            model="deepseek-chat",
        )

        mock_connector = Mock()
        mock_result = Mock()
        mock_result.rows = [{"name": "Tom Hanks"}, {"name": "Leonardo DiCaprio"}]
        mock_connector.execute.return_value = mock_result

        execution = Execution(mock_connector)
        scoring = Scoring()

        pipeline = InferencePipeline(
            generator=generation,
            dst=dst,
            template_service=template_service,
            template_name="cypher",
            extract_query=True,
            execution=execution,
            scoring=scoring,
            method="llm",
            lang="cypher",
            model="deepseek-chat",
            workers=1,
        )

        records = [
            Record(id="q001", question="Find all actors in the database", answer=["Tom Hanks", "Leonardo DiCaprio"]),
        ]

        print("\n=== DeepSeek Full Pipeline Test ===")

        records = pipeline.generate(records, movie_schema)
        res = dst.get("q001", "llm", "cypher", "deepseek-chat")
        print(f"[Generated Query]: {res.gen.query}")
        assert res.gen.query is not None

        records = pipeline.execute(records)
        res = dst.get("q001", "llm", "cypher", "deepseek-chat")
        print(f"[Execution Success]: {res.exec.success}")
        print(f"[Execution Result]: {res.exec.result}")

        records = pipeline.evaluate(records)
        res = dst.get("q001", "llm", "cypher", "deepseek-chat")
        print(f"[Exact Match]: {res.eval.exact_match}")
        print(f"[F1]: {res.eval.f1}")


class TestExecution:

    @pytest.fixture
    def mock_connector(self):
        connector = Mock()
        result = Mock()
        result.rows = [{"name": "Alice"}, {"name": "Bob"}]
        connector.execute.return_value = result
        return connector

    def test_execute_success(self, mock_connector):
        execution = Execution(mock_connector)

        result = Result(
            question_id="q001",
            method="llm",
            lang="cypher",
            model="gpt-4o",
            gen=GenerationResult(query="MATCH (n) RETURN n.name"),
        )

        exec_result = execution.execute(result)

        assert exec_result.success is True
        assert exec_result.result == ["Alice", "Bob"]

    def test_execute_no_query(self, mock_connector):
        execution = Execution(mock_connector)

        result = Result(
            question_id="q001",
            method="llm",
            lang="cypher",
            model="gpt-4o",
        )

        exec_result = execution.execute(result)

        assert exec_result.success is False
        assert "no query" in exec_result.error

    def test_execute_error(self, mock_connector):
        mock_connector.execute.side_effect = Exception("Connection failed")
        execution = Execution(mock_connector)

        result = Result(
            question_id="q001",
            method="llm",
            lang="cypher",
            model="gpt-4o",
            gen=GenerationResult(query="MATCH (n) RETURN n"),
        )

        exec_result = execution.execute(result)

        assert exec_result.success is False
        assert "Connection failed" in exec_result.error

    def test_extract_answer_single_column(self, mock_connector):
        execution = Execution(mock_connector)
        rows = [{"name": "Alice"}, {"name": "Bob"}]
        result = execution._extract_answer(rows)
        assert result == ["Alice", "Bob"]

    def test_extract_answer_multiple_columns(self, mock_connector):
        execution = Execution(mock_connector)
        rows = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = execution._extract_answer(rows)
        assert result == [["Alice", 30], ["Bob", 25]]

    def test_extract_answer_empty(self, mock_connector):
        execution = Execution(mock_connector)
        result = execution._extract_answer([])
        assert result == []


class TestInferencePipelineUnit:

    @pytest.fixture
    def dst(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "dst.db"
            repo = ResultRepository(str(db_path))
            yield repo
            repo.close()

    def test_init(self, dst):
        pipeline = InferencePipeline(
            generator=Mock(),
            dst=dst,
            execution=Mock(),
            method="llm",
            lang="cypher",
            model="gpt-4o",
        )
        assert pipeline.lang == "cypher"
        assert pipeline.model == "gpt-4o"
