import pytest
from pathlib import Path
from unittest.mock import Mock

from nl2graph.llm.pipeline import LLMPipeline
from nl2graph.llm.generation import Generation
from nl2graph.eval import Execution
from nl2graph.base.llm.service import LLMService
from nl2graph.base.templates.service import TemplateService
from nl2graph.base.configs import ConfigService
from nl2graph.eval import Record, RunResult, GenerationResult, ExecutionResult, Scoring
from nl2graph.graph.schema.property_graph import PropertyGraphSchema, NodeSchema, EdgeSchema, PropertySchema


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


class TestLLMPipelineReal:

    def test_full_pipeline_openai(self, config_service, llm_service, template_service, movie_schema):
        api_key = config_service.get_env("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        generation = Generation(
            llm_service=llm_service,
            template_service=template_service,
            provider="openai",
            model="gpt-4o-mini",
            lang="cypher",
            prompt_template="cypher",
        )

        mock_connector = Mock()
        mock_result = Mock()
        mock_result.rows = [{"name": "Christopher Nolan"}]
        mock_connector.execute.return_value = mock_result

        execution = Execution(mock_connector)
        scoring = Scoring()

        runner = LLMPipeline(
            generation=generation,
            execution=execution,
            scoring=scoring,
            lang="cypher",
            model="gpt-4o-mini",
            workers=1,
        )

        records = [
            Record(question="Who directed Inception?", answer=["Christopher Nolan"]),
        ]

        print("\n=== OpenAI Full Pipeline Test ===")

        records = runner.generate(records, movie_schema)
        gen_result = records[0].runs["cypher--gpt-4o-mini"].gen
        print(f"[Generated Query Raw]: {gen_result.query_raw}")
        print(f"[Generated Query]: {gen_result.query}")
        assert gen_result.query is not None

        records = runner.execute(records)
        exec_result = records[0].runs["cypher--gpt-4o-mini"].exec
        print(f"[Execution Success]: {exec_result.success}")
        print(f"[Execution Result]: {exec_result.result}")
        assert exec_result.success is True

        records = runner.evaluate(records)
        eval_result = records[0].runs["cypher--gpt-4o-mini"].eval
        print(f"[Exact Match]: {eval_result.exact_match}")
        print(f"[F1]: {eval_result.f1}")
        assert eval_result is not None

    def test_full_pipeline_deepseek(self, config_service, llm_service, template_service, movie_schema):
        api_key = config_service.get_env("DEEPSEEK_API_KEY")
        if not api_key:
            pytest.skip("DEEPSEEK_API_KEY not set")

        generation = Generation(
            llm_service=llm_service,
            template_service=template_service,
            provider="deepseek",
            model="deepseek-chat",
            lang="cypher",
            prompt_template="cypher",
        )

        mock_connector = Mock()
        mock_result = Mock()
        mock_result.rows = [{"name": "Tom Hanks"}, {"name": "Leonardo DiCaprio"}]
        mock_connector.execute.return_value = mock_result

        execution = Execution(mock_connector)
        scoring = Scoring()

        runner = LLMPipeline(
            generation=generation,
            execution=execution,
            scoring=scoring,
            lang="cypher",
            model="deepseek-chat",
            workers=1,
        )

        records = [
            Record(question="Find all actors in the database", answer=["Tom Hanks", "Leonardo DiCaprio"]),
        ]

        print("\n=== DeepSeek Full Pipeline Test ===")

        records = runner.generate(records, movie_schema)
        gen_result = records[0].runs["cypher--deepseek-chat"].gen
        print(f"[Generated Query Raw]: {gen_result.query_raw}")
        print(f"[Generated Query]: {gen_result.query}")
        assert gen_result.query is not None

        records = runner.execute(records)
        exec_result = records[0].runs["cypher--deepseek-chat"].exec
        print(f"[Execution Success]: {exec_result.success}")
        print(f"[Execution Result]: {exec_result.result}")

        records = runner.evaluate(records)
        eval_result = records[0].runs["cypher--deepseek-chat"].eval
        print(f"[Exact Match]: {eval_result.exact_match}")
        print(f"[F1]: {eval_result.f1}")


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

        record = Record(question="Q", answer=["A"])
        record.runs["cypher--gpt-4o"] = RunResult(
            method="llm",
            lang="cypher",
            model="gpt-4o",
            gen=GenerationResult(query="MATCH (n) RETURN n.name"),
        )

        result = execution.execute(record, "cypher--gpt-4o")

        assert result.success is True
        assert result.result == ["Alice", "Bob"]

    def test_execute_no_query(self, mock_connector):
        execution = Execution(mock_connector)

        record = Record(question="Q", answer=["A"])
        record.runs["cypher--gpt-4o"] = RunResult(
            method="llm",
            lang="cypher",
            model="gpt-4o",
        )

        result = execution.execute(record, "cypher--gpt-4o")

        assert result.success is False
        assert "no query" in result.error

    def test_execute_error(self, mock_connector):
        mock_connector.execute.side_effect = Exception("Connection failed")
        execution = Execution(mock_connector)

        record = Record(question="Q", answer=["A"])
        record.runs["cypher--gpt-4o"] = RunResult(
            method="llm",
            lang="cypher",
            model="gpt-4o",
            gen=GenerationResult(query="MATCH (n) RETURN n"),
        )

        result = execution.execute(record, "cypher--gpt-4o")

        assert result.success is False
        assert "Connection failed" in result.error

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


class TestLLMPipelineUnit:

    def test_init(self):
        runner = LLMPipeline(
            generation=Mock(),
            execution=Mock(),
            lang="cypher",
            model="gpt-4o",
        )
        assert runner.run_id == "cypher--gpt-4o"
        assert runner.lang == "cypher"
        assert runner.model == "gpt-4o"

    def test_ensure_run_creates_run(self):
        runner = LLMPipeline(
            generation=Mock(),
            execution=Mock(),
            lang="cypher",
            model="gpt-4o",
        )

        record = Record(question="Q", answer=["A"])
        runner._ensure_run(record)

        assert "cypher--gpt-4o" in record.runs
        assert record.runs["cypher--gpt-4o"].method == "llm"
