import pytest
from pathlib import Path

from nl2graph.llm.generation import Generation
from nl2graph.base.llm.service import LLMService
from nl2graph.base.templates.service import TemplateService
from nl2graph.base.configs import ConfigService
from nl2graph.eval.entity import Record
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
    query_template.write_text("""You are a Cypher query expert. Generate a Cypher query for the following question.

Schema:
{{ schema }}

Question: {{ question }}

Generate only the Cypher query, wrapped in ```cypher``` code block.""")

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"templates:\n  prompts: {prompts_dir}\n")

    config = ConfigService(config_dir=[config_file], env_path=".env.nonexistent")
    return TemplateService(config)


@pytest.fixture
def movie_schema():
    return PropertyGraphSchema(
        name="MovieDB",
        nodes=[
            NodeSchema(
                label="Person",
                properties=[
                    PropertySchema(name="name", data_type="string"),
                    PropertySchema(name="born", data_type="int"),
                ],
            ),
            NodeSchema(
                label="Movie",
                properties=[
                    PropertySchema(name="title", data_type="string"),
                    PropertySchema(name="released", data_type="int"),
                ],
            ),
        ],
        edges=[
            EdgeSchema(label="ACTED_IN", source_label="Person", target_label="Movie"),
            EdgeSchema(label="DIRECTED", source_label="Person", target_label="Movie"),
        ],
    )


class TestGenerationReal:

    def test_generate_with_openai(self, config_service, llm_service, template_service, movie_schema):
        api_key = config_service.get_env("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        generator = Generation(
            llm_service=llm_service,
            template_service=template_service,
            provider="openai",
            model="gpt-4o-mini",
            lang="cypher",
            prompt_template="cypher",
        )

        record = Record(
            question="Who directed the movie The Matrix?",
            answer=["Lana Wachowski", "Lilly Wachowski"],
        )

        result = generator.generate(record, movie_schema)

        assert result.query_raw is not None
        assert result.query is not None
        print(f"\n[OpenAI Raw]: {result.query_raw}")
        print(f"[OpenAI Query]: {result.query}")

    def test_generate_with_deepseek(self, config_service, llm_service, template_service, movie_schema):
        api_key = config_service.get_env("DEEPSEEK_API_KEY")
        if not api_key:
            pytest.skip("DEEPSEEK_API_KEY not set")

        generator = Generation(
            llm_service=llm_service,
            template_service=template_service,
            provider="deepseek",
            model="deepseek-chat",
            lang="cypher",
            prompt_template="cypher",
        )

        record = Record(
            question="Find all actors who acted in movies released after 2000",
            answer=[],
        )

        result = generator.generate(record, movie_schema)

        assert result.query_raw is not None
        assert result.query is not None
        print(f"\n[DeepSeek Raw]: {result.query_raw}")
        print(f"[DeepSeek Query]: {result.query}")


class TestExtractQuery:

    def test_extract_query_with_code_block(self):
        from nl2graph.llm.generation import Generation
        from unittest.mock import Mock

        generator = Generation(
            llm_service=Mock(),
            template_service=Mock(),
            provider="test",
            model="test",
            lang="cypher",
            prompt_template="test",
        )

        assert generator._extract_query("```cypher\nMATCH (n) RETURN n\n```") == "MATCH (n) RETURN n"
        assert generator._extract_query("```\nSELECT * FROM t\n```") == "SELECT * FROM t"
        assert generator._extract_query("```sparql\nSELECT ?x\n```") == "SELECT ?x"

    def test_extract_query_with_inline_code(self):
        from nl2graph.llm.generation import Generation
        from unittest.mock import Mock

        generator = Generation(
            llm_service=Mock(),
            template_service=Mock(),
            provider="test",
            model="test",
            lang="cypher",
            prompt_template="test",
        )

        assert generator._extract_query("`MATCH (n) RETURN n`") == "MATCH (n) RETURN n"

    def test_extract_query_plain_text(self):
        from nl2graph.llm.generation import Generation
        from unittest.mock import Mock

        generator = Generation(
            llm_service=Mock(),
            template_service=Mock(),
            provider="test",
            model="test",
            lang="cypher",
            prompt_template="test",
        )

        assert generator._extract_query("MATCH (n) RETURN n") == "MATCH (n) RETURN n"
