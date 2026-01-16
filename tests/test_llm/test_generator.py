import pytest
from pathlib import Path
from unittest.mock import Mock

from nl2graph.generation.llm.generation import Generation
from nl2graph.base.llm.service import LLMService
from nl2graph.base.llm.entity import LLMMessage, LLMUsage, LLMResponse
from nl2graph.base.configs import ConfigService
from nl2graph.data import GenerationOutput


@pytest.fixture
def config_service():
    config_file = Path(__file__).parent.parent.parent / "configs" / "configs.yaml"
    env_file = Path(__file__).parent.parent.parent / ".env"
    return ConfigService(config_dir=[config_file], env_path=str(env_file))


@pytest.fixture
def llm_service(config_service):
    return LLMService(config=config_service)


class TestGeneration:

    def test_init(self):
        mock_service = Mock(spec=LLMService)
        gen = Generation(
            llm_service=mock_service,
            provider="openai",
            model="gpt-4o-mini",
        )
        assert gen.provider == "openai"
        assert gen.model == "gpt-4o-mini"

    def test_generate_mock(self):
        mock_service = Mock(spec=LLMService)
        mock_client = Mock()
        mock_response = LLMResponse(
            message=LLMMessage.assistant("MATCH (n) RETURN n"),
            usage=LLMUsage(input_tokens=10, output_tokens=5, cached_tokens=0),
            duration=0.5,
        )
        mock_client.chat.return_value = mock_response
        mock_service.get_client.return_value = mock_client

        gen = Generation(
            llm_service=mock_service,
            provider="openai",
            model="gpt-4o-mini",
        )

        result = gen.generate("Generate a cypher query to find all nodes")

        assert isinstance(result, GenerationOutput)
        assert result.content == "MATCH (n) RETURN n"
        assert result.stats["duration"] == 0.5
        assert result.stats["input_tokens"] == 10
        assert result.stats["output_tokens"] == 5
        assert result.stats["cached_tokens"] == 0
        mock_service.get_client.assert_called_once_with("openai", "gpt-4o-mini")
        mock_client.chat.assert_called_once()


class TestGenerationIntegration:

    def test_generate_with_openai(self, config_service, llm_service):
        api_key = config_service.get_env("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        gen = Generation(
            llm_service=llm_service,
            provider="openai",
            model="gpt-4o-mini",
        )

        prompt = "Say 'hello' and nothing else."
        result = gen.generate(prompt)

        assert isinstance(result, GenerationOutput)
        assert result.content is not None
        assert len(result.content) > 0
        assert result.stats is not None
        assert result.stats["duration"] > 0
        print(f"\n[OpenAI]: {result.content}")

    def test_generate_with_deepseek(self, config_service, llm_service):
        api_key = config_service.get_env("DEEPSEEK_API_KEY")
        if not api_key:
            pytest.skip("DEEPSEEK_API_KEY not set")

        gen = Generation(
            llm_service=llm_service,
            provider="deepseek",
            model="deepseek-chat",
        )

        prompt = "Say 'hello' and nothing else."
        result = gen.generate(prompt)

        assert isinstance(result, GenerationOutput)
        assert result.content is not None
        assert len(result.content) > 0
        assert result.stats is not None
        assert result.stats["duration"] > 0
        print(f"\n[DeepSeek]: {result.content}")


class TestExtractQuery:

    @pytest.fixture
    def generation(self):
        mock_service = Mock(spec=LLMService)
        mock_client = Mock()
        mock_service.get_client.return_value = mock_client
        return Generation(
            llm_service=mock_service,
            provider="openai",
            model="gpt-4o-mini",
            extract_query=True,
        )

    def test_extract_from_code_block(self, generation):
        raw = "Here's the query:\n```cypher\nMATCH (n) RETURN n\n```"
        assert generation._extract_query(raw) == "MATCH (n) RETURN n"

    def test_extract_from_code_block_no_lang(self, generation):
        raw = "```\nSELECT * FROM t\n```"
        assert generation._extract_query(raw) == "SELECT * FROM t"

    def test_extract_from_sparql_block(self, generation):
        raw = "```sparql\nSELECT ?x WHERE { ?x a :Person }\n```"
        assert generation._extract_query(raw) == "SELECT ?x WHERE { ?x a :Person }"

    def test_extract_from_inline_code(self, generation):
        raw = "Use this query: `MATCH (n) RETURN n`"
        assert generation._extract_query(raw) == "MATCH (n) RETURN n"

    def test_extract_plain_text(self, generation):
        raw = "MATCH (n) RETURN n"
        assert generation._extract_query(raw) == "MATCH (n) RETURN n"


class TestBuildPrompt:

    def test_build_prompt_with_template(self, tmp_path):
        from nl2graph.base.templates.service import TemplateService
        from nl2graph.base.configs import ConfigService
        from nl2graph.data.schema.cypher import CypherSchema, NodeSchema

        prompts_dir = tmp_path / "templates" / "prompts"
        prompts_dir.mkdir(parents=True)
        template_file = prompts_dir / "cypher.jinja2"
        template_file.write_text("Schema: {{ schema }}\nQuestion: {{ question }}")

        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"templates:\n  prompts: {prompts_dir}\n")

        config = ConfigService(config_dir=[config_file], env_path=".env.nonexistent")
        template_service = TemplateService(config)

        schema = CypherSchema(
            name="TestDB",
            nodes=[NodeSchema(label="Person", properties=[])],
            edges=[],
        )

        mock_llm_service = Mock(spec=LLMService)
        mock_client = Mock()
        mock_llm_service.get_client.return_value = mock_client

        gen = Generation(
            llm_service=mock_llm_service,
            provider="openai",
            model="gpt-4o-mini",
            template_service=template_service,
            template_name="cypher",
        )

        prompt = gen._build_prompt("Find all persons", schema)

        assert "Find all persons" in prompt
        assert "Person" in prompt
