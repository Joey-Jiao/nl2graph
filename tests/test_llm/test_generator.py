import pytest
from pathlib import Path
from unittest.mock import Mock

from nl2graph.llm.generation import Generation
from nl2graph.base.llm.service import LLMService
from nl2graph.base.configs import ConfigService


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
        mock_response = Mock()
        mock_response.content = "MATCH (n) RETURN n"
        mock_client.chat.return_value = mock_response
        mock_service.get_client.return_value = mock_client

        gen = Generation(
            llm_service=mock_service,
            provider="openai",
            model="gpt-4o-mini",
        )

        result = gen.generate("Generate a cypher query to find all nodes")

        assert result == "MATCH (n) RETURN n"
        mock_service.get_client.assert_called_once_with("openai", "gpt-4o-mini")
        mock_client.chat.assert_called_once()

    def test_generate_batch_mock(self):
        mock_service = Mock(spec=LLMService)
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = "MATCH (n) RETURN n"
        mock_client.chat.return_value = mock_response
        mock_service.get_client.return_value = mock_client

        gen = Generation(
            llm_service=mock_service,
            provider="openai",
            model="gpt-4o-mini",
        )

        results = gen.generate_batch(["prompt1", "prompt2"])

        assert len(results) == 2
        assert results[0] == "MATCH (n) RETURN n"
        assert results[1] == "MATCH (n) RETURN n"


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

        assert result is not None
        assert len(result) > 0
        print(f"\n[OpenAI]: {result}")

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

        assert result is not None
        assert len(result) > 0
        print(f"\n[DeepSeek]: {result}")
