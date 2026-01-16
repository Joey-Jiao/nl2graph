import pytest
from pathlib import Path

from nl2graph.base.llm.entity import ClientConfig, LLMMessage, LLMUsage, LLMResponse
from nl2graph.base.llm.clients.openai import OpenAIClient
from nl2graph.base.llm.clients.deepseek import DeepSeekClient
from nl2graph.base.configs import ConfigService


class TestClientConfig:

    def test_create_minimal(self):
        config = ClientConfig(provider="openai", model="gpt-4o")
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.api_key is None
        assert config.endpoint is None
        assert config.timeout == 30

    def test_create_full(self):
        config = ClientConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_key="sk-xxx",
            endpoint="https://api.deepseek.com",
            timeout=60,
        )
        assert config.api_key == "sk-xxx"
        assert config.endpoint == "https://api.deepseek.com"
        assert config.timeout == 60


class TestLLMMessage:

    def test_system_message(self):
        msg = LLMMessage.system("You are a helpful assistant")
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant"

    def test_user_message(self):
        msg = LLMMessage.user("Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_assistant_message(self):
        msg = LLMMessage.assistant("Hi there")
        assert msg.role == "assistant"
        assert msg.content == "Hi there"


class TestLLMUsage:

    def test_create_default(self):
        usage = LLMUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cached_tokens == 0

    def test_create_with_values(self):
        usage = LLMUsage(input_tokens=100, output_tokens=50, cached_tokens=20)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cached_tokens == 20


class TestLLMResponse:

    def test_create(self):
        msg = LLMMessage.assistant("Hello")
        usage = LLMUsage(input_tokens=10, output_tokens=5)
        response = LLMResponse(message=msg, usage=usage, duration=0.5)
        assert response.message.content == "Hello"
        assert response.usage.input_tokens == 10
        assert response.duration == 0.5


@pytest.fixture
def config_service():
    config_file = Path(__file__).parent.parent.parent / "configs" / "configs.yaml"
    env_file = Path(__file__).parent.parent.parent / ".env"
    return ConfigService(config_dir=[config_file], env_path=str(env_file))


class TestOpenAIClientReal:

    def test_chat_simple(self, config_service):
        api_key = config_service.get_env("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        config = ClientConfig(
            provider="openai",
            model="gpt-4o-mini",
            api_key=api_key,
        )
        client = OpenAIClient(config)

        messages = [LLMMessage.user("Say 'hello' and nothing else")]
        result = client.chat(messages)

        assert isinstance(result, LLMResponse)
        assert result.message.role == "assistant"
        assert result.message.content is not None
        assert len(result.message.content) > 0
        assert result.duration > 0
        assert result.usage.input_tokens >= 0
        print(f"\n[OpenAI Response]: {result.message.content}")

    def test_chat_cypher_generation(self, config_service):
        api_key = config_service.get_env("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        config = ClientConfig(
            provider="openai",
            model="gpt-4o-mini",
            api_key=api_key,
        )
        client = OpenAIClient(config)

        messages = [
            LLMMessage.system("You are a Cypher query expert. Generate only the query, no explanation."),
            LLMMessage.user("Generate a Cypher query to find all Person nodes"),
        ]
        result = client.chat(messages)

        assert result.message.role == "assistant"
        assert result.message.content is not None
        print(f"\n[OpenAI Cypher]: {result.message.content}")


class TestDeepSeekClientReal:

    def test_chat_simple(self, config_service):
        api_key = config_service.get_env("DEEPSEEK_API_KEY")
        if not api_key:
            pytest.skip("DEEPSEEK_API_KEY not set")

        config = ClientConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_key=api_key,
            endpoint="https://api.deepseek.com",
        )
        client = DeepSeekClient(config)

        messages = [LLMMessage.user("Say 'hello' and nothing else")]
        result = client.chat(messages)

        assert isinstance(result, LLMResponse)
        assert result.message.role == "assistant"
        assert result.message.content is not None
        assert len(result.message.content) > 0
        assert result.duration > 0
        assert result.usage.input_tokens >= 0
        print(f"\n[DeepSeek Response]: {result.message.content}")

    def test_chat_cypher_generation(self, config_service):
        api_key = config_service.get_env("DEEPSEEK_API_KEY")
        if not api_key:
            pytest.skip("DEEPSEEK_API_KEY not set")

        config = ClientConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_key=api_key,
            endpoint="https://api.deepseek.com",
        )
        client = DeepSeekClient(config)

        messages = [
            LLMMessage.system("You are a Cypher query expert. Generate only the query, no explanation."),
            LLMMessage.user("Generate a Cypher query to find all movies directed by Christopher Nolan"),
        ]
        result = client.chat(messages)

        assert result.message.role == "assistant"
        assert result.message.content is not None
        print(f"\n[DeepSeek Cypher]: {result.message.content}")
