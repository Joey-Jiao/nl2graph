import pytest
from unittest.mock import Mock, patch

from nl2graph.base.llm.entity import LLMMessage, ClientConfig
from nl2graph.base.llm.adapters.openai import OpenAIAdapter
from nl2graph.base.llm.adapters.deepseek import DeepSeekAdapter
from nl2graph.base.llm.clients.openai import OpenAIClient
from nl2graph.base.llm.clients.deepseek import DeepSeekClient


class TestLLMMessage:
    def test_system_message(self):
        msg = LLMMessage.system("You are a helpful assistant.")
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant."

    def test_user_message(self):
        msg = LLMMessage.user("Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_assistant_message(self):
        msg = LLMMessage.assistant("Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_tool_message(self):
        msg = LLMMessage.tool("result: 42")
        assert msg.role == "tool"
        assert msg.content == "result: 42"

    def test_developer_message(self):
        msg = LLMMessage.developer("debug info")
        assert msg.role == "developer"
        assert msg.content == "debug info"


class TestClientConfig:
    def test_client_config_required_fields(self):
        config = ClientConfig(provider="openai", model="gpt-4")
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.timeout == 30

    def test_client_config_optional_fields(self):
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


class TestOpenAIAdapter:
    def test_to_chat_messages(self):
        messages = [
            LLMMessage.system("You are helpful."),
            LLMMessage.user("Hello"),
        ]
        result = OpenAIAdapter.to_chat_messages(messages)
        assert result == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

    def test_extract_chat_message(self):
        mock_resp = Mock()
        mock_resp.output_text = "I am an assistant."
        result = OpenAIAdapter.extract_chat_message(mock_resp)
        assert result.role == "assistant"
        assert result.content == "I am an assistant."


class TestDeepSeekAdapter:
    def test_to_chat_messages(self):
        messages = [
            LLMMessage.user("What is 2+2?"),
        ]
        result = DeepSeekAdapter.to_chat_messages(messages)
        assert result == [{"role": "user", "content": "What is 2+2?"}]

    def test_extract_chat_message(self):
        mock_resp = Mock()
        mock_resp.choices = [Mock()]
        mock_resp.choices[0].message.content = "The answer is 4."
        result = DeepSeekAdapter.extract_chat_message(mock_resp)
        assert result.role == "assistant"
        assert result.content == "The answer is 4."


class TestOpenAIClient:
    @patch("nl2graph.base.llm.clients.openai.OpenAI")
    def test_chat(self, mock_openai_cls):
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        mock_resp = Mock()
        mock_resp.output_text = "Hello from GPT!"
        mock_client.responses.create.return_value = mock_resp

        config = ClientConfig(provider="openai", model="gpt-4", api_key="sk-test")
        client = OpenAIClient(config)

        messages = [LLMMessage.user("Hi")]
        result = client.chat(messages)

        mock_client.responses.create.assert_called_once()
        assert result.role == "assistant"
        assert result.content == "Hello from GPT!"


class TestDeepSeekClient:
    @patch("nl2graph.base.llm.clients.deepseek.OpenAI")
    def test_chat(self, mock_openai_cls):
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        mock_resp = Mock()
        mock_resp.choices = [Mock()]
        mock_resp.choices[0].message.content = "Hello from DeepSeek!"
        mock_client.chat.completions.create.return_value = mock_resp

        config = ClientConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_key="sk-test",
            endpoint="https://api.deepseek.com",
        )
        client = DeepSeekClient(config)

        messages = [LLMMessage.user("Hi")]
        result = client.chat(messages)

        mock_client.chat.completions.create.assert_called_once()
        assert result.role == "assistant"
        assert result.content == "Hello from DeepSeek!"
