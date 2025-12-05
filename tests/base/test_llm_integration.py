import os
import pytest
from dotenv import load_dotenv

from nl2graph.base.llm.entity import LLMMessage, ClientConfig
from nl2graph.base.llm.clients.deepseek import DeepSeekClient
from nl2graph.base.llm.clients.openai import OpenAIClient

load_dotenv()


@pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not set"
)
class TestDeepSeekClientIntegration:
    def test_chat_real_call(self):
        config = ClientConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            endpoint="https://api.deepseek.com",
            timeout=30,
        )
        client = DeepSeekClient(config)

        messages = [
            LLMMessage.user("Say 'hello' and nothing else.")
        ]
        result = client.chat(messages)

        assert result.role == "assistant"
        assert len(result.content) > 0
        print(f"\n[DeepSeek Response]: {result.content}")


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
class TestOpenAIClientIntegration:
    def test_chat_real_call(self):
        config = ClientConfig(
            provider="openai",
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=30,
        )
        client = OpenAIClient(config)

        messages = [
            LLMMessage.user("Say 'hello' and nothing else.")
        ]
        result = client.chat(messages)

        assert result.role == "assistant"
        assert len(result.content) > 0
        print(f"\n[OpenAI Response]: {result.content}")
