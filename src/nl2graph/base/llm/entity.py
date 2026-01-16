from typing import Dict, List, Optional, Union, Literal

from pydantic import BaseModel, Field

RoleType = Literal["system", "user", "assistant", "tool", "developer"]


class ClientConfig(BaseModel):
    provider: str = Field(..., description="provider name e.g, openai, deepseek")
    model: str = Field(..., description="model name, e.g., o4-mini-deep-research")
    api_key: Optional[str] = Field(None, description="API key")
    endpoint: Optional[str] = Field(None, description="base URL / host")
    timeout: int = Field(30, description="timeout in seconds")


class LLMUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


class LLMResponse(BaseModel):
    message: "LLMMessage"
    usage: LLMUsage
    duration: float


class LLMMessage(BaseModel):
    role: RoleType
    content: str

    @staticmethod
    def system(text: str) -> "LLMMessage":
        return LLMMessage(role="system", content=text)

    @staticmethod
    def user(text: str) -> "LLMMessage":
        return LLMMessage(role="user", content=text)

    @staticmethod
    def assistant(text: str) -> "LLMMessage":
        return LLMMessage(role="assistant", content=text)

    @staticmethod
    def tool(text: str) -> "LLMMessage":
        return LLMMessage(role="tool", content=text)

    @staticmethod
    def developer(text: str) -> "LLMMessage":
        return LLMMessage(role="developer", content=text)
