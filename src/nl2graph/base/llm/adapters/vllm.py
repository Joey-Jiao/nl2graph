from typing import List, Any, Optional

from .base import BaseAdapter
from ..entity import LLMMessage


class VLLMAdapter(BaseAdapter):
    def ir2provider(self, messages: List[LLMMessage]):
        prompt = "".join(f"{message.role.upper()}: {message.content}\n" for message in messages)
        return {"prompt": prompt}

    def provider2ir(self, resp) -> LLMMessage:
        return LLMMessage.assistant(resp.get("text", ""))
