from typing import Optional, List
from pathlib import Path

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str = Field(..., description="Model name")
    path: Path = Field(..., description="Path to model weights")
    tokenizer_path: Optional[Path] = Field(None, description="Path to tokenizer (defaults to model path)")
    max_length: int = Field(512, description="Maximum sequence length")
    special_tokens: List[str] = Field(default_factory=list, description="Special tokens to add")
