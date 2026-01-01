from .llm.generation import Generation as LLMGeneration
from .seq2seq.generation import Generation as Seq2SeqGeneration

__all__ = [
    "LLMGeneration",
    "Seq2SeqGeneration",
]
