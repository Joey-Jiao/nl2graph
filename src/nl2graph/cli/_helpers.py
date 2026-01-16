from typing import Optional, List

from ..data import Record
from ..data.repository import SourceRepository


def load_records(src: SourceRepository, hop: Optional[int], split: Optional[str]) -> List[Record]:
    filters = {}
    if hop is not None:
        filters["hop"] = hop
    if split is not None:
        filters["split"] = split

    if filters:
        return list(src.iter_by_filter(**filters))
    return list(src.iter_all())


def detect_provider(model: str) -> Optional[str]:
    if model.startswith("gpt"):
        return "openai"
    elif model.startswith("deepseek"):
        return "deepseek"
    return None
