from pathlib import Path

from pydantic import BaseModel


class Template(BaseModel):
    name: str
    category: str
    path: Path
