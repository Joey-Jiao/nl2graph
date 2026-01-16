import json
from pathlib import Path
from typing import Optional, Union

from .base import BaseSchema
from .cypher import CypherSchema
from .sparql import SparqlSchema
from .gremlin import GremlinSchema


def load_schema(path: Union[str, Path], lang: str) -> Optional[BaseSchema]:
    path = Path(path)
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    if lang == "sparql":
        return SparqlSchema.from_dict(data)
    elif lang == "gremlin":
        return GremlinSchema.from_dict(data)
    else:
        return CypherSchema.from_dict(data)
