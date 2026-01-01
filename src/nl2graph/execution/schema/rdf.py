from typing import List, Dict, Optional

from pydantic import BaseModel

from .base import BaseSchema


class ClassSchema(BaseModel):
    uri: str
    label: Optional[str] = None
    parent: Optional[str] = None


class PropertyDef(BaseModel):
    uri: str
    label: Optional[str] = None
    domain: Optional[str] = None
    range: Optional[str] = None
    is_object_property: bool = False


class RDFSchema(BaseSchema, BaseModel):
    name: str
    prefixes: Dict[str, str] = {}
    classes: List[ClassSchema] = []
    properties: List[PropertyDef] = []

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_prompt_string(self) -> str:
        lines = [f"RDF Graph: {self.name}", ""]

        if self.prefixes:
            lines.append("Prefixes:")
            for prefix, uri in sorted(self.prefixes.items()):
                lines.append(f"  {prefix}: <{uri}>")
            lines.append("")

        lines.append("Classes:")
        for cls in sorted(self.classes, key=lambda x: x.uri):
            label = f" ({cls.label})" if cls.label else ""
            parent = f" rdfs:subClassOf {cls.parent}" if cls.parent else ""
            lines.append(f"  {cls.uri}{label}{parent}")

        lines.append("")
        lines.append("Properties:")
        for prop in sorted(self.properties, key=lambda x: x.uri):
            label = f" ({prop.label})" if prop.label else ""
            domain = f" domain={prop.domain}" if prop.domain else ""
            range_str = f" range={prop.range}" if prop.range else ""
            prop_type = " [ObjectProperty]" if prop.is_object_property else " [DatatypeProperty]"
            lines.append(f"  {prop.uri}{label}{domain}{range_str}{prop_type}")

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict) -> "RDFSchema":
        classes = [ClassSchema(**c) for c in data.get("classes", [])]
        properties = [PropertyDef(**p) for p in data.get("properties", [])]
        return cls(
            name=data["name"],
            prefixes=data.get("prefixes", {}),
            classes=classes,
            properties=properties,
        )
