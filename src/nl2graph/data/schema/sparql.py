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
    domain: List[str] = []
    range: List[str] = []
    is_object_property: bool = False


class SparqlSchema(BaseSchema, BaseModel):
    name: str
    return_hint: Optional[str] = None
    entity_rule: Optional[str] = None
    prefixes: Dict[str, str] = {}
    classes: List[ClassSchema] = []
    properties: List[PropertyDef] = []

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_prompt_string(self) -> str:
        lines = [f"RDF Graph: {self.name}", ""]

        if self.return_hint:
            lines.append(f"Return: {self.return_hint}")
            lines.append("")

        if self.entity_rule:
            lines.append(f"Entity Rule: {self.entity_rule}")
            lines.append("")

        if self.prefixes:
            lines.append("Prefixes:")
            for prefix, uri in sorted(self.prefixes.items()):
                lines.append(f"  {prefix}: <{uri}>")
            lines.append("")

        if self.classes:
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
    def from_dict(cls, data: dict) -> "SparqlSchema":
        classes = [ClassSchema(**c) for c in data.get("classes", [])]

        properties = []
        for p in data.get("object_properties", []):
            properties.append(PropertyDef(
                uri=p["uri"],
                label=p.get("label"),
                domain=p.get("domain", []),
                range=p.get("range", []),
                is_object_property=True,
            ))
        for p in data.get("datatype_properties", []):
            properties.append(PropertyDef(
                uri=p["uri"],
                label=p.get("label"),
                domain=p.get("domain", []),
                range=p.get("range", []),
                is_object_property=False,
            ))
        for p in data.get("properties", []):
            properties.append(PropertyDef(**p))

        return cls(
            name=data["name"],
            return_hint=data.get("return_hint"),
            entity_rule=data.get("entity_rule"),
            prefixes=data.get("prefixes", {}),
            classes=classes,
            properties=properties,
        )
