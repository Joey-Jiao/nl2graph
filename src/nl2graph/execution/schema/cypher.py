from typing import List

from pydantic import BaseModel

from .base import BaseSchema


class PropertySchema(BaseModel):
    name: str
    data_type: str


class NodeSchema(BaseModel):
    label: str
    properties: List[PropertySchema] = []


class EdgeSchema(BaseModel):
    label: str
    source_label: str
    target_label: str
    properties: List[PropertySchema] = []


class CypherSchema(BaseSchema, BaseModel):
    name: str
    nodes: List[NodeSchema] = []
    edges: List[EdgeSchema] = []

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_prompt_string(self) -> str:
        lines = [f"Graph: {self.name}", "", "Nodes:"]
        for node in sorted(self.nodes, key=lambda x: x.label):
            props = ", ".join(f"{p.name}: {p.data_type}" for p in node.properties)
            lines.append(f"  ({node.label}) [{props}]" if props else f"  ({node.label})")

        lines.append("")
        lines.append("Edges:")
        for edge in sorted(self.edges, key=lambda x: x.label):
            props = ", ".join(f"{p.name}: {p.data_type}" for p in edge.properties)
            edge_str = f"  (:{edge.source_label})-[:{edge.label}]->(:{edge.target_label})"
            if props:
                edge_str += f" [{props}]"
            lines.append(edge_str)

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict) -> "CypherSchema":
        nodes = []
        for n in data.get("nodes", []) or data.get("entities", []):
            props = []
            raw_props = n.get("properties", {})
            if isinstance(raw_props, dict):
                props = [PropertySchema(name=k, data_type=str(v)) for k, v in raw_props.items()]
            elif isinstance(raw_props, list):
                props = [PropertySchema(**p) for p in raw_props]
            nodes.append(NodeSchema(label=n["label"], properties=props))

        edges = []
        for e in data.get("edges", []) or data.get("relations", []):
            props = []
            raw_props = e.get("properties", {})
            if isinstance(raw_props, dict):
                props = [PropertySchema(name=k, data_type=str(v)) for k, v in raw_props.items()]
            elif isinstance(raw_props, list):
                props = [PropertySchema(**p) for p in raw_props]
            edges.append(EdgeSchema(
                label=e["label"],
                source_label=e.get("source_label") or e.get("subj_label"),
                target_label=e.get("target_label") or e.get("obj_label"),
                properties=props,
            ))

        return cls(name=data["name"], nodes=nodes, edges=edges)
