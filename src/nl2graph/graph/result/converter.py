from typing import Any
from datetime import date, datetime, time


def convert_neo4j_value(value: Any) -> Any:
    if value is None:
        return None

    type_name = type(value).__name__

    if type_name == "Date":
        return date(value.year, value.month, value.day)
    elif type_name == "DateTime":
        return datetime(
            value.year, value.month, value.day,
            value.hour, value.minute, value.second, value.nanosecond // 1000
        )
    elif type_name == "Time":
        return time(value.hour, value.minute, value.second, value.nanosecond // 1000)
    elif type_name == "Duration":
        return {
            "months": value.months,
            "days": value.days,
            "seconds": value.seconds,
            "nanoseconds": value.nanoseconds,
        }
    elif type_name == "Node":
        return {
            "id": value.element_id,
            "labels": list(value.labels),
            "properties": dict(value),
        }
    elif type_name == "Relationship":
        return {
            "id": value.element_id,
            "type": value.type,
            "start": value.start_node.element_id,
            "end": value.end_node.element_id,
            "properties": dict(value),
        }
    elif type_name == "Path":
        return {
            "nodes": [convert_neo4j_value(n) for n in value.nodes],
            "relationships": [convert_neo4j_value(r) for r in value.relationships],
        }
    elif isinstance(value, list):
        return [convert_neo4j_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: convert_neo4j_value(v) for k, v in value.items()}

    return value


def convert_rdf_value(value: Any) -> Any:
    if value is None:
        return None

    type_name = type(value).__name__

    if type_name == "URIRef":
        return str(value)
    elif type_name == "Literal":
        py_value = value.toPython()
        if isinstance(py_value, type(value)):
            return str(value)
        return py_value
    elif type_name == "BNode":
        return f"_:{value}"
    elif isinstance(value, list):
        return [convert_rdf_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: convert_rdf_value(v) for k, v in value.items()}

    return value


def convert_gremlin_value(value: Any) -> Any:
    if value is None:
        return None

    type_name = type(value).__name__

    if type_name == "Vertex":
        return {
            "id": value.id,
            "label": value.label,
        }
    elif type_name == "Edge":
        return {
            "id": value.id,
            "label": value.label,
            "inV": value.inV.id,
            "outV": value.outV.id,
        }
    elif type_name == "Path":
        return {
            "labels": [list(labels) for labels in value.labels],
            "objects": [convert_gremlin_value(o) for o in value.objects],
        }
    elif isinstance(value, list):
        return [convert_gremlin_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: convert_gremlin_value(v) for k, v in value.items()}

    return value
