from typing import Optional, List, TYPE_CHECKING
from pathlib import Path

from ..entity import QueryLanguage, ConnectionConfig
from ..schema.base import BaseSchema
from ..schema.rdf import RDFSchema, ClassSchema, PropertyDef
from ..result.entity import QueryResult
from ..result.converter import convert_rdf_value
from .base import BaseConnector

if TYPE_CHECKING:
    from rdflib import Graph


class RDFLibConnector(BaseConnector):
    query_language = QueryLanguage.SPARQL

    def __init__(self, config: ConnectionConfig, data_path: Optional[str] = None, data_format: str = "turtle"):
        super().__init__(config)
        self._graph: Optional["Graph"] = None
        self.data_path = data_path
        self.data_format = data_format

    def connect(self) -> None:
        from rdflib import Graph

        self._graph = Graph()
        if self.data_path:
            path = Path(self.data_path)
            if path.exists():
                self._graph.parse(str(path), format=self.data_format)

    def close(self) -> None:
        self._graph = None

    def load_data(self, data: str, format: str = "turtle") -> None:
        from rdflib import Graph

        if self._graph is None:
            self._graph = Graph()
        self._graph.parse(data=data, format=format)

    def load_file(self, path: str, format: Optional[str] = None) -> None:
        from rdflib import Graph

        if self._graph is None:
            self._graph = Graph()
        self._graph.parse(path, format=format)

    def execute(self, query: str, timeout: Optional[int] = None) -> QueryResult:
        result = self._graph.query(query)

        if hasattr(result, "bindings"):
            if not result.bindings:
                return QueryResult(columns=[], rows=[], raw=result)

            columns = [str(v) for v in result.vars]
            rows = []
            for binding in result.bindings:
                row = {}
                for var in result.vars:
                    val = binding.get(var)
                    row[str(var)] = convert_rdf_value(val)
                rows.append(row)
            return QueryResult(columns=columns, rows=rows, raw=result)

        elif hasattr(result, "askAnswer"):
            return QueryResult(
                columns=["result"],
                rows=[{"result": result.askAnswer}],
                raw=result,
            )

        return QueryResult(columns=[], rows=[], raw=result)

    def get_schema(self) -> RDFSchema:
        from rdflib import RDF, RDFS, OWL

        prefixes = {}
        for prefix, uri in self._graph.namespaces():
            if prefix:
                prefixes[prefix] = str(uri)

        classes = []
        for cls in self._graph.subjects(RDF.type, RDFS.Class):
            label = self._graph.value(cls, RDFS.label)
            parent = self._graph.value(cls, RDFS.subClassOf)
            classes.append(ClassSchema(
                uri=str(cls),
                label=str(label) if label else None,
                parent=str(parent) if parent else None,
            ))

        for cls in self._graph.subjects(RDF.type, OWL.Class):
            if not any(c.uri == str(cls) for c in classes):
                label = self._graph.value(cls, RDFS.label)
                parent = self._graph.value(cls, RDFS.subClassOf)
                classes.append(ClassSchema(
                    uri=str(cls),
                    label=str(label) if label else None,
                    parent=str(parent) if parent else None,
                ))

        properties = []
        for prop in self._graph.subjects(RDF.type, RDF.Property):
            label = self._graph.value(prop, RDFS.label)
            domain = self._graph.value(prop, RDFS.domain)
            range_val = self._graph.value(prop, RDFS.range)
            properties.append(PropertyDef(
                uri=str(prop),
                label=str(label) if label else None,
                domain=str(domain) if domain else None,
                range=str(range_val) if range_val else None,
                is_object_property=False,
            ))

        for prop in self._graph.subjects(RDF.type, OWL.ObjectProperty):
            if not any(p.uri == str(prop) for p in properties):
                label = self._graph.value(prop, RDFS.label)
                domain = self._graph.value(prop, RDFS.domain)
                range_val = self._graph.value(prop, RDFS.range)
                properties.append(PropertyDef(
                    uri=str(prop),
                    label=str(label) if label else None,
                    domain=str(domain) if domain else None,
                    range=str(range_val) if range_val else None,
                    is_object_property=True,
                ))

        for prop in self._graph.subjects(RDF.type, OWL.DatatypeProperty):
            if not any(p.uri == str(prop) for p in properties):
                label = self._graph.value(prop, RDFS.label)
                domain = self._graph.value(prop, RDFS.domain)
                range_val = self._graph.value(prop, RDFS.range)
                properties.append(PropertyDef(
                    uri=str(prop),
                    label=str(label) if label else None,
                    domain=str(domain) if domain else None,
                    range=str(range_val) if range_val else None,
                    is_object_property=False,
                ))

        return RDFSchema(
            name=self.config.name,
            prefixes=prefixes,
            classes=classes,
            properties=properties,
        )
