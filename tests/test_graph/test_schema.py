import pytest

from nl2graph.data.schema.cypher import (
    PropertySchema,
    NodeSchema,
    EdgeSchema,
    CypherSchema,
)


class TestPropertySchema:

    def test_create(self):
        prop = PropertySchema(name="age", data_type="int")
        assert prop.name == "age"
        assert prop.data_type == "int"


class TestNodeSchema:

    def test_create_minimal(self):
        node = NodeSchema(label="Person")
        assert node.label == "Person"
        assert node.properties == []

    def test_create_with_properties(self):
        node = NodeSchema(
            label="Person",
            properties=[
                PropertySchema(name="name", data_type="string"),
                PropertySchema(name="age", data_type="int"),
            ],
        )
        assert len(node.properties) == 2


class TestEdgeSchema:

    def test_create_minimal(self):
        edge = EdgeSchema(
            label="KNOWS",
            source_label="Person",
            target_label="Person",
        )
        assert edge.label == "KNOWS"
        assert edge.source_label == "Person"
        assert edge.target_label == "Person"
        assert edge.properties == []

    def test_create_with_properties(self):
        edge = EdgeSchema(
            label="WORKS_AT",
            source_label="Person",
            target_label="Company",
            properties=[PropertySchema(name="since", data_type="date")],
        )
        assert len(edge.properties) == 1


class TestCypherSchema:

    @pytest.fixture
    def sample_schema(self):
        return CypherSchema(
            name="TestGraph",
            nodes=[
                NodeSchema(
                    label="Person",
                    properties=[
                        PropertySchema(name="name", data_type="string"),
                        PropertySchema(name="age", data_type="int"),
                    ],
                ),
                NodeSchema(label="Movie", properties=[]),
            ],
            edges=[
                EdgeSchema(
                    label="ACTED_IN",
                    source_label="Person",
                    target_label="Movie",
                ),
            ],
        )

    def test_to_dict(self, sample_schema):
        result = sample_schema.to_dict()
        assert result["name"] == "TestGraph"
        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1

    def test_to_prompt_string(self, sample_schema):
        result = sample_schema.to_prompt_string()
        assert "Graph: TestGraph" in result
        assert "(Movie)" in result
        assert "(Person)" in result
        assert "name: string" in result
        assert "ACTED_IN" in result

    def test_from_dict_basic(self):
        data = {
            "name": "MyGraph",
            "nodes": [
                {"label": "Person", "properties": {"name": "string"}},
            ],
            "edges": [
                {
                    "label": "KNOWS",
                    "source_label": "Person",
                    "target_label": "Person",
                },
            ],
        }
        schema = CypherSchema.from_dict(data)
        assert schema.name == "MyGraph"
        assert len(schema.nodes) == 1
        assert schema.nodes[0].label == "Person"
        assert len(schema.nodes[0].properties) == 1

    def test_from_dict_with_entities_and_relations(self):
        data = {
            "name": "AltGraph",
            "entities": [
                {"label": "Actor"},
            ],
            "relations": [
                {
                    "label": "STARS_IN",
                    "subj_label": "Actor",
                    "obj_label": "Film",
                },
            ],
        }
        schema = CypherSchema.from_dict(data)
        assert len(schema.nodes) == 1
        assert schema.nodes[0].label == "Actor"
        assert len(schema.edges) == 1
        assert schema.edges[0].source_label == "Actor"
        assert schema.edges[0].target_label == "Film"

    def test_from_dict_properties_as_list(self):
        data = {
            "name": "ListPropsGraph",
            "nodes": [
                {
                    "label": "Item",
                    "properties": [
                        {"name": "id", "data_type": "int"},
                        {"name": "value", "data_type": "float"},
                    ],
                },
            ],
            "edges": [],
        }
        schema = CypherSchema.from_dict(data)
        assert len(schema.nodes[0].properties) == 2
        assert schema.nodes[0].properties[0].name == "id"

    def test_empty_schema(self):
        schema = CypherSchema(name="Empty")
        assert schema.nodes == []
        assert schema.edges == []
        result = schema.to_prompt_string()
        assert "Graph: Empty" in result
