import pytest
from pathlib import Path

from nl2graph.base.templates.renderer import TemplateRenderer
from nl2graph.base.templates.entity import Template

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestTemplateRenderer:
    @pytest.fixture
    def renderer(self):
        return TemplateRenderer(template_dir=FIXTURES_DIR)

    def test_render_template(self, renderer):
        result = renderer.render("test_template.jinja2", name="Alice", task="testing")
        assert "Hello, Alice!" in result
        assert "Your task is: testing" in result

    def test_render_missing_variable_raises(self, renderer):
        with pytest.raises(Exception):
            renderer.render("test_template.jinja2", name="Bob")


class TestTemplate:
    def test_template_creation(self):
        template = Template(name="test", category="prompts", path=Path("/tmp/test.jinja2"))
        assert template.name == "test"
        assert template.category == "prompts"
        assert template.path == Path("/tmp/test.jinja2")
