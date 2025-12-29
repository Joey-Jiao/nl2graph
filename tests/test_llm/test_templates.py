import pytest
from pathlib import Path

from nl2graph.base.templates.entity import Template
from nl2graph.base.templates.renderer import TemplateRenderer
from nl2graph.base.templates.service import TemplateService
from nl2graph.base.configs import ConfigService


class TestTemplate:

    def test_create(self):
        template = Template(
            name="query",
            category="prompts",
            path=Path("templates/prompts/query.jinja2"),
        )
        assert template.name == "query"
        assert template.category == "prompts"
        assert template.path == Path("templates/prompts/query.jinja2")


class TestTemplateRenderer:

    @pytest.fixture
    def renderer(self, tmp_path):
        template_file = tmp_path / "test.jinja2"
        template_file.write_text("Hello {{ name }}!")
        return TemplateRenderer(tmp_path)

    def test_render_simple(self, renderer):
        result = renderer.render("test.jinja2", name="World")
        assert result == "Hello World!"

    def test_render_missing_variable_raises(self, tmp_path):
        template_file = tmp_path / "strict.jinja2"
        template_file.write_text("Value: {{ missing }}")
        renderer = TemplateRenderer(tmp_path)
        with pytest.raises(Exception):
            renderer.render("strict.jinja2")

    def test_render_with_conditionals(self, tmp_path):
        template_file = tmp_path / "conditional.jinja2"
        template_file.write_text("{% if show %}Visible{% endif %}")
        renderer = TemplateRenderer(tmp_path)
        assert renderer.render("conditional.jinja2", show=True) == "Visible"
        assert renderer.render("conditional.jinja2", show=False) == ""

    def test_render_with_loop(self, tmp_path):
        template_file = tmp_path / "loop.jinja2"
        template_file.write_text("{% for item in items %}{{ item }},{% endfor %}")
        renderer = TemplateRenderer(tmp_path)
        result = renderer.render("loop.jinja2", items=["a", "b", "c"])
        assert result == "a,b,c,"


class TestTemplateService:

    @pytest.fixture
    def service(self, tmp_path):
        prompts_dir = tmp_path / "templates" / "prompts"
        prompts_dir.mkdir(parents=True)

        query_template = prompts_dir / "query.jinja2"
        query_template.write_text("Question: {{ question }}\nLang: {{ lang }}")

        summary_template = prompts_dir / "summary.jinja2"
        summary_template.write_text("Summarize: {{ text }}")

        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"templates:\n  prompts: {prompts_dir}\n")

        config = ConfigService(
            config_dir=[config_file],
            env_path=".env.nonexistent",
        )
        return TemplateService(config)

    def test_ls_categories(self, service):
        categories = service.ls_categories()
        assert "prompts" in categories

    def test_ls_templates(self, service):
        templates = service.ls_templates("prompts")
        assert "query" in templates
        assert "summary" in templates

    def test_render(self, service):
        result = service.render("prompts", "query", question="What is X?", lang="cypher")
        assert "Question: What is X?" in result
        assert "Lang: cypher" in result

    def test_render_with_jinja2_suffix(self, service):
        result = service.render("prompts", "query.jinja2", question="Q", lang="sparql")
        assert "Question: Q" in result

    def test_render_nonexistent_category(self, service):
        with pytest.raises(KeyError):
            service.render("nonexistent", "template", arg="value")

    def test_render_nonexistent_template(self, service):
        with pytest.raises(KeyError):
            service.render("prompts", "nonexistent", arg="value")

    def test_empty_templates_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("other: value\n")
        config = ConfigService(
            config_dir=[config_file],
            env_path=".env.nonexistent",
        )
        service = TemplateService(config)
        assert service.ls_categories() == []
