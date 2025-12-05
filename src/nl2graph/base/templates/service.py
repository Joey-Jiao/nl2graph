from typing import Dict
from pathlib import Path

from ..configs import ConfigService
from .entity import Template
from .renderer import TemplateRenderer


class TemplateService:
    def __init__(self, config: ConfigService):
        self._renderers: Dict[str, TemplateRenderer] = {}
        self._templates: Dict[str, Dict[str, Template]] = {}
        self._load_categories(config)

    def _load_categories(self, config: ConfigService):
        templates_config = config.get("templates", {})
        if not templates_config:
            return

        for category, dir_path in templates_config.items():
            template_dir = Path(dir_path)
            if not template_dir.exists():
                continue

            self._renderers[category] = TemplateRenderer(template_dir)
            self._templates[category] = {}

            for f in template_dir.glob("*.jinja2"):
                name = f.stem
                self._templates[category][name] = Template(
                    name=name,
                    category=category,
                    path=f,
                )

    def ls_categories(self):
        return list(self._templates.keys())

    def ls_templates(self, category: str):
        return list(self._templates.get(category, {}).keys())

    def render(self, category: str, name: str, **kwargs) -> str:
        if category not in self._renderers:
            raise KeyError(f"category not found: '{category}'")
        template_name = name.removesuffix(".jinja2")
        if template_name not in self._templates.get(category, {}):
            raise KeyError(f"template not found: '{category}/{name}'")

        return self._renderers[category].render(f"{template_name}.jinja2", **kwargs)
