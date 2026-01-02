from pathlib import Path

import punq

from .configs import ConfigService
from .models.service import ModelService
from .llm.service import LLMService
from .templates.service import TemplateService
from ..execution.service import GraphService
from ..evaluation import Scoring


class ApplicationContext:
    def __init__(self, container: punq.Container):
        self._container = container

    def resolve(self, cls):
        return self._container.resolve(cls)

    def register(self, *args, **kwargs):
        return self._container.register(*args, **kwargs)


def get_context(
    config_dir: str = "configs",
    env_path: str = ".env",
) -> ApplicationContext:
    container = punq.Container()

    config_files = list(Path(config_dir).glob("*.yaml"))
    config_service = ConfigService(config_dir=config_files, env_path=env_path)
    container.register(ConfigService, instance=config_service)

    llm_service = LLMService(config=config_service)
    container.register(LLMService, instance=llm_service)

    template_service = TemplateService(config=config_service)
    container.register(TemplateService, instance=template_service)

    model_service = ModelService(config=config_service)
    container.register(ModelService, instance=model_service)

    graph_service = GraphService(config=config_service)
    container.register(GraphService, instance=graph_service)

    scoring = Scoring()
    container.register(Scoring, instance=scoring)

    return ApplicationContext(container)
