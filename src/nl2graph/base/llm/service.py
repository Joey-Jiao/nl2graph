from ..configs import ConfigService
from .entity import ClientConfig
from .clients.deepseek import DeepSeekClient
from .clients.openai import OpenAIClient


class LLMService:
    def __init__(self, config: ConfigService):
        self._config = {}
        provider_config = config.get("llm", {})

        if not provider_config:
            return

        for provider_name, _ in provider_config.items():
            self._config[provider_name] = {}
            model_config = config.get(f"llm.{provider_name}", {})

            for model_name, _ in model_config.items():
                provider_name_upper = provider_name.upper()
                api_key = config.get_env(f"{provider_name_upper}_API_KEY", default=None)
                endpoint = config.get(f"llm.{provider_name}.{model_name}.endpoint", default=None)
                timeout = config.get(f"llm.{provider_name}.{model_name}.timeout", default=30)

                client_config = ClientConfig(
                    api_key=api_key,
                    endpoint=endpoint,
                    timeout=timeout,
                    model=model_name,
                    provider=provider_name,
                )
                self._config[provider_name][model_name] = client_config

    def ls_providers(self):
        return list(self._config.keys())

    def ls_models(self, provider: str):
        return list(self._config.get(provider, {}).keys())

    def get_client_config(self, provider: str, model: str) -> ClientConfig:
        return self._config.get(provider, {}).get(model)

    def get_client(self, provider: str, model: str):
        client_config = self.get_client_config(provider, model)
        if not client_config:
            return None
        if provider == "deepseek":
            return DeepSeekClient(client_config)
        elif provider == "openai":
            return OpenAIClient(client_config)
        return None
