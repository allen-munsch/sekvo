from typing import TYPE_CHECKING

from sekvo.config.settings import ENV_NAME, SEKVO_ENV_KEY, SekvoSettings
from sekvo.providers import ProviderRegistry
from sekvo.providers.base import BaseProvider

if TYPE_CHECKING:
    from sekvo.config.settings import AnthropicConfig


@ProviderRegistry.register("anthropic")
class AnthropicProvider(BaseProvider):

    def __init__(self, env_name: str | None = ENV_NAME) -> None:
        settings = SekvoSettings.from_env(env_name) if env_name else SekvoSettings()
        if not settings.anthropic:
            raise ValueError(f"Anthropic configuration not found for env: {env_name} set {SEKVO_ENV_KEY} like 'export {SEKVO_ENV_KEY}=anthropic-dev'")
        self.config: AnthropicConfig  = settings.anthropic

    def validate_config(self) -> None:
        required = {"api_key", "model"}
        if not all(k in self.config for k in required):
            raise ValueError(f"Missing required config: {required}")

    @property
    def client(self):  # noqa: ANN201
        try:
            import anthropic
            return anthropic.Client(
                api_key=self.config.api_key,
                # **self.config.additional_params.model_dump()
            )
        except ImportError:
            raise ImportError(
                "Anthropic package not found. Install with: pip install '.[anthropic]'"
            ) from None

    async def generate(self, prompt: str, system_prompt: str) -> str:
        # Prompt the model to summarize the text
        params = self.config.additional_params
        response = self.client.messages.create(
            model=params.model,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.content[0].text
