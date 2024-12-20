from .base import BaseProvider
from ..config.provider_config import ProviderConfig
from . import ProviderRegistry

@ProviderRegistry.register("openai")
class OpenAIProvider(BaseProvider):
    def validate_config(self) -> None:
        required = {"api_key", "model"}
        if not all(k in self.config for k in required):
            raise ValueError(f"Missing required config: {required}")
            
    async def generate(self, prompt: str) -> str:
        # Implementation here
        pass
