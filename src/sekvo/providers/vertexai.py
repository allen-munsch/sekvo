from .base import BaseProvider
from ..config.provider_config import ProviderConfig
from . import ProviderRegistry

@ProviderRegistry.register("vertexai")
class VertexAIProvider(BaseProvider):
    def validate_config(self) -> None:
        required = {"project_id", "location", "model"}
        if not all(k in self.config for k in required):
            raise ValueError(f"Missing required config: {required}")
            
    async def generate(self, prompt: str) -> str:
        # Implementation here
        pass
