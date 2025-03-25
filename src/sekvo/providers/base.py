import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from sekvo.config.settings import SekvoSettings
from sekvo.core.prompt_pipe import Prompt

# Import the BaseTool type from SimpleMind
from simplemind.providers._base_tools import BaseTool

T = TypeVar("T", bound=BaseModel)

class BaseProvider(ABC):
    """Base provider that integrates SimpleMind's provider capabilities with Sekvo's piping functionality."""
    
    NAME: str = ""
    DEFAULT_MODEL: str = ""
    supports_streaming: bool = False
    supports_structured_responses: bool = True
    
    def __init__(self, env_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize the provider using either env_name for config loading or direct config."""
        self.env_name = env_name
        self._config = config
        self.validate_config()
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the provider configuration."""
        if self._config is None:
            # Load config from environment if not directly provided
            settings = SekvoSettings.from_env(self.env_name) if self.env_name else SekvoSettings()
            # Get the config based on provider name
            provider_config = getattr(settings, self.NAME.lower(), None)
            if not provider_config:
                raise ValueError(f"{self.NAME} configuration not found for env: {self.env_name}")
            self._config = provider_config.model_dump()
        return self._config
    
    async def __call__(self, prompt: str) -> str:
        """Make provider callable for use in pipelines"""
        return await self.generate(prompt)
    
    async def pipe(self, prompt: str) -> str:
        """Alias for generate to be more explicit about piping"""
        return await self.generate(prompt)
    
    async def __ror__(self, prompt: Union[str, "Prompt"]) -> str:
        """Handle right side of pipe operation (prompt | provider)"""
        if asyncio.iscoroutine(prompt):
            prompt = await prompt
        
        if isinstance(prompt, str):
            return await self.generate(prompt, None)
        
        return await self.generate(prompt.text, prompt.system_prompt)
    
    @abstractmethod
    def validate_config(self) -> None:
        """Validate provider-specific configuration."""
        raise NotImplementedError
    
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response from the provider."""
        raise NotImplementedError
    
    def structured_response(self, prompt: str, response_model: Type[T], **kwargs) -> T:
        """Get a structured response."""
        raise NotImplementedError("This provider does not support structured responses")
    
    def make_tools(self, tools: List[Union[Callable, BaseTool]]) -> List[BaseTool]:
        """Convert tools to provider-specific tool format."""
        raise NotImplementedError("This provider does not support tools")
    

class ProviderRegistry:
    _providers: Dict[str, Type[BaseProvider]] = {}
    
    @classmethod
    def register(cls, name: str):
        def wrapper(provider_cls: Type[BaseProvider]):
            cls._providers[name] = provider_cls
            return provider_cls
        return wrapper
    
    @classmethod
    def get_provider(cls, name: str) -> Type[BaseProvider]:
        if name not in cls._providers:
            raise ValueError(f"Provider {name} not found")
        return cls._providers[name]