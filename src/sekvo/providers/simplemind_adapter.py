"""
Adapter for SimpleMind providers.
This module provides classes that adapt SimpleMind providers to work with Sekvo.
"""
import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

from pydantic import BaseModel
from simplemind.providers._base_tools import BaseTool
from simplemind.providers.amazon import Amazon

# Import the supported providers from SimpleMind
from simplemind.providers.anthropic import Anthropic
from simplemind.providers.deepseek import Deepseek
from simplemind.providers.gemini import Gemini
from simplemind.providers.groq import Groq
from simplemind.providers.ollama import Ollama
from simplemind.providers.openai import OpenAI
from simplemind.providers.xai import XAI

from sekvo.config.settings import ENV_NAME, SEKVO_ENV_KEY
from sekvo.providers import ProviderRegistry
from sekvo.providers.base import BaseProvider

T = TypeVar("T", bound=BaseModel)

# Mapping of provider names to their SimpleMind classes
PROVIDER_MAP = {
    "anthropic": Anthropic,
    "openai": OpenAI,
    "groq": Groq,
    "gemini": Gemini,
    "ollama": Ollama,
    "xai": XAI,
    "amazon": Amazon,
    "deepseek": Deepseek,
}

# Default models for each provider
DEFAULT_MODELS = {
    "anthropic": "claude-3-opus-20240229",
    "openai": "gpt-4-turbo-preview",
    "groq": "llama3-8b-8192",
    "gemini": "gemini-1.5-flash-latest",
    "ollama": "llama3.2",
    "xai": "grok-beta",
    "amazon": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "deepseek": "deepseek-chat",
}

class SimpleMindAdapter(BaseProvider):
    """Adapter that wraps SimpleMind providers to work with Sekvo."""

    def __init__(self, provider_name: str, env_name: str | None = ENV_NAME, config: dict[str, Any] | None = None) -> None:
        """
        Initialize a provider adapter for SimpleMind.
        
        Args:
            provider_name: The name of the provider to adapt (e.g., "anthropic", "openai")
            env_name: The environment name to load configuration from
            config: Optional direct configuration dictionary
        """
        self.NAME = provider_name.lower()
        self.DEFAULT_MODEL = DEFAULT_MODELS.get(self.NAME, "")
        self._simplemind_provider = None
        super().__init__(env_name=env_name, config=config)

    @property
    def simplemind_provider(self) -> Any:
        """Lazy-initialize the SimpleMind provider."""
        if self._simplemind_provider is None:
            provider_class = PROVIDER_MAP.get(self.NAME)
            if not provider_class:
                raise ValueError(f"Unsupported provider: {self.NAME}")

            # Handle different initialization parameters for different providers
            if self.NAME == "ollama":
                self._simplemind_provider = provider_class(host_url=self.config.get("host_url"))
            elif self.NAME == "amazon":
                self._simplemind_provider = provider_class(profile_name=self.config.get("profile_name"))
            else:
                self._simplemind_provider = provider_class(api_key=self.config.get("api_key"))

            # Copy streaming support flags from SimpleMind provider
            self.supports_streaming = getattr(self._simplemind_provider, "supports_streaming", False)
            self.supports_structured_responses = getattr(self._simplemind_provider, "supports_structured_responses", True)

        return self._simplemind_provider

    def validate_config(self) -> None:
        """Validate provider-specific configuration."""
        if self.NAME == "ollama":
            if not self.config.get("host_url"):
                raise ValueError(f"Missing required Ollama host URL. Set {SEKVO_ENV_KEY} like 'export {SEKVO_ENV_KEY}=ollama-dev'")
        elif self.NAME == "amazon":
            if not self.config.get("profile_name"):
                raise ValueError(f"Missing required AWS profile name. Set {SEKVO_ENV_KEY} like 'export {SEKVO_ENV_KEY}=amazon-dev'")
        else:
            if not self.config.get("api_key"):
                raise ValueError(f"Missing required {self.NAME.capitalize()} API key. Set {SEKVO_ENV_KEY} like 'export {SEKVO_ENV_KEY}={self.NAME}-dev'")

    async def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate response from the provider."""
        params = self.config.get("additional_params", {})
        model = params.get("model", self.DEFAULT_MODEL)
        max_tokens = params.get("max_tokens", 1000)
        temperature = params.get("temperature", 0.7)

        # Build appropriate kwargs based on provider
        kwargs = {
            "llm_model": model,
            "temperature": temperature,
        }

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        # System prompts are handled differently by different providers
        if system_prompt:
            if self.NAME == "anthropic":
                kwargs["system"] = system_prompt
            else:
                # For providers like OpenAI that use messages
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                kwargs["messages"] = messages

        # Use the SimpleMind provider's generate_text method
        response = self.simplemind_provider.generate_text(
            prompt=prompt,
            **kwargs
        )

        return response

    async def generate_stream(self, prompt: str, system_prompt: str | None = None) -> AsyncIterator[str]:
        """
        Generate streaming response from the provider.
        """
        params = self.config.get("additional_params", {})
        model = params.get("model", self.DEFAULT_MODEL)
        max_tokens = params.get("max_tokens", 1000)
        temperature = params.get("temperature", 0.7)

        # Build kwargs
        kwargs = {
            "llm_model": model,
            "temperature": temperature,
        }

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        # Handle system prompt
        if system_prompt:
            if self.NAME == "anthropic":
                kwargs["system"] = system_prompt
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                kwargs["messages"] = messages

        try:
            if self.supports_streaming and hasattr(self.simplemind_provider, 'generate_stream_text'):
                # The key fix: Adapt the regular generator to an async generator
                # by yielding each value from the regular generator
                for token in self.simplemind_provider.generate_stream_text(
                    prompt=prompt,
                    **kwargs
                ):
                    if token:
                        yield token
                        # Small delay to avoid overwhelming the terminal
                        await asyncio.sleep(0.001)
            else:
                # Fallback for non-streaming providers
                complete_response = self.simplemind_provider.generate_text(
                    prompt=prompt,
                    **kwargs
                )
                yield complete_response
        except Exception as e:
            yield f"Error in streaming: {e!s}"

    def structured_response(self, prompt: str, response_model: type[T], **kwargs) -> T:
        """Get a structured response using SimpleMind's implementation."""
        params = self.config.get("additional_params", {})
        model = params.get("model", self.DEFAULT_MODEL)

        # Check if the provider supports structured responses
        if not getattr(self.simplemind_provider, "supports_structured_responses", True):
            raise NotImplementedError(f"Provider {self.NAME} does not support structured responses")

        return self.simplemind_provider.structured_response(
            prompt=prompt,
            response_model=response_model,
            llm_model=model,
            **kwargs
        )

    def make_tools(self, tools: list[Callable | BaseTool]) -> list[BaseTool]:
        """Convert tools to provider-specific tool format."""
        # Check if the provider supports tools (has the make_tools method)
        if hasattr(self.simplemind_provider, "make_tools"):
            return self.simplemind_provider.make_tools(tools)
        else:
            raise NotImplementedError(f"Provider {self.NAME} does not support tools")


# Create concrete provider classes with decorator registration
@ProviderRegistry.register("anthropic")
class AnthropicProvider(SimpleMindAdapter):
    """Anthropic provider adapter"""
    def __init__(self, env_name: str | None = ENV_NAME, provider_name: str | None = "anthropic", config: dict[str, Any] | None = None) -> None:
        super().__init__(provider_name=provider_name, env_name=env_name, config=config)
        self.supports_streaming = True

@ProviderRegistry.register("openai")
class OpenAIProvider(SimpleMindAdapter):
    """OpenAI provider adapter"""
    def __init__(self, env_name: str | None = ENV_NAME, provider_name: str | None = "openai", config: dict[str, Any] | None = None) -> None:
        super().__init__(provider_name=provider_name, env_name=env_name, config=config)

@ProviderRegistry.register("groq")
class GroqProvider(SimpleMindAdapter):
    """Groq provider adapter"""
    def __init__(self, env_name: str | None = ENV_NAME, provider_name: str | None = "groq", config: dict[str, Any] | None = None) -> None:
        super().__init__(provider_name=provider_name, env_name=env_name, config=config)

@ProviderRegistry.register("gemini")
class GeminiProvider(SimpleMindAdapter):
    """Gemini provider adapter"""
    def __init__(self, env_name: str | None = ENV_NAME, provider_name: str | None = "gemini", config: dict[str, Any] | None = None) -> None:
        super().__init__(provider_name=provider_name, env_name=env_name, config=config)

@ProviderRegistry.register("ollama")
class OllamaProvider(SimpleMindAdapter):
    """Ollama provider adapter"""
    def __init__(self, env_name: str | None = ENV_NAME, provider_name: str | None = "ollama", config: dict[str, Any] | None = None) -> None:
        super().__init__(provider_name=provider_name, env_name=env_name, config=config)

@ProviderRegistry.register("xai")
class XAIProvider(SimpleMindAdapter):
    """XAI provider adapter"""
    def __init__(self, env_name: str | None = ENV_NAME, provider_name: str | None = "xai", config: dict[str, Any] | None = None) -> None:
        super().__init__(provider_name=provider_name, env_name=env_name, config=config)

@ProviderRegistry.register("amazon")
class AmazonProvider(SimpleMindAdapter):
    """Amazon Bedrock provider adapter"""
    def __init__(self, env_name: str | None = ENV_NAME, provider_name: str | None = "amazon", config: dict[str, Any] | None = None) -> None:
        super().__init__(provider_name=provider_name, env_name=env_name, config=config)

@ProviderRegistry.register("deepseek")
class DeepseekProvider(SimpleMindAdapter):
    """Deepseek provider adapter"""
    def __init__(self, env_name: str | None = ENV_NAME, provider_name: str | None = "deepseek", config: dict[str, Any] | None = None) -> None:
        super().__init__(provider_name=provider_name, env_name=env_name, config=config)
