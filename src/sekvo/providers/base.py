import asyncio
from abc import ABC, abstractmethod
from typing import Any, Union

from sekvo.core.prompt_pipe import Prompt


class BaseProvider(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.validate_config()

    async def __call__(self, prompt: str) -> str:
        """Make provider callable for use in pipelines"""
        return await self.generate(prompt)

    async def pipe(self, prompt: str) -> str:
        """Alias for generate to be more explicit about piping"""
        return await self.generate(prompt)

    async def __ror__(self, prompt: Union[str, "Prompt"]) -> str:
        """Handle right side of pipe operation (prompt | provider)"""
        # If it's a coroutine, await it first
        if asyncio.iscoroutine(prompt):
            prompt = await prompt
        if isinstance(prompt, str):
            return await self.generate(prompt)
        return await self.generate(prompt.text, prompt.system_prompt)

    @abstractmethod
    def validate_config(self) -> None:
        """Validate provider-specific configuration."""
        raise NotImplementedError

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate response from the provider."""
        raise NotImplementedError

    @property
    @abstractmethod
    async def client(self, prompt: str) -> str:
        """Generate response from the provider."""
        raise NotImplementedError
