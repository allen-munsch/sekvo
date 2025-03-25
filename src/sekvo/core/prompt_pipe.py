import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import List, TypeVar, Union, Optional, Dict, Any

from simplemind.models import Message as SimpleMindMessage
from simplemind.models import Conversation as SimpleMindConversation

T = TypeVar("T", bound="BaseProvider")


@dataclass
class Metrics:
    provider: str
    input_tokens: int
    output_tokens: int
    duration: float
    timestamp: float = field(default_factory=time.time)


class BasePrompt:
    """Base class for all prompt types"""

    def __init__(
        self, text: str, system_prompt: str = "You are a helpful assistant."
    ) -> None:
        self.text = text
        self.system_prompt = system_prompt
        self._pipeline = []

    async def __or__(
        self, other: Union[T, "BasePrompt", str]
    ) -> Union[str, "BasePrompt"]:
        """Enable piping syntax with |"""
        from sekvo.providers.base import BaseProvider

        # If we're piping to a provider
        if isinstance(other, BaseProvider):
            if not self._pipeline:
                return await other.generate(self.text, self.system_prompt)

            current_text = self.text
            for p in self._pipeline:
                current_text = await p.generate(current_text, self.system_prompt)
            return await other.generate(current_text, self.system_prompt)

        # If we're piping to another prompt or string
        other_text = other.text if isinstance(other, BasePrompt) else str(other)
        combined_text = f"{self.text}\n{other_text}"
        return type(self)(combined_text, self.system_prompt)

    async def execute(self) -> str:
        """Execute the pipeline explicitly"""
        if not self._pipeline:
            return self.text

        current_text = self.text
        for provider in self._pipeline:
            current_text = await provider.generate(current_text, self.system_prompt)
        return current_text

    def to_simplemind_conversation(self) -> SimpleMindConversation:
        """Convert this prompt to a SimpleMind conversation."""
        conversation = SimpleMindConversation()
        if self.system_prompt:
            conversation.prepend_system_message(self.system_prompt)
        conversation.add_message(role="user", text=self.text)
        return conversation

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(text='{self.text}', system_prompt='{self.system_prompt}')"


class Prompt(BasePrompt):
    """Standard prompt with basic piping functionality"""
    pass


class ParallelPrompt(BasePrompt):
    """Process through multiple providers simultaneously"""

    async def __or__(self, providers: List[T]) -> List[str]:
        tasks = [
            provider.generate(self.text, self.system_prompt) for provider in providers
        ]
        return await asyncio.gather(*tasks)


class MetricsPrompt(BasePrompt):
    """Prompt that collects performance metrics"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metrics: List[Metrics] = []

    async def __or__(self, provider: T) -> str:
        start = time.time()
        result = await super().__or__(provider)
        duration = time.time() - start

        self.metrics.append(Metrics(
            provider=provider.__class__.__name__,
            input_tokens=len(self.text.split()),  # Simple approximation
            output_tokens=len(result.split()),    # Simple approximation
            duration=duration
        ))
        return result


class ValidatedPrompt(BasePrompt):
    """Prompt with validation and error handling"""
    async def __or__(self, provider: T) -> str:
        try:
            result = await super().__or__(provider)
            if self.validate_json(result):
                return result
            return await self.retry(provider)
        except Exception as e:
            return await self.fallback(provider, e)

    def validate_json(self, text: str) -> bool:
        """Check if response is valid JSON"""
        try:
            json.loads(text)
            return True
        except:
            return False

    async def retry(self, provider: T, max_retries: int = 3) -> str:
        """Retry failed attempts"""
        for _ in range(max_retries):
            try:
                result = await provider.generate(self.text, self.system_prompt)
                if self.validate_json(result):
                    return result
            except:
                continue
        return '{"error": "Max retries exceeded"}'

    async def fallback(self, provider: T, error: Exception) -> str:
        """Handle errors with fallback response"""
        return json.dumps({"error": str(error)})


class FilterPrompt(BasePrompt):
    """Prompt with filtering and transformation capabilities"""

    def filter(self, func: Callable[[str], bool]) -> "FilterPrompt":
        if not func(self.text):
            self.text = ""
        return self

    def transform(self, func: Callable[[str], str]) -> "FilterPrompt":
        self.text = func(self.text)
        return self


class BatchPrompt(BasePrompt):
    """Process multiple prompts with rate limiting"""

    def __init__(
        self, texts: List[str], system_prompt: str = "You are a helpful assistant."
    ):
        super().__init__(texts[0], system_prompt)
        self.texts = texts
        self.batch_size = 10
        self.rate_limit = 1  # seconds between batches

    async def __or__(self, provider: T) -> List[str]:
        results = []
        for i in range(0, len(self.texts), self.batch_size):
            batch = self.texts[i : i + self.batch_size]
            batch_results = await asyncio.gather(
                *[provider.generate(text, self.system_prompt) for text in batch]
            )
            results.extend(batch_results)
            if i + self.batch_size < len(self.texts):
                await asyncio.sleep(self.rate_limit)
        return results