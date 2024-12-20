from unittest.mock import patch

import pytest

from sekvo.core.prompt_pipe import (
    BatchPrompt,
    FilterPrompt,
    MetricsPrompt,
    ParallelPrompt,
    Prompt,
    ValidatedPrompt,
)
from sekvo.providers.anthropic.generate import AnthropicProvider


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline(mock_env) -> None:
    """Test complete pipeline with multiple providers"""
    provider1 = AnthropicProvider(env_name="anthropic-test1")
    provider2 = AnthropicProvider(env_name="anthropic-test2")

    # Mock the generate methods to show the chaining
    async def mock_generate1(prompt: str, system_prompt: str | None = None) -> str:
        return f"provider1({prompt})"

    async def mock_generate2(prompt: str, system_prompt: str | None = None) -> str:
        return f"provider2({prompt})"

    with patch.object(provider1, 'generate', side_effect=mock_generate1), \
         patch.object(provider2, 'generate', side_effect=mock_generate2):

        # Test single provider
        result = await (Prompt("test prompt") | provider1)
        assert result == "provider1(test prompt)"

        # Test chained providers
        result = await (Prompt("test prompt") | provider1 | provider2)
        assert result == "provider2(provider1(test prompt))"

        # Test with system prompts
        result = await provider1.generate(
            prompt="test prompt",
            system_prompt="custom system prompt"
        )
        assert result == "provider1(test prompt)"

        # Test piping prompts together
        first = await (Prompt("first") | provider1)
        second = await (Prompt("second") | provider1)

        # Pipe the two processed prompts together and through another provider
        result = await (Prompt(first) | Prompt(second) | provider1)
        assert result == "provider1(provider1(first)\nprovider1(second))"

        # Test inline composition
        result = await (
            Prompt(await (Prompt("first") | provider1)) |
            Prompt(await (Prompt("second") | provider1)) |
            provider1
        )
        assert result == "provider1(provider1(first)\nprovider1(second))"


@pytest.mark.asyncio
async def test_parallel_prompt(mock_env) -> None:
    """Test parallel processing of prompts"""
    provider1 = AnthropicProvider(env_name="anthropic-test1")
    provider2 = AnthropicProvider(env_name="anthropic-test2")

    async def mock_generate1(prompt: str, system_prompt: str | None = None) -> str:
        return f"provider1({prompt})"

    async def mock_generate2(prompt: str, system_prompt: str | None = None) -> str:
        return f"provider2({prompt})"

    with patch.object(provider1, 'generate', side_effect=mock_generate1), \
         patch.object(provider2, 'generate', side_effect=mock_generate2):

        result = await (ParallelPrompt("test prompt") | [provider1, provider2])
        assert result == ["provider1(test prompt)", "provider2(test prompt)"]


@pytest.mark.asyncio
async def test_metrics_prompt(mock_env) -> None:
    """Test metrics collection"""
    provider = AnthropicProvider(env_name="anthropic-test1")

    async def mock_generate(prompt: str, system_prompt: str | None = None) -> str:
        return f"response({prompt})"

    with patch.object(provider, 'generate', side_effect=mock_generate):
        prompt = MetricsPrompt("test")
        result = await (prompt | provider)

        assert len(prompt.metrics) == 1
        assert prompt.metrics[0].provider == "AnthropicProvider"
        assert prompt.metrics[0].input_tokens == 1  # "test"
        assert prompt.metrics[0].duration > 0

@pytest.mark.asyncio
async def test_validated_prompt(mock_env) -> None:
    """Test JSON validation and retry logic"""
    provider = AnthropicProvider(env_name="anthropic-test1")

    call_count = 0
    async def mock_generate(prompt: str, system_prompt: str | None = None) -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return "invalid json"
        return '{"result": "valid json"}'

    with patch.object(provider, 'generate', side_effect=mock_generate):
        prompt = ValidatedPrompt("test")
        result = await (prompt | provider)

        assert result == '{"result": "valid json"}'
        assert call_count == 3  # Two failures + one success

@pytest.mark.asyncio
async def test_filter_prompt(mock_env) -> None:
    """Test filtering and transformation"""
    provider = AnthropicProvider(env_name="anthropic-test1")

    async def mock_generate(prompt: str, system_prompt: str | None = None) -> str:
        return f"response({prompt})"

    with patch.object(provider, 'generate', side_effect=mock_generate):
        prompt = FilterPrompt("test prompt")
        result = await (
            prompt
            .transform(str.upper)
            .filter(lambda x: len(x) > 0) |
            provider
        )

        assert result == "response(TEST PROMPT)"

@pytest.mark.asyncio
async def test_batch_prompt(mock_env) -> None:
    """Test batch processing"""
    provider = AnthropicProvider(env_name="anthropic-test1")

    async def mock_generate(prompt: str, system_prompt: str | None = None) -> str:
        return f"response({prompt})"

    with patch.object(provider, 'generate', side_effect=mock_generate):
        prompt = BatchPrompt(["test1", "test2", "test3"])
        results = await (prompt | provider)

        assert len(results) == 3
        assert results == [
            "response(test1)",
            "response(test2)",
            "response(test3)"
        ]
