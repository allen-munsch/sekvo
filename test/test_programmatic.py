from unittest.mock import patch

import anthropic
import pytest

from sekvo.providers.anthropic.generate import AnthropicProvider


@pytest.mark.asyncio
async def test_direct_provider_usage(mock_env) -> None:
    """Test direct provider usage"""
    provider = AnthropicProvider()

    # Test authentication error case
    with pytest.raises(anthropic.AuthenticationError):
        await provider.generate("tell me a joke", '')

    # Test successful case with mock
    with patch.object(provider, 'generate', return_value="This is a test response"):
        result = await provider.generate("tell me a joke", '')
        assert result == "This is a test response"

@pytest.mark.asyncio
async def test_provider_with_options(mock_env) -> None:
    """Test provider with custom options"""
    provider = AnthropicProvider(env_name="test")

    async def mock_generate(prompt: str, system_prompt: str | None = None) -> str:
        return "This is a test response"

    with patch.object(provider, 'generate', side_effect=mock_generate):
        result = await provider.generate(
            prompt="tell me a joke",
            system_prompt="You are a comedian"
        )
        assert result == "This is a test response"
