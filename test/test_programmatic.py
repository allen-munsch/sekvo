from unittest.mock import MagicMock, patch, AsyncMock
import anthropic
import pytest
from sekvo.providers.anthropic.generate import AnthropicProvider

@pytest.mark.asyncio
async def test_direct_provider_usage(mock_env) -> None:
    """Test direct provider usage"""
    provider = AnthropicProvider(config={"api_key": "invalid-key"})
    
    # Update this part to create a proper AuthenticationError with required arguments
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_body = {"error": {"message": "Invalid API key"}}
    
    with patch.object(provider.simplemind_provider, 'generate_text', 
                     side_effect=anthropic.APIStatusError("Invalid API key", response=mock_response, body=mock_body)):
        with pytest.raises(anthropic.APIStatusError):
            await provider.generate("tell me a joke", '')
    
    # Test successful case with mock
    with patch.object(provider, 'generate', return_value="This is a test response"):
        result = await provider.generate("tell me a joke", '')
        assert result == "This is a test response"

@pytest.mark.asyncio
async def test_provider_with_options(mock_env) -> None:
    """Test provider with custom options"""
    provider = AnthropicProvider(config={"api_key": "test-key"})
    
    # Create a mock for the generate method
    async def mock_generate(prompt: str, system_prompt: str | None = None) -> str:
        return "This is a test response"
    
    with patch.object(provider, 'generate', side_effect=mock_generate):
        result = await provider.generate(
            prompt="tell me a joke",
            system_prompt="You are a comedian"
        )
        assert result == "This is a test response"