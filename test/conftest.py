from unittest.mock import patch, MagicMock, PropertyMock
import pytest
from click.testing import CliRunner

from sekvo.cli.main import load_provider_commands
from sekvo.config.provider_config import AnthropicConfig
from sekvo.config.settings import SekvoSettings
from sekvo.config.types import AdditionalParams
from sekvo.providers import ProviderRegistry
from sekvo.providers.anthropic.generate import AnthropicProvider
from simplemind.providers.anthropic import Anthropic


@pytest.fixture
def mock_env():
    """Set up test environment variables and mock settings loading"""
    # Create mock settings instance with anthropic config
    mock_settings = SekvoSettings(
        anthropic=AnthropicConfig(
            api_key="asdf",
            additional_params=AdditionalParams(
                model="test-model",
                max_tokens=1000,
                temperature=0.7,
                api_key="test-key",
            ),
        ),
        ollama=MagicMock(host_url="http://localhost:11434")
    )

    with (
        patch.dict(
            "os.environ",
            {
                "SEKVO_ANTHROPIC_TEST1_API_KEY": "test-key1",
                "SEKVO_ANTHROPIC_TEST2_API_KEY": "test-key2",
                "SEKVO_ENV": "test",
                "SEKVO_ANTHROPIC_API_KEY": "test-key",
                "SEKVO_OLLAMA_HOST_URL": "http://localhost:11434",
            },
            clear=False,
        ),
        patch("pathlib.Path.exists", return_value=True),
        patch("sekvo.config.settings.load_dotenv"),
        patch(
            "sekvo.config.settings.SekvoSettings.from_env", return_value=mock_settings
        ),
        # Fix for test_lazy_provider_initialization
        patch("simplemind.providers.anthropic.Anthropic", autospec=True, return_value=MagicMock()),
    ):
        yield


@pytest.fixture
def mock_provider_client():
    """Mock the anthropic client for exception cases"""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_body = {"error": {"message": "Invalid API key"}}
    
    class MockAuthError(Exception):
        def __init__(self, message, response, body):
            self.message = message
            self.response = response
            self.body = body
            super().__init__(message)
    
    # Add the mock exception to the anthropic module
    with patch.object(Anthropic, 'generate_text', side_effect=MockAuthError("Invalid API key", response=mock_response, body=mock_body)):
        yield


@pytest.fixture(autouse=True, scope="session")
def setup_commands():
    class MockAnthropicProvider(AnthropicProvider):
        async def generate(self, *args, **kwargs):
            return "This is a test response"

    # Store the original provider
    original_provider = None
    if "anthropic" in ProviderRegistry._providers:
        original_provider = ProviderRegistry._providers["anthropic"]

    # Register our mock provider
    ProviderRegistry._providers["anthropic"] = MockAnthropicProvider

    # Load commands with our mock provider
    load_provider_commands()

    yield


@pytest.fixture
def cli_runner():
    """Create a Click CLI runner"""
    return CliRunner()


@pytest.fixture
def mock_anthropic():
    """Mock Anthropic class for provider initialization testing"""
    with patch('simplemind.providers.anthropic.Anthropic', autospec=True) as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock