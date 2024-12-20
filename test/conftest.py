from unittest.mock import patch

import pytest
from click.testing import CliRunner

from sekvo.cli.main import load_provider_commands
from sekvo.config.provider_config import AnthropicConfig
from sekvo.config.settings import SekvoSettings
from sekvo.config.types import AdditionalParams
from sekvo.providers import ProviderRegistry
from sekvo.providers.anthropic.generate import AnthropicProvider


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
        )
    )

    with (
        patch.dict(
            "os.environ",
            {
                "SEKVO_ANTHROPIC_TEST1_API_KEY": "test-key1",
                "SEKVO_ANTHROPIC_TEST2_API_KEY": "test-key2",
                "SEKVO_ENV": "test",
            },
        ),
        patch("pathlib.Path.exists", return_value=True),
        patch("sekvo.config.settings.load_dotenv"),
        patch(
            "sekvo.config.settings.SekvoSettings.from_env", return_value=mock_settings
        ),
    ):
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
