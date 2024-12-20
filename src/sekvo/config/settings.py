import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from sekvo.config.provider_config import AnthropicConfig, OpenAIConfig, VertexAIConfig

SEKVO_ENV_KEY = "SEKVO_ENVIRONMENT"
ENV_NAME = os.environ.get(SEKVO_ENV_KEY, None)


class EnvManager:
    @staticmethod
    def parse_env_name(env_name: str) -> tuple[str, str]:
        """Parse environment name into provider and environment"""
        try:
            provider, env = env_name.split("-", 1)
            return provider, env
        except ValueError:
            raise ValueError(
                "Environment name must be in format 'provider-env' (e.g., 'anthropic-dev')"
            ) from None

    @staticmethod
    def load_env(env_name: str | None = None, env_dir: str | Path = ".env") -> str:
        """Load environment variables with provider-specific support"""
        env_dir = Path(env_dir)
        provider = None

        # Load base configuration first
        base_env = env_dir / ".env"
        if base_env.exists():
            load_dotenv(base_env)

        if env_name:
            provider, env = EnvManager.parse_env_name(env_name)

            # Load provider base config
            provider_env = env_dir / f".env.{provider}"
            if provider_env.exists():
                load_dotenv(provider_env)

            # Load provider-specific environment
            env_file = env_dir / f".env.{provider}.{env}"
            if env_file.exists():
                load_dotenv(env_file, override=True)
            else:
                raise FileNotFoundError(f"Environment file not found: {env_file}")

        return provider if provider else ""


class SekvoSettings(BaseSettings):
    anthropic: AnthropicConfig | None = None
    openai: OpenAIConfig | None = None
    vertexai: VertexAIConfig | None = None

    model_config = SettingsConfigDict(
        env_prefix="SEKVO_",
        env_nested_delimiter="-",
        case_sensitive=False,
        extra="allow",
    )

    @classmethod
    def from_env(
        cls, env_name: str | None = None, env_dir: str | Path = ".env"
    ) -> "SekvoSettings":
        """Create settings from provider-specific environment"""
        provider = EnvManager.load_env(env_name, env_dir)
        settings = cls()

        if provider:
            # Return only the specified provider's configuration
            provider_config = getattr(settings, provider, None)
            if provider_config is None:
                raise ValueError(
                    f"No configuration found for provider: {provider} for"
                    f" {env_name} in {env_dir} do the ENV vars exist "
                    "in the .env directory?"
                )

            # Create new settings with only the specified provider
            return cls(**{provider: provider_config})

        return settings
