from typing import Optional
from pydantic import Field, AliasChoices
from .types import ProviderConfig, AdditionalParams

class AnthropicConfig(ProviderConfig):
    api_key: str = Field(
        validation_alias=AliasChoices('anthropic_api_key', 'api_key')
    )
    additional_params: AdditionalParams = AdditionalParams(
        temperature=0.7,
        max_tokens=1000,
        model="claude-3-opus-20240229"
    )

class OpenAIConfig(ProviderConfig):
    api_key: str = Field(
        validation_alias=AliasChoices('openai_api_key', 'api_key')
    )
    organization_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices('openai_org_id', 'organization_id')
    )
    additional_params: AdditionalParams = AdditionalParams(
        temperature=0.7,
        max_tokens=1000,
        model= "gpt-4-turbo-preview"
    )


class VertexAIConfig(ProviderConfig):
    project_id: str = Field(
        validation_alias=AliasChoices('vertex_project_id', 'project_id')
    )
    location: str = Field(
        default="us-central1",
        validation_alias=AliasChoices('vertex_location', 'location')
    )
    additional_params: AdditionalParams = AdditionalParams(
        temperature=0.7,
        max_tokens=1000,
        model= "gemini-pro"
    )