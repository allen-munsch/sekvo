from typing import Dict, Any, Set, Optional
from pydantic import BaseModel, Field, AliasChoices

class AdditionalParams(BaseModel):
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    top_p: Optional[float] = 1.0
    stream: bool = False
    model: str = ''

class ProviderConfig(BaseModel):
    api_key: str = Field(alias='api_key')
    additional_params: AdditionalParams = AdditionalParams()