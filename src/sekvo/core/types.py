from typing import TypeVar, Dict, Any
from pydantic import BaseModel

ConfigT = TypeVar('ConfigT', bound=BaseModel)
ProviderT = TypeVar('ProviderT')
