from typing import Dict, Type
from .base import BaseProvider

class ProviderRegistry:
    _providers: Dict[str, Type[BaseProvider]] = {}
    
    @classmethod
    def register(cls, name: str):
        def wrapper(provider_cls: Type[BaseProvider]):
            cls._providers[name] = provider_cls
            return provider_cls
        return wrapper
    
    @classmethod
    def get_provider(cls, name: str) -> Type[BaseProvider]:
        if name not in cls._providers:
            raise ValueError(f"Provider {name} not found")
        return cls._providers[name]
