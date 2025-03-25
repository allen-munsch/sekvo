"""
Provider registry and imports.
"""
from typing import Dict, Type

from sekvo.providers.base import BaseProvider

class ProviderRegistry:
    """Provider registry for registering and retrieving providers."""
    
    # Dictionary to store provider classes - use lowercase name for backward compatibility
    _providers: Dict[str, Type[BaseProvider]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Register a provider class with the given name."""
        def decorator(provider_class: Type[BaseProvider]) -> Type[BaseProvider]:
            cls._providers[name.lower()] = provider_class
            return provider_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseProvider]:
        """Get a provider class by name."""
        provider_class = cls._providers.get(name.lower())
        if not provider_class:
            available = ", ".join(cls._providers.keys())
            raise ValueError(f"Provider '{name}' not registered. Available providers: {available}")
        return provider_class