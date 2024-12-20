#!/bin/bash

# Create directory structure
mkdir -p providers config core

# Create __init__.py files
touch __init__.py
touch providers/__init__.py
touch config/__init__.py
touch core/__init__.py

# Create base provider class
cat > providers/base.py << 'EOF'
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseProvider(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validate_config()
        
    @abstractmethod
    def validate_config(self) -> None:
        """Validate provider-specific configuration."""
        pass
        
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate response from the provider."""
        pass
EOF

# Create provider registry
cat > providers/__init__.py << 'EOF'
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
EOF

# Create provider implementations
cat > providers/anthropic.py << 'EOF'
from .base import BaseProvider
from ..config.provider_config import ProviderConfig
from . import ProviderRegistry

@ProviderRegistry.register("anthropic")
class AnthropicProvider(BaseProvider):
    def validate_config(self) -> None:
        required = {"api_key", "model"}
        if not all(k in self.config for k in required):
            raise ValueError(f"Missing required config: {required}")
            
    async def generate(self, prompt: str) -> str:
        # Implementation here
        pass
EOF

cat > providers/openai.py << 'EOF'
from .base import BaseProvider
from ..config.provider_config import ProviderConfig
from . import ProviderRegistry

@ProviderRegistry.register("openai")
class OpenAIProvider(BaseProvider):
    def validate_config(self) -> None:
        required = {"api_key", "model"}
        if not all(k in self.config for k in required):
            raise ValueError(f"Missing required config: {required}")
            
    async def generate(self, prompt: str) -> str:
        # Implementation here
        pass
EOF

cat > providers/vertexai.py << 'EOF'
from .base import BaseProvider
from ..config.provider_config import ProviderConfig
from . import ProviderRegistry

@ProviderRegistry.register("vertexai")
class VertexAIProvider(BaseProvider):
    def validate_config(self) -> None:
        required = {"project_id", "location", "model"}
        if not all(k in self.config for k in required):
            raise ValueError(f"Missing required config: {required}")
            
    async def generate(self, prompt: str) -> str:
        # Implementation here
        pass
EOF

# Create configuration files
cat > config/base.py << 'EOF'
from typing import Dict, Any
from pydantic import BaseModel

class BaseConfig(BaseModel):
    """Base configuration class for all configs"""
    pass
EOF

cat > config/provider_config.py << 'EOF'
from typing import Dict, Any
from pydantic import BaseModel
from .base import BaseConfig

class ProviderConfig(BaseConfig):
    provider_name: str
    api_key: str
    model: str
    additional_params: Dict[str, Any] = {}
EOF

cat > config/settings.py << 'EOF'
from typing import Dict, Any
from pathlib import Path
import os
import json

class Settings:
    @staticmethod
    def load_provider_configs() -> Dict[str, Dict[str, Any]]:
        """Load provider configurations from environment or file"""
        # Implementation here - load from env vars or config file
        return {}
EOF

# Create core utilities
cat > core/types.py << 'EOF'
from typing import TypeVar, Dict, Any
from pydantic import BaseModel

ConfigT = TypeVar('ConfigT', bound=BaseModel)
ProviderT = TypeVar('ProviderT')
EOF

cat > core/exceptions.py << 'EOF'
class ProviderError(Exception):
    """Base exception for provider-related errors"""
    pass

class ConfigurationError(Exception):
    """Raised when there's an error in configuration"""
    pass

class ProviderNotFoundError(ProviderError):
    """Raised when a requested provider is not found"""
    pass
EOF

# Make script executable
chmod +x providers/__init__.py

echo "Project structure created successfully!"
