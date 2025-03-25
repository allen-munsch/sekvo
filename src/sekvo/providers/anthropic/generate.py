# In src/sekvo/providers/anthropic/generate.py
import warnings
from typing import Optional, Dict, Any

# Import the new implementation
from sekvo.providers.simplemind_adapter import AnthropicProvider as NewAnthropicProvider

# Create a compatibility class that inherits from the new implementation
class AnthropicProvider(NewAnthropicProvider):
    """
    Anthropic provider (legacy import path).
    This class exists for backwards compatibility.
    """
    
    def __init__(self, env_name: Optional[str] = None, provider_name='anthropic', config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize with deprecation warning"""
        warnings.warn(
            "Importing from sekvo.providers.anthropic.generate is deprecated since version 0.1.1"
            "Please import from sekvo.providers.simplemind_adapter instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(provider_name=provider_name, env_name=env_name, config=config)