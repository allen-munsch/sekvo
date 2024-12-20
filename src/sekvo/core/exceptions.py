class ProviderError(Exception):
    """Base exception for provider-related errors"""
    pass

class ConfigurationError(Exception):
    """Raised when there's an error in configuration"""
    pass

class ProviderNotFoundError(ProviderError):
    """Raised when a requested provider is not found"""
    pass
