"""
Custom exceptions for Deeploi.
"""


class DeeploiError(Exception):
    """Base exception for all Deeploi errors."""
    pass


class UnsupportedModelError(DeeploiError):
    """Raised when model type is not supported."""
    pass


class InvalidSampleError(DeeploiError):
    """Raised when sample input is invalid."""
    pass


class SchemaValidationError(DeeploiError):
    """Raised when input data does not match schema."""
    pass


class ArtifactLoadError(DeeploiError):
    """Raised when loading artifact from disk fails."""
    pass


class PredictionError(DeeploiError):
    """Raised when prediction fails."""
    pass


class SerializationError(DeeploiError):
    """Raised when serialization/deserialization fails."""
    pass
