"""
Deeploi: One-line deployment for trained tabular ML models.

Simple, powerful, local-first model serving for scikit-learn, XGBoost, and LightGBM.

Basic usage:
    from deeploi import deploy
    
    deploy(model, sample=X_train)

Or for more control:
    from deeploi import package, load
    
    pkg = package(model, sample=X_train)
    pkg.save("artifacts/model")
    pkg.serve(port=8000)
    
    loaded = load("artifacts/model")
    predictions = loaded.predict(X_test)
"""

from deeploi.api import deploy, package, load
from deeploi.constants import __version__
from deeploi.package import DeeploiPackage
from deeploi.exceptions import (
    DeeploiError,
    UnsupportedModelError,
    InvalidSampleError,
    SchemaValidationError,
    ArtifactLoadError,
    PredictionError,
    SerializationError,
)

__all__ = [
    # Public API
    "deploy",
    "package",
    "load",
    # Core class
    "DeeploiPackage",
    # Exceptions
    "DeeploiError",
    "UnsupportedModelError",
    "InvalidSampleError",
    "SchemaValidationError",
    "ArtifactLoadError",
    "PredictionError",
    "SerializationError",
]
