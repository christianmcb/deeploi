"""
Public API entry points for Deeploi.

Provides:
- deploy(): One-liner for immediate serving
- package(): Reusable package object
- load(): Load saved artifacts
"""

import pandas as pd
from typing import Any, Union, Optional
from deeploi.package import DeeploiPackage
from deeploi.loader import load as load_artifact


def package(
    model: Any,
    sample: pd.DataFrame,
) -> DeeploiPackage:
    """
    Create a reusable DeeploiPackage from a trained model.
    
    Args:
        model: Trained scikit-learn or XGBoost model
        sample: Sample DataFrame with feature columns for schema inference
    
    Returns:
        DeeploiPackage instance
    
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from deeploi import package
        >>> 
        >>> model = RandomForestClassifier().fit(X_train, y_train)
        >>> pkg = package(model, X_train)
        >>> pkg.save("artifacts/iris_rf")
        >>> preds = pkg.predict(X_test)
    """
    return DeeploiPackage.from_model(model, sample)


def deploy(
    model: Any,
    sample: pd.DataFrame,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """
    One-liner to deploy a trained model as a local API.
    
    Infers schema, packages the model, and starts a FastAPI server.
    
    Args:
        model: Trained scikit-learn or XGBoost model
        sample: Sample DataFrame with feature columns
        host: Host to bind to (default: 127.0.0.1)
        port: Port to bind to (default: 8000)
    
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from deeploi import deploy
        >>> 
        >>> model = RandomForestClassifier().fit(X_train, y_train)
        >>> deploy(model, X_train, port=8000)
        # Server starts at http://127.0.0.1:8000
        # Try: curl -X POST http://127.0.0.1:8000/predict ...
    """
    pkg = package(model, sample)
    pkg.serve(host=host, port=port)


def load(path: str) -> DeeploiPackage:
    """
    Load a saved DeeploiPackage from disk.
    
    Args:
        path: Directory containing saved artifact
    
    Returns:
        Reconstructed DeeploiPackage instance
    
    Example:
        >>> from deeploi import load
        >>> 
        >>> pkg = load("artifacts/iris_rf")
        >>> preds = pkg.predict(X_test)
    """
    return load_artifact(path)
