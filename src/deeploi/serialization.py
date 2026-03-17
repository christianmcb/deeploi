"""
Model serialization and deserialization.
"""

import joblib
import os
from typing import Any
from deeploi.exceptions import SerializationError, ArtifactLoadError
from deeploi.constants import MODEL_FILE


def save_model(model: Any, filepath: str) -> None:
    """
    Save model using joblib.
    
    Args:
        model: The model object
        filepath: Path to save to
    
    Raises:
        SerializationError: If save fails
    """
    try:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        joblib.dump(model, filepath, compress=3)
    except Exception as e:
        raise SerializationError(f"Failed to save model: {str(e)}")


def load_model(filepath: str) -> Any:
    """
    Load model from joblib file.
    
    Args:
        filepath: Path to load from
    
    Returns:
        Loaded model object
    
    Raises:
        ArtifactLoadError: If load fails
    """
    try:
        return joblib.load(filepath)
    except Exception as e:
        raise ArtifactLoadError(f"Failed to load model from {filepath}: {str(e)}")
