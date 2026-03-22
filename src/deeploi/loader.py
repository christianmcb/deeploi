"""
Loading saved artifacts and reconstructing DeeploiPackage.
"""

import os
from typing import Optional
from deeploi.package import DeeploiPackage
from deeploi.types import Schema, Metadata
from deeploi.serialization import load_model
from deeploi.utils import load_json
from deeploi.exceptions import ArtifactLoadError
from deeploi.constants import (
    __version__,
    MODEL_FILE,
    METADATA_FILE,
    SCHEMA_FILE,
    MANIFEST_FILE,
)


def load(path: str) -> DeeploiPackage:
    """
    Load a saved DeeploiPackage from disk.
    
    Args:
        path: Directory containing saved artifact
    
    Returns:
        Reconstructed DeeploiPackage instance
    
    Raises:
        ArtifactLoadError: If artifact is corrupted or incomplete
    """
    if not os.path.isdir(path):
        raise ArtifactLoadError(
            f"Artifact directory not found: {path}. "
            "Pass a valid artifact directory created with pkg.save(path)."
        )
    
    try:
        # Load manifest to verify format
        manifest_path = os.path.join(path, MANIFEST_FILE)
        if os.path.exists(manifest_path):
            manifest = load_json(manifest_path)
        else:
            # Create manifest for backward compat
            manifest = {"deeploi_version": __version__}
        
        # Load model
        model_path = os.path.join(path, MODEL_FILE)
        if not os.path.exists(model_path):
            raise ArtifactLoadError(
                f"Missing required artifact file: {MODEL_FILE} at {model_path}. "
                "The artifact appears incomplete. Re-save it with pkg.save(path)."
            )
        model = load_model(model_path)
        
        # Load metadata
        metadata_path = os.path.join(path, METADATA_FILE)
        if not os.path.exists(metadata_path):
            raise ArtifactLoadError(
                f"Missing required artifact file: {METADATA_FILE} at {metadata_path}. "
                "The artifact appears incomplete. Re-save it with pkg.save(path)."
            )
        metadata_dict = load_json(metadata_path)
        metadata = Metadata.from_dict(metadata_dict)
        
        # Load schema
        schema_path = os.path.join(path, SCHEMA_FILE)
        if not os.path.exists(schema_path):
            raise ArtifactLoadError(
                f"Missing required artifact file: {SCHEMA_FILE} at {schema_path}. "
                "The artifact appears incomplete. Re-save it with pkg.save(path)."
            )
        schema_dict = load_json(schema_path)
        schema = Schema.from_dict(schema_dict)
        
        # Reconstruct package
        return DeeploiPackage(
            model=model,
            schema=schema,
            metadata=metadata,
        )
    
    except ArtifactLoadError:
        raise
    except Exception as e:
        raise ArtifactLoadError(
            "Failed to load artifact from "
            f"{path}. Ensure {MODEL_FILE}, {METADATA_FILE}, and {SCHEMA_FILE} are valid. "
            f"Original error: {str(e)}"
        )


def artifact_exists(path: str) -> bool:
    """Check if a valid artifact exists at path."""
    if not os.path.isdir(path):
        return False
    
    required_files = [MODEL_FILE, METADATA_FILE, SCHEMA_FILE]
    return all(
        os.path.exists(os.path.join(path, f))
        for f in required_files
    )
