"""
Core DeeploiPackage class for managing trained models.
"""

import os
import pandas as pd
from typing import Any, List, Dict, Optional, Union
from deeploi.types import Schema, Metadata, PredictionResponse
from deeploi.exceptions import (
    UnsupportedModelError,
    InvalidSampleError,
    SchemaValidationError,
    PredictionError,
)
from deeploi.inspector import inspect_model
from deeploi.schema import infer_schema, validate_batch
from deeploi.metadata import create_metadata
from deeploi.serialization import save_model, load_model
from deeploi.utils import (
    validate_dataframe,
    to_records,
    from_records,
    select_columns,
    save_json,
    load_json,
    save_text,
    load_text,
    ensure_dir,
)
from deeploi.constants import (
    MODEL_FILE,
    METADATA_FILE,
    SCHEMA_FILE,
    MANIFEST_FILE,
    REQUIREMENTS_FILE,
    CLASSIFICATION,
)


class DeeploiPackage:
    """
    A packaged trained model ready for deployment.
    
    Encapsulates:
    - The trained model (sklearn or XGBoost)
    - Schema (feature metadata, column order)
    - Metadata (framework, task type, versions)
    - Methods for predict, save, load, serve
    """
    
    def __init__(
        self,
        model: Any,
        schema: Schema,
        metadata: Metadata,
    ):
        """
        Initialize DeeploiPackage.
        
        Args:
            model: Trained scikit-learn or XGBoost model
            schema: Input schema from inference
            metadata: Model metadata
        """
        self.model = model
        self.schema = schema
        self.metadata = metadata
    
    @classmethod
    def from_model(cls, model: Any, sample: pd.DataFrame) -> "DeeploiPackage":
        """
        Create a DeeploiPackage from a trained model and sample input.
        
        Args:
            model: Trained scikit-learn or XGBoost model
            sample: Sample DataFrame with feature columns
        
        Returns:
            DeeploiPackage instance
        
        Raises:
            UnsupportedModelError: If model is not supported
            InvalidSampleError: If sample is invalid
        """
        # Validate sample
        sample = validate_dataframe(sample)
        
        # Inspect model
        framework, model_type, task_type, supports_proba = inspect_model(model)
        
        # Infer schema
        schema = infer_schema(sample)
        
        # Create metadata
        metadata = create_metadata(framework, task_type, supports_proba, model)
        
        return cls(model=model, schema=schema, metadata=metadata)
    
    def predict(
        self,
        X: Union[pd.DataFrame, List[Dict[str, Any]]],
        include_probabilities: bool = False,
    ) -> PredictionResponse:
        """
        Make predictions on input data.
        
        Args:
            X: Input data (DataFrame or list of record dicts)
            include_probabilities: Whether to include predict_proba (for classifiers)
        
        Returns:
            PredictionResponse with predictions and optional probabilities
        
        Raises:
            SchemaValidationError: If input doesn't match schema
            PredictionError: If prediction fails
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(X, list):
                validate_batch(X, self.schema)
                df = from_records(X)
            else:
                df = validate_dataframe(X)
            
            # Reorder columns and select
            df = select_columns(df, self.schema.column_order)
            
        except (InvalidSampleError, SchemaValidationError) as e:
            raise SchemaValidationError(f"Input validation failed: {str(e)}")
        
        try:
            # Make predictions
            preds = self.model.predict(df)
            response = PredictionResponse(predictions=preds.tolist())
            
            # Add probabilities if requested and available
            if include_probabilities and self.metadata.supports_predict_proba:
                probs = self.model.predict_proba(df)
                
                # Get class labels if available
                if hasattr(self.model, "classes_"):
                    classes = self.model.classes_
                else:
                    # Infer from proba shape
                    classes = list(range(probs.shape[1]))
                
                # Convert to list of dicts
                response.probabilities = [
                    {str(cls): float(prob) for cls, prob in zip(classes, row)}
                    for row in probs
                ]
            
            return response
        
        except Exception as e:
            raise PredictionError(f"Prediction failed: {str(e)}")
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Get class probabilities (for classifiers only).
        
        Args:
            X: Input data
        
        Returns:
            Dict with probabilities
        
        Raises:
            PredictionError: If model doesn't support predict_proba
        """
        if not self.metadata.supports_predict_proba:
            raise PredictionError(
                f"Model {self.metadata.estimator_class} does not support predict_proba"
            )
        
        return self.predict(X, include_probabilities=True).to_dict()
    
    def save(self, path: str) -> None:
        """
        Save package to disk.
        
        Creates artifact structure:
        path/
        ├── model.joblib
        ├── metadata.json
        ├── schema.json
        ├── deeploi.json
        └── requirements.txt
        
        Args:
            path: Directory path to save to
        """
        ensure_dir(path)
        
        # Save model
        model_path = os.path.join(path, MODEL_FILE)
        save_model(self.model, model_path)
        
        # Save metadata
        metadata_path = os.path.join(path, METADATA_FILE)
        save_json(self.metadata.to_dict(), metadata_path)
        
        # Save schema
        schema_path = os.path.join(path, SCHEMA_FILE)
        save_json(self.schema.to_dict(), schema_path)
        
        # Save manifest
        manifest = {
            "deeploi_version": self.metadata.deeploi_version,
            "framework": self.metadata.framework,
            "serializer": "joblib",
        }
        manifest_path = os.path.join(path, MANIFEST_FILE)
        save_json(manifest, manifest_path)
        
        # Save requirements
        requirements_path = os.path.join(path, REQUIREMENTS_FILE)
        requirements_text = self._generate_requirements()
        save_text(requirements_text, requirements_path)
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt content from library versions."""
        lines = []
        for lib, version in self.metadata.library_versions.items():
            if version != "not installed" and version != "unknown":
                lines.append(f"{lib}=={version}")
        return "\n".join(sorted(lines)) + "\n"
    
    def serve(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """
        Launch FastAPI server for this package.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        from deeploi.serving import create_app
        import uvicorn
        
        app = create_app(self)
        uvicorn.run(app, host=host, port=port)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DeeploiPackage("
            f"framework={self.metadata.framework}, "
            f"estimator={self.metadata.estimator_class}, "
            f"task={self.metadata.task_type}"
            f")"
        )
