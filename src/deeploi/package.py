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
    DOCKERFILE_FILE,
    DOCKERIGNORE_FILE,
    DOCKER_APP_FILE,
    CLASSIFICATION,
    PACKAGE_NAME_MAPPING,
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
    def from_model(
        cls,
        model: Any,
        sample: Optional[pd.DataFrame] = None,
    ) -> "DeeploiPackage":
        """
        Create a DeeploiPackage from a trained model and sample input.
        
        Args:
            model: Trained scikit-learn or XGBoost model
            sample: Optional sample DataFrame with feature columns
        
        Returns:
            DeeploiPackage instance
        
        Raises:
            UnsupportedModelError: If model is not supported
            InvalidSampleError: If sample is invalid
        """
        # Inspect model
        framework, model_type, task_type, supports_proba = inspect_model(model)
        
        # Validate sample when provided and otherwise fall back to model metadata.
        validated_sample = validate_dataframe(sample) if sample is not None else None

        # Infer schema
        schema = infer_schema(sample=validated_sample, model=model)
        
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
    
    def save(
        self,
        path: str,
        generate_docker: bool = False,
        docker_port: int = 8000,
        docker_python_image: str = "3.11-slim",
    ) -> None:
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
            generate_docker: Whether to also generate Docker files
            docker_port: Container port used if generate_docker=True
            docker_python_image: Python image tag if generate_docker=True
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

        if generate_docker:
            self.generate_docker(
                path=path,
                port=docker_port,
                python_image=docker_python_image,
            )

    def generate_docker(
        self,
        path: str,
        port: int = 8000,
        python_image: str = "3.11-slim",
    ) -> None:
        """
        Generate Docker files inside a saved artifact directory.

        Generated files:
        path/
        ├── Dockerfile
        ├── .dockerignore
        └── serve.py

        Args:
            path: Saved artifact directory
            port: Container port exposed by Uvicorn
            python_image: Python base image tag (e.g. "3.11-slim")

        Raises:
            ValueError: If artifact is missing required files or args are invalid
        """
        if port <= 0:
            raise ValueError("port must be a positive integer")

        if not python_image or not isinstance(python_image, str):
            raise ValueError("python_image must be a non-empty string")

        required_files = [MODEL_FILE, METADATA_FILE, SCHEMA_FILE, REQUIREMENTS_FILE]
        missing_files = [
            name
            for name in required_files
            if not os.path.exists(os.path.join(path, name))
        ]
        if missing_files:
            missing = ", ".join(sorted(missing_files))
            raise ValueError(
                f"Cannot generate Docker files. Missing artifact files: {missing}. "
                f"Call save('{path}') first."
            )

        dockerfile_content = self._generate_dockerfile(port=port, python_image=python_image)
        dockerignore_content = self._generate_dockerignore()
        serve_app_content = self._generate_serve_app()

        save_text(dockerfile_content, os.path.join(path, DOCKERFILE_FILE))
        save_text(dockerignore_content, os.path.join(path, DOCKERIGNORE_FILE))
        save_text(serve_app_content, os.path.join(path, DOCKER_APP_FILE))
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt content from library versions."""
        lines = []
        for lib, version in self.metadata.library_versions.items():
            if version != "not installed" and version != "unknown":
                # Map internal package names to PyPI package names
                pypi_name = PACKAGE_NAME_MAPPING.get(lib, lib)
                lines.append(f"{pypi_name}=={version}")
        return "\n".join(sorted(lines)) + "\n"

    def _generate_dockerfile(self, port: int, python_image: str) -> str:
        """Generate Dockerfile content for artifact serving."""
        return (
            f"FROM python:{python_image}\n"
            "\n"
            "WORKDIR /app\n"
            "\n"
            "COPY requirements.txt /app/requirements.txt\n"
            "RUN pip install --no-cache-dir --upgrade pip && \\\n"
            "    pip install --no-cache-dir -r /app/requirements.txt && \\\n"
            "    pip install --no-cache-dir deeploi\n"
            "\n"
            "COPY . /app/artifact\n"
            "COPY serve.py /app/serve.py\n"
            "\n"
            f"EXPOSE {port}\n"
            "\n"
            f"CMD [\"uvicorn\", \"serve:app\", \"--host\", \"0.0.0.0\", \"--port\", \"{port}\"]\n"
        )

    def _generate_dockerignore(self) -> str:
        """Generate .dockerignore content for clean Docker builds."""
        return (
            "__pycache__/\n"
            "*.pyc\n"
            ".pytest_cache/\n"
            ".DS_Store\n"
            ".git/\n"
        )

    def _generate_serve_app(self) -> str:
        """Generate lightweight ASGI entrypoint for Docker."""
        return (
            "from deeploi import load\n"
            "from deeploi.serving import create_app\n"
            "\n"
            "pkg = load(\"/app/artifact\")\n"
            "app = create_app(pkg)\n"
        )
    
    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        require_auth: Optional[bool] = None,
        api_key: Optional[str] = None,
        auth_header_name: Optional[str] = None,
        auto_generate_api_key: Optional[bool] = None,
    ) -> None:
        """
        Launch FastAPI server for this package.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            require_auth: Whether to require API key auth. If None, uses
                DEEPLOI_AUTH_ENABLED.
            api_key: API key value. If None, uses DEEPLOI_AUTH_API_KEY.
            auth_header_name: Header name for API key. If None, uses
                DEEPLOI_AUTH_HEADER (default: X-API-Key).
            auto_generate_api_key: Whether to auto-generate an API key when auth
                is enabled and no key is provided. If None, uses
                DEEPLOI_AUTH_AUTO_GENERATE (default: True).
        """
        from deeploi.serving import create_app
        import uvicorn
        
        app = create_app(
            self,
            require_auth=require_auth,
            api_key=api_key,
            auth_header_name=auth_header_name,
            auto_generate_api_key=auto_generate_api_key,
        )
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
