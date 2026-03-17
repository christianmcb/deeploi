"""
FastAPI server factory for serving predictions.
"""

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from deeploi.package import DeeploiPackage
from deeploi.exceptions import SchemaValidationError, PredictionError, InvalidSampleError


class PredictionRequest(BaseModel):
    """Input request for prediction."""
    records: List[Dict[str, Any]] = Field(
        ...,
        description="List of records with feature values",
        json_schema_extra={
            "example": [
                {"feature_1": 1.2, "feature_2": "A"},
                {"feature_1": 2.3, "feature_2": "B"},
            ]
        },
    )


class PredictionResponseModel(BaseModel):
    """Output response for prediction."""
    predictions: List[Any] = Field(..., description="List of predictions")
    probabilities: Optional[List[Dict[str, float]]] = Field(
        None,
        description="Class probabilities (for classifiers with predict_proba)",
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="Deeploi version")


class MetadataResponse(BaseModel):
    """Model metadata response."""
    framework: str
    estimator_class: str
    task_type: str
    supports_predict_proba: bool
    python_version: str
    deeploi_version: str
    created_at: str


def create_app(package: DeeploiPackage) -> FastAPI:
    """
    Create a FastAPI app for the given package.
    
    Args:
        package: DeeploiPackage instance
    
    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Deeploi Model Server",
        description=f"{package.metadata.estimator_class} ({package.metadata.framework})",
        version="0.1.0",
    )
    
    @app.get("/health", response_model=HealthResponse)
    def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="ok",
            version=package.metadata.deeploi_version,
        )
    
    @app.get("/meta", response_model=MetadataResponse)
    def metadata():
        """Model metadata endpoint."""
        return MetadataResponse(
            framework=package.metadata.framework,
            estimator_class=package.metadata.estimator_class,
            task_type=package.metadata.task_type,
            supports_predict_proba=package.metadata.supports_predict_proba,
            python_version=package.metadata.python_version,
            deeploi_version=package.metadata.deeploi_version,
            created_at=package.metadata.created_at,
        )
    
    @app.post("/predict", response_model=PredictionResponseModel)
    def predict(request: PredictionRequest):
        """Make predictions on input data."""
        try:
            response = package.predict(request.records, include_probabilities=True)
            return PredictionResponseModel(
                predictions=response.predictions,
                probabilities=response.probabilities,
            )
        except (SchemaValidationError, InvalidSampleError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except PredictionError as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict_proba", response_model=PredictionResponseModel)
    def predict_proba(request: PredictionRequest):
        """Get class probabilities (classifiers only)."""
        if not package.metadata.supports_predict_proba:
            raise HTTPException(
                status_code=400,
                detail=f"Model does not support predict_proba",
            )
        
        try:
            response = package.predict(request.records, include_probabilities=True)
            return PredictionResponseModel(
                predictions=response.predictions,
                probabilities=response.probabilities,
            )
        except (SchemaValidationError, InvalidSampleError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except PredictionError as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app
