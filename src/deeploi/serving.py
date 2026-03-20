"""
FastAPI server factory for serving predictions.
"""

import json
import pathlib
import os
import secrets
import warnings
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from deeploi.package import DeeploiPackage
from deeploi.constants import __version__

_DASHBOARD = pathlib.Path(__file__).parent / "dashboard.html"
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
    model_type: str
    task_type: str
    supports_predict_proba: bool
    python_version: str
    deeploi_version: str
    created_at: str
    features: List[str]
    dtypes: List[str]


def _build_dashboard_html(
    auth_enabled: bool,
    auth_header_name: str,
    auth_key: Optional[str],
) -> str:
    """Inject server auth configuration into the static dashboard."""
    if not _DASHBOARD.exists():
        return (
            "<html><head><title>Deeploi API</title></head><body>"
            "<h1>Deeploi API</h1>"
            "<p><a href='/docs'>API Documentation</a></p>"
            "</body></html>"
        )

    content = _DASHBOARD.read_text()
    config = {
        "authEnabled": auth_enabled,
        "authHeaderName": auth_header_name,
        "authKey": auth_key or "",
    }
    return content.replace(
        "</head>",
        (
            "<script>window.__DEEPLOI_CONFIG__ = "
            f"{json.dumps(config)};"
            "</script></head>"
        ),
        1,
    )


def _is_authorized(
    auth_enabled: bool,
    provided_key: Optional[str],
    expected_key: Optional[str],
) -> bool:
    """Check whether the provided API key satisfies auth requirements."""
    if not auth_enabled:
        return True
    return provided_key == expected_key


def _parse_env_bool(var_name: str) -> Optional[bool]:
    """Parse a boolean env var. Returns None when unset."""
    raw = os.getenv(var_name)
    if raw is None:
        return None

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"Invalid value for {var_name}: '{raw}'. "
        "Use true/false, 1/0, yes/no, or on/off."
    )


def create_app(
    package: DeeploiPackage,
    require_auth: Optional[bool] = None,
    api_key: Optional[str] = None,
    auth_header_name: Optional[str] = None,
    auto_generate_api_key: Optional[bool] = None,
) -> FastAPI:
    """
    Create a FastAPI app for the given package.
    
    Args:
        package: DeeploiPackage instance
        require_auth: Whether to require API key auth. If None, reads
            DEEPLOI_AUTH_ENABLED (default False when unset).
        api_key: API key value. If None and auth is enabled, reads
            DEEPLOI_AUTH_API_KEY.
        auth_header_name: Header name for API key. If None, reads
            DEEPLOI_AUTH_HEADER (default: X-API-Key).
        auto_generate_api_key: Whether to auto-generate an API key when auth is
            enabled but no key is provided. If None, reads
            DEEPLOI_AUTH_AUTO_GENERATE (default True when unset).
    
    Returns:
        FastAPI application
    """
    env_require_auth = _parse_env_bool("DEEPLOI_AUTH_ENABLED")
    auth_enabled = require_auth if require_auth is not None else env_require_auth
    if auth_enabled is None:
        auth_enabled = False

    resolved_api_key = api_key if api_key is not None else os.getenv("DEEPLOI_AUTH_API_KEY")
    resolved_auth_header = (
        auth_header_name
        if auth_header_name is not None
        else os.getenv("DEEPLOI_AUTH_HEADER", "X-API-Key")
    )
    env_auto_generate = _parse_env_bool("DEEPLOI_AUTH_AUTO_GENERATE")
    resolved_auto_generate = auto_generate_api_key
    if resolved_auto_generate is None:
        resolved_auto_generate = True if env_auto_generate is None else env_auto_generate

    if auth_enabled and not resolved_api_key:
        if resolved_auto_generate:
            resolved_api_key = secrets.token_urlsafe(32)
            warnings.warn(
                "Authentication is enabled but no API key was provided. "
                "A secure API key was auto-generated for this process only. "
                f"Use header '{resolved_auth_header}' with key: {resolved_api_key}",
                stacklevel=2,
            )
        else:
            raise ValueError(
                "Authentication is enabled but no API key was provided. "
                "Set api_key or DEEPLOI_AUTH_API_KEY, or enable auto-generation."
            )

    app = FastAPI(
        title="Deeploi Model Server",
        description=f"{package.metadata.estimator_class} ({package.metadata.framework})",
        version=__version__,
    )

    app.state.auth_enabled = auth_enabled
    app.state.auth_header_name = resolved_auth_header
    app.state.generated_api_key = resolved_api_key if auth_enabled else None

    # Enable CORS for browser-based API calls
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/routes", include_in_schema=False)
    def list_routes():
        """Debug: list all registered routes."""
        routes = [
            {"path": route.path, "methods": route.methods}
            for route in app.routes
            if hasattr(route, "path")
        ]
        return {"routes": routes}

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard(
        api_key_query: Optional[str] = Query(default=None, alias="api_key"),
        x_api_key: Optional[str] = Header(default=None, alias=resolved_auth_header),
    ):
        """Serve the interactive API dashboard."""
        provided_key = x_api_key or api_key_query
        if not _is_authorized(auth_enabled, provided_key, resolved_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        return _build_dashboard_html(
            auth_enabled=auth_enabled,
            auth_header_name=resolved_auth_header,
            auth_key=provided_key if auth_enabled else None,
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
            model_type=package.metadata.estimator_class,
            task_type=package.metadata.task_type,
            supports_predict_proba=package.metadata.supports_predict_proba,
            python_version=package.metadata.python_version,
            deeploi_version=package.metadata.deeploi_version,
            created_at=package.metadata.created_at,
            features=package.schema.column_order,
            dtypes=[f.dtype for f in package.schema.features],
        )
    
    @app.post("/predict", response_model=PredictionResponseModel)
    def predict(request: PredictionRequest, x_api_key: Optional[str] = Header(default=None, alias=resolved_auth_header)):
        """Make predictions on input data."""
        if auth_enabled and x_api_key != resolved_api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

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
    def predict_proba(request: PredictionRequest, x_api_key: Optional[str] = Header(default=None, alias=resolved_auth_header)):
        """Get class probabilities (classifiers only)."""
        if auth_enabled and x_api_key != resolved_api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

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
