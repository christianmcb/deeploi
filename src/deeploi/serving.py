"""
FastAPI server factory for serving predictions.
"""

import json
import pathlib
import os
import secrets
import time
import io
import csv
import warnings
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Header, Query, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
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
    feature_count: int
    training_set_size: Optional[int] = None
    class_labels: Optional[List[str]] = None
    feature_importance: Optional[Dict[str, float]] = None


def _extract_training_set_size(model: Any) -> Optional[int]:
    """Best-effort extraction of the number of training rows seen during fit."""
    candidate_attrs = [
        "n_samples_fit_",
        "n_samples_",
        "_n_samples",
    ]
    for attr in candidate_attrs:
        value = getattr(model, attr, None)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return None


def _extract_class_labels(model: Any) -> Optional[List[str]]:
    """Best-effort class label extraction for classifiers."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None
    try:
        return [str(label) for label in list(classes)]
    except Exception:
        return None


def _extract_feature_importance(model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
    """Best-effort extraction of feature importance values."""
    # sklearn / lightgbm / catboost wrappers commonly expose feature_importances_
    raw_importances = getattr(model, "feature_importances_", None)
    if raw_importances is not None:
        try:
            values = list(raw_importances)
            if len(values) == len(feature_names):
                return {
                    name: float(value)
                    for name, value in zip(feature_names, values)
                }
        except Exception:
            pass

    # xgboost core booster fallback
    booster = getattr(model, "get_booster", None)
    if callable(booster):
        try:
            score = booster().get_score(importance_type="gain")
        except Exception:
            score = None
        if score:
            mapped: Dict[str, float] = {}
            for feature_name in feature_names:
                mapped[feature_name] = float(score.get(feature_name, 0.0))
            if any(value > 0 for value in mapped.values()):
                return mapped

    return None


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
    app.state.prediction_history = []
    app.state.max_prediction_history = 200

    def _record_prediction_history(
        endpoint: str,
        request_payload: Any,
        status_code: int,
        response_body: Dict[str, Any],
        latency_ms: int,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "status": status_code,
            "ok": 200 <= status_code < 300,
            "latency_ms": latency_ms,
            "request": request_payload,
            "response": response_body,
        }
        app.state.prediction_history.append(entry)

        overflow = len(app.state.prediction_history) - app.state.max_prediction_history
        if overflow > 0:
            del app.state.prediction_history[:overflow]

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
            raise HTTPException(
                status_code=401,
                detail=(
                    "Invalid or missing API key. "
                    f"Provide header '{resolved_auth_header}' or query param 'api_key'"
                ),
            )

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
        feature_names = package.schema.column_order
        return MetadataResponse(
            framework=package.metadata.framework,
            estimator_class=package.metadata.estimator_class,
            model_type=package.metadata.estimator_class,
            task_type=package.metadata.task_type,
            supports_predict_proba=package.metadata.supports_predict_proba,
            python_version=package.metadata.python_version,
            deeploi_version=package.metadata.deeploi_version,
            created_at=package.metadata.created_at,
            features=feature_names,
            dtypes=[f.dtype for f in package.schema.features],
            feature_count=len(feature_names),
            training_set_size=_extract_training_set_size(package.model),
            class_labels=_extract_class_labels(package.model),
            feature_importance=_extract_feature_importance(package.model, feature_names),
        )

    @app.get("/history/summary")
    def prediction_history_summary(
        x_api_key: Optional[str] = Header(default=None, alias=resolved_auth_header),
    ):
        """Return aggregate stats derived from in-memory prediction history."""
        if auth_enabled and x_api_key != resolved_api_key:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Invalid or missing API key. "
                    f"Provide header '{resolved_auth_header}'"
                ),
            )

        history = app.state.prediction_history
        total = len(history)
        ok_count = sum(1 for item in history if item.get("ok"))
        error_count = total - ok_count
        latencies = [
            int(item.get("latency_ms", 0))
            for item in history
            if isinstance(item.get("latency_ms", None), int)
        ]
        avg_latency_ms = round(sum(latencies) / len(latencies), 2) if latencies else None

        return {
            "predictions_served": total,
            "successful_requests": ok_count,
            "failed_requests": error_count,
            "avg_response_time_ms": avg_latency_ms,
        }

    @app.get("/history")
    def prediction_history(
        limit: int = Query(default=25, ge=1, le=200),
        x_api_key: Optional[str] = Header(default=None, alias=resolved_auth_header),
    ):
        """Return recent prediction request/response events from in-memory history."""
        if auth_enabled and x_api_key != resolved_api_key:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Invalid or missing API key. "
                    f"Provide header '{resolved_auth_header}'"
                ),
            )

        items = list(reversed(app.state.prediction_history[-limit:]))
        return {
            "count": len(items),
            "limit": limit,
            "items": items,
        }

    @app.delete("/history")
    def clear_prediction_history(
        x_api_key: Optional[str] = Header(default=None, alias=resolved_auth_header),
    ):
        """Clear in-memory prediction history entries."""
        if auth_enabled and x_api_key != resolved_api_key:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Invalid or missing API key. "
                    f"Provide header '{resolved_auth_header}'"
                ),
            )

        cleared = len(app.state.prediction_history)
        app.state.prediction_history.clear()
        return {"cleared": cleared}
    
    @app.post("/predict", response_model=PredictionResponseModel)
    def predict(request: PredictionRequest, x_api_key: Optional[str] = Header(default=None, alias=resolved_auth_header)):
        """Make predictions on input data."""
        if auth_enabled and x_api_key != resolved_api_key:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Invalid or missing API key. "
                    f"Provide header '{resolved_auth_header}'"
                ),
            )

        started = time.perf_counter()
        try:
            response = package.predict(request.records, include_probabilities=True)
            response_body = {
                "predictions": response.predictions,
                "probabilities": response.probabilities,
            }
            latency_ms = int((time.perf_counter() - started) * 1000)
            _record_prediction_history(
                endpoint="/predict",
                request_payload={"records": request.records},
                status_code=200,
                response_body=response_body,
                latency_ms=latency_ms,
            )
            return PredictionResponseModel(
                predictions=response.predictions,
                probabilities=response.probabilities,
            )
        except (SchemaValidationError, InvalidSampleError) as e:
            latency_ms = int((time.perf_counter() - started) * 1000)
            _record_prediction_history(
                endpoint="/predict",
                request_payload={"records": request.records},
                status_code=400,
                response_body={"detail": str(e)},
                latency_ms=latency_ms,
            )
            raise HTTPException(status_code=400, detail=str(e))
        except PredictionError as e:
            latency_ms = int((time.perf_counter() - started) * 1000)
            _record_prediction_history(
                endpoint="/predict",
                request_payload={"records": request.records},
                status_code=500,
                response_body={"detail": str(e)},
                latency_ms=latency_ms,
            )
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict_proba", response_model=PredictionResponseModel)
    def predict_proba(request: PredictionRequest, x_api_key: Optional[str] = Header(default=None, alias=resolved_auth_header)):
        """Get class probabilities (classifiers only)."""
        if auth_enabled and x_api_key != resolved_api_key:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Invalid or missing API key. "
                    f"Provide header '{resolved_auth_header}'"
                ),
            )

        if not package.metadata.supports_predict_proba:
            _record_prediction_history(
                endpoint="/predict_proba",
                request_payload={"records": request.records},
                status_code=400,
                response_body={
                    "detail": (
                        f"Model {package.metadata.estimator_class} does not support predict_proba. "
                        "Use POST /predict instead."
                    )
                },
                latency_ms=0,
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model {package.metadata.estimator_class} does not support predict_proba. "
                    "Use POST /predict instead."
                ),
            )
        
        started = time.perf_counter()
        try:
            response = package.predict(request.records, include_probabilities=True)
            response_body = {
                "predictions": response.predictions,
                "probabilities": response.probabilities,
            }
            latency_ms = int((time.perf_counter() - started) * 1000)
            _record_prediction_history(
                endpoint="/predict_proba",
                request_payload={"records": request.records},
                status_code=200,
                response_body=response_body,
                latency_ms=latency_ms,
            )
            return PredictionResponseModel(
                predictions=response.predictions,
                probabilities=response.probabilities,
            )
        except (SchemaValidationError, InvalidSampleError) as e:
            latency_ms = int((time.perf_counter() - started) * 1000)
            _record_prediction_history(
                endpoint="/predict_proba",
                request_payload={"records": request.records},
                status_code=400,
                response_body={"detail": str(e)},
                latency_ms=latency_ms,
            )
            raise HTTPException(status_code=400, detail=str(e))
        except PredictionError as e:
            latency_ms = int((time.perf_counter() - started) * 1000)
            _record_prediction_history(
                endpoint="/predict_proba",
                request_payload={"records": request.records},
                status_code=500,
                response_body={"detail": str(e)},
                latency_ms=latency_ms,
            )
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict-csv", response_model=PredictionResponseModel)
    async def predict_csv(
        file: UploadFile = File(..., description="CSV file upload for batch prediction"),
        include_probabilities: bool = Query(default=True),
        x_api_key: Optional[str] = Header(default=None, alias=resolved_auth_header),
    ):
        """Make batch predictions from an uploaded CSV file."""
        if auth_enabled and x_api_key != resolved_api_key:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Invalid or missing API key. "
                    f"Provide header '{resolved_auth_header}'"
                ),
            )

        started = time.perf_counter()
        filename = file.filename or "uploaded.csv"
        if not filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail=f"Expected a .csv file upload. Received: {filename}",
            )

        try:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded CSV file is empty")

            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Could not decode CSV as UTF-8. Save the file as UTF-8 CSV and retry. "
                        "If exported from Excel, choose UTF-8 CSV format."
                    ),
                )

            if not text.strip():
                raise HTTPException(status_code=400, detail="Uploaded CSV file is empty")

            header_reader = csv.reader(io.StringIO(text))
            header = next(header_reader, None)
            if not header:
                raise HTTPException(
                    status_code=400,
                    detail="CSV header row is missing. Add a first row with feature names.",
                )

            normalized_header = [str(col).strip() for col in header]
            if any(not col for col in normalized_header):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "CSV contains empty column names in the header row. "
                        "Provide a non-empty name for every feature column."
                    ),
                )

            duplicate_columns = sorted(
                {
                    col
                    for col in normalized_header
                    if normalized_header.count(col) > 1
                }
            )
            if duplicate_columns:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "CSV contains duplicate column names: "
                        f"{', '.join(duplicate_columns)}. "
                        "Each feature column must be unique."
                    ),
                )

            if all(col.replace(".", "", 1).isdigit() for col in normalized_header):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "CSV header appears to contain data values instead of feature names. "
                        "Include a header row matching model feature names."
                    ),
                )

            dataframe = pd.read_csv(io.StringIO(text))
            if dataframe.empty:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded CSV contains no rows. Provide at least one data row.",
                )

            response = package.predict(
                dataframe,
                include_probabilities=include_probabilities,
            )
            response_body = {
                "predictions": response.predictions,
                "probabilities": response.probabilities,
            }
            latency_ms = int((time.perf_counter() - started) * 1000)
            _record_prediction_history(
                endpoint="/predict-csv",
                request_payload={
                    "csv_filename": filename,
                    "row_count": int(len(dataframe)),
                    "columns": [str(column) for column in dataframe.columns],
                },
                status_code=200,
                response_body=response_body,
                latency_ms=latency_ms,
            )
            return PredictionResponseModel(
                predictions=response.predictions,
                probabilities=response.probabilities,
            )
        except HTTPException:
            raise
        except (SchemaValidationError, InvalidSampleError) as e:
            latency_ms = int((time.perf_counter() - started) * 1000)
            _record_prediction_history(
                endpoint="/predict-csv",
                request_payload={"csv_filename": filename},
                status_code=400,
                response_body={"detail": str(e)},
                latency_ms=latency_ms,
            )
            raise HTTPException(status_code=400, detail=str(e))
        except PredictionError as e:
            latency_ms = int((time.perf_counter() - started) * 1000)
            _record_prediction_history(
                endpoint="/predict-csv",
                request_payload={"csv_filename": filename},
                status_code=500,
                response_body={"detail": str(e)},
                latency_ms=latency_ms,
            )
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            latency_ms = int((time.perf_counter() - started) * 1000)
            _record_prediction_history(
                endpoint="/predict-csv",
                request_payload={"csv_filename": filename},
                status_code=400,
                response_body={"detail": str(e)},
                latency_ms=latency_ms,
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not parse CSV upload. Ensure the file is valid UTF-8 CSV with a header row. "
                    f"Original error: {str(e)}"
                ),
            )
    
    return app
