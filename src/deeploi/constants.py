"""
Constants for Deeploi.
"""

__version__ = "0.3.0"

# Supported frameworks
SKLEARN = "sklearn"
XGBOOST = "xgboost"
LIGHTGBM = "lightgbm"
CATBOOST = "catboost"
SUPPORTED_FRAMEWORKS = {SKLEARN, XGBOOST, LIGHTGBM, CATBOOST}

# Task types
CLASSIFICATION = "classification"
REGRESSION = "regression"
TASK_TYPES = {CLASSIFICATION, REGRESSION}

# Model types
SKLEARN_CLASSIFIER = "sklearn_classifier"
SKLEARN_REGRESSOR = "sklearn_regressor"
XGBOOST_CLASSIFIER = "xgboost_classifier"
XGBOOST_REGRESSOR = "xgboost_regressor"
LIGHTGBM_CLASSIFIER = "lightgbm_classifier"
LIGHTGBM_REGRESSOR = "lightgbm_regressor"
CATBOOST_CLASSIFIER = "catboost_classifier"
CATBOOST_REGRESSOR = "catboost_regressor"

# Artifact file names
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
SCHEMA_FILE = "schema.json"
MANIFEST_FILE = "deeploi.json"
REQUIREMENTS_FILE = "requirements.txt"
DOCKERFILE_FILE = "Dockerfile"
DOCKERIGNORE_FILE = ".dockerignore"
DOCKER_APP_FILE = "serve.py"

# Default serving config
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# Request/response schemas
REQUEST_RECORDS_KEY = "records"
PREDICTIONS_KEY = "predictions"
PROBABILITIES_KEY = "probabilities"

# Endpoints
HEALTH_ENDPOINT = "/health"
METADATA_ENDPOINT = "/meta"
PREDICT_ENDPOINT = "/predict"
PREDICT_PROBA_ENDPOINT = "/predict_proba"
PREDICT_CSV_ENDPOINT = "/predict-csv"
HISTORY_ENDPOINT = "/history"
HISTORY_SUMMARY_ENDPOINT = "/history/summary"

# Package name mappings (internal name -> PyPI package name)
PACKAGE_NAME_MAPPING = {
    "sklearn": "scikit-learn",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
}
