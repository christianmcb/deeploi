"""
Constants for Deeploi.
"""

__version__ = "0.1.0"

# Supported frameworks
SKLEARN = "sklearn"
XGBOOST = "xgboost"
SUPPORTED_FRAMEWORKS = {SKLEARN, XGBOOST}

# Task types
CLASSIFICATION = "classification"
REGRESSION = "regression"
TASK_TYPES = {CLASSIFICATION, REGRESSION}

# Model types
SKLEARN_CLASSIFIER = "sklearn_classifier"
SKLEARN_REGRESSOR = "sklearn_regressor"
XGBOOST_CLASSIFIER = "xgboost_classifier"
XGBOOST_REGRESSOR = "xgboost_regressor"

# Artifact file names
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
SCHEMA_FILE = "schema.json"
MANIFEST_FILE = "deeploi.json"
REQUIREMENTS_FILE = "requirements.txt"

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
