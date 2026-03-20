# Deeploi Docs

This document covers the advanced usage patterns for Deeploi.

## Core API

Deeploi exposes three main entry points:

- `deploy(model, sample=None, ...)`
- `package(model, sample=None)`
- `load(path)`

### `deploy()`

Use `deploy()` when you want the shortest path from trained model to running API.

```python
from deeploi import deploy

deploy(model, host="127.0.0.1", port=8000)
```

If you want full schema fidelity, you can still pass a sample explicitly:

```python
deploy(model, sample=X_train, host="127.0.0.1", port=8000)
```

Optional auth-enabled deployment:

```python
deploy(
    model,
    host="127.0.0.1",
    port=8000,
    require_auth=True,
    api_key="my-secret-key",
)
```

### `package()`

Use `package()` when you want a reusable Python object for prediction, saving, and serving.

```python
from deeploi import package

pkg = package(model)
preds = pkg.predict(X_test)
pkg.save("artifacts/model_v1")
pkg.serve(port=8000)
```

If you want Deeploi to infer exact dtypes and nullability from a DataFrame, pass a sample explicitly:

```python
pkg = package(model, sample=X_train)
```

### `load()`

Use `load()` when you want to restore a saved artifact from disk.

```python
from deeploi import load

pkg = load("artifacts/model_v1")
preds = pkg.predict(X_test)
pkg.serve(port=8000)
```

## API Endpoints

Deeploi serves these endpoints:

- `POST /predict`
- `POST /predict_proba`
- `GET /meta`
- `GET /health`
- `GET /` dashboard

### `POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2
      }
    ]
  }'
```

Typical response:

```json
{
  "predictions": [0],
  "probabilities": [
    {"0": 0.91, "1": 0.09}
  ]
}
```

### `POST /predict_proba`

This is available for classifiers that support `predict_proba`.

### `GET /meta`

Returns framework, model type, task type, feature list, and version metadata.

### `GET /health`

Returns basic service health information.

## Authentication

Authentication is optional and off by default.

### Explicit auth in Python

```python
deploy(model, require_auth=True, api_key="my-secret-key")
```

Or:

```python
pkg.serve(require_auth=True, api_key="my-secret-key")
```

### Environment-based auth

```bash
export DEEPLOI_AUTH_ENABLED=true
export DEEPLOI_AUTH_API_KEY="your-secret-key"
export DEEPLOI_AUTH_HEADER="X-API-Key"
```

When these variables are set, `deploy(...)` and `pkg.serve(...)` pick them up automatically.

### Auto-generated API keys

If auth is enabled and no key is provided, Deeploi auto-generates a secure process-local key by default.

You can disable that behavior with:

```bash
export DEEPLOI_AUTH_AUTO_GENERATE=false
```

Or explicitly in Python:

```python
pkg.serve(require_auth=True, auto_generate_api_key=False)
```

### Dashboard access when auth is enabled

The dashboard is also protected.

Open it with:

```text
http://127.0.0.1:8000/?api_key=your-secret-key
```

Once loaded, the dashboard automatically reuses that key for Playground requests.

### Authenticated curl example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2
      }
    ]
  }'
```

## Docker Artifact Generation

Generate Docker files when saving:

```python
pkg.save("artifacts/iris_rf", generate_docker=True)
```

Or generate them later:

```python
pkg.generate_docker("artifacts/iris_rf", port=8000)
```

Build and run:

```bash
cd artifacts/iris_rf
docker build -t iris-model .
docker run --rm -p 8000:8000 iris-model
```

## Artifact Layout

Saved artifacts contain:

```text
artifacts/model_v1/
├── model.joblib
├── metadata.json
├── schema.json
├── deeploi.json
└── requirements.txt
```

If Docker generation is enabled, you also get:

```text
artifacts/model_v1/
├── Dockerfile
├── .dockerignore
└── serve.py
```

## Error Handling

```python
from deeploi import package, UnsupportedModelError, InvalidSampleError

try:
    pkg = package(my_model, my_sample)
except UnsupportedModelError:
    print("Only sklearn and XGBoost are supported")
except InvalidSampleError:
    print("Sample must be a non-empty DataFrame")
```

Prediction-time issues surface as schema validation errors or prediction errors, depending on whether the request shape or the model execution failed.

## Schema Inference Rules

Deeploi now infers schema in this order:

1. Use `sample` when provided
2. Otherwise use fitted model feature metadata such as `feature_names_in_`
3. Otherwise fall back to generated names like `feature_0`, `feature_1`, ... when only `n_features_in_` is available

If none of those are available, Deeploi raises an error and you must provide `sample=...`.

## Supported Use Cases

Current core support includes:

- scikit-learn classifiers and regressors
- XGBoost classifiers and regressors
- pandas DataFrame-based schema inference
- local FastAPI serving
- save and load artifacts
- interactive dashboard
- optional API key protection
- Docker file generation for artifacts

## Requirements

- Python 3.8+
- pandas >= 1.0
- scikit-learn >= 0.24
- xgboost >= 1.0
- fastapi >= 0.68
- uvicorn >= 0.15