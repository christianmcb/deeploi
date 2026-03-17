# Deeploi

One-line deployment for trained tabular ML models.

**Deeploi** = instant API for sklearn and XGBoost → no config, no boilerplate, no DevOps required.

```python
from deeploi import deploy

deploy(model, sample=X_train)
```

That's it. Your model is now serving predictions at `http://127.0.0.1:8000`.

## Features

✅ **Instant API**  
One command launches a FastAPI server with `/predict`, `/health`, `/meta` endpoints.

✅ **Schema Inference**  
Learns feature names, dtypes, order from your training sample.

✅ **Artifact Packaging**  
Save, version, and reload models with metadata.

✅ **Prediction Probabilities**  
Classification models get `/predict_proba` endpoint automatic.

✅ **sklearn & XGBoost**  
Classifiers and regressors, both frameworks.

✅ **Local-First**  
Run anywhere — no cloud, no containers, no registry.

## Quick Start

### Installation

```bash
pip install deeploi
```

### The One-Liner

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from deeploi import deploy

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

model = RandomForestClassifier(n_estimators=10).fit(X, y)

deploy(model, sample=X)
```

Server starts at `http://127.0.0.1:8000`.

### Test It

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

Response:
```json
{
  "predictions": [0],
  "probabilities": [
    {"0": 0.91, "1": 0.09}
  ]
}
```

## Core API

### Three Functions

#### 1. `deploy()` — One-liner, immediate serving

```python
from deeploi import deploy

deploy(model, sample=X_train, host="127.0.0.1", port=8000)
```

Infers schema → packages model → starts server (blocking).

#### 2. `package()` — Reusable object

```python
from deeploi import package

pkg = package(model, sample=X_train)

# Use it
preds = pkg.predict(X_test)

# or save it
pkg.save("artifacts/iris_rf")

# or serve it later
pkg.serve(port=8000)
```

#### 3. `load()` — Reload saved artifacts

```python
from deeploi import load

pkg = load("artifacts/iris_rf")
preds = pkg.predict(X_test)
```

## Endpoints

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

### `GET /meta`

```bash
curl http://127.0.0.1:8000/meta
```

```json
{
  "framework": "sklearn",
  "estimator_class": "RandomForestClassifier",
  "task_type": "classification",
  "supports_predict_proba": true,
  "python_version": "3.11.0",
  "deeploi_version": "0.1.0",
  "created_at": "2026-03-17T12:00:00Z"
}
```

### `POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records": [{"col_1": 1.0, "col_2": 2.0}]}'
```

**Regression response:**
```json
{
  "predictions": [123.45, 118.91],
  "probabilities": null
}
```

**Classification response:**
```json
{
  "predictions": [0, 1],
  "probabilities": [
    {"0": 0.91, "1": 0.09},
    {"0": 0.12, "1": 0.88}
  ]
}
```

### `POST /predict_proba`

Classification models only. Same as `/predict` with probabilities.

```bash
curl -X POST http://127.0.0.1:8000/predict_proba \
  -H "Content-Type: application/json" \
  -d '{"records": [{"col_1": 1.0, "col_2": 2.0}]}'
```

## Artifact Structure

When you call `pkg.save("path/to/artifact")`, you get:

```
path/to/artifact/
├── model.joblib           # Serialized model
├── metadata.json          # Versions, task type, timestamps
├── schema.json            # Features, dtypes, column order
├── deeploi.json           # Manifest
└── requirements.txt       # Dependencies
```

**metadata.json:**
```json
{
  "framework": "sklearn",
  "estimator_class": "RandomForestClassifier",
  "task_type": "classification",
  "supports_predict_proba": true,
  "created_at": "2026-03-17T12:00:00Z",
  "python_version": "3.11.0",
  "deeploi_version": "0.1.0",
  "library_versions": {
    "sklearn": "1.5.0",
    "xgboost": "2.0.0",
    "pandas": "2.0.0"
  }
}
```

**schema.json:**
```json
{
  "features": [
    {"name": "sepal length (cm)", "dtype": "float64", "nullable": false},
    {"name": "sepal width (cm)", "dtype": "float64", "nullable": false},
    {"name": "petal length (cm)", "dtype": "float64", "nullable": false},
    {"name": "petal width (cm)", "dtype": "float64", "nullable": false}
  ],
  "column_order": [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
  ]
}
```

## Examples

### Scikit-learn Regressor

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from deeploi import package

X, y = load_diabetes(return_X_y=True, as_frame=True)
model = RandomForestRegressor().fit(X, y)

pkg = package(model, X)
preds = pkg.predict(X[:5])
print(preds.to_json())
```

### XGBoost Classifier

```python
import pandas as pd
import xgboost as xgb
from deeploi import deploy

df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

model = xgb.XGBClassifier().fit(X, y)

deploy(model, sample=X, port=9000)
```

### Save & Load

```python
from deeploi import package, load

pkg = package(model, X_train)
pkg.save("artifacts/v1")

# Later...
pkg = load("artifacts/v1")
preds = pkg.predict(X_test)
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

## What's in v0.1.0

**Supported:**
- sklearn classifiers and regressors
- XGBoost classifiers and regressors
- pandas DataFrame inputs
- Local FastAPI serving
- Model save/load
- Schema inference
- Prediction probabilities (classifiers)

**Not in v0.1.0:**
- S3 / cloud storage
- Docker generation
- Authentication
- Batch inference
- Async workers
- Model registry
- Monitoring

## Requirements

- Python 3.8+
- pandas >= 1.0
- scikit-learn >= 0.24
- xgboost >= 1.0
- fastapi >= 0.68
- uvicorn >= 0.15

## License

MIT. See [LICENSE](./LICENSE).

---

**Next Steps:**

1. Try the [Iris example](examples/sklearn_classifier.py)
2. Package your own model
3. Deploy locally
4. Hit `/predict`

Questions? Open an issue on [GitHub](https://github.com/deeploi/deeploi).
