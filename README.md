# Deeploi &mdash; Deploy ML Models in Seconds 🚀

[![PyPI version](https://img.shields.io/pypi/v/deeploi.svg)](https://pypi.org/project/deeploi/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**Turn your trained tabular ML model into a production-ready API with a single line of code. No DevOps. No boilerplate. No headaches.**

---

## Why Deeploi?

- **Instant API:** Serve your scikit-learn or XGBoost model as a blazing-fast REST API in one command.
- **Zero Config:** No YAML, no Docker, no cloud lock-in. Just your model and your data.
- **For Data Scientists:** Focus on modeling, not infrastructure.
- **For Startups & Teams:** Ship ML features faster, without waiting for MLOps.
- **Local-First:** Run anywhere—laptop, server, or cloud VM.

---

## Get Started in 10 Seconds

```python
from deeploi import deploy

deploy(model, sample=X_train)
```

Your model is now live at `http://127.0.0.1:8000` with `/predict`, `/health`, and `/meta` endpoints.

---

## Features

- ✅ **One-line deployment** for tabular models
- ✅ **Automatic schema inference** from your training data
- ✅ **Model packaging & versioning** for reproducibility
- ✅ **Prediction probabilities** for classifiers
- ✅ **Supports scikit-learn & XGBoost**
- ✅ **No cloud or container required**

---

## Quick Start

```bash
pip install deeploi
```

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from deeploi import deploy

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
model = RandomForestClassifier().fit(X, y)
deploy(model, sample=X)
```

---

## How It Works

1. **Train your model** as usual.
2. **Deploy instantly** with `deploy(model, sample=X_train)`.
3. **Call your API** for predictions.

---

## API Endpoints

- `POST /predict` — Get predictions
- `POST /predict_proba` — Get class probabilities (classifiers)
- `GET /meta` — Model metadata
- `GET /health` — Health check

---

## Save, Load, and Reuse

```python
from deeploi import package, load

pkg = package(model, X_train)
pkg.save("artifacts/v1")
pkg = load("artifacts/v1")
preds = pkg.predict(X_test)
```

---

## Who is Deeploi for?

- Data scientists who want to demo or share models instantly
- ML engineers who need fast, local serving for tabular models
- Startups and teams who want to skip MLOps complexity

---

## Advanced Usage

### Three Core Functions

#### 1. `deploy()` — One-liner, immediate serving

```python
from deeploi import deploy

deploy(model, sample=X_train, host="127.0.0.1", port=8000)
```

#### 2. `package()` — Reusable object

```python
from deeploi import package

pkg = package(model, sample=X_train)
preds = pkg.predict(X_test)
pkg.save("artifacts/iris_rf")
pkg.serve(port=8000)
```

#### 3. `load()` — Reload saved artifacts

```python
from deeploi import load

pkg = load("artifacts/iris_rf")
preds = pkg.predict(X_test)
```

---

## Example: Test the API

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

---

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

---

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

---

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

---

## Requirements

- Python 3.8+
- pandas >= 1.0
- scikit-learn >= 0.24
- xgboost >= 1.0
- fastapi >= 0.68
- uvicorn >= 0.15

---

## License

MIT &mdash; see [LICENSE](./LICENSE).

---

**Ready to deploy your model?**  
Try the [examples](examples/) or run `pip install deeploi` now!

Questions? Open an issue on [GitHub](https://github.com/deeploi/deeploi).
