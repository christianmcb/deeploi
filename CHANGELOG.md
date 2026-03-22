# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and the project follows semantic versioning.

## [0.3.0] - 2026-03-21

### Added

- Automatic schema-aware input coercion during prediction (for example, numeric strings like "42" are converted to numeric dtypes when possible).
- Actionable dtype mismatch feedback that explains when input "looks like X" but the model expects "Y", including concrete failing values.
- Focused tests for coercion behavior and improved artifact/Docker validation errors.
- LightGBM classifier and regressor support in one-line deployment and packaging workflows.
- CatBoost classifier and regressor support in one-line deployment and packaging workflows.
- Broader support for sklearn-compatible third-party estimators that implement the standard estimator contract.
- Explicit support coverage for HistGradientBoosting, ExtraTrees, and CalibratedClassifierCV.
- Explicit support coverage for NGBoost, LightGBM/CatBoost rankers, and imbalanced-learn meta-estimators.
- In-memory prediction history summary endpoint (`GET /history/summary`) for dashboard-ready aggregate stats.
- Enriched metadata endpoint fields including `feature_count`, `training_set_size` (best-effort), `class_labels` (when available), and `feature_importance` (when available).
- Optional CSV batch prediction endpoint (`POST /predict-csv`) with multipart file upload.
- Optional dependency bundles (`tabular`, `ranking`, `imbalanced`, `all`) for lighter installs.
- Supported-models matrix documentation for quick framework/feature compatibility lookup.
- Added endpoint constants for `/predict-csv`, `/history`, and `/history/summary`.

### Changed

- Docker generation validation now gives clearer guidance for invalid ports, invalid Python image tags, missing artifact directories, and incomplete artifact contents.
- Artifact load and serialization errors now include recovery steps (for example, re-saving artifacts with `pkg.save(path)`).
- Neural network support remains intentionally deferred to v0.4+ to keep v0.3.x tabular-first and simple.
- Dashboard overview now shows professional quick stats backed by server memory (`predictions served`, `avg response time`) and richer model metadata context.
- Dashboard metadata panel now visualizes top feature importance when supported by the model.
- CSV upload validation now provides clearer feedback for UTF-8 encoding issues, duplicate columns, and missing/invalid header rows.

### Fixed

- Prediction-time validation now attempts intelligent dtype coercion before failing, reducing avoidable errors for common JSON payload patterns.

## [0.2.0] - 2026-03-20

### Added

- Optional API key authentication for prediction endpoints.
- Protected dashboard access when authentication is enabled.
- Automatic process-local API key generation when auth is enabled without an explicit key.
- Advanced package documentation in `DOCS.md`.
- Initial `CHANGELOG.md` for release tracking.

### Changed

- `deploy(model)` and `package(model)` now work without `sample` for common fitted sklearn and XGBoost models when feature metadata is available.
- Dashboard requests automatically reuse the validated API key after opening the protected dashboard.
- README was simplified to focus on the primary ways to access Deeploi.
- Docker guidance was promoted in the README as a primary workflow.
- Package metadata documentation URL now points to `DOCS.md`.

### Fixed

- Removed outdated documentation claims that authentication was unsupported.
- Updated dashboard quickstart and usage examples to reflect current API behavior.
- Updated advanced docs and examples to match the new optional-sample deployment flow.

## [0.1.0] - Initial Release

### Added

- One-line local deployment for trained tabular ML models with `deploy(...)`.
- Support for scikit-learn and XGBoost classifiers and regressors.
- FastAPI-based serving with `/predict`, `/predict_proba`, `/meta`, and `/health` endpoints.
- Automatic schema inference from pandas DataFrame samples.
- Reusable packaging flow with `package(...)`, `save(...)`, and `load(...)`.
- Interactive dashboard for API exploration and testing.
- Prediction probability support for compatible classifiers.
- Docker artifact generation for saved model packages.
