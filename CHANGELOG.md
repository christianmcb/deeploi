# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and the project follows semantic versioning.

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
