"""
Tests for LightGBM model support.
"""

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_iris

from deeploi import package

lightgbm = pytest.importorskip("lightgbm")


class TestLightGBMClassifiers:
    """Test support for LightGBM classifiers."""

    def test_lightgbm_classifier_creation(self):
        """Test creating package from LightGBM classifier."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target

        model = lightgbm.LGBMClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        pkg = package(model, X)

        assert pkg is not None
        assert pkg.metadata.framework == "lightgbm"
        assert pkg.metadata.task_type == "classification"
        assert pkg.metadata.supports_predict_proba


class TestLightGBMRegressors:
    """Test support for LightGBM regressors."""

    def test_lightgbm_regressor_creation(self):
        """Test creating package from LightGBM regressor."""
        X, y = load_diabetes(return_X_y=True, as_frame=True)

        model = lightgbm.LGBMRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)

        pkg = package(model, X)

        assert pkg is not None
        assert pkg.metadata.framework == "lightgbm"
        assert pkg.metadata.task_type == "regression"
        assert not pkg.metadata.supports_predict_proba


class TestLightGBMPredictions:
    """Test predictions with LightGBM models."""

    def test_lightgbm_classifier_predictions(self):
        """Package predictions should match LightGBM classifier predictions."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target

        model = lightgbm.LGBMClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        pkg = package(model, X)

        pkg_preds = pkg.predict(X[:10]).predictions
        model_preds = model.predict(X[:10]).tolist()

        assert pkg_preds == model_preds

    def test_lightgbm_regressor_predictions(self):
        """Package predictions should match LightGBM regressor predictions."""
        X, y = load_diabetes(return_X_y=True, as_frame=True)

        model = lightgbm.LGBMRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)

        pkg = package(model, X)

        pkg_preds = pkg.predict(X[:10]).predictions
        model_preds = model.predict(X[:10]).tolist()

        np.testing.assert_array_almost_equal(pkg_preds, model_preds, decimal=6)
