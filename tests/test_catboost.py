"""
Tests for CatBoost model support.
"""

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_iris

from deeploi import package

catboost = pytest.importorskip("catboost")


class TestCatBoostClassifiers:
    """Test support for CatBoost classifiers."""

    def test_catboost_classifier_creation(self):
        """Test creating package from CatBoost classifier."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target

        model = catboost.CatBoostClassifier(iterations=20, random_seed=42, verbose=False)
        model.fit(X, y)

        pkg = package(model, X)

        assert pkg is not None
        assert pkg.metadata.framework == "catboost"
        assert pkg.metadata.task_type == "classification"
        assert pkg.metadata.supports_predict_proba


class TestCatBoostRegressors:
    """Test support for CatBoost regressors."""

    def test_catboost_regressor_creation(self):
        """Test creating package from CatBoost regressor."""
        X, y = load_diabetes(return_X_y=True, as_frame=True)

        model = catboost.CatBoostRegressor(iterations=30, random_seed=42, verbose=False)
        model.fit(X, y)

        pkg = package(model, X)

        assert pkg is not None
        assert pkg.metadata.framework == "catboost"
        assert pkg.metadata.task_type == "regression"
        assert not pkg.metadata.supports_predict_proba


class TestCatBoostPredictions:
    """Test predictions with CatBoost models."""

    def test_catboost_classifier_predictions(self):
        """Package predictions should match CatBoost classifier predictions."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target

        model = catboost.CatBoostClassifier(iterations=20, random_seed=42, verbose=False)
        model.fit(X, y)

        pkg = package(model, X)

        pkg_preds = pkg.predict(X[:10]).predictions
        model_preds = model.predict(X[:10]).reshape(-1).tolist()

        assert pkg_preds == model_preds

    def test_catboost_regressor_predictions(self):
        """Package predictions should match CatBoost regressor predictions."""
        X, y = load_diabetes(return_X_y=True, as_frame=True)

        model = catboost.CatBoostRegressor(iterations=30, random_seed=42, verbose=False)
        model.fit(X, y)

        pkg = package(model, X)

        pkg_preds = pkg.predict(X[:10]).predictions
        model_preds = model.predict(X[:10]).reshape(-1).tolist()

        np.testing.assert_array_almost_equal(pkg_preds, model_preds, decimal=6)
