"""Tests for NGBoost model support."""

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_iris

from deeploi import package

ngboost = pytest.importorskip("ngboost")


class TestNGBoostClassifiers:
    """Test support for NGBoost classifiers."""

    def test_ngboost_classifier_creation(self):
        """Package NGBoost classifier successfully."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target

        model = ngboost.NGBClassifier(verbose=False)
        model.fit(X, y)

        pkg = package(model, X)

        assert pkg.metadata.framework == "sklearn"
        assert pkg.metadata.task_type == "classification"


class TestNGBoostRegressors:
    """Test support for NGBoost regressors."""

    def test_ngboost_regressor_creation(self):
        """Package NGBoost regressor successfully."""
        X, y = load_diabetes(return_X_y=True, as_frame=True)

        model = ngboost.NGBRegressor(verbose=False)
        model.fit(X, y)

        pkg = package(model, X)

        assert pkg.metadata.framework == "sklearn"
        assert pkg.metadata.task_type == "regression"


class TestNGBoostPredictions:
    """Test NGBoost predictions match underlying model output."""

    def test_ngboost_classifier_predictions(self):
        """Packaged classifier predictions should match model predictions."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target

        model = ngboost.NGBClassifier(verbose=False)
        model.fit(X, y)

        pkg = package(model, X)

        pkg_preds = pkg.predict(X[:10]).predictions
        model_preds = model.predict(X[:10]).tolist()

        assert pkg_preds == model_preds

    def test_ngboost_regressor_predictions(self):
        """Packaged regressor predictions should match model predictions."""
        X, y = load_diabetes(return_X_y=True, as_frame=True)

        model = ngboost.NGBRegressor(verbose=False)
        model.fit(X, y)

        pkg = package(model, X)

        pkg_preds = pkg.predict(X[:10]).predictions
        model_preds = model.predict(X[:10]).tolist()

        np.testing.assert_array_almost_equal(pkg_preds, model_preds, decimal=6)
