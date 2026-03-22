"""Tests for LightGBM and CatBoost ranker support."""

import numpy as np
import pytest
from sklearn.datasets import load_diabetes

from deeploi import package

lightgbm = pytest.importorskip("lightgbm")
catboost = pytest.importorskip("catboost")


class TestLightGBMRanker:
    """Test LightGBM ranker support."""

    def test_lightgbm_ranker_creation_and_prediction(self):
        """LightGBM ranker should package and predict."""
        X, y = load_diabetes(return_X_y=True, as_frame=True)

        model = lightgbm.LGBMRanker(n_estimators=20, random_state=42)
        group = [len(X)]
        model.fit(X, y, group=group)

        pkg = package(model, X)

        assert pkg.metadata.framework == "lightgbm"
        assert pkg.metadata.task_type == "regression"
        assert not pkg.metadata.supports_predict_proba

        pkg_preds = pkg.predict(X[:10]).predictions
        model_preds = model.predict(X[:10]).tolist()

        np.testing.assert_array_almost_equal(pkg_preds, model_preds, decimal=6)


class TestCatBoostRanker:
    """Test CatBoost ranker support."""

    def test_catboost_ranker_creation_and_prediction(self):
        """CatBoost ranker should package and predict."""
        X, y = load_diabetes(return_X_y=True, as_frame=True)

        model = catboost.CatBoostRanker(iterations=30, random_seed=42, verbose=False)
        group_id = np.ones(len(X), dtype=int)
        model.fit(X, y, group_id=group_id)

        pkg = package(model, X)

        assert pkg.metadata.framework == "catboost"
        assert pkg.metadata.task_type == "regression"
        assert not pkg.metadata.supports_predict_proba

        pkg_preds = pkg.predict(X[:10]).predictions
        model_preds = model.predict(X[:10]).reshape(-1).tolist()

        np.testing.assert_array_almost_equal(pkg_preds, model_preds, decimal=6)
