"""Tests for imbalanced-learn meta-estimator support."""

import pytest
from sklearn.datasets import load_iris

from deeploi import package

imblearn_ensemble = pytest.importorskip("imblearn.ensemble")


class TestImbalancedLearnMetaEstimators:
    """Test common imbalanced-learn wrappers."""

    def test_balanced_random_forest_classifier_support(self):
        """BalancedRandomForestClassifier should package and predict."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target

        model = imblearn_ensemble.BalancedRandomForestClassifier(
            n_estimators=20,
            random_state=42,
        )
        model.fit(X, y)

        pkg = package(model, X)

        assert pkg.metadata.framework == "sklearn"
        assert pkg.metadata.task_type == "classification"
        assert pkg.metadata.supports_predict_proba

        response = pkg.predict(X[:5], include_probabilities=True)
        assert len(response.predictions) == 5
        assert response.probabilities is not None

    def test_easy_ensemble_classifier_support(self):
        """EasyEnsembleClassifier should package and predict."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target

        model = imblearn_ensemble.EasyEnsembleClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        pkg = package(model, X)

        assert pkg.metadata.framework == "sklearn"
        assert pkg.metadata.task_type == "classification"

        response = pkg.predict(X[:5])
        assert len(response.predictions) == 5
