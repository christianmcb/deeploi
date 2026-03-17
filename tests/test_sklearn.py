"""
Tests for scikit-learn model support.
"""

import pytest
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import load_iris, load_diabetes
from sklearn.tree import DecisionTreeClassifier

from deeploi import package
from deeploi.exceptions import UnsupportedModelError


class TestSklearnClassifiers:
    """Test support for sklearn classifiers."""
    
    @pytest.mark.parametrize("classifier_class", [
        RandomForestClassifier,
        GradientBoostingClassifier,
        DecisionTreeClassifier,
        LogisticRegression,
    ])
    def test_sklearn_classifiers(self, classifier_class):
        """Test that various sklearn classifiers work."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        
        model = classifier_class(random_state=42)
        if hasattr(model, 'max_iter'):
            model.max_iter = 100
        model.fit(X, y)
        
        pkg = package(model, X)
        
        assert pkg is not None
        assert pkg.metadata.task_type == "classification"
        assert pkg.metadata.framework == "sklearn"
    
    def test_classifier_predict_proba_support(self):
        """Test that classifier predict_proba is detected."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        
        # RandomForest has predict_proba
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(X, y)
        pkg_rf = package(rf, X)
        assert pkg_rf.metadata.supports_predict_proba
        
        # LogisticRegression also has it
        lr = LogisticRegression(max_iter=200, random_state=42)
        lr.fit(X, y)
        pkg_lr = package(lr, X)
        assert pkg_lr.metadata.supports_predict_proba


class TestSklearnRegressors:
    """Test support for sklearn regressors."""
    
    @pytest.mark.parametrize("regressor_class", [
        RandomForestRegressor,
    ])
    def test_sklearn_regressors(self, regressor_class):
        """Test that various sklearn regressors work."""
        X, y = load_diabetes(return_X_y=True, as_frame=True)
        
        model = regressor_class(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        pkg = package(model, X)
        
        assert pkg is not None
        assert pkg.metadata.task_type == "regression"
        assert pkg.metadata.framework == "sklearn"
        assert not pkg.metadata.supports_predict_proba


class TestSklearnPredictions:
    """Test predictions with sklearn models."""
    
    def test_classifier_predictions_match(self):
        """Test that package predictions match model predictions."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        pkg = package(model, X)
        
        # Get predictions from both
        pkg_preds = pkg.predict(X[:10]).predictions
        model_preds = model.predict(X[:10]).tolist()
        
        assert pkg_preds == model_preds
    
    def test_regressor_predictions_match(self):
        """Test that package predictions match model predictions."""
        X, y = load_diabetes(return_X_y=True, as_frame=True)
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        pkg = package(model, X)
        
        # Get predictions from both
        pkg_preds = pkg.predict(X[:10]).predictions
        model_preds = model.predict(X[:10]).tolist()
        
        # Should be very close (floating point)
        np.testing.assert_array_almost_equal(pkg_preds, model_preds, decimal=5)
