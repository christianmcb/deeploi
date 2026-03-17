"""
Tests for XGBoost model support.
"""

import pytest
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split

from deeploi import package


class TestXGBoostClassifiers:
    """Test support for XGBoost classifiers."""
    
    def test_xgboost_classifier_creation(self):
        """Test creating package from XGBoost classifier."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        
        model = xgb.XGBClassifier(n_estimators=5, random_state=42, verbosity=0)
        model.fit(X, y)
        
        pkg = package(model, X)
        
        assert pkg is not None
        assert pkg.metadata.framework == "xgboost"
        assert pkg.metadata.task_type == "classification"
        assert pkg.metadata.supports_predict_proba
    
    def test_xgboost_classifier_frameworks_detected(self):
        """Test that XGBoost is detected correctly."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        
        model = xgb.XGBClassifier(n_estimators=5, random_state=42, verbosity=0)
        model.fit(X, y)
        
        pkg = package(model, X)
        
        assert pkg.metadata.framework == "xgboost"
        assert pkg.metadata.estimator_class == "XGBClassifier"


class TestXGBoostRegressors:
    """Test support for XGBoost regressors."""
    
    def test_xgboost_regressor_creation(self):
        """Test creating package from XGBoost regressor."""
        from sklearn.datasets import load_diabetes
        
        X, y = load_diabetes(return_X_y=True, as_frame=True)
        
        model = xgb.XGBRegressor(n_estimators=5, random_state=42, verbosity=0)
        model.fit(X, y)
        
        pkg = package(model, X)
        
        assert pkg is not None
        assert pkg.metadata.framework == "xgboost"
        assert pkg.metadata.task_type == "regression"
        assert not pkg.metadata.supports_predict_proba


class TestXGBoostPredictions:
    """Test predictions with XGBoost models."""
    
    def test_xgboost_classifier_predictions(self):
        """Test XGBoost classifier predictions."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        
        model = xgb.XGBClassifier(n_estimators=5, random_state=42, verbosity=0)
        model.fit(X, y)
        
        pkg = package(model, X)
        
        # Get predictions from both
        pkg_preds = pkg.predict(X[:10]).predictions
        model_preds = model.predict(X[:10]).tolist()
        
        assert pkg_preds == model_preds
    
    def test_xgboost_classifier_probabilities(self):
        """Test XGBoost classifier probability predictions."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        
        model = xgb.XGBClassifier(n_estimators=5, random_state=42, verbosity=0)
        model.fit(X, y)
        
        pkg = package(model, X)
        
        response = pkg.predict(X[:3], include_probabilities=True)
        
        assert len(response.predictions) == 3
        assert response.probabilities is not None
        assert len(response.probabilities) == 3
    
    def test_xgboost_regressor_predictions(self):
        """Test XGBoost regressor predictions."""
        from sklearn.datasets import load_diabetes
        
        X, y = load_diabetes(return_X_y=True, as_frame=True)
        
        model = xgb.XGBRegressor(n_estimators=5, random_state=42, verbosity=0)
        model.fit(X, y)
        
        pkg = package(model, X)
        
        # Get predictions from both
        pkg_preds = pkg.predict(X[:10]).predictions
        model_preds = model.predict(X[:10]).tolist()
        
        # Should be very close (floating point)
        np.testing.assert_array_almost_equal(pkg_preds, model_preds, decimal=1)


class TestXGBoostVsBinary:
    """Test XGBoost binary classification."""
    
    def test_xgboost_binary_classification(self):
        """Test XGBoost binary classification."""
        cancer = load_breast_cancer(as_frame=True)
        X = cancer.data
        y = cancer.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = xgb.XGBClassifier(n_estimators=5, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        
        pkg = package(model, X_train)
        
        # Test prediction
        response = pkg.predict(X_test[:5], include_probabilities=True)
        
        assert len(response.predictions) == 5
        assert response.probabilities is not None
        assert len(response.probabilities) == 5
        
        # Each probability should have 2 classes
        for proba_dict in response.probabilities:
            assert len(proba_dict) == 2
