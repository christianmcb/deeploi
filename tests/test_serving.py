"""
Tests for FastAPI serving endpoint.
"""

import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_diabetes
import numpy as np

from deeploi import package
from deeploi.serving import create_app


class TestHealthEndpoint:
    """Test /health endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        pkg = package(model, X)
        app = create_app(pkg)
        
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestMetadataEndpoint:
    """Test /meta endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        pkg = package(model, X)
        app = create_app(pkg)
        
        return TestClient(app)
    
    def test_metadata_endpoint(self, client):
        """Test metadata endpoint returns correct info."""
        response = client.get("/meta")
        
        assert response.status_code == 200
        data = response.json()
        assert data["framework"] == "sklearn"
        assert data["estimator_class"] == "RandomForestClassifier"
        assert data["task_type"] == "classification"
        assert data["supports_predict_proba"] is True


class TestPredictEndpoint:
    """Test /predict endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        pkg = package(model, X)
        app = create_app(pkg)
        
        return TestClient(app)
    
    def test_predict_single_record(self, client):
        """Test prediction with a single record."""
        payload = {
            "records": [
                {
                    "sepal length (cm)": 5.1,
                    "sepal width (cm)": 3.5,
                    "petal length (cm)": 1.4,
                    "petal width (cm)": 0.2,
                }
            ]
        }
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        assert data["predictions"][0] in [0, 1, 2]
    
    def test_predict_multiple_records(self, client):
        """Test prediction with multiple records."""
        payload = {
            "records": [
                {
                    "sepal length (cm)": 5.1,
                    "sepal width (cm)": 3.5,
                    "petal length (cm)": 1.4,
                    "petal width (cm)": 0.2,
                },
                {
                    "sepal length (cm)": 4.9,
                    "sepal width (cm)": 3.0,
                    "petal length (cm)": 1.4,
                    "petal width (cm)": 0.2,
                },
            ]
        }
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 2
        assert "probabilities" in data
    
    def test_predict_invalid_input(self, client):
        """Test prediction with invalid input."""
        payload = {
            "records": [
                {
                    "sepal length (cm)": 5.1,
                    # Missing other columns
                }
            ]
        }
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 400
    
    def test_predict_probability_response(self, client):
        """Test that probabilities are included in response."""
        payload = {
            "records": [
                {
                    "sepal length (cm)": 5.1,
                    "sepal width (cm)": 3.5,
                    "petal length (cm)": 1.4,
                    "petal width (cm)": 0.2,
                }
            ]
        }
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "probabilities" in data
        assert data["probabilities"] is not None
        assert len(data["probabilities"]) == 1


class TestPredictProbaEndpoint:
    """Test /predict_proba endpoint."""
    
    @pytest.fixture
    def classifier_client(self):
        """Create a test client for a classifier."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        pkg = package(model, X)
        app = create_app(pkg)
        
        return TestClient(app)
    
    @pytest.fixture
    def regressor_client(self):
        """Create a test client for a regressor."""
        X, y = load_diabetes(return_X_y=True, as_frame=True)
        
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        pkg = package(model, X)
        app = create_app(pkg)
        
        return TestClient(app)
    
    def test_predict_proba_classifier(self, classifier_client):
        """Test predict_proba works for classifier."""
        payload = {
            "records": [
                {
                    "sepal length (cm)": 5.1,
                    "sepal width (cm)": 3.5,
                    "petal length (cm)": 1.4,
                    "petal width (cm)": 0.2,
                }
            ]
        }
        
        response = classifier_client.post("/predict_proba", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "probabilities" in data
    
    def test_predict_proba_regressor_fails(self, regressor_client):
        """Test predict_proba fails for regressor."""
        payload = {
            "records": [
                {"s0": 0.0, "s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": 4.0, 
                 "s5": 5.0, "s6": 6.0, "s7": 7.0, "s8": 8.0, "s9": 9.0}
            ]
        }
        
        response = regressor_client.post("/predict_proba", json=payload)
        
        assert response.status_code == 400


class TestRegressionEndpoint:
    """Test endpoints for regression models."""
    
    @pytest.fixture
    def regressor_client(self):
        """Create a test client for a regressor."""
        X, y = load_diabetes(return_X_y=True, as_frame=True)
        
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        pkg = package(model, X)
        app = create_app(pkg)
        
        return TestClient(app)
    
    def test_regression_predict(self, regressor_client):
        """Test regression prediction endpoint."""
        X, y = load_diabetes(return_X_y=True, as_frame=True)
        
        payload = {
            "records": [
                {
                    "age": X.iloc[0]["age"],
                    "sex": X.iloc[0]["sex"],
                    "bmi": X.iloc[0]["bmi"],
                    "bp": X.iloc[0]["bp"],
                    "s1": X.iloc[0]["s1"],
                    "s2": X.iloc[0]["s2"],
                    "s3": X.iloc[0]["s3"],
                    "s4": X.iloc[0]["s4"],
                    "s5": X.iloc[0]["s5"],
                    "s6": X.iloc[0]["s6"],
                }
            ]
        }
        
        response = regressor_client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        assert isinstance(data["predictions"][0], (int, float))
