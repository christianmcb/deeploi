"""
Tests for the core DeeploiPackage class.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_diabetes
import xgboost as xgb

from deeploi import package, load, deploy
from deeploi.package import DeeploiPackage
from deeploi.exceptions import (
    UnsupportedModelError,
    InvalidSampleError,
    SchemaValidationError,
)


class TestPackageCreation:
    """Test creating packages from models."""
    
    def test_sklearn_classifier_package_creation(self):
        """Test creating a package from sklearn classifier."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        pkg = package(model, X)
        
        assert pkg is not None
        assert isinstance(pkg, DeeploiPackage)
        assert pkg.metadata.framework == "sklearn"
        assert pkg.metadata.task_type == "classification"
        assert pkg.metadata.supports_predict_proba
    
    def test_sklearn_regressor_package_creation(self):
        """Test creating a package from sklearn regressor."""
        X, y = load_diabetes(return_X_y=True, as_frame=True)
        
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        pkg = package(model, X)
        
        assert pkg is not None
        assert pkg.metadata.framework == "sklearn"
        assert pkg.metadata.task_type == "regression"
        assert not pkg.metadata.supports_predict_proba
    
    def test_xgboost_classifier_package_creation(self):
        """Test creating a package from XGBoost classifier."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        
        model = xgb.XGBClassifier(n_estimators=5, random_state=42, verbosity=0)
        model.fit(X, y)
        
        pkg = package(model, X)
        
        assert pkg is not None
        assert pkg.metadata.framework == "xgboost"
        assert pkg.metadata.task_type == "classification"
        assert pkg.metadata.supports_predict_proba
    
    def test_package_with_invalid_sample(self):
        """Test that invalid samples raise errors."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        
        model = RandomForestClassifier(n_estimators=5)
        model.fit(X, y)
        
        # Empty DataFrame
        with pytest.raises(InvalidSampleError):
            package(model, pd.DataFrame())

    def test_package_without_sample_uses_model_feature_names(self):
        """A fitted model with feature metadata should not require sample."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        pkg = package(model)

        assert pkg.schema.column_order == list(X.columns)
        assert all(feature.dtype == "unknown" for feature in pkg.schema.features)

    def test_package_without_sample_generates_generic_names_for_array_fit(self):
        """Fallback to generic names when only n_features_in_ is available."""
        iris = load_iris()
        X, y = iris.data, iris.target

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        pkg = package(model)

        assert pkg.schema.column_order == ["feature_0", "feature_1", "feature_2", "feature_3"]

    def test_package_without_sample_raises_without_model_metadata(self):
        """A model without fitted feature metadata should still require sample."""
        with pytest.raises(InvalidSampleError):
            DeeploiPackage.from_model(RandomForestClassifier(), sample=None)


class TestPrediction:
    """Test prediction functionality."""
    
    @pytest.fixture
    def sklearn_classifier_pkg(self):
        """Fixture for sklearn classifier package."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        return package(model, X)
    
    def test_classifier_predict(self, sklearn_classifier_pkg):
        """Test classifier prediction."""
        iris = load_iris(as_frame=True)
        X = iris.data
        
        response = sklearn_classifier_pkg.predict(X[:5])
        
        assert len(response.predictions) == 5
        assert all(p in [0, 1, 2] for p in response.predictions)
        assert response.probabilities is None
    
    def test_classifier_predict_with_probabilities(self, sklearn_classifier_pkg):
        """Test classifier prediction with probabilities."""
        iris = load_iris(as_frame=True)
        X = iris.data
        
        response = sklearn_classifier_pkg.predict(X[:5], include_probabilities=True)
        
        assert len(response.predictions) == 5
        assert response.probabilities is not None
        assert len(response.probabilities) == 5
        # Each probability dict should have 3 classes
        assert all(len(p) == 3 for p in response.probabilities)
    
    def test_predict_with_records(self, sklearn_classifier_pkg):
        """Test prediction with list of records."""
        record = {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2,
        }
        
        response = sklearn_classifier_pkg.predict([record])
        
        assert len(response.predictions) == 1
    
    def test_predict_with_invalid_input(self, sklearn_classifier_pkg):
        """Test prediction with invalid input."""
        with pytest.raises(SchemaValidationError):
            # Missing required column
            sklearn_classifier_pkg.predict([{"sepal length (cm)": 5.1}])
    
    def test_predict_with_extra_columns(self, sklearn_classifier_pkg):
        """Test prediction with extra columns is rejected."""
        record = {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2,
            "extra_column": 999,
        }
        
        with pytest.raises(SchemaValidationError):
            sklearn_classifier_pkg.predict([record])


class TestSaveAndLoad:
    """Test artifact save/load functionality."""
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading an artifact."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        pkg = package(model, X)
        
        # Save
        artifact_path = str(tmp_path / "test_artifact")
        pkg.save(artifact_path)
        
        # Check files exist
        import os
        assert os.path.exists(os.path.join(artifact_path, "model.joblib"))
        assert os.path.exists(os.path.join(artifact_path, "metadata.json"))
        assert os.path.exists(os.path.join(artifact_path, "schema.json"))
        
        # Load
        loaded_pkg = load(artifact_path)
        
        assert loaded_pkg is not None
        assert loaded_pkg.metadata.framework == pkg.metadata.framework
        assert loaded_pkg.metadata.task_type == pkg.metadata.task_type
    
    def test_loaded_package_predicts_same(self, tmp_path):
        """Test that loaded package makes same predictions."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        pkg = package(model, X)
        original_pred = pkg.predict(X[:5])
        
        # Save and load
        artifact_path = str(tmp_path / "test_artifact")
        pkg.save(artifact_path)
        loaded_pkg = load(artifact_path)
        
        # Check predictions are the same
        loaded_pred = loaded_pkg.predict(X[:5])
        
        assert original_pred.predictions == loaded_pred.predictions

    def test_generate_docker_files(self, tmp_path):
        """Test generating Docker files for a saved artifact."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        pkg = package(model, X)
        artifact_path = str(tmp_path / "docker_artifact")
        pkg.save(artifact_path)

        pkg.generate_docker(artifact_path, port=9001)

        import os
        dockerfile = os.path.join(artifact_path, "Dockerfile")
        dockerignore = os.path.join(artifact_path, ".dockerignore")
        serve_app = os.path.join(artifact_path, "serve.py")

        assert os.path.exists(dockerfile)
        assert os.path.exists(dockerignore)
        assert os.path.exists(serve_app)

        with open(dockerfile, "r") as f:
            dockerfile_content = f.read()

        assert "FROM python:3.11-slim" in dockerfile_content
        assert "EXPOSE 9001" in dockerfile_content
        assert '"--port", "9001"' in dockerfile_content

    def test_save_with_generate_docker(self, tmp_path):
        """Test generating Docker files directly from save()."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        pkg = package(model, X)
        artifact_path = str(tmp_path / "docker_on_save")
        pkg.save(artifact_path, generate_docker=True, docker_port=9100)

        import os
        assert os.path.exists(os.path.join(artifact_path, "Dockerfile"))
        assert os.path.exists(os.path.join(artifact_path, ".dockerignore"))
        assert os.path.exists(os.path.join(artifact_path, "serve.py"))


class TestSchema:
    """Test schema inference and validation."""
    
    def test_schema_inference(self):
        """Test that schema is correctly inferred from sample."""
        iris = load_iris(as_frame=True)
        X = iris.data
        
        pkg = package(RandomForestClassifier(n_estimators=5).fit(X, np.zeros(len(X))), X)
        
        schema = pkg.schema
        
        assert len(schema.features) == 4
        assert schema.column_order == list(X.columns)
        
        # Check feature specs
        for feature in schema.features:
            assert feature.name in X.columns
            assert feature.dtype == str(X[feature.name].dtype)


class TestMetadata:
    """Test metadata generation."""
    
    def test_metadata_contains_versions(self):
        """Test that metadata contains version information."""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        
        model = RandomForestClassifier(n_estimators=5)
        model.fit(X, y)
        
        pkg = package(model, X)
        
        assert pkg.metadata.deeploi_version is not None
        assert pkg.metadata.python_version is not None
        assert pkg.metadata.created_at is not None
        assert len(pkg.metadata.library_versions) > 0
