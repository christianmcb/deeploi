"""
Tests for schema inference and validation.
"""

import pytest
import pandas as pd
import numpy as np

from deeploi.schema import infer_schema, validate_input, validate_batch
from deeploi.types import Schema, FeatureSpec
from deeploi.exceptions import InvalidSampleError, SchemaValidationError


class TestSchemaInference:
    """Test schema inference from DataFrames."""
    
    def test_infer_simple_schema(self):
        """Test inferring schema from simple DataFrame."""
        df = pd.DataFrame({
            "col_a": [1.0, 2.0, 3.0],
            "col_b": ["x", "y", "z"],
        })
        
        schema = infer_schema(df)
        
        assert len(schema.features) == 2
        assert schema.column_order == ["col_a", "col_b"]
        
        feature_a = next(f for f in schema.features if f.name == "col_a")
        assert feature_a.dtype == "float64"
        assert not feature_a.nullable
    
    def test_infer_schema_with_nulls(self):
        """Test that nullable is detected."""
        df = pd.DataFrame({
            "col_a": [1.0, np.nan, 3.0],
            "col_b": [1, 2, 3],
        })
        
        schema = infer_schema(df)
        
        feature_a = next(f for f in schema.features if f.name == "col_a")
        assert feature_a.nullable
        
        feature_b = next(f for f in schema.features if f.name == "col_b")
        assert not feature_b.nullable
    
    def test_infer_schema_preserves_column_order(self):
        """Test that column order is preserved."""
        df = pd.DataFrame({
            "z": [1, 2, 3],
            "a": [4, 5, 6],
            "m": [7, 8, 9],
        })
        
        schema = infer_schema(df)
        
        assert schema.column_order == ["z", "a", "m"]
    
    def test_infer_schema_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises error."""
        df = pd.DataFrame()
        
        with pytest.raises(InvalidSampleError):
            infer_schema(df)


class TestSchemaValidation:
    """Test input validation against schema."""
    
    @pytest.fixture
    def simple_schema(self):
        """Fixture for a simple schema."""
        features = [
            FeatureSpec(name="col_a", dtype="float64", nullable=False),
            FeatureSpec(name="col_b", dtype="int64", nullable=False),
            FeatureSpec(name="col_c", dtype="object", nullable=True),
        ]
        return Schema(features=features, column_order=["col_a", "col_b", "col_c"])
    
    def test_validate_valid_input(self, simple_schema):
        """Test that valid input passes validation."""
        record = {"col_a": 1.0, "col_b": 2, "col_c": "hello"}
        
        # Should not raise
        validate_input(record, simple_schema)
    
    def test_validate_missing_required_column(self, simple_schema):
        """Test that missing required columns raise error."""
        record = {"col_a": 1.0, "col_c": "hello"}  # Missing col_b
        
        with pytest.raises(InvalidSampleError) as excinfo:
            validate_input(record, simple_schema)
        
        assert "col_b" in str(excinfo.value)
    
    def test_validate_extra_columns(self, simple_schema):
        """Test that extra columns raise error."""
        record = {
            "col_a": 1.0,
            "col_b": 2,
            "col_c": "hello",
            "col_d": "extra",
        }
        
        with pytest.raises(InvalidSampleError) as excinfo:
            validate_input(record, simple_schema)
        
        assert "col_d" in str(excinfo.value)
    
    def test_validate_nullable_field_missing(self, simple_schema):
        """Test that nullable fields can be missing."""
        record = {"col_a": 1.0, "col_b": 2}  # col_c is nullable, so OK
        
        # Should not raise
        validate_input(record, simple_schema)
    
    def test_validate_batch(self, simple_schema):
        """Test batch validation."""
        records = [
            {"col_a": 1.0, "col_b": 2, "col_c": "hello"},
            {"col_a": 2.0, "col_b": 3, "col_c": "world"},
        ]
        
        # Should not raise
        validate_batch(records, simple_schema)
    
    def test_validate_batch_with_error(self, simple_schema):
        """Test batch validation with error in one record."""
        records = [
            {"col_a": 1.0, "col_b": 2, "col_c": "hello"},
            {"col_a": 2.0},  # Missing col_b and col_c
        ]
        
        with pytest.raises(InvalidSampleError) as excinfo:
            validate_batch(records, simple_schema)
        
        assert "Record 2" in str(excinfo.value)
    
    def test_validate_empty_batch(self, simple_schema):
        """Test that empty batch raises error."""
        with pytest.raises(InvalidSampleError):
            validate_batch([], simple_schema)


class TestSchemaConversion:
    """Test Schema serialization and deserialization."""
    
    def test_schema_to_dict(self):
        """Test converting schema to dict."""
        features = [
            FeatureSpec(name="a", dtype="float64"),
            FeatureSpec(name="b", dtype="int64"),
        ]
        schema = Schema(features=features, column_order=["a", "b"])
        
        schema_dict = schema.to_dict()
        
        assert schema_dict["column_order"] == ["a", "b"]
        assert len(schema_dict["features"]) == 2
        assert schema_dict["features"][0]["name"] == "a"
    
    def test_schema_from_dict(self):
        """Test creating schema from dict."""
        schema_dict = {
            "features": [
                {"name": "a", "dtype": "float64", "nullable": False},
                {"name": "b", "dtype": "int64", "nullable": True},
            ],
            "column_order": ["a", "b"],
        }
        
        schema = Schema.from_dict(schema_dict)
        
        assert len(schema.features) == 2
        assert schema.column_order == ["a", "b"]
        assert schema.features[1].nullable
    
    def test_schema_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        features = [
            FeatureSpec(name="x", dtype="float64"),
            FeatureSpec(name="y", dtype="object"),
        ]
        original = Schema(features=features, column_order=["x", "y"])
        
        # Serialize and deserialize
        json_str = original.to_json()
        restored = Schema.from_json(json_str)
        
        assert restored.column_order == original.column_order
        assert len(restored.features) == len(original.features)
