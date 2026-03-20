"""
Schema inference from sample DataFrames or fitted model metadata.
"""

import pandas as pd
from typing import Any, List, Optional
from deeploi.types import Schema, FeatureSpec
from deeploi.exceptions import InvalidSampleError
from deeploi.utils import validate_dataframe


def infer_schema(sample: Optional[pd.DataFrame] = None, model: Optional[Any] = None) -> Schema:
    """
    Infer schema from a sample DataFrame or fitted model metadata.
    
    Args:
        sample: Sample DataFrame with feature columns
        model: Fitted model exposing feature metadata
    
    Returns:
        Schema object with feature specs and column order
    
    Raises:
        InvalidSampleError: If neither a valid sample nor model metadata is available
    """
    if sample is None:
        if model is None:
            raise InvalidSampleError(
                "A sample DataFrame is required unless the model exposes fitted feature metadata"
            )
        return infer_schema_from_model(model)

    sample = validate_dataframe(sample)
    
    features = []
    for col in sample.columns:
        dtype_str = str(sample[col].dtype)
        nullable = sample[col].isnull().any()
        
        feature = FeatureSpec(
            name=col,
            dtype=dtype_str,
            nullable=nullable
        )
        features.append(feature)
    
    column_order = list(sample.columns)
    
    return Schema(features=features, column_order=column_order)


def infer_schema_from_model(model: Any) -> Schema:
    """Infer schema from fitted model metadata when no sample is provided."""
    feature_names = _get_model_feature_names(model)
    if not feature_names:
        raise InvalidSampleError(
            "Could not infer schema from model metadata. "
            "Pass sample=... or fit the model with feature names."
        )

    features = [
        FeatureSpec(name=name, dtype="unknown", nullable=False)
        for name in feature_names
    ]
    return Schema(features=features, column_order=feature_names)


def _get_model_feature_names(model: Any) -> Optional[List[str]]:
    """Extract feature names from common fitted estimator attributes."""
    feature_names_in = getattr(model, "feature_names_in_", None)
    if feature_names_in is not None:
        return [str(name) for name in feature_names_in]

    booster = getattr(model, "get_booster", None)
    if callable(booster):
        try:
            xgb_feature_names = booster().feature_names
        except Exception:
            xgb_feature_names = None
        if xgb_feature_names:
            return [str(name) for name in xgb_feature_names]

    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None:
        return [f"feature_{index}" for index in range(int(n_features))]

    return None


def validate_input(data: dict, schema: Schema) -> None:
    """
    Validate a single input record against schema.
    
    Args:
        data: Dictionary with feature values
        schema: Schema to validate against
    
    Raises:
        InvalidSampleError: If validation fails
    """
    # Check for missing required columns
    required_columns = {f.name for f in schema.features if not f.nullable}
    provided_columns = set(data.keys())
    
    missing = required_columns - provided_columns
    if missing:
        raise InvalidSampleError(f"Missing required columns: {missing}")
    
    # Check for extra columns
    allowed_columns = {f.name for f in schema.features}
    extra = provided_columns - allowed_columns
    if extra:
        raise InvalidSampleError(f"Unexpected columns: {extra}")


def validate_batch(records: List[dict], schema: Schema) -> None:
    """
    Validate a batch of records against schema.
    
    Args:
        records: List of dictionaries
        schema: Schema to validate against
    
    Raises:
        InvalidSampleError: If any record fails validation
    """
    if not records:
        raise InvalidSampleError("No records provided")
    
    for i, record in enumerate(records):
        try:
            validate_input(record, schema)
        except InvalidSampleError as e:
            raise InvalidSampleError(f"Record {i}: {str(e)}")


def get_feature_names(schema: Schema) -> List[str]:
    """Get feature names in correct order from schema."""
    return schema.column_order
