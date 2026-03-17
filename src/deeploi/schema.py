"""
Schema inference from sample DataFrames.
"""

import pandas as pd
from typing import List, Optional
from deeploi.types import Schema, FeatureSpec
from deeploi.exceptions import InvalidSampleError
from deeploi.utils import validate_dataframe


def infer_schema(sample: pd.DataFrame) -> Schema:
    """
    Infer schema from a sample DataFrame.
    
    Args:
        sample: Sample DataFrame with feature columns
    
    Returns:
        Schema object with feature specs and column order
    
    Raises:
        InvalidSampleError: If sample is invalid
    """
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
