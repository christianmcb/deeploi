"""
DataFrame utilities for sample validation and conversion.
"""

import pandas as pd
import numpy as np
from typing import Any, List, Dict
from deeploi.exceptions import InvalidSampleError


def validate_dataframe(df: Any) -> pd.DataFrame:
    """Validate and convert input to DataFrame."""
    if df is None:
        raise InvalidSampleError(
            "Input data is None. Provide a pandas DataFrame or a list of record dictionaries"
        )
    
    if isinstance(df, pd.DataFrame):
        if df.empty:
            raise InvalidSampleError(
                "Input DataFrame is empty. Provide at least one row for prediction or schema inference"
            )
        return df
    
    try:
        df = pd.DataFrame(df)
        if df.empty:
            raise InvalidSampleError(
                "Converted DataFrame is empty. Provide at least one non-empty record"
            )
        return df
    except Exception as e:
        raise InvalidSampleError(
            "Could not convert input to DataFrame. "
            f"Received type: {type(df).__name__}. Original error: {str(e)}"
        )


def infer_dtypes(df: pd.DataFrame) -> Dict[str, str]:
    """Infer data types from DataFrame."""
    dtypes = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        dtypes[col] = dtype
    return dtypes


def to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to list of records (dicts)."""
    return df.to_dict(orient="records")


def from_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list of records to DataFrame."""
    if not records:
        raise InvalidSampleError(
            "Records list is empty. Send at least one object in the 'records' array"
        )
    return pd.DataFrame(records)


def select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Select specific columns from DataFrame."""
    missing = set(columns) - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        expected_cols = ", ".join(columns)
        provided_cols = ", ".join(str(c) for c in df.columns)
        raise InvalidSampleError(
            "Missing columns in input: "
            f"{missing_cols}. Expected order: [{expected_cols}]. "
            f"Provided columns: [{provided_cols}]"
        )
    return df[columns]
