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
        raise InvalidSampleError("Sample cannot be None")
    
    if isinstance(df, pd.DataFrame):
        if df.empty:
            raise InvalidSampleError("Sample DataFrame is empty")
        return df
    
    try:
        df = pd.DataFrame(df)
        if df.empty:
            raise InvalidSampleError("Sample DataFrame is empty")
        return df
    except Exception as e:
        raise InvalidSampleError(f"Could not convert sample to DataFrame: {str(e)}")


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
        raise InvalidSampleError("Records list is empty")
    return pd.DataFrame(records)


def select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Select specific columns from DataFrame."""
    missing = set(columns) - set(df.columns)
    if missing:
        raise InvalidSampleError(f"Missing columns in input: {missing}")
    return df[columns]
