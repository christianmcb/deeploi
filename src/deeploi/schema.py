"""
Schema inference from sample DataFrames or fitted model metadata.
"""

import pandas as pd
from typing import Any, Dict, List, Optional
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

    feature_names = getattr(model, "feature_names_", None)
    if feature_names is not None:
        try:
            return [str(name) for name in feature_names]
        except Exception:
            pass

    booster = getattr(model, "get_booster", None)
    if callable(booster):
        try:
            xgb_feature_names = booster().feature_names
        except Exception:
            xgb_feature_names = None
        if xgb_feature_names:
            return [str(name) for name in xgb_feature_names]

    lgbm_feature_names = getattr(model, "feature_name_", None)
    if lgbm_feature_names:
        return [str(name) for name in lgbm_feature_names]

    booster_ = getattr(model, "booster_", None)
    if booster_ is not None and hasattr(booster_, "feature_name"):
        try:
            booster_feature_names = booster_.feature_name()
        except Exception:
            booster_feature_names = None
        if booster_feature_names:
            return [str(name) for name in booster_feature_names]

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
    if not isinstance(data, dict):
        raise InvalidSampleError(
            "Each record must be a JSON object (Python dict) of feature names to values"
        )

    # Check for missing required columns
    required_columns = {f.name for f in schema.features if not f.nullable}
    provided_columns = set(data.keys())
    
    missing = required_columns - provided_columns
    if missing:
        missing_cols = ", ".join(sorted(missing))
        expected_cols = ", ".join(schema.column_order)
        raise InvalidSampleError(
            "Missing required columns: "
            f"{missing_cols}. Expected features: [{expected_cols}]"
        )
    
    # Check for extra columns
    allowed_columns = {f.name for f in schema.features}
    extra = provided_columns - allowed_columns
    if extra:
        extra_cols = ", ".join(sorted(extra))
        allowed = ", ".join(schema.column_order)
        raise InvalidSampleError(
            "Unexpected columns: "
            f"{extra_cols}. Allowed features: [{allowed}]"
        )


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
        raise InvalidSampleError(
            "No records provided. Send at least one record under the 'records' field"
        )
    
    for i, record in enumerate(records):
        try:
            validate_input(record, schema)
        except InvalidSampleError as e:
            raise InvalidSampleError(f"Record {i + 1}: {str(e)}")


def get_feature_names(schema: Schema) -> List[str]:
    """Get feature names in correct order from schema."""
    return schema.column_order


def coerce_dataframe_to_schema(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """
    Attempt to coerce DataFrame columns to schema dtypes.

    Args:
        df: Input DataFrame aligned to schema columns
        schema: Target schema with feature dtypes

    Returns:
        Coerced DataFrame

    Raises:
        InvalidSampleError: If coercion fails for any required column
    """
    feature_map: Dict[str, FeatureSpec] = {feature.name: feature for feature in schema.features}
    coerced_df = df.copy()

    for column in schema.column_order:
        feature = feature_map.get(column)
        if feature is None or column not in coerced_df.columns:
            continue

        expected_dtype = (feature.dtype or "unknown").lower()
        if expected_dtype == "unknown":
            continue

        series = coerced_df[column]
        coerced_df[column] = _coerce_series(series, column, expected_dtype, feature.nullable)

    return coerced_df


def _coerce_series(
    series: pd.Series,
    column: str,
    expected_dtype: str,
    nullable: bool,
) -> pd.Series:
    """Coerce a single series to expected dtype with actionable errors."""
    if _is_integer_dtype(expected_dtype):
        return _coerce_integer_series(series, column, expected_dtype, nullable)

    if _is_float_dtype(expected_dtype):
        return _coerce_float_series(series, column, expected_dtype, nullable)

    if _is_bool_dtype(expected_dtype):
        return _coerce_bool_series(series, column, expected_dtype, nullable)

    if _is_datetime_dtype(expected_dtype):
        return _coerce_datetime_series(series, column, expected_dtype, nullable)

    if _is_string_like_dtype(expected_dtype):
        return _coerce_string_series(series, column, expected_dtype, nullable)

    # Unknown or uncommon dtypes are left untouched to avoid over-aggressive casting.
    return series


def _coerce_integer_series(
    series: pd.Series,
    column: str,
    expected_dtype: str,
    nullable: bool,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    failed_mask = series.notna() & numeric.isna()
    if failed_mask.any():
        _raise_type_mismatch(column, series, failed_mask, expected_dtype)

    non_integer_mask = numeric.notna() & ((numeric % 1) != 0)
    if non_integer_mask.any():
        _raise_type_mismatch(column, series, non_integer_mask, expected_dtype)

    if not nullable and numeric.isna().any():
        raise InvalidSampleError(
            f"Column '{column}' contains null or empty values, but model expects non-null '{expected_dtype}'."
        )

    if numeric.isna().any():
        return numeric.astype("Float64")
    return numeric.astype("int64")


def _coerce_float_series(
    series: pd.Series,
    column: str,
    expected_dtype: str,
    nullable: bool,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    failed_mask = series.notna() & numeric.isna()
    if failed_mask.any():
        _raise_type_mismatch(column, series, failed_mask, expected_dtype)

    if not nullable and numeric.isna().any():
        raise InvalidSampleError(
            f"Column '{column}' contains null or empty values, but model expects non-null '{expected_dtype}'."
        )

    return numeric.astype("float64")


def _coerce_bool_series(
    series: pd.Series,
    column: str,
    expected_dtype: str,
    nullable: bool,
) -> pd.Series:
    def _parse_bool(value: Any) -> Optional[bool]:
        if pd.isna(value):
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "t", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "f", "0", "no", "n", "off"}:
                return False
        return None

    parsed = series.map(_parse_bool)
    failed_mask = series.notna() & parsed.isna()
    if failed_mask.any():
        _raise_type_mismatch(column, series, failed_mask, expected_dtype)

    if not nullable and parsed.isna().any():
        raise InvalidSampleError(
            f"Column '{column}' contains null or empty values, but model expects non-null '{expected_dtype}'."
        )

    if parsed.isna().any():
        return parsed.astype("boolean")
    return parsed.astype("bool")


def _coerce_datetime_series(
    series: pd.Series,
    column: str,
    expected_dtype: str,
    nullable: bool,
) -> pd.Series:
    converted = pd.to_datetime(series, errors="coerce")
    failed_mask = series.notna() & converted.isna()
    if failed_mask.any():
        _raise_type_mismatch(column, series, failed_mask, expected_dtype)

    if not nullable and converted.isna().any():
        raise InvalidSampleError(
            f"Column '{column}' contains null or empty values, but model expects non-null '{expected_dtype}'."
        )

    return converted


def _coerce_string_series(
    series: pd.Series,
    column: str,
    expected_dtype: str,
    nullable: bool,
) -> pd.Series:
    if not nullable and series.isna().any():
        raise InvalidSampleError(
            f"Column '{column}' contains null or empty values, but model expects non-null '{expected_dtype}'."
        )

    converted = series.copy()
    not_null = series.notna()
    converted.loc[not_null] = series.loc[not_null].astype(str)
    return converted


def _raise_type_mismatch(
    column: str,
    original: pd.Series,
    failed_mask: pd.Series,
    expected_dtype: str,
) -> None:
    """Raise a detailed type mismatch error with concrete failing examples."""
    failed_values = original[failed_mask].head(3).tolist()
    raise InvalidSampleError(
        f"Column '{column}' looks like '{original.dtype}' with values {failed_values}, "
        f"but model expects '{expected_dtype}'. Deeploi attempted automatic coercion and failed."
    )


def _is_integer_dtype(dtype: str) -> bool:
    return "int" in dtype and "interval" not in dtype


def _is_float_dtype(dtype: str) -> bool:
    return "float" in dtype or dtype in {"double", "real"}


def _is_bool_dtype(dtype: str) -> bool:
    return dtype in {"bool", "boolean"}


def _is_datetime_dtype(dtype: str) -> bool:
    return "datetime" in dtype or dtype in {"date", "timestamp"}


def _is_string_like_dtype(dtype: str) -> bool:
    return dtype in {"object", "string", "str", "category"}
