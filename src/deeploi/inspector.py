"""
Model inspection to detect type, framework, and capabilities.
"""

from typing import Tuple, Optional, Any
from deeploi.exceptions import UnsupportedModelError
from deeploi.constants import (
    SKLEARN, XGBOOST, LIGHTGBM, CATBOOST,
    SKLEARN_CLASSIFIER, SKLEARN_REGRESSOR,
    XGBOOST_CLASSIFIER, XGBOOST_REGRESSOR,
    LIGHTGBM_CLASSIFIER, LIGHTGBM_REGRESSOR,
    CATBOOST_CLASSIFIER, CATBOOST_REGRESSOR,
    CLASSIFICATION, REGRESSION
)


def inspect_model(model: Any) -> Tuple[str, str, str, bool]:
    """
    Inspect a model and return (framework, model_type, task_type, supports_predict_proba).
    
    Returns:
        Tuple of (framework, model_type_str, task_type, supports_predict_proba)
    
    Raises:
        UnsupportedModelError: If model is not supported.
    """
    framework = _detect_framework(model)
    model_type, task_type, supports_proba = _detect_model_type(model, framework)
    
    return framework, model_type, task_type, supports_proba


def _detect_framework(model: Any) -> str:
    """Detect if model is sklearn-compatible, xgboost, lightgbm, or catboost."""
    model_class = model.__class__
    module_name = model_class.__module__
    class_name = model_class.__name__
    
    # Check class name patterns first (more reliable)
    xgb_patterns = ["XGB", "xgb"]
    for pattern in xgb_patterns:
        if pattern in class_name:
            return XGBOOST

    lgbm_patterns = ["LGBM", "LGB", "LightGBM"]
    for pattern in lgbm_patterns:
        if pattern in class_name:
            return LIGHTGBM

    cat_patterns = ["CatBoost", "Catboost", "CATBOOST"]
    for pattern in cat_patterns:
        if pattern in class_name:
            return CATBOOST
    
    # Then check module name
    if "sklearn" in module_name:
        return SKLEARN
    elif "ngboost" in module_name:
        # NGBoost follows the sklearn estimator interface.
        return SKLEARN
    elif "imblearn" in module_name:
        # imbalanced-learn meta-estimators are sklearn-compatible wrappers.
        return SKLEARN
    elif "xgboost" in module_name:
        return XGBOOST
    elif "lightgbm" in module_name:
        return LIGHTGBM
    elif "catboost" in module_name:
        return CATBOOST
    # Support sklearn-compatible third-party estimators that follow the estimator contract.
    elif hasattr(model, "_estimator_type") and hasattr(model, "predict"):
        return SKLEARN
    
    raise UnsupportedModelError(
        f"Model framework not supported. "
        f"Expected sklearn-compatible, xgboost, lightgbm, or catboost, got {module_name} ({class_name})"
    )


def _detect_model_type(model: Any, framework: str) -> Tuple[str, str, bool]:
    """
    Detect model type and whether it supports predict_proba.
    
    Returns:
        Tuple of (model_type_str, task_type, supports_predict_proba)
    """
    base_name = model.__class__.__name__
    
    # Classification indicators
    is_classifier = _is_classifier(model, base_name)
    
    if framework == SKLEARN:
        if is_classifier:
            task_type = CLASSIFICATION
            supports_proba = hasattr(model, "predict_proba")
            return SKLEARN_CLASSIFIER, task_type, supports_proba
        else:
            task_type = REGRESSION
            return SKLEARN_REGRESSOR, task_type, False
    
    elif framework == XGBOOST:
        if is_classifier:
            task_type = CLASSIFICATION
            supports_proba = True  # XGBoost classifiers support predict_proba
            return XGBOOST_CLASSIFIER, task_type, supports_proba
        else:
            task_type = REGRESSION
            return XGBOOST_REGRESSOR, task_type, False

    elif framework == LIGHTGBM:
        if is_classifier:
            task_type = CLASSIFICATION
            supports_proba = hasattr(model, "predict_proba")
            return LIGHTGBM_CLASSIFIER, task_type, supports_proba
        else:
            task_type = REGRESSION
            return LIGHTGBM_REGRESSOR, task_type, False

    elif framework == CATBOOST:
        if is_classifier:
            task_type = CLASSIFICATION
            supports_proba = hasattr(model, "predict_proba")
            return CATBOOST_CLASSIFIER, task_type, supports_proba
        else:
            task_type = REGRESSION
            return CATBOOST_REGRESSOR, task_type, False
    
    raise UnsupportedModelError(f"Unknown framework: {framework}")


def _is_classifier(model: Any, class_name: str) -> bool:
    """Determine if model is a classifier based on class attributes and name."""
    # Check for sklearn/xgboost classifier indicator
    if hasattr(model, "_estimator_type"):
        return model._estimator_type == "classifier"
    
    # Check by class name patterns
    classifier_patterns = ["classifier", "Classifier"]
    for pattern in classifier_patterns:
        if pattern in class_name:
            return True
    
    # Default: if not a classifier, assume regressor
    return False


def get_estimator_info(model: Any) -> dict:
    """Get metadata about the estimator."""
    return {
        "class_name": model.__class__.__name__,
        "module": model.__class__.__module__,
        "has_predict_proba": hasattr(model, "predict_proba"),
        "has_classes": hasattr(model, "classes_"),
    }
