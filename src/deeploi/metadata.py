"""
Metadata generation and management.
"""

from datetime import datetime, timezone
from typing import Dict, Any
from deeploi.types import Metadata
from deeploi.constants import __version__
from deeploi.utils import get_python_version, get_library_versions
from deeploi.inspector import get_estimator_info


def create_metadata(
    framework: str,
    task_type: str,
    supports_predict_proba: bool,
    model: Any
) -> Metadata:
    """
    Create metadata object for a model.
    
    Args:
        framework: Framework name (sklearn, xgboost)
        task_type: Task type (classification, regression)
        supports_predict_proba: Whether model supports predict_proba
        model: The model object
    
    Returns:
        Metadata object
    """
    estimator_info = get_estimator_info(model)
    
    return Metadata(
        framework=framework,
        estimator_class=estimator_info["class_name"],
        problem_type=task_type,  # For backward compat
        task_type=task_type,
        supports_predict_proba=supports_predict_proba,
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        python_version=get_python_version(),
        deeploi_version=__version__,
        library_versions=get_library_versions(),
    )
