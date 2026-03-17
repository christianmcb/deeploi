"""
Type definitions and data models for Deeploi.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from enum import Enum
import json


class TaskType(str, Enum):
    """Enum for task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class Framework(str, Enum):
    """Enum for supported frameworks."""
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"


@dataclass
class FeatureSpec:
    """Specification for a single feature."""
    name: str
    dtype: str
    nullable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSpec":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Schema:
    """Schema for model input."""
    features: List[FeatureSpec]
    column_order: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "features": [f.to_dict() for f in self.features],
            "column_order": self.column_order,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Schema":
        """Create from dictionary."""
        features = [FeatureSpec.from_dict(f) for f in data.get("features", [])]
        column_order = data.get("column_order", [])
        return cls(features=features, column_order=column_order)

    @classmethod
    def from_json(cls, json_str: str) -> "Schema":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class Metadata:
    """Metadata for a model artifact."""
    framework: str
    estimator_class: str
    problem_type: str
    task_type: str
    supports_predict_proba: bool
    created_at: str
    python_version: str
    deeploi_version: str
    library_versions: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Metadata":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "Metadata":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class PredictionResponse:
    """Standard prediction response."""
    predictions: List[Any]
    probabilities: Optional[List[Dict[str, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"predictions": self.predictions}
        if self.probabilities is not None:
            result["probabilities"] = self.probabilities
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
