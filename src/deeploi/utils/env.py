"""
Utility functions for environment and package detection.
"""

import sys
from typing import Dict


def get_python_version() -> str:
    """Get Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_library_versions() -> Dict[str, str]:
    """Get versions of key dependencies."""
    versions = {}
    
    packages = ["pandas", "sklearn", "xgboost", "fastapi", "pydantic"]
    for package_name in packages:
        try:
            if package_name == "sklearn":
                import sklearn
                versions[package_name] = sklearn.__version__
            else:
                module = __import__(package_name)
                versions[package_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[package_name] = "not installed"
    
    return versions
