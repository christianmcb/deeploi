"""
Hashing utilities for artifact versioning.
"""

import hashlib
import json
from typing import Any


def hash_object(obj: Any) -> str:
    """Create a SHA256 hash of a JSON-serializable object."""
    json_str = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


def hash_file(filepath: str, chunk_size: int = 65536) -> str:
    """Create a SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
