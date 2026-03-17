"""
File I/O utilities.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: str) -> str:
    """Ensure directory exists, create if needed."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data as JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_text(content: str, filepath: str) -> None:
    """Save text to file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w") as f:
        f.write(content)


def load_text(filepath: str) -> str:
    """Load text from file."""
    with open(filepath, "r") as f:
        return f.read()


def list_files(directory: str, extension: str = None) -> list:
    """List files in directory, optionally filtered by extension."""
    if not os.path.exists(directory):
        return []
    
    files = []
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        if os.path.isfile(path):
            if extension is None or item.endswith(extension):
                files.append(path)
    return files
