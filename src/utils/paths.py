"""Path resolution helpers."""

from pathlib import Path
from typing import Any


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a path from config relative to project root."""
    root = config["paths"]["root"]
    value = config["paths"].get(key)
    if value is None:
        raise KeyError(f"Missing path config: {key}")
    return (root / value).resolve()
