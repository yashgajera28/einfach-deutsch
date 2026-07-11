"""Configuration loading helpers."""

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and resolve environment overrides."""
    if path is None:
        path = os.environ.get("EINFACH_DEUTSCH_CONFIG", "configs/config.yaml")
    path = Path(path)

    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    config["paths"]["root"] = Path(config["paths"]["root"]).resolve()
    return config
