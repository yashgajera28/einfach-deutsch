"""Logging setup."""

import logging
from typing import Any


def setup_logging(config: dict[str, Any]) -> None:
    """Configure root logger from config."""
    level = config.get("level", "INFO")
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
