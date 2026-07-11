"""Dataset builder placeholder."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_dataset(config: dict[str, Any], demo: bool = False, sources: list[str] | None = None) -> None:
    """Build the parallel dataset from configured sources."""
    out = Path(config["paths"]["data_processed"])
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Dataset builder placeholder; demo=%s sources=%s", demo, sources)
