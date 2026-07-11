"""Evaluation runner placeholder."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_evaluation(config: dict[str, Any], test_file: str, output_dir: str) -> None:
    """Run evaluation on a test set and save results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Evaluation runner not yet implemented; test_file=%s output=%s", test_file, out)
