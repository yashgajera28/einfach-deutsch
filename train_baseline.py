"""Train the baseline mT5-small simplification model."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.baseline_trainer import train
from src.utils.config import load_config
from src.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline mT5-small model")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config")
    parser.add_argument("--demo", action="store_true", help="Run a small demo training")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("logging", {}))
    logger = logging.getLogger(__name__)

    logger.info("Starting baseline training")
    train(config, demo=args.demo, output_dir=args.output_dir)
    logger.info("Baseline training complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
