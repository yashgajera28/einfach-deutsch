"""Run the full data pipeline from raw sources to processed datasets."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_collection.dataset_builder import build_dataset
from src.utils.config import load_config
from src.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Einfach Deutsch datasets")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config")
    parser.add_argument("--demo", action="store_true", help="Use small demo corpus only")
    parser.add_argument("--sources", nargs="+", default=None, help="Sources to include")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("logging", {}))
    logger = logging.getLogger(__name__)

    logger.info("Starting data preparation")
    build_dataset(config, demo=args.demo, sources=args.sources)
    logger.info("Data preparation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
