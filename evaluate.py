"""Run evaluation on a test set."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.runner import run_evaluation
from src.utils.config import load_config
from src.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate simplification model")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config")
    parser.add_argument("--test-file", required=True, help="CSV with source/reference/prediction columns")
    parser.add_argument("--output-dir", default="outputs/evaluation", help="Directory for results")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("logging", {}))
    logger = logging.getLogger(__name__)

    logger.info("Starting evaluation")
    run_evaluation(config, args.test_file, args.output_dir)
    logger.info("Evaluation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
