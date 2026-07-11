"""Start the FastAPI backend server."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from src.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Einfach Deutsch API")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config")
    parser.add_argument("--host", default=None, help="Host")
    parser.add_argument("--port", default=None, type=int, help="Port")
    args = parser.parse_args()

    config = load_config(args.config)
    host = args.host or config["api"]["host"]
    port = args.port or config["api"]["port"]

    uvicorn.run("src.api.main:app", host=host, port=port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
