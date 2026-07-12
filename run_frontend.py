"""Start the NiceGUI frontend."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Einfach Deutsch frontend")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config")
    parser.add_argument("--port", default=None, type=int, help="Port")
    parser.add_argument("--host", default="0.0.0.0", help="Server address")
    args = parser.parse_args()

    config = load_config(args.config)
    port = args.port or config["frontend"]["port"]

    from src.frontend.app import index
    from nicegui import ui

    ui.run(
        host=args.host,
        port=port,
        title="Einfach Deutsch",
        favicon="📝",
        reload=False,
        show=False,
        storage_secret="einfach-deutsch-secret",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
