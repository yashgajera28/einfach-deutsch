"""Start the Streamlit frontend."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Einfach Deutsch frontend")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config")
    parser.add_argument("--server.port", dest="port", default=None, type=int, help="Port")
    parser.add_argument("--host", default="0.0.0.0", help="Server address")
    args = parser.parse_args()

    config = load_config(args.config)
    port = args.port or config["frontend"]["port"]

    app_path = Path(__file__).parent / "src" / "frontend" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        f"--server.port={port}",
        f"--server.address={args.host}",
    ]
    env = {"PYTHONPATH": str(Path(__file__).parent), **dict(os.environ)}
    subprocess.run(cmd, check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
