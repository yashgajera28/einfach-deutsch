"""Demo script showing how to use the unified Simplifier."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.simplifier import Simplifier


def main() -> None:
    """Print a sample Simplifier usage example without running inference."""
    print("Sample Simplifier usage:")
    print()
    print('    from src.models.simplifier import Simplifier')
    print('    from src.utils.config import load_config')
    print()
    print('    config = load_config("configs/config.yaml")')
    print('    simplifier = Simplifier(config, backend="lora")')
    print('    result = simplifier.simplify(')
    print('        "Der Antrag wurde vom Gericht abgelehnt.",')
    print('        level="B1",')
    print('        preserve_entities=True,')
    print('        explain=True,')
    print('    )')
    print('    print(result["simplified"])')
    print()
    print("Available backends: baseline, lora")
    print("Default backend in this demo: lora")
    print("Default level from config: B1")


if __name__ == "__main__":
    main()
