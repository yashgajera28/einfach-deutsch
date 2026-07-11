"""Helpers for human evaluation of simplifications."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any


_ANNOTATION_COLUMNS = [
    "original",
    "simplified",
    "simplicity_score",
    "meaning_preservation",
    "fluency",
]


def create_annotation_template(output_path: str | Path) -> None:
    """Write an empty CSV annotation template."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_ANNOTATION_COLUMNS)
        writer.writeheader()


def sample_for_annotation(
    pairs: list[dict[str, Any]], n: int = 50, seed: int = 42
) -> list[dict[str, Any]]:
    """Sample a fixed number of source/simplification pairs for annotation."""
    if n <= 0:
        return []
    rng = random.Random(seed)
    sample = pairs.copy()
    rng.shuffle(sample)
    return sample[: min(n, len(sample))]
