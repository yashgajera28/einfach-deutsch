"""Download or read Klexikon-aligned German simplification data.

Klexikon is a manually simplified German encyclopedia for children.
There is no official trivial bulk download, so this module attempts to read
a local alignment file and falls back to a curated demo set.
"""

import json
import logging
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

_KLEXIKON_ALIGNMENT_URLS: list[str] = [
    "https://raw.githubusercontent.com/dkelm/Klexikon/master/klexikon.json",
    "https://raw.githubusercontent.com/christos-c/klexikon/main/klexikon.json",
]

_KLEXIKON_DEMO_PAIRS: list[dict[str, str]] = [
    {
        "complex": "Die Erde ist der dritte Planet in unserem Sonnensystem.",
        "simple": "Die Erde ist der dritte Planet von der Sonne.",
    },
    {
        "complex": "Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff.",
        "simple": "Wasser besteht aus Wasserstoff und Sauerstoff.",
    },
    {
        "complex": "Pflanzen benötigen Sonnenlicht, um durch Photosynthese Nahrung zu produzieren.",
        "simple": "Pflanzen brauchen Licht, um zu wachsen.",
    },
    {
        "complex": "Im Winter dreht sich die Erde mit der Nordhalbkugel von der Sonne weg.",
        "simple": "Im Winter ist es kalt, weil die Erde von der Sonne abgewandt ist.",
    },
    {
        "complex": "Die Menschheit nutzt seit der Industrialisierung vermehrt fossile Brennstoffe.",
        "simple": "Seit der Industrialisierung verbrennen die Menschen mehr Kohle, Öl und Gas.",
    },
]


def _download_alignment(url: str, timeout: int = 30) -> list[dict[str, Any]] | None:
    """Download and parse a JSON alignment file."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.debug("Could not download %s: %s", url, exc)
        return None

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        logger.debug("Invalid JSON at %s: %s", url, exc)
        return None

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("data") or data.get("pairs") or data.get("articles") or []
    return None


def read_local_alignment(path: Path) -> list[dict[str, str]]:
    """Read a local Klexikon alignment file.

    Supports JSON and JSONL formats. Expects objects with ``complex`` and
    ``simple`` fields or ``source``/``target`` synonyms.
    """
    text = path.read_text(encoding="utf-8").strip()
    raw: list[dict[str, Any]] = []
    if text.startswith("["):
        raw = json.loads(text)
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    pairs: list[dict[str, str]] = []
    for item in raw:
        complex_text = item.get("complex") or item.get("source") or item.get("original")
        simple_text = item.get("simple") or item.get("target") or item.get("simplified")
        if complex_text and simple_text:
            pairs.append({"complex": str(complex_text), "simple": str(simple_text), "source": "klexikon"})
    return pairs


def fetch_klexikon_pairs(
    local_path: Path | str | None = None,
    urls: list[str] | None = None,
) -> list[dict[str, str]]:
    """Return Klexikon sentence pairs, falling back to a demo set.

    Args:
        local_path: Optional path to a local alignment file.
        urls: Optional list of remote alignment URLs to try.

    Returns:
        Aligned complex/simple pairs with ``source`` set to ``klexikon``.
    """
    if local_path is not None:
        path = Path(local_path)
        if path.exists():
            logger.info("Reading Klexikon data from %s", path)
            return read_local_alignment(path)

    for url in urls or _KLEXIKON_ALIGNMENT_URLS:
        logger.info("Attempting to download Klexikon data from %s", url)
        data = _download_alignment(url)
        if data:
            pairs: list[dict[str, str]] = []
            for item in data:
                complex_text = item.get("complex") or item.get("source") or item.get("original")
                simple_text = item.get("simple") or item.get("target") or item.get("simplified")
                if complex_text and simple_text:
                    pairs.append({"complex": str(complex_text), "simple": str(simple_text), "source": "klexikon"})
            if pairs:
                return pairs

    logger.info("No Klexikon source available; using demo pairs")
    return [dict(pair, source="klexikon") for pair in _KLEXIKON_DEMO_PAIRS]
