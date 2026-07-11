"""Build the parallel simplification dataset from configured sources."""

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from src.data_collection.bureaucratic import generate_synthetic_pairs, scrape_muster_vorlage
from src.data_collection.klexikon import fetch_klexikon_pairs
from src.data_collection.wiki_align import build_wikipedia_pairs, fetch_demo_pairs
from src.preprocessing.quality import apply_quality_filters
from src.utils.paths import resolve_path

logger = logging.getLogger(__name__)

_SOURCE_LOADERS: dict[str, Any] = {
    "wikipedia": build_wikipedia_pairs,
    "bureaucratic": scrape_muster_vorlage,
    "klexikon": fetch_klexikon_pairs,
}

_DEMO_LOADERS: dict[str, Any] = {
    "wikipedia": fetch_demo_pairs,
    "bureaucratic": generate_synthetic_pairs,
    "klexikon": fetch_klexikon_pairs,
}


def _load_demo_pairs(demo_path: Path) -> list[dict[str, str]]:
    """Load additional demo pairs from the JSONL file if present."""
    if not demo_path.exists():
        return []
    pairs: list[dict[str, str]] = []
    with demo_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "complex" in item and "simple" in item:
                item.setdefault("source", "demo")
                pairs.append(item)
    return pairs


def _article_id(pair: dict[str, str]) -> str:
    """Return a stable article identifier for overlap-aware splitting."""
    text = (pair.get("complex", "") + "\n" + pair.get("simple", ""))[:400]
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _split_by_article(
    pairs: list[dict[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, str]]]:
    """Split pairs into train/val/test without article overlap.

    Articles are greedily assigned so the total number of pairs per split
    stays close to the requested ratios, even when article sizes vary.

    Args:
        pairs: All filtered pairs.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        seed: Random seed.

    Returns:
        Mapping from split name to list of pairs.
    """
    random.seed(seed)

    by_article: dict[str, list[dict[str, str]]] = {}
    for pair in pairs:
        by_article.setdefault(_article_id(pair), []).append(pair)

    groups = list(by_article.values())
    random.shuffle(groups)

    total = len(pairs)
    targets = {
        "train": total * train_ratio,
        "validation": total * val_ratio,
        "test": total * (1.0 - train_ratio - val_ratio),
    }

    splits: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "test": []}
    for group in groups:
        # Assign the group to the split that is currently most below target.
        best_split = min(
            targets,
            key=lambda name: len(splits[name]) / max(targets[name], 1),
        )
        splits[best_split].extend(group)

    for split_name, split_pairs in splits.items():
        random.shuffle(split_pairs)

    return splits


def build_dataset(
    config: dict[str, Any],
    demo: bool = False,
    sources: list[str] | None = None,
) -> DatasetDict:
    """Build the parallel dataset from configured sources.

    Args:
        config: Project configuration dictionary.
        demo: If True, skip large downloads and use only curated/demo pairs.
        sources: List of source names to include; defaults to all sources.

    Returns:
        A HuggingFace ``DatasetDict`` with ``train``, ``validation``, and
        ``test`` splits.
    """
    sources = sources or ["wikipedia", "bureaucratic", "klexikon"]
    max_tokens = config.get("models", {}).get("baseline", {}).get("max_length", 256)

    loaders = _DEMO_LOADERS if demo else _SOURCE_LOADERS
    all_pairs: list[dict[str, str]] = []

    if demo:
        demo_path = Path(config.get("paths", {}).get("data_raw", "data/raw")) / "demo_pairs.jsonl"
        demo_path = demo_path.resolve()
        all_pairs.extend(_load_demo_pairs(demo_path))

    for source in sources:
        loader = loaders.get(source)
        if loader is None:
            logger.warning("Unknown source: %s", source)
            continue

        try:
            pairs = loader()
        except Exception as exc:
            logger.error("Failed to load source %s: %s", source, exc)
            continue

        if not pairs:
            logger.warning("Source %s returned no pairs", source)
            continue

        for pair in pairs:
            pair.setdefault("source", source)
            all_pairs.append(pair)

    logger.info("Loaded %d raw pairs from %d source(s)", len(all_pairs), len(sources))
    filtered = apply_quality_filters(all_pairs, max_tokens=max_tokens)
    logger.info("Retained %d pairs after quality filtering", len(filtered))

    if len(filtered) < 3:
        raise RuntimeError("Not enough pairs to create train/val/test splits")

    splits = _split_by_article(filtered)
    dataset = DatasetDict({name: Dataset.from_list(rows) for name, rows in splits.items()})

    out_dir = resolve_path(config, "data_processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out_dir))
    logger.info("Saved DatasetDict to %s", out_dir)

    return dataset
