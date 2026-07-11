"""Quality filters for parallel simplification data."""

from typing import Any


def remove_identical(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop pairs where the simple text equals the complex text."""
    return [p for p in pairs if p.get("complex", "").strip() != p.get("simple", "").strip()]


def filter_non_german(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove pairs that contain mostly non-German characters."""
    allowed = set(
        " abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "äöüßÄÖÜ"
        "0123456789"
        ".,;:!?-'\"()[]{}«»"
    )
    kept = []
    for pair in pairs:
        text = (pair.get("complex", "") + " " + pair.get("simple", "")).strip()
        if not text:
            continue
        ok = sum(1 for ch in text if ch in allowed or ch.isspace())
        if ok / len(text) >= 0.7:
            kept.append(pair)
    return kept


def filter_length_ratio(
    pairs: list[dict[str, Any]],
    min_ratio: float = 0.5,
    max_ratio: float = 2.0,
) -> list[dict[str, Any]]:
    """Keep pairs whose token-count ratio stays within the given bounds."""
    kept = []
    for pair in pairs:
        complex_toks = len(pair.get("complex", "").split())
        simple_toks = len(pair.get("simple", "").split())
        if complex_toks == 0 or simple_toks == 0:
            continue
        ratio = complex_toks / simple_toks
        if min_ratio <= ratio <= max_ratio:
            kept.append(pair)
    return kept


def filter_max_length(pairs: list[dict[str, Any]], max_tokens: int = 256) -> list[dict[str, Any]]:
    """Drop pairs where either side exceeds the maximum token length."""
    return [
        p
        for p in pairs
        if len(p.get("complex", "").split()) <= max_tokens
        and len(p.get("simple", "").split()) <= max_tokens
    ]


def apply_quality_filters(
    pairs: list[dict[str, Any]],
    max_tokens: int = 256,
) -> list[dict[str, Any]]:
    """Run the default quality-filter pipeline."""
    pairs = remove_identical(pairs)
    pairs = filter_non_german(pairs)
    pairs = filter_length_ratio(pairs)
    pairs = filter_max_length(pairs, max_tokens=max_tokens)
    return pairs
