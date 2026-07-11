"""Lazy spaCy German NLP pipeline."""

from __future__ import annotations

import logging
import os
import re
import warnings
from typing import Any

logger = logging.getLogger(__name__)

_SPACY_MODEL = os.environ.get("SPACY_MODEL", "de_core_news_lg")

_nlp: Any | None = None


def _load_nlp() -> Any | None:
    """Load the German spaCy pipeline on first call."""
    global _nlp
    if _nlp is not None:
        return _nlp

    try:
        import spacy
    except Exception as exc:
        warnings.warn(f"spaCy is not installed: {exc}", stacklevel=2)
        _nlp = None
        return _nlp

    try:
        _nlp = spacy.load(_SPACY_MODEL, disable=["textcat"])
    except Exception as exc:
        warnings.warn(
            f"Could not load spaCy model '{_SPACY_MODEL}': {exc}",
            stacklevel=2,
        )
        _nlp = None

    return _nlp


def get_nlp() -> Any | None:
    """Return the lazy-loaded German spaCy pipeline or ``None``."""
    return _load_nlp()


def sentences(text: str) -> list[str]:
    """Segment ``text`` into sentences.

    Falls back to a simple regex split if spaCy is unavailable.
    """
    if not text:
        return []

    nlp = _load_nlp()
    if nlp is None:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]

    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def entities(text: str) -> list[tuple[str, str]]:
    """Extract named entities as ``(text, label)`` tuples.

    Returns an empty list if the spaCy model is not available.
    """
    if not text:
        return []

    nlp = _load_nlp()
    if nlp is None:
        return []

    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


_LEGAL_REFERENCE_RE = re.compile(
    r"(?:§\s*\d+[a-z]?|Art\.\s*\d+[a-z]?|Artikel\s*\d+[a-z]?)"
    r"(?:\s*Abs(?:atz)?\.?\s*\d+[a-z]?)?"
    r"(?:\s*Satz\s*\d+)?"
    r"(?:\s*Nr\.\s*\d+)?"
    r"(?:\s*[A-Z][a-z]+\w*\.?\s*\d+)?",
    re.IGNORECASE,
)


_DATE_RE = re.compile(
    r"\b\d{1,2}\.\s*(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember|\d{1,2})\s*\d{4}\b",
    re.IGNORECASE,
)


_AMOUNT_RE = re.compile(
    r"(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d{1,2})?\s*(?:€|EUR|USD|\$|GBP|£|%|Prozent|kg|g|mg|km|m|cm|mm|l|ml)(?=\s|$|[.,;])",
    re.IGNORECASE,
)


def extract_legal_references(text: str) -> list[str]:
    """Extract legal references such as ``§ 123 Abs. 1 Satz 2``.

    Returns an empty list if the input is empty.
    """
    if not text:
        return []
    matches = _LEGAL_REFERENCE_RE.findall(text)
    return [m.strip() for m in matches if m.strip() and re.search(r"\d", m)]


def extract_amounts(text: str) -> list[str]:
    """Extract amounts including currency, percentages, and units.

    Returns an empty list if the input is empty.
    """
    if not text:
        return []
    return _AMOUNT_RE.findall(text)


def extract_dates(text: str) -> list[str]:
    """Extract date strings from ``text``.

    Returns an empty list if the input is empty.
    """
    if not text:
        return []
    return _DATE_RE.findall(text)


def extract_entities(text: str) -> list[dict[str, str]]:
    """Extract named entities as ``{text, label}`` dictionaries.

    Falls back to regex-based dates, amounts, and legal references when
    the spaCy model is not available.
    """
    if not text:
        return []

    nlp = _load_nlp()
    if nlp is None:
        return _rule_based_entities(text)

    doc = nlp(text)
    result: list[dict[str, str]] = [
        {"text": ent.text, "label": ent.label_} for ent in doc.ents
    ]
    seen = {item["text"].lower() for item in result}
    for item in _rule_based_entities(text):
        if item["text"].lower() not in seen:
            result.append(item)
    return result


def _rule_based_entities(text: str) -> list[dict[str, str]]:
    """Return date, amount, and legal reference entities without spaCy."""
    result: list[dict[str, str]] = []
    for value in extract_dates(text):
        result.append({"text": value, "label": "DATE"})
    for value in extract_amounts(text):
        result.append({"text": value, "label": "AMOUNT"})
    for value in extract_legal_references(text):
        result.append({"text": value, "label": "LEGAL"})
    return result
