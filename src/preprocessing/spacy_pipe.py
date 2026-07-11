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
    r"§\s*\d+[a-z]?\s*(?:Abs\.\s*\d+)?\s*(?:Satz\s*\d+)?",
    re.IGNORECASE,
)


def extract_legal_references(text: str) -> list[str]:
    """Extract legal references such as ``§ 123 Abs. 1 Satz 2``.

    Returns an empty list if the input is empty.
    """
    if not text:
        return []
    return _LEGAL_REFERENCE_RE.findall(text)
