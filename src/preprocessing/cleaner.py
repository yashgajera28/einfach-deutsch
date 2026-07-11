"""Text cleaning helpers for Wikipedia markup and generic HTML."""

import html
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

try:
    import mwparserfromhell
except Exception:  # pragma: no cover
    mwparserfromhell = None  # type: ignore[assignment]

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore[assignment,misc]

_WIKI_TAG_REMOVE = {"ref", "table", "gallery", "math"}


def _collapse_whitespace(text: str) -> str:
    """Collapse consecutive whitespace into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def clean_wiki_markup(text: str) -> str:
    """Remove references, tables, infoboxes, and templates from wikitext.

    Uses ``mwparserfromhell`` when available and falls back to regular
    expressions and HTML-tag stripping otherwise.
    """
    if not text:
        return ""

    text = html.unescape(text)

    if mwparserfromhell is not None:
        try:
            code: Any = mwparserfromhell.parse(text)
            for template in code.filter_templates():
                code.remove(template)
            for tag in code.filter_tags():
                if str(tag.name).lower().strip() in _WIKI_TAG_REMOVE:
                    code.remove(tag)
            return _collapse_whitespace(code.strip_code())
        except Exception as exc:
            logger.warning("mwparserfromhell cleaning failed: %s", exc)

    cleaned = re.sub(r"<ref\b[^>]*>.*?</ref>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(
        r"<gallery\b[^>]*>.*?</gallery>", "", cleaned, flags=re.IGNORECASE | re.DOTALL
    )
    cleaned = re.sub(r"<math\b[^>]*>.*?</math>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    while re.search(r"\{\{[^{}]*\}\}", cleaned):
        cleaned = re.sub(r"\{\{[^{}]*\}\}", "", cleaned)
    cleaned = re.sub(r"\{\|.*?\|\}", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(
        r"\[\[(?:File|Image|Datei):[^\]]*\]\]", "", cleaned, flags=re.IGNORECASE
    )
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", cleaned)
    cleaned = re.sub(r"'{2,6}", "", cleaned)
    cleaned = re.sub(r"={2,6}\s*(.*?)\s*={2,6}", r"\1", cleaned)
    return _collapse_whitespace(cleaned)


def clean_html(text: str) -> str:
    """Extract plain text from HTML and collapse whitespace."""
    if not text:
        return ""

    text = html.unescape(text)
    if BeautifulSoup is not None:
        soup = BeautifulSoup(text, "html.parser")
        plain = soup.get_text(separator=" ")
    else:
        plain = re.sub(r"<[^>]+>", " ", text)
    return _collapse_whitespace(plain)
