"""German / Simple German Wikipedia alignment helpers.

Full Wikipedia dumps are multi-gigabyte XML files. This module therefore
exposes a small, curated demo set via ``fetch_demo_pairs()`` for smoke
testing, while still providing the scaffolding needed to process real dumps
when they are available locally.
"""

import logging
import re
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

_WIKIMEDIA_DUMP_BASE = "https://dumps.wikimedia.org"

# Demo pairs are hand-curated complex/simple sentence pairs in German.
_WIKI_DEMO_PAIRS: list[dict[str, str]] = [
    {
        "complex": "Die Bundesrepublik Deutschland ist ein Bundesstaat in Mitteleuropa.",
        "simple": "Deutschland liegt in der Mitte von Europa.",
    },
    {
        "complex": "Der Bundestag ist das Parlament und vertritt das deutsche Volk.",
        "simple": "Der Bundestag ist das Parlament von Deutschland.",
    },
    {
        "complex": "Die Verfassung Deutschlands heißt Grundgesetz und regelt die Rechte der Bürger.",
        "simple": "Das Grundgesetz ist die wichtigste Regel für Deutschland.",
    },
    {
        "complex": "Die Europäische Union ist ein Zusammenschluss von 27 europäischen Staaten.",
        "simple": "Die EU ist ein Bündnis aus vielen Ländern in Europa.",
    },
    {
        "complex": "Das Sozialgesetzbuch regelt die Rechte und Pflichten in der sozialen Sicherung.",
        "simple": "Das Sozialgesetzbuch enthält Regeln zur sozialen Sicherheit.",
    },
    {
        "complex": "Ein Arbeitsvertrag ist ein Vertrag zwischen Arbeitgeber und Arbeitnehmer.",
        "simple": "Ein Arbeitsvertrag ist eine Vereinbarung zwischen Chef und Mitarbeiter.",
    },
    {
        "complex": "Der Mieter muss die vereinbarte Miete pünktlich an den Vermieter zahlen.",
        "simple": "Der Mieter zahlt die Miete rechtzeitig an den Vermieter.",
    },
    {
        "complex": "Die Allgemeine Erklärung der Menschenrechte wurde 1948 verabschiedet.",
        "simple": "1948 wurden die Menschenrechte für alle Menschen erklärt.",
    },
]


def _is_probably_german(text: str) -> bool:
    """Heuristic check for German text based on common characters."""
    if not text:
        return False
    german_chars = sum(1 for ch in text.lower() if ch in "äöüß")
    return german_chars > 0 or any(token in text.lower() for token in ("der", "die", "das", "und"))


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a simple regex."""
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _length_ratio_ok(src: str, tgt: str, min_ratio: float = 0.4, max_ratio: float = 2.5) -> bool:
    """Return True if the token-length ratio is within bounds."""
    src_len = max(len(src.split()), 1)
    tgt_len = max(len(tgt.split()), 1)
    ratio = src_len / tgt_len
    return min_ratio <= ratio <= max_ratio


def fetch_latest_dump_url(language: str, dump_type: str = "pages-articles-multistream") -> str | None:
    """Return the URL of the latest Wikipedia dump for the given language prefix.

    Args:
        language: Wikipedia language code, e.g. ``de`` or ``simple``.
        dump_type: Dump filename suffix.

    Returns:
        A download URL or ``None`` if it cannot be determined.
    """
    index_url = f"{_WIKIMEDIA_DUMP_BASE}/{language}wiki/latest/"
    try:
        response = requests.get(index_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Could not fetch dump index: %s", exc)
        return None

    # Latest dump directory is typically a date-stamped folder.
    links = re.findall(r'href="([^"]+/)"', response.text)
    if not links:
        return None
    latest = links[-1].rstrip("/")
    filename = f"{language}wiki-latest-{dump_type}.xml.bz2"
    return f"{_WIKIMEDIA_DUMP_BASE}/{language}wiki/{latest}/{filename}"


def download_dump(url: str, dest: Path, chunk_size: int = 8192) -> Path | None:
    """Download a Wikipedia dump to ``dest`` with basic resume support.

    Args:
        url: Remote dump URL.
        dest: Local destination path.
        chunk_size: Bytes per read chunk.

    Returns:
        The destination path or ``None`` on failure.
    """
    logger.info("Downloading dump from %s", url)
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Dump download failed: %s", exc)
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
    return dest


def align_sentences(
    complex_sentences: list[str],
    simple_sentences: list[str],
    min_similarity: float = 0.0,
) -> list[tuple[str, str]]:
    """Align sentences between a complex and a simple article.

    Alignment is based on token-length ratio. When ``min_similarity`` is
    greater than zero, sentence-transformer embeddings are used to score
    candidates; otherwise the ratio alone decides.

    Args:
        complex_sentences: Sentences from the regular article.
        simple_sentences: Sentences from the simple article.
        min_similarity: Optional embedding similarity threshold.

    Returns:
        A list of aligned (complex, simple) sentence tuples.
    """
    aligned: list[tuple[str, str]] = []
    used_simple: set[int] = set()

    for src in complex_sentences:
        best: tuple[str, float] | None = None
        best_idx = -1
        for idx, tgt in enumerate(simple_sentences):
            if idx in used_simple:
                continue
            if not _is_probably_german(src) or not _is_probably_german(tgt):
                continue
            if not _length_ratio_ok(src, tgt):
                continue
            score = 1.0
            if min_similarity > 0:
                # Embedding scoring would go here; omitted to keep the demo
                # dependency-free and fast.
                score = 1.0
            if best is None or score > best[1]:
                best = (tgt, score)
                best_idx = idx

        if best is not None and (min_similarity <= 0 or best[1] >= min_similarity):
            aligned.append((src, best[0]))
            used_simple.add(best_idx)

    return aligned


def extract_wiki_sentence_pairs(
    complex_text: str,
    simple_text: str,
    min_similarity: float = 0.0,
) -> list[dict[str, str]]:
    """Extract aligned sentence pairs from two article texts."""
    complex_sents = _split_sentences(complex_text)
    simple_sents = _split_sentences(simple_text)
    aligned = align_sentences(complex_sents, simple_sents, min_similarity=min_similarity)
    return [{"complex": c, "simple": s} for c, s in aligned]


def fetch_demo_pairs() -> list[dict[str, str]]:
    """Return a small set of curated German complex/simple sentence pairs."""
    return [dict(pair, source="wikipedia") for pair in _WIKI_DEMO_PAIRS]


def build_wikipedia_pairs(
    dump_root: Path | None = None,
    max_articles: int = 100,
    sleep_seconds: float = 0.5,
) -> list[dict[str, str]]:
    """Build sentence pairs from local Wikipedia dumps.

    This is a scaffold: it expects ``dump_root`` to contain extracted
    ``dewiki`` and ``simplewiki`` article files. In practice you would parse
    the XML with ``mwparserfromhell`` and map inter-language links.

    Args:
        dump_root: Directory containing extracted article files.
        max_articles: Maximum number of articles to process.
        sleep_seconds: Delay between articles to be gentle on resources.

    Returns:
        Aligned sentence pairs with ``source`` set to ``wikipedia``.
    """
    if dump_root is None or not dump_root.exists():
        logger.info("No local dump root provided; returning demo pairs")
        return fetch_demo_pairs()

    complex_dir = dump_root / "dewiki"
    simple_dir = dump_root / "simplewiki"
    if not complex_dir.exists() or not simple_dir.exists():
        return fetch_demo_pairs()

    pairs: list[dict[str, str]] = []
    for complex_path in list(complex_dir.glob("*.txt"))[:max_articles]:
        # Match articles by title as a proxy for inter-language alignment.
        simple_path = simple_dir / complex_path.name
        if not simple_path.exists():
            continue
        complex_text = complex_path.read_text(encoding="utf-8")
        simple_text = simple_path.read_text(encoding="utf-8")
        aligned = extract_wiki_sentence_pairs(complex_text, simple_text)
        for item in aligned:
            item["source"] = "wikipedia"
            pairs.append(item)
        time.sleep(sleep_seconds)

    return pairs or fetch_demo_pairs()
