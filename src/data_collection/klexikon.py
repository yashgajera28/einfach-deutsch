"""Download Klexikon-aligned German simplification data from Hugging Face.

Klexikon is a manually simplified German encyclopedia for children. The
``dennlinger/klexikon`` dataset on Hugging Face provides document-aligned
Wikipedia / Klexikon sentence lists. This module aligns them with a
sentence-transformer similarity model and flattens them into high-quality
complex/simple sentence pairs.
"""

import json
import logging
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

_KLEXIKON_HF_NAME: str = "dennlinger/klexikon"

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


def _load_similarity_model() -> Any | None:
    """Load a multilingual sentence-transformer model for alignment, if available."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        logger.debug("sentence-transformers unavailable: %s", exc)
        return None

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    try:
        model = SentenceTransformer(model_name)
        logger.info("Loaded sentence-transformer model for Klexikon alignment")
        return model
    except Exception as exc:
        logger.warning("Could not load %s: %s", model_name, exc)
        return None


def _encode_batches(texts: list[str], model: Any, batch_size: int = 256) -> Any:
    """Encode a list of texts in batches using the sentence-transformer model."""
    import numpy as np

    embeddings: list[Any] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)
    return np.vstack(embeddings)


def _align_articles(
    articles: list[tuple[list[str], list[str]]],
    model: Any,
    threshold: float = 0.45,
) -> list[tuple[str, str]]:
    """Align sentences across all articles in a single batched encoding pass."""
    import numpy as np

    wiki_records: list[tuple[int, int, str]] = []
    klex_records: list[tuple[int, int, str]] = []

    for art_idx, (wiki_sents, klex_sents) in enumerate(articles):
        for sent_idx, text in enumerate(wiki_sents):
            if len(text.split()) >= 4:
                wiki_records.append((art_idx, sent_idx, text))
        for sent_idx, text in enumerate(klex_sents):
            if len(text.split()) >= 4:
                klex_records.append((art_idx, sent_idx, text))

    if not wiki_records or not klex_records:
        return []

    wiki_texts = [r[2] for r in wiki_records]
    klex_texts = [r[2] for r in klex_records]

    wiki_embs = _encode_batches(wiki_texts, model)
    klex_embs = _encode_batches(klex_texts, model)

    wiki_embs = wiki_embs / (np.linalg.norm(wiki_embs, axis=1, keepdims=True) + 1e-8)
    klex_embs = klex_embs / (np.linalg.norm(klex_embs, axis=1, keepdims=True) + 1e-8)

    similarity = wiki_embs @ klex_embs.T

    by_article: dict[int, tuple[list[tuple[int, int, float]], list[tuple[int, int]]]] = {}
    for i, (art_idx, sent_idx, _) in enumerate(wiki_records):
        by_article.setdefault(art_idx, ([], []))[0].append((i, sent_idx, -1.0))

    klex_by_article: dict[int, list[tuple[int, int]]] = {}
    for j, (art_idx, sent_idx, _) in enumerate(klex_records):
        klex_by_article.setdefault(art_idx, []).append((j, sent_idx))

    aligned: list[tuple[str, str]] = []
    used_klex: set[int] = set()

    for art_idx, (wiki_entries, _) in by_article.items():
        klex_entries = klex_by_article.get(art_idx, [])
        if not klex_entries:
            continue

        klex_indices = [j for j, _ in klex_entries]
        for i, wiki_sent_idx, _ in wiki_entries:
            best_j = -1
            best_score = -1.0
            for j in klex_indices:
                if j in used_klex:
                    continue
                score = similarity[i, j]
                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j >= 0 and best_score >= threshold:
                complex_text = wiki_records[i][2]
                simple_text = klex_records[best_j][2]
                aligned.append((complex_text, simple_text))
                used_klex.add(best_j)

    return aligned


def _pairs_from_hf() -> list[dict[str, str]] | None:
    """Load Klexikon sentence pairs from the Hugging Face dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        logger.debug("datasets library unavailable: %s", exc)
        return None

    try:
        ds = load_dataset(_KLEXIKON_HF_NAME)
    except Exception as exc:
        logger.warning("Could not load %s from Hugging Face: %s", _KLEXIKON_HF_NAME, exc)
        return None

    split = "train"
    if split not in ds:
        split = next(iter(ds.keys()))

    articles_data = ds[split]

    articles: list[tuple[list[str], list[str]]] = []
    for article in articles_data:
        wiki_sents = article.get("wiki_sentences") or article.get("wiki_text") or []
        klex_sents = article.get("klexikon_sentences") or article.get("klexikon_text") or []

        if isinstance(wiki_sents, str):
            wiki_sents = [s.strip() for s in wiki_sents.split(".") if s.strip()]
        if isinstance(klex_sents, str):
            klex_sents = [s.strip() for s in klex_sents.split(".") if s.strip()]

        wiki_sents = [str(s).strip() for s in wiki_sents if str(s).strip()]
        klex_sents = [str(s).strip() for s in klex_sents if str(s).strip()]

        if wiki_sents and klex_sents:
            articles.append((wiki_sents, klex_sents))

    if not articles:
        return None

    model = _load_similarity_model()

    if model is not None:
        aligned = _align_articles(articles, model)
    else:
        from src.data_collection.wiki_align import align_sentences

        aligned = []
        for wiki_sents, klex_sents in articles:
            aligned.extend(align_sentences(wiki_sents, klex_sents, min_similarity=0.0))

    if not aligned:
        return None

    pairs = [{"complex": c, "simple": s, "source": "klexikon"} for c, s in aligned]
    logger.info("Loaded %d Klexikon pairs from Hugging Face", len(pairs))
    return pairs


def fetch_klexikon_pairs(
    local_path: Path | str | None = None,
    urls: list[str] | None = None,
) -> list[dict[str, str]]:
    """Return Klexikon sentence pairs.

    Tries, in order: a local alignment file, the Hugging Face dataset,
    remote alignment URLs, and finally the built-in demo pairs.

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

    hf_pairs = _pairs_from_hf()
    if hf_pairs:
        return hf_pairs

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
