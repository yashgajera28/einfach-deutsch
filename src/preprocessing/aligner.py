"""Sentence alignment for complex/simple German text."""

from __future__ import annotations

import logging
import math
import os
import warnings
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get(
    "SENTENCE_TRANSFORMER_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

_MIN_LENGTH_RATIO = 0.5
_MAX_LENGTH_RATIO = 2.0

_embedder: Any | None = None
_embedder_error_logged: bool = False


def _length_ratio_ok(src: str, tgt: str) -> bool:
    """Return True if the token-length ratio is within the default bounds."""
    src_len = max(len(src.split()), 1)
    tgt_len = max(len(tgt.split()), 1)
    ratio = src_len / tgt_len
    return _MIN_LENGTH_RATIO <= ratio <= _MAX_LENGTH_RATIO


def _cosine_similarity(a: Any, b: Any) -> float:
    """Compute cosine similarity between two vectors."""
    try:
        import numpy as np

        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if norm == 0.0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / norm)
    except Exception:
        a_list = [float(x) for x in a]
        b_list = [float(x) for x in b]
        dot = sum(x * y for x, y in zip(a_list, b_list))
        norm_a = math.sqrt(sum(x * x for x in a_list))
        norm_b = math.sqrt(sum(x * x for x in b_list))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


def _load_embedder(allow_download: bool = False) -> Any | None:
    """Lazy-load the sentence transformer; never downloads by default."""
    global _embedder, _embedder_error_logged
    if _embedder is not None:
        return _embedder

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        if not _embedder_error_logged:
            warnings.warn(f"sentence-transformers not installed: {exc}", stacklevel=2)
            _embedder_error_logged = True
        return None

    try:
        _embedder = SentenceTransformer(
            DEFAULT_MODEL, local_files_only=not allow_download
        )
    except Exception as exc:
        if not _embedder_error_logged:
            mode = "download" if allow_download else "offline"
            warnings.warn(
                f"Could not load sentence transformer ({mode}): {exc}",
                stacklevel=2,
            )
            _embedder_error_logged = True
        return None

    return _embedder


def align_sentences(
    complex_sents: list[str],
    simple_sents: list[str],
    threshold: float = 0.6,
    allow_download: bool = False,
) -> list[tuple[str, str, float]]:
    """Align complex and simple sentences.

    First filters candidate pairs by token-length ratio (0.5--2.0). When the
    sentence transformer model is available, embeddings are computed and only
    pairs with cosine similarity >= ``threshold`` are kept. If the model is not
    installed or cannot be loaded offline, the length-ratio filter is used as a
    lightweight fallback.

    Args:
        complex_sents: Sentences from the complex source.
        simple_sents: Sentences from the simplified source.
        threshold: Minimum cosine similarity for an alignment.
        allow_download: If True, allow the transformer library to download the
            model on first use.

    Returns:
        A list of ``(complex_sentence, simple_sentence, score)`` tuples.
    """
    if not complex_sents or not simple_sents:
        return []

    candidates: list[tuple[int, int]] = [
        (i, j)
        for i, c in enumerate(complex_sents)
        for j, s in enumerate(simple_sents)
        if _length_ratio_ok(c, s)
    ]

    embedder = _load_embedder(allow_download=allow_download)
    if embedder is None:
        return [(complex_sents[i], simple_sents[j], 1.0) for i, j in candidates]

    all_sents = list(complex_sents) + list(simple_sents)
    try:
        embeddings = embedder.encode(
            all_sents, convert_to_numpy=True, show_progress_bar=False
        )
    except Exception as exc:
        warnings.warn(
            f"Embedding failed: {exc}; falling back to length ratio.",
            stacklevel=2,
        )
        return [(complex_sents[i], simple_sents[j], 1.0) for i, j in candidates]

    complex_embs = embeddings[: len(complex_sents)]
    simple_embs = embeddings[len(complex_sents) :]

    scored: dict[tuple[int, int], float] = {}
    for i, j in candidates:
        scored[(i, j)] = _cosine_similarity(complex_embs[i], simple_embs[j])

    aligned: list[tuple[str, str, float]] = []
    used_simple: set[int] = set()
    for (i, j), score in sorted(scored.items(), key=lambda kv: kv[1], reverse=True):
        if j in used_simple or score < threshold:
            continue
        aligned.append((complex_sents[i], simple_sents[j], round(score, 4)))
        used_simple.add(j)

    return aligned
