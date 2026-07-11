"""Evaluation helpers for German text simplification."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

from src.evaluation.readability import lix_score, readability_delta, wstf_score

__all__ = ["lix_score", "readability_delta", "wstf_score"]

try:
    from src.evaluation.bleu import compute_corpus_bleu, compute_sentence_bleu

    __all__.extend(["compute_corpus_bleu", "compute_sentence_bleu"])
except ImportError as exc:  # pragma: no cover
    logger.warning("BLEU metrics unavailable: %s", exc)
    compute_corpus_bleu: Any = None  # type: ignore[misc,assignment]
    compute_sentence_bleu: Any = None  # type: ignore[misc,assignment]

try:
    from src.evaluation.human_eval import create_annotation_template, sample_for_annotation

    __all__.extend(["create_annotation_template", "sample_for_annotation"])
except ImportError as exc:  # pragma: no cover
    logger.warning("Human evaluation helpers unavailable: %s", exc)
    create_annotation_template: Any = None  # type: ignore[misc,assignment]
    sample_for_annotation: Any = None  # type: ignore[misc,assignment]

try:
    from src.evaluation.sari import compute_sari, compute_sari_batch

    __all__.extend(["compute_sari", "compute_sari_batch"])
except ImportError as exc:  # pragma: no cover
    logger.warning("SARI metrics unavailable: %s", exc)
    compute_sari: Any = None  # type: ignore[misc,assignment]
    compute_sari_batch: Any = None  # type: ignore[misc,assignment]
