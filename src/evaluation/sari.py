"""SARI metric for text simplification."""

from __future__ import annotations

from collections import Counter
from typing import Iterable


def _ngrams(tokens: list[str], n: int) -> Counter:
    """Return n-gram counts."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _f1_score(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _score_additions(
    source_ngrams: Counter,
    prediction_ngrams: Counter,
    references_ngrams: list[Counter],
    n: int,
) -> float:
    """Score added n-grams between source and prediction."""
    scores = []
    for ref in references_ngrams:
        added = prediction_ngrams - source_ngrams
        num_additions = sum(added.values())
        if num_additions == 0:
            scores.append(1.0)
            continue
        matched = sum((added & ref).values())
        precision = matched / num_additions
        recall = matched / sum((ref - source_ngrams).values()) if sum((ref - source_ngrams).values()) > 0 else 0.0
        scores.append(_f1_score(precision, recall))
    return sum(scores) / len(scores)


def _score_deletions(
    source_ngrams: Counter,
    prediction_ngrams: Counter,
    references_ngrams: list[Counter],
    n: int,
) -> float:
    """Score deleted n-grams between source and prediction."""
    scores = []
    for ref in references_ngrams:
        deleted = source_ngrams - prediction_ngrams
        num_deletions = sum(deleted.values())
        if num_deletions == 0:
            scores.append(1.0)
            continue
        matched = sum((deleted & (source_ngrams - ref)).values())
        precision = matched / num_deletions
        recall_denominator = sum((source_ngrams - ref).values())
        recall = matched / recall_denominator if recall_denominator > 0 else 0.0
        scores.append(_f1_score(precision, recall))
    return sum(scores) / len(scores)


def _score_keeps(
    source_ngrams: Counter,
    prediction_ngrams: Counter,
    references_ngrams: list[Counter],
    n: int,
) -> float:
    """Score kept n-grams between source and prediction."""
    scores = []
    for ref in references_ngrams:
        kept_source = source_ngrams & prediction_ngrams
        kept_ref = source_ngrams & ref
        num_kept = sum(kept_source.values())
        if num_kept == 0:
            scores.append(1.0 if sum(kept_ref.values()) == 0 else 0.0)
            continue
        matched = sum((kept_source & kept_ref).values())
        precision = matched / num_kept
        recall = matched / sum(kept_ref.values()) if sum(kept_ref.values()) > 0 else 0.0
        scores.append(_f1_score(precision, recall))
    return sum(scores) / len(scores)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization."""
    return text.lower().split()


def compute_sari(source: str, prediction: str, references: list[str]) -> float:
    """Compute SARI score for a single source/prediction/references triple."""
    source_tokens = _tokenize(source)
    prediction_tokens = _tokenize(prediction)
    reference_tokens = [_tokenize(ref) for ref in references]

    add_scores: list[float] = []
    del_scores: list[float] = []
    keep_scores: list[float] = []

    for n in (1, 2):
        src_ngrams = _ngrams(source_tokens, n)
        pred_ngrams = _ngrams(prediction_tokens, n)
        ref_ngrams = [_ngrams(ref, n) for ref in reference_tokens]

        add_scores.append(_score_additions(src_ngrams, pred_ngrams, ref_ngrams, n))
        del_scores.append(_score_deletions(src_ngrams, pred_ngrams, ref_ngrams, n))
        keep_scores.append(_score_keeps(src_ngrams, pred_ngrams, ref_ngrams, n))

    return (sum(add_scores) + sum(del_scores) + sum(keep_scores)) / 6 * 100


def compute_sari_batch(
    sources: list[str], predictions: list[str], references: list[list[str]]
) -> dict[str, float | list[float]]:
    """Compute mean SARI and per-example scores for a batch."""
    scores = [
        compute_sari(src, pred, refs)
        for src, pred, refs in zip(sources, predictions, references)
    ]
    return {"mean": sum(scores) / len(scores) if scores else 0.0, "scores": scores}
