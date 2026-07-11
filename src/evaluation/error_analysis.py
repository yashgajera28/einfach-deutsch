"""Rule-based error categorization for simplification outputs."""

import re
from collections import defaultdict
from typing import Any


def _tokens(text: str) -> set[str]:
    """Return normalized tokens."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def _ngram_overlap(a: str, b: str) -> float:
    """Compute unigram overlap between two strings."""
    a_tokens = _tokens(a)
    b_tokens = _tokens(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def categorize_error(source: str, prediction: str, reference: str) -> str:
    """Assign a rule-based error category."""
    source_tokens = _tokens(source)
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)

    pred_in_ref = len(pred_tokens & ref_tokens) / len(pred_tokens) if pred_tokens else 0.0
    source_in_pred = len(source_tokens & pred_tokens) / len(source_tokens) if source_tokens else 0.0

    if pred_in_ref < 0.4 and source_in_pred > 0.7:
        return "insufficient_simplification"

    extra = pred_tokens - source_tokens - ref_tokens
    if len(extra) / max(len(pred_tokens), 1) > 0.25:
        return "hallucination"

    missing = ref_tokens - pred_tokens
    source_missing = source_tokens - pred_tokens
    if len(missing) / max(len(ref_tokens), 1) > 0.35 or len(source_missing) / max(len(source_tokens), 1) > 0.35:
        return "fact_loss"

    if source_in_pred < 0.4 and pred_in_ref > 0.6:
        return "over_simplification"

    if re.search(r"\b(die|der|das|den|dem|des|ein|eine|einen|einem|einer|eines)\s+(die|der|das|den|dem|des|ein|eine|einen|einem|einer|eines)\b", prediction, re.IGNORECASE):
        return "grammar_error"

    return "insufficient_simplification"


def confusion_matrix_by_domain(
    pairs: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Aggregate error categories by domain."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for pair in pairs:
        domain = pair.get("domain", "unknown")
        category = categorize_error(
            pair.get("source", ""), pair.get("prediction", ""), pair.get("reference", "")
        )
        matrix[domain][category] += 1
    return {domain: dict(categories) for domain, categories in matrix.items()}
