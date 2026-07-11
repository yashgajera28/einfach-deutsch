"""BLEU wrapper around sacrebleu."""

from sacrebleu import corpus_bleu, sentence_bleu


def compute_sentence_bleu(prediction: str, references: list[str]) -> float:
    """Compute sentence-level BLEU."""
    score = sentence_bleu(prediction, references)
    return float(score.score)


def compute_corpus_bleu(predictions: list[str], references: list[list[str]]) -> float:
    """Compute corpus-level BLEU."""
    score = corpus_bleu(predictions, references)
    return float(score.score)
