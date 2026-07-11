"""Evaluation runner for German text simplification."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.bleu import compute_corpus_bleu, compute_sentence_bleu
from src.evaluation.readability import readability_delta
from src.evaluation.sari import compute_sari, compute_sari_batch

logger = logging.getLogger(__name__)


def _load_test_data(test_file: str | Path) -> pd.DataFrame:
    """Load test CSV with source, reference, and prediction columns."""
    df = pd.read_csv(test_file)
    required = {"source", "reference", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {test_file}: {missing}")
    return df


def run_evaluation(
    config: dict[str, Any], test_file: str | Path, output_dir: str | Path
) -> None:
    """Run evaluation on a test set and save reports."""
    test_file = Path(test_file)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = _load_test_data(test_file)
    sources = df["source"].astype(str).tolist()
    predictions = df["prediction"].astype(str).tolist()
    references: list[list[str]] = [
        [ref.strip() for ref in str(refs).split("|") if ref.strip()]
        for refs in df["reference"]
    ]

    logger.info("Evaluating %d examples", len(df))

    sari_result = compute_sari_batch(sources, predictions, references)
    sari_scores = sari_result["scores"]
    mean_sari = sari_result["mean"]

    bleu_scores = [
        compute_sentence_bleu(pred, refs) for pred, refs in zip(predictions, references)
    ]
    corpus_bleu = compute_corpus_bleu(predictions, references)

    readability_rows = [
        readability_delta(src, pred) for src, pred in zip(sources, predictions)
    ]

    per_example = pd.DataFrame(
        {
            "source": sources,
            "prediction": predictions,
            "reference": df["reference"].astype(str),
            "sari": sari_scores,
            "bleu": bleu_scores,
        }
    )
    per_example = pd.concat(
        [per_example, pd.DataFrame(readability_rows)], axis=1
    )
    per_example.to_csv(out / "per_example.csv", index=False)

    report = {
        "num_examples": len(df),
        "sari": float(mean_sari),
        "bleu": float(corpus_bleu),
        "readability": {
            "mean_lix_before": float(
                sum(r["lix_before"] for r in readability_rows) / len(readability_rows)
            ),
            "mean_lix_after": float(
                sum(r["lix_after"] for r in readability_rows) / len(readability_rows)
            ),
            "mean_lix_delta": float(
                sum(r["lix_delta"] for r in readability_rows) / len(readability_rows)
            ),
            "mean_wstf_before": float(
                sum(r["wstf_before"] for r in readability_rows) / len(readability_rows)
            ),
            "mean_wstf_after": float(
                sum(r["wstf_after"] for r in readability_rows) / len(readability_rows)
            ),
            "mean_wstf_delta": float(
                sum(r["wstf_delta"] for r in readability_rows) / len(readability_rows)
            ),
        },
    }

    with (out / "report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    logger.info("SARI: %.2f, BLEU: %.2f", mean_sari, corpus_bleu)
    logger.info("Saved reports to %s", out)
