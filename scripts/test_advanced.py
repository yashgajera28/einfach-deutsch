"""Tests for advanced simplification features without model inference."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import spacy_pipe

SAMPLE_TEXT = (
    "Am 15. März 2023 wurde der Vertrag von der Partei gekündigt, "
    "weil gemäß § 123 Abs. 1 Satz 2 eine Kostenpauschale von 1.234,56 € fällig war."
)


def _make_simplifier(backend: str = "baseline") -> Any:
    """Create a Simplifier instance without loading any model weights."""
    from src.models.simplifier import Simplifier

    config: dict[str, Any] = {
        "models": {
            "baseline": {"name": "google/mt5-small", "max_length": 256, "prefix": "vereinfachen: "},
            "lora": {
                "name": "LeoLM/leo-hessianai-7b",
                "max_seq_length": 512,
                "control_tokens": ["<A2>", "<B1>", "<B2>"],
            },
        },
        "simplification": {"levels": ["A2", "B1", "B2"], "default_level": "B1"},
        "readability": {"lix_green_max": 35, "lix_yellow_max": 50},
    }
    with patch.object(Simplifier, "_load_model", lambda self: None):
        simplifier = Simplifier(config, backend=backend)
    return simplifier


def test_level_formatting() -> None:
    """Verify level tokens are embedded in both backend prompts."""
    simplifier = _make_simplifier("baseline")
    baseline = simplifier._format_baseline_input("Der Text.", "B1")
    assert "<B1>" in baseline
    assert "vereinfachen:" in baseline

    simplifier = _make_simplifier("lora")
    lora = simplifier._format_lora_input("Der Text.", "A2")
    assert "<A2>" in lora
    assert "Vereinfache:" in lora
    print("[OK] level formatting works for baseline and lora")


def test_entity_extraction() -> None:
    """Verify regex-based extraction of dates, amounts, and legal references."""
    dates = spacy_pipe.extract_dates(SAMPLE_TEXT)
    amounts = spacy_pipe.extract_amounts(SAMPLE_TEXT)
    legal = spacy_pipe.extract_legal_references(SAMPLE_TEXT)

    assert any("15. März 2023" in d for d in dates), f"dates: {dates}"
    assert any("1.234,56 €" in a for a in amounts), f"amounts: {amounts}"
    assert any("123" in ref for ref in legal), f"legal: {legal}"
    print("[OK] entity extraction (dates, amounts, legal references)")


def test_extract_entities_fallback() -> None:
    """Verify extract_entities returns structured items without a spaCy model."""
    entities = spacy_pipe.extract_entities(SAMPLE_TEXT)
    labels = {item["label"] for item in entities}
    assert "DATE" in labels or "AMOUNT" in labels or "LEGAL" in labels
    assert all("text" in item and "label" in item for item in entities)
    print("[OK] extract_entities fallback structure")


def test_preserve_entities() -> None:
    """Verify missing factual phrases are re-injected into the simplified text."""
    simplifier = _make_simplifier("baseline")
    simplified = "Der Vertrag wurde gekündigt, weil eine Kostenpauschale fällig war."
    updated, preserved = simplifier._preserve_entities(SAMPLE_TEXT, simplified)

    assert any("15. März 2023" in p for p in preserved), f"preserved: {preserved}"
    assert any("1.234,56 €" in p for p in preserved), f"preserved: {preserved}"
    assert "§ 123" in updated or "Wichtige Begriffe" in updated, f"updated: {updated}"
    print("[OK] entity preservation re-injects missing phrases")


def test_explanation_rules() -> None:
    """Verify explanation heuristics fire for passive, split, and connector changes."""
    simplifier = _make_simplifier("baseline")

    original = "Der Antrag wurde von der Behörde genehmigt."
    simplified = "Die Behörde genehmigte den Antrag."
    expl = simplifier._build_explanation(original, simplified)
    assert any("Passive" in e for e in expl), f"explanations: {expl}"

    original = "Der lange und komplizierte Vertrag wurde unterschrieben."
    simplified = "Der Vertrag wurde unterschrieben. Er war lang und kompliziert."
    expl = simplifier._build_explanation(original, simplified)
    assert any("split" in e.lower() for e in expl), f"explanations: {expl}"

    original = "Deshalb wurde die Regel geändert."
    simplified = "Daher wurde die Regel geändert."
    expl = simplifier._build_explanation(original, simplified)
    assert any("connector" in e.lower() for e in expl), f"explanations: {expl}"

    original = "Die Arbeitsunfähigkeitsbescheinigung liegt vor."
    simplified = "Das Attest liegt vor."
    expl = simplifier._build_explanation(original, simplified)
    assert any("compound" in e.lower() for e in expl), f"explanations: {expl}"

    print("[OK] explanation heuristics (passive, split, connector, compound)")


def test_extract_entities_for_display() -> None:
    """Verify the frontend-facing entity helper returns structured data."""
    simplifier = _make_simplifier("baseline")
    entities = simplifier.extract_entities_for_display(SAMPLE_TEXT)
    assert isinstance(entities, list)
    assert all("text" in item and "label" in item for item in entities)
    print("[OK] extract_entities_for_display")


def main() -> int:
    """Run all advanced feature tests."""
    print("Running advanced feature tests (no model inference)...\n")
    try:
        test_level_formatting()
        test_entity_extraction()
        test_extract_entities_fallback()
        test_preserve_entities()
        test_explanation_rules()
        test_extract_entities_for_display()
    except AssertionError as exc:
        print(f"\n[FAIL] test failed: {exc}")
        return 1
    except Exception as exc:
        print(f"\n[ERROR] unexpected error: {exc}")
        return 1

    print("\nAll advanced feature tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
