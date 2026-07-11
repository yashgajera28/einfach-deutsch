"""Test text extraction from sample PDF, DOCX, and TXT files."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))


def _print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def test_txt(sample_path: Path) -> None:
    """Test plain text extraction."""
    try:
        from src.api.extract import extract_text_from_txt
        text = extract_text_from_txt(sample_path.read_bytes())
    except Exception as exc:
        print(f"SKIP TXT: {exc}")
        return

    _print_section("TXT extraction")
    print(text[:500])


def test_pdf(sample_path: Path) -> None:
    """Test PDF extraction."""
    if not sample_path.exists():
        print(f"SKIP: {sample_path} not found. Run scripts/create_sample_pdf.py first.")
        return

    try:
        from src.api.extract import extract_text_from_pdf
        pages = extract_text_from_pdf(sample_path.read_bytes())
    except Exception as exc:
        print(f"SKIP PDF: {exc}")
        return

    _print_section("PDF extraction")
    for page_num, text in pages.items():
        print(f"--- Page {page_num} ---")
        print(text[:500])


def test_docx(sample_path: Path) -> None:
    """Test DOCX extraction."""
    if not sample_path.exists():
        print(f"SKIP: {sample_path} not found. Run scripts/create_sample_docx.py first.")
        return

    try:
        from src.api.extract import extract_text_from_docx
        text = extract_text_from_docx(sample_path.read_bytes())
    except Exception as exc:
        print(f"SKIP DOCX: {exc}")
        return

    _print_section("DOCX extraction")
    print(text[:500])


def main() -> None:
    """Run extraction tests against sample files."""
    evaluation_dir = Path(__file__).parents[1] / "data" / "evaluation"

    test_txt(evaluation_dir / "sample_german.txt")
    test_pdf(evaluation_dir / "sample_german.pdf")
    test_docx(evaluation_dir / "sample_german.docx")


if __name__ == "__main__":
    main()
