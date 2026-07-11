"""Text extraction helpers for PDF, DOCX, and plain text uploads."""

from __future__ import annotations

import io
from typing import Any


def _require_pypdf2() -> Any:
    """Import PyPDF2 or raise a clear RuntimeError."""
    try:
        import PyPDF2

        return PyPDF2
    except ImportError as exc:
        raise RuntimeError("PyPDF2 is not installed. Install it to extract PDF text.") from exc


def _require_pytesseract() -> Any:
    """Import pytesseract or raise a clear RuntimeError."""
    try:
        import pytesseract

        return pytesseract
    except ImportError as exc:
        raise RuntimeError("pytesseract is not installed. Install it for OCR fallback.") from exc


def _require_pdf2image() -> Any:
    """Import pdf2image or raise a clear RuntimeError."""
    try:
        from pdf2image import convert_from_bytes

        return convert_from_bytes
    except ImportError as exc:
        raise RuntimeError("pdf2image is not installed. Install it for OCR fallback.") from exc


def _require_docx() -> Any:
    """Import python-docx or raise a clear RuntimeError."""
    try:
        import docx

        return docx
    except ImportError as exc:
        raise RuntimeError("python-docx is not installed. Install it to extract DOCX text.") from exc


def _ocr_page(file_bytes: bytes, page_num: int) -> str:
    """Run German OCR on a single PDF page."""
    pytesseract = _require_pytesseract()
    convert_from_bytes = _require_pdf2image()

    images = convert_from_bytes(
        file_bytes,
        first_page=page_num,
        last_page=page_num,
        dpi=200,
    )
    if not images:
        return ""
    return pytesseract.image_to_string(images[0], lang="deu")


def extract_text_from_pdf(file_bytes: bytes) -> dict[int, str]:
    """Extract text from a PDF byte stream.

    Falls back to OCR for pages with fewer than 50 characters of text.
    """
    PyPDF2 = _require_pypdf2()
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))

    pages: dict[int, str] = {}
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) < 50:
            text = _ocr_page(file_bytes, index).strip()
        pages[index] = text
    return pages


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX byte stream."""
    docx = _require_docx()
    document = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode a plain text byte stream as UTF-8, falling back to latin-1."""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")


def extract_text(filename: str, file_bytes: bytes) -> dict[int, str] | str:
    """Dispatch extraction based on the file extension."""
    extension = filename.lower().rsplit(".", 1)[-1]
    if extension == "pdf":
        return extract_text_from_pdf(file_bytes)
    if extension == "docx":
        return extract_text_from_docx(file_bytes)
    if extension in {"txt", "text", "md"}:
        return extract_text_from_txt(file_bytes)
    raise ValueError(f"Unsupported file extension: {extension}")
