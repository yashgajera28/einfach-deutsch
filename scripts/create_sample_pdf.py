"""Generate a minimal German PDF sample for extraction tests."""

from __future__ import annotations

from pathlib import Path


def _winansi_escape(text: str) -> bytes:
    """Escape PDF string parentheses and encode as WinAnsiEncoding (latin-1)."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return escaped.encode("latin-1")


def _make_manual_pdf(text: str) -> bytes:
    """Build a minimal one-page PDF with the given text."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    y_start = 700
    y_step = 16

    content_stream = f"BT\n/F1 12 Tf\n72 {y_start} Td\n".encode("latin-1")
    for line in lines:
        encoded = _winansi_escape(line)
        content_stream += b"(" + encoded + b") Tj\n"
        content_stream += f"0 -{y_step} Td\n".encode("latin-1")
    content_stream += b"ET\n"

    content_obj = (
        b"4 0 obj\n<< /Length "
        + str(len(content_stream)).encode("ascii")
        + b" >>\nstream\n"
        + content_stream
        + b"endstream\nendobj\n"
    )

    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
        content_obj,
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n",
    ]

    header = b"%PDF-1.4\n"
    offsets: list[int] = []
    body = b""
    for obj in objects:
        offsets.append(len(body))
        body += obj

    xref_offset = len(header) + len(body)
    xref = b"xref\n0 6\n0000000000 65535 f \n"
    for offset in offsets:
        xref += f"{len(header) + offset:010d} 00000 n \n".encode("ascii")

    trailer = (
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n"
        + str(xref_offset).encode("ascii")
        + b"\n%%EOF\n"
    )

    return header + body + xref + trailer


def _make_fpdf_pdf(text: str, pdf_path: Path) -> None:
    """Create a PDF using fpdf2 when it is available."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_doc_option("core_fonts_encoding", "latin-1")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=12)

    for line in text.splitlines():
        pdf.multi_cell(0, 8, txt=line)

    pdf.output(str(pdf_path))


def main() -> None:
    """Create sample_german.pdf from sample_german.txt."""
    evaluation_dir = Path(__file__).parents[1] / "data" / "evaluation"
    text_path = evaluation_dir / "sample_german.txt"
    pdf_path = evaluation_dir / "sample_german.pdf"

    text = text_path.read_text(encoding="utf-8")

    try:
        _make_fpdf_pdf(text, pdf_path)
    except ImportError:
        pdf_path.write_bytes(_make_manual_pdf(text))

    print(f"Created {pdf_path}")


if __name__ == "__main__":
    main()
