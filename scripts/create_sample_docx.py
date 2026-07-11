"""Generate a minimal German DOCX sample for extraction tests."""

from __future__ import annotations

import zipfile
from pathlib import Path


def _escape_xml(text: str) -> str:
    """Escape XML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _make_docx_bytes(text: str) -> bytes:
    """Build a minimal DOCX archive from plain text paragraphs."""
    paragraphs = "\n".join(
        f"<w:p><w:r><w:t>{_escape_xml(line)}</w:t></w:r></w:p>"
        for line in text.splitlines()
    )

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    document_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
</Relationships>"""

    document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {paragraphs}
  </w:body>
</w:document>"""

    buffer: bytes
    import io

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/_rels/document.xml.rels", document_rels)
        zf.writestr("word/document.xml", document)
    return archive.getvalue()


def _make_docx_with_library(text: str, docx_path: Path) -> None:
    """Create a DOCX using python-docx when it is available."""
    import docx

    document = docx.Document()
    for line in text.splitlines():
        document.add_paragraph(line)
    document.save(str(docx_path))


def main() -> None:
    """Create sample_german.docx from sample_german.txt."""
    evaluation_dir = Path(__file__).parents[1] / "data" / "evaluation"
    text_path = evaluation_dir / "sample_german.txt"
    docx_path = evaluation_dir / "sample_german.docx"

    text = text_path.read_text(encoding="utf-8")

    try:
        _make_docx_with_library(text, docx_path)
    except ImportError:
        docx_path.write_bytes(_make_docx_bytes(text))

    print(f"Created {docx_path}")


if __name__ == "__main__":
    main()
