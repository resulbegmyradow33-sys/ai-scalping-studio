"""Local document ingestion: provenance, extraction and inert failure records."""

from __future__ import annotations

import ast
import hashlib
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from scalping.knowledge import (
    DocumentIngestionError,
    DocumentIngestor,
    DocumentRecord,
    UnsupportedFormat,
)

KNOWLEDGE_SRC = Path(__file__).resolve().parents[1] / "src" / "scalping" / "knowledge"

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _docx(path: Path, paragraphs: list[tuple[str, str]]) -> Path:
    """Write a minimal DOCX with ``(style, text)`` paragraphs."""
    body = "".join(
        (
            "<w:p>"
            f'<w:pPr><w:pStyle w:val="{escape(style)}"/></w:pPr>'
            f'<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
            "</w:p>"
        )
        for style, text in paragraphs
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{WORD_NS}"><w:body>{body}</w:body></w:document>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ingest_text_records_provenance_and_preserves_bytes(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_bytes(b"Liquidity sweeps precede reversals.\r\nSecond line.\r\n")
    ingestor = DocumentIngestor(tmp_path / "knowledge")

    record = ingestor.ingest(source)

    assert isinstance(record, DocumentRecord)
    assert record.status == "indexed"
    assert record.doc_id == record.checksum == _sha(source)
    assert record.format == "text"
    assert record.parser == "stdlib-text"
    assert record.parser_version
    assert record.size_bytes == source.stat().st_size
    assert record.mtime.startswith("20")
    assert record.source_path == str(source.resolve())
    stored = Path(record.stored_path)
    assert stored == tmp_path / "knowledge" / "sources" / record.checksum / "notes.txt"
    assert stored.read_bytes() == source.read_bytes()
    assert record.duplicate is False
    assert [section.text for section in record.sections] == [
        "Liquidity sweeps precede reversals.\nSecond line."
    ]
    assert record.error is None


def test_duplicate_checksum_reuses_the_single_copy(tmp_path: Path) -> None:
    source = tmp_path / "book.md"
    source.write_text("# Liquidity\n\nSweeps can reverse.", encoding="utf-8")
    root = tmp_path / "knowledge"

    first = DocumentIngestor(root).ingest(source)
    second = DocumentIngestor(root).ingest(source)

    assert second.doc_id == first.doc_id
    assert second.duplicate is True
    assert second.stored_path == first.stored_path
    copies = list((root / "sources" / first.checksum).iterdir())
    assert copies == [Path(first.stored_path)]
    assert len(list((root / "sources").iterdir())) == 1


def test_same_checksum_under_another_name_reuses_the_copy(tmp_path: Path) -> None:
    first = tmp_path / "chapter-one.md"
    second = tmp_path / "renamed-copy.md"
    first.write_text("# Same bytes\n\nNo second copy.", encoding="utf-8")
    second.write_bytes(first.read_bytes())
    root = tmp_path / "knowledge"
    ingestor = DocumentIngestor(root)

    one = ingestor.ingest(first)
    two = ingestor.ingest(second)

    assert two.doc_id == one.doc_id
    assert two.duplicate is True
    assert two.stored_path == one.stored_path
    assert two.filename == "renamed-copy.md"
    assert len(list((root / "sources" / one.checksum).iterdir())) == 1


def test_markdown_headings_become_sections(tmp_path: Path) -> None:
    source = tmp_path / "research.md"
    source.write_text(
        "# Market structure\n"
        "\n"
        "Swing highs and lows define structure.\n"
        "\n"
        "## Liquidity\n"
        "\n"
        "A sweep takes the prior low before reversing.\n"
        "\n"
        "```text\n"
        "# not a heading inside a fence\n"
        "```\n",
        encoding="utf-8",
    )

    record = DocumentIngestor(tmp_path / "k").ingest(source)

    assert record.status == "indexed"
    assert [section.heading for section in record.sections] == [
        "Market structure",
        "Market structure > Liquidity",
    ]
    assert "not a heading inside a fence" in record.sections[1].text
    assert record.sections[0].label == "Market structure"


def test_heading_only_markdown_is_still_text(tmp_path: Path) -> None:
    source = tmp_path / "outline.md"
    source.write_text("# Scope\n\n## Data\n", encoding="utf-8")

    record = DocumentIngestor(tmp_path / "k").ingest(source)

    assert record.status == "indexed"
    assert [section.text for section in record.sections] == ["Scope\nData"]


def test_html_strips_scripts_and_styles(tmp_path: Path) -> None:
    source = tmp_path / "page.html"
    source.write_text(
        "<html><head><title>Ignored</title>"
        "<style>.x{color:red}</style>"
        "<script>stealCredentials()</script></head>"
        "<body><h1>Order flow</h1><p>Delta measures aggressor imbalance.</p>"
        "<h2>Limits</h2><p>Only with genuine tick data.</p></body></html>",
        encoding="utf-8",
    )

    record = DocumentIngestor(tmp_path / "k").ingest(source)

    assert record.status == "indexed"
    text = "\n".join(section.text for section in record.sections)
    assert "Delta measures aggressor imbalance." in text
    assert "stealCredentials" not in text
    assert "color:red" not in text
    assert [section.heading for section in record.sections] == ["Order flow", "Order flow > Limits"]


def test_docx_paragraph_text_and_headings(tmp_path: Path) -> None:
    source = _docx(
        tmp_path / "notes.docx",
        [
            ("Heading1", "Regime detection"),
            ("Normal", "Volatility expansion follows compression."),
            ("Normal", "Test on M15, not on the final test set."),
            ("Heading2", "Caveats"),
            ("Normal", "Regime labels are not trade signals."),
        ],
    )

    record = DocumentIngestor(tmp_path / "k").ingest(source)

    assert record.status == "indexed"
    assert record.format == "docx"
    assert record.parser == "zipfile+ElementTree"
    assert [section.heading for section in record.sections] == [
        "Regime detection",
        "Regime detection > Caveats",
    ]
    assert record.sections[0].text == (
        "Volatility expansion follows compression.\n\nTest on M15, not on the final test set."
    )
    assert record.sections[1].text == "Regime labels are not trade signals."


def test_pdf_is_reported_unsupported_without_a_dependency(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.4\n% not parsed here\n")

    record = DocumentIngestor(tmp_path / "k").ingest(source)

    assert record.status == "unsupported"
    assert record.format == "pdf"
    assert record.sections == ()
    assert record.error is not None
    assert "optional parser" in record.error
    assert Path(record.stored_path).read_bytes() == source.read_bytes()
    with pytest.raises(UnsupportedFormat, match="optional parser"):
        DocumentIngestor(tmp_path / "k").extract(source)


def test_extraction_failure_is_recorded_and_original_kept(tmp_path: Path) -> None:
    source = tmp_path / "broken.docx"
    source.write_bytes(b"this is not a zip archive")

    record = DocumentIngestor(tmp_path / "k").ingest(source)

    assert record.status == "extraction_failed"
    assert record.error is not None
    assert "BadZipFile" in record.error or "zip" in record.error
    assert record.sections == ()
    assert Path(record.stored_path).read_bytes() == source.read_bytes()


def test_invalid_utf8_text_is_an_extraction_failure(tmp_path: Path) -> None:
    source = tmp_path / "latin1.txt"
    source.write_bytes("café".encode("latin-1"))

    record = DocumentIngestor(tmp_path / "k").ingest(source)

    assert record.status == "extraction_failed"
    assert record.error is not None
    assert "UTF-8" in record.error


def test_source_path_traversal_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "inside.txt"
    source.write_text("ok", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    traversal = tmp_path / "nested" / ".." / "inside.txt"
    assert traversal.exists()

    with pytest.raises(ValueError, match="must not contain"):
        DocumentIngestor(tmp_path / "k").ingest(traversal)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DocumentIngestor(tmp_path / "k").ingest(tmp_path / "absent.txt")


def test_directory_is_not_a_document(tmp_path: Path) -> None:
    directory = tmp_path / "folder.txt"
    directory.mkdir()

    with pytest.raises(DocumentIngestionError, match="not a regular file"):
        DocumentIngestor(tmp_path / "k").ingest(directory)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_no_network_or_credential_imports() -> None:
    forbidden = {
        "socket",
        "ssl",
        "urllib",
        "http",
        "requests",
        "httpx",
        "aiohttp",
        "openai",
        "anthropic",
        "boto3",
        "keyring",
        "dotenv",
    }
    for path in KNOWLEDGE_SRC.glob("*.py"):
        assert not (_imports(path) & forbidden), f"{path.name} imports a forbidden module"


def test_no_eval_exec_or_dynamic_import() -> None:
    forbidden = {"eval", "exec", "compile", "__import__"}
    for path in KNOWLEDGE_SRC.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden, f"{path.name} calls {node.func.id}"


def test_no_document_to_order_path() -> None:
    """The knowledge package may not import trading or execution modules."""
    forbidden_prefixes = (
        "scalping.paper",
        "scalping.strategy",
        "scalping.api",
        "scalping.backtest",
    )
    for path in KNOWLEDGE_SRC.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module
            elif isinstance(node, ast.Import):
                module = node.names[0].name
            if module is not None:
                assert not module.startswith(forbidden_prefixes), f"{path.name} imports {module}"
