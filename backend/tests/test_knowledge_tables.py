"""Structural tables: DOCX/HTML/Markdown extraction, storage and search.

The tests use local, synthetic documents only. No network, model or evaluation
is involved; one test asserts that property statically over the package.
"""

from __future__ import annotations

import ast
import sqlite3
import zipfile
from pathlib import Path

import pytest

from scalping.knowledge import (
    MERGED_CELL,
    SCHEMA_VERSION,
    DocumentIngestor,
    DocumentRecord,
    KnowledgeIndex,
    TableRecord,
    extract_tables,
    flatten_table,
)
from scalping.knowledge.index import _MIGRATIONS, _SCHEMA_VERSION_DDL

KNOWLEDGE_SRC = Path(__file__).resolve().parents[1] / "src" / "scalping" / "knowledge"

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

DOCX_TABLE = (
    "<w:tbl>"
    "<w:tr><w:trPr><w:tblHeader/></w:trPr>"
    "<w:tc><w:p><w:r><w:t>Regime</w:t></w:r></w:p></w:tc>"
    "<w:tc><w:p><w:r><w:t>Mean R</w:t></w:r></w:p></w:tc>"
    "<w:tc><w:p><w:r><w:t>Trades</w:t></w:r></w:p></w:tc>"
    "</w:tr>"
    "<w:tr>"
    "<w:tc><w:p><w:r><w:t>Trend day</w:t></w:r></w:p></w:tc>"
    "<w:tc><w:p><w:r><w:t>0.21</w:t></w:r></w:p></w:tc>"
    "<w:tc><w:p><w:r><w:t>45</w:t></w:r></w:p></w:tc>"
    "</w:tr>"
    "<w:tr>"
    '<w:tc><w:tcPr><w:gridSpan w:val="2"/><w:vMerge w:val="restart"/></w:tcPr>'
    "<w:p><w:r><w:t>Compression</w:t></w:r></w:p></w:tc>"
    "<w:tc><w:p><w:r><w:t>9</w:t></w:r></w:p></w:tc>"
    "</w:tr>"
    "<w:tr>"
    "<w:tc><w:tcPr><w:vMerge/></w:tcPr><w:p/></w:tc>"
    "<w:tc><w:p><w:r><w:t>7</w:t></w:r></w:p></w:tc>"
    "<w:tc><w:p><w:r><w:t>3</w:t></w:r></w:p></w:tc>"
    "</w:tr>"
    "</w:tbl>"
)


def _docx(path: Path, body: str) -> Path:
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{WORD_NS}"><w:body>{body}</w:body></w:document>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)
    return path


def _heading(text: str) -> str:
    return f'<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>{text}</w:t></w:r></w:p>'


def _paragraph(text: str) -> str:
    return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"


def _markdown(tmp_path: Path, text: str, name: str = "paper.md") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _knowledge(tmp_path: Path) -> KnowledgeIndex:
    return KnowledgeIndex(tmp_path / "knowledge.sqlite3")


# --------------------------------------------------------------- DOCX mapping


def test_docx_table_keeps_header_and_marks_merged_cells(tmp_path: Path) -> None:
    source = _docx(
        tmp_path / "results.docx",
        _heading("Results")
        + _paragraph("Table 1: Returns by regime")
        + DOCX_TABLE
        + _paragraph("After the table"),
    )
    record = DocumentIngestor(tmp_path / "knowledge").ingest(source)
    assert record.status == "indexed"

    extraction = extract_tables(record.stored_path, sections=record.sections)
    assert extraction.problems == ()
    assert len(extraction.tables) == 1
    table = extraction.tables[0]

    assert table.kind == "docx"
    assert table.caption == "Table 1: Returns by regime"
    assert table.heading == "Results"
    assert table.section_ordinal == 0
    assert table.header == ("Regime", "Mean R", "Trades")
    assert table.rows == (
        ("Trend day", "0.21", "45"),
        ("Compression", MERGED_CELL, "9"),
        (MERGED_CELL, "7", "3"),
    )
    assert table.row_count == 4
    assert table.col_count == 3

    payload = Path(record.stored_path).read_bytes()
    with zipfile.ZipFile(record.stored_path) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    assert document_xml[table.char_start : table.char_start + 6] == "<w:tbl"
    assert document_xml[table.char_end - 8 : table.char_end] == "</w:tbl>"
    assert payload  # the preserved copy is the authority, not the store


def test_docx_ingest_stores_tables_and_keeps_paragraph_text(tmp_path: Path) -> None:
    source = _docx(tmp_path / "results.docx", _heading("Results") + DOCX_TABLE)
    record = DocumentIngestor(tmp_path / "knowledge").ingest(source)

    assert [table.kind for table in record.tables] == ["docx"]
    assert record.to_dict()["table_count"] == 1
    # The existing paragraph extractor owns the text; table cells appear once
    # and are not added a second time by table extraction.
    text = "\n".join(section.text for section in record.sections)
    assert text.count("0.21") == 1
    assert "Trend day" in text


def test_docx_without_a_table_yields_no_records(tmp_path: Path) -> None:
    source = _docx(tmp_path / "plain.docx", _paragraph("No table here."))
    extraction = extract_tables(source)
    assert extraction.tables == ()
    assert extraction.problems == ()


# --------------------------------------------------------------- HTML mapping


def test_html_caption_thead_colspan_and_rowspan(tmp_path: Path) -> None:
    html = (
        "<html><body><h2>Costs</h2>"
        "<table><caption>Spread by session</caption>"
        '<thead><tr><th>Session</th><th colspan="2">Cost</th></tr></thead>'
        "<tbody>"
        "<tr><td>London</td><td>12</td><td>18</td></tr>"
        '<tr><td rowspan="2">New York</td><td>15</td><td>20</td></tr>'
        "<tr><td>16</td><td>22</td></tr>"
        "</tbody></table></body></html>"
    )
    source = tmp_path / "costs.html"
    source.write_text(html, encoding="utf-8")

    extraction = extract_tables(source)
    assert extraction.problems == ()
    table = extraction.tables[0]

    assert table.kind == "html"
    assert table.caption == "Spread by session"
    assert table.heading == "Costs"
    assert table.header == ("Session", "Cost", MERGED_CELL)
    assert table.rows == (
        ("London", "12", "18"),
        ("New York", "15", "20"),
        (MERGED_CELL, "16", "22"),
    )
    assert table.row_count == 4
    assert table.col_count == 3
    assert html[table.char_start : table.char_start + 6] == "<table"
    assert html[table.char_end - 8 : table.char_end] == "</table>"


def test_html_malformed_tables_are_reported_not_raised(tmp_path: Path) -> None:
    html = (
        "<table></table>"
        "<table><tr><td>open without end tag"
        "<table><caption>Kept</caption>"
        "<tr><td>one</td><td>two</td></tr></table>"
    )
    source = tmp_path / "broken.html"
    source.write_text(html, encoding="utf-8")

    extraction = extract_tables(source)

    reasons = [problem.reason for problem in extraction.problems]
    assert "table has no rows" in reasons
    assert any("not closed" in reason for reason in reasons)
    # The unclosed table is still read up to its last row; the nested table is
    # a separate record. Neither aborts extraction.
    assert [table.caption for table in extraction.tables] == ["", "Kept"]
    assert extraction.tables[0].rows == (("open without end tag",),)


# ----------------------------------------------------------- Markdown mapping


def test_markdown_pipe_table_uses_header_and_skips_fences(tmp_path: Path) -> None:
    text = (
        "# Costs\n"
        "\n"
        "Table 1: Session costs\n"
        "\n"
        "| Session | Min | Max |\n"
        "|:--------|----:|:---:|\n"
        "| London  | 12  | 18  |\n"
        "| New York | 15 | 20 |\n"
        "\n"
        "```text\n"
        "| not | a table |\n"
        "|-----|---------|\n"
        "| x   | y       |\n"
        "```\n"
    )
    source = _markdown(tmp_path, text)

    extraction = extract_tables(source)
    assert extraction.problems == ()
    assert len(extraction.tables) == 1
    table = extraction.tables[0]

    assert table.kind == "markdown"
    assert table.caption == "Table 1: Session costs"
    assert table.heading == "Costs"
    assert table.header == ("Session", "Min", "Max")
    assert table.rows == (("London", "12", "18"), ("New York", "15", "20"))
    assert table.row_count == 3
    assert table.col_count == 3
    assert text[table.char_start : table.char_start + 1] == "|"
    assert text[table.char_end - 1 : table.char_end] == "|"


def test_markdown_ragged_row_is_padded_and_reported(tmp_path: Path) -> None:
    text = "| a | b |\n|---|---|\n| one |\n| x | y | z |\n"
    source = _markdown(tmp_path, text, "ragged.md")

    extraction = extract_tables(source)
    assert len(extraction.tables) == 1
    assert extraction.tables[0].rows == (("one", ""), ("x", "y"))
    assert [problem.reason for problem in extraction.problems] == [
        "row 1 has 1 cells, expected 2",
        "row 2 has 3 cells, expected 2",
    ]


def test_markdown_delimiter_mismatch_is_reported(tmp_path: Path) -> None:
    text = "| a | b |\n|---|\n| one |\n"
    source = _markdown(tmp_path, text, "mismatch.md")

    extraction = extract_tables(source)
    assert extraction.tables == ()
    assert "delimiter has 1 columns, header has 2" in extraction.problems[0].reason


def test_pipe_lines_without_a_delimiter_are_not_tables(tmp_path: Path) -> None:
    source = _markdown(tmp_path, "a | b\nplain prose\n---\n")
    extraction = extract_tables(source)
    assert extraction.tables == ()
    assert extraction.problems == ()


def test_section_ordinal_follows_the_heading_path(tmp_path: Path) -> None:
    source = _markdown(
        tmp_path,
        "# One\n\nNo table.\n\n# Two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n",
        "sections.md",
    )
    record = DocumentIngestor(tmp_path / "knowledge").ingest(source)
    extraction = extract_tables(record.stored_path, sections=record.sections)
    assert extraction.tables[0].heading == "Two"
    assert extraction.tables[0].section_ordinal == 1


# ------------------------------------------------------------------ behaviour


def test_unsupported_formats_and_missing_files(tmp_path: Path) -> None:
    text = tmp_path / "notes.txt"
    text.write_text("a\tb\n1\t2\n", encoding="utf-8")
    extraction = extract_tables(text)
    assert extraction.tables == () and extraction.problems == ()

    with pytest.raises(FileNotFoundError):
        extract_tables(tmp_path / "absent.md")
    with pytest.raises(TypeError):
        extract_tables(123)  # type: ignore[arg-type]


def test_extraction_is_deterministic(tmp_path: Path) -> None:
    text = "| a | b |\n|---|---|\n| 1 | 2 |\n\n| c | d |\n|---|---|\n| 3 | 4 |\n"
    source = _markdown(tmp_path, text, "two.md")

    first = extract_tables(source)
    second = extract_tables(source)
    assert first == second
    assert [table.char_start for table in first.tables] == sorted(
        table.char_start for table in first.tables
    )
    assert len(first.tables) == 2


# -------------------------------------------------------------- index storage


def _ingest_table_doc(tmp_path: Path, text: str) -> tuple[DocumentRecord, KnowledgeIndex]:
    source = _markdown(tmp_path, text, "tables.md")
    record = DocumentIngestor(tmp_path / "knowledge").ingest(source)
    index = _knowledge(tmp_path)
    index.add_document(record)
    return record, index


def test_index_round_trip_preserves_table_shape(tmp_path: Path) -> None:
    text = (
        "# Costs\n\n"
        "| Session | Min | Max |\n"
        "|---|---|---|\n"
        "| London | 12 | 18 |\n\n"
        "| Region | Fee |\n"
        "|---|---|\n"
        "| Asia | 4 |\n"
    )
    record, index = _ingest_table_doc(tmp_path, text)
    tables = index.tables(record.doc_id)

    assert [table.kind for table in tables] == ["markdown", "markdown"]
    assert [table.caption for table in tables] == ["", ""]
    first = index.table(record.doc_id, 0)
    assert first is not None
    assert first.header == ("Session", "Min", "Max")
    assert first.rows == (("London", "12", "18"),)
    assert isinstance(first, TableRecord)
    assert index.table(record.doc_id, 9) is None
    assert index.tables("0" * 64) == []
    assert flatten_table(first) == "Session\tMin\tMax\nLondon\t12\t18"
    assert first.to_dict()["row_count"] == 2
    index.close()


def test_table_search_finds_a_cell_value(tmp_path: Path) -> None:
    text = "# Costs\n\n| Session | Cost |\n|---|---|\n| London | 12 |\n| New York | 15 |\n"
    record, index = _ingest_table_doc(tmp_path, text)

    hits = index.search_tables("London")
    assert [hit.doc_id for hit in hits] == [record.doc_id]
    assert hits[0].table_ordinal == 0
    assert hits[0].caption == ""
    assert "London" in hits[0].text
    assert index.tables_search_backend in {"fts5", "like"}

    if index.tables_search_backend != "like":
        index._tables_backend = "like"  # exercise the documented fallback
    fallback = index.search_tables("london", limit=5)
    assert fallback and fallback[0].score == 0.0
    assert index.search_tables("missing_term") == []
    with pytest.raises(ValueError):
        index.search_tables("   ")
    with pytest.raises(ValueError):
        index.search_tables("London", limit=0)

    # Paragraph search remains chunk-only, and the table's source lines are
    # already part of the section text (documented, not duplicated by storage).
    assert index.search("London")
    index.close()


def test_migration_from_v1_keeps_documents_and_backfills_tables(tmp_path: Path) -> None:
    text = "# Costs\n\n| Session | Cost |\n|---|---|\n| London | 12 |\n"
    source = _markdown(tmp_path, text, "costs.md")
    ingested = DocumentIngestor(tmp_path / "knowledge").ingest(source)

    path = tmp_path / "v1.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA_VERSION_DDL)
    for statement in _MIGRATIONS[1]:
        conn.execute(statement)
    conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (1, '2026-01-01')")
    conn.execute(
        """
        INSERT INTO documents (
            doc_id, checksum, filename, source_path, stored_path, size_bytes, mtime,
            format, parser, parser_version, imported_at, duplicate, status, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ingested.doc_id,
            ingested.checksum,
            ingested.filename,
            ingested.source_path,
            ingested.stored_path,
            ingested.size_bytes,
            ingested.mtime,
            ingested.format,
            ingested.parser,
            "1",
            ingested.imported_at,
            0,
            "indexed",
            None,
        ),
    )
    conn.commit()
    conn.close()

    index = KnowledgeIndex(path)
    assert index.schema_version == SCHEMA_VERSION == 2
    assert [document.doc_id for document in index.list_documents()] == [ingested.doc_id]
    assert index.tables(ingested.doc_id) == []

    # Re-importing the same bytes is a duplicate, but the tables are backfilled
    # once so an old database does not need to be rebuilt.
    duplicate = DocumentIngestor(tmp_path / "knowledge").ingest(source)
    assert duplicate.duplicate is True
    index.add_document(duplicate)
    tables = index.tables(ingested.doc_id)
    assert len(tables) == 1
    assert tables[0].rows == (("London", "12"),)
    index.add_document(duplicate)
    assert len(index.tables(ingested.doc_id)) == 1
    index.close()


def test_tables_are_removed_with_their_document(tmp_path: Path) -> None:
    record, index = _ingest_table_doc(tmp_path, "| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert index.tables(record.doc_id)
    with index.transaction() as tx:
        tx.execute("DELETE FROM documents WHERE doc_id = ?", (record.doc_id,))
    assert index.tables(record.doc_id) == []
    index.close()


def test_no_network_eval_or_trading_imports() -> None:
    forbidden_modules = {
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
    forbidden_calls = {"eval", "exec", "compile", "__import__"}
    forbidden_prefixes = (
        "scalping.paper",
        "scalping.strategy",
        "scalping.api",
        "scalping.backtest",
    )
    for path in KNOWLEDGE_SRC.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                names = []
            for name in names:
                root = name.split(".")[0]
                assert root not in forbidden_modules, f"{path.name} imports {name}"
                assert not name.startswith(forbidden_prefixes), f"{path.name} imports {name}"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls, f"{path.name} calls {node.func.id}"
