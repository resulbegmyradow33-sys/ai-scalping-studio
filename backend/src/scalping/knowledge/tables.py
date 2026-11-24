"""Structural table extraction from DOCX, HTML and Markdown.

The stdlib parsers in :mod:`scalping.knowledge.ingest` already flatten table
cell text into ordinary paragraphs. This module keeps the table structure as
well: a :class:`TableRecord` is a rectangular grid of cell strings plus the
caption, heading path, source section and character offsets needed to cite it.

Tables are inert data. Nothing here interprets a cell as a signal, a rule or an
order, calls a model, touches the network or evaluates text.

Merged cells
------------
A merged cell is never copied into the slots it covers. The first slot keeps the
cell text; every covered slot holds :data:`MERGED_CELL` (``"[[merged]]"``), so a
reader can tell "the author merged this" from "the author repeated this value".
The policy is documented per format in ``docs/knowledge-ingestion.md``.

Character offsets
-----------------
``char_start``/``char_end`` slice the *raw textual source* the extractor read:

* Markdown and HTML: the decoded file text.
* DOCX: the decoded ``word/document.xml`` payload, because the DOCX body has no
  stable plain-text rendering to point at.

The offset invariant is ``char_start < char_end`` and the slice starts at the
table's opening markup (``|`` for Markdown, ``<table`` for HTML, ``<w:tbl`` for
DOCX). Offsets are advisory citations, not a promise that re-parsing the slice
alone reproduces the record.

Malformed input
---------------
A malformed table never raises: :func:`extract_tables` returns the tables it
could read plus a :class:`TableProblem` per rejection with a reason. Only a
missing path or an unknown argument type raises.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scalping.knowledge.ingest import Section

__all__ = [
    "MERGED_CELL",
    "TABLE_KINDS",
    "TableExtraction",
    "TableProblem",
    "TableRecord",
    "extract_tables",
    "flatten_table",
]

#: Marker stored where a merged cell covers a grid slot. Not translatable: the
#: exact string is part of the stored contract and is tested.
MERGED_CELL = "[[merged]]"

#: Formats this module understands. ``kind`` is also the stored column value.
TABLE_KINDS: frozenset[str] = frozenset({"docx", "html", "markdown"})

_SUFFIX_KIND: dict[str, str] = {
    ".docx": "docx",
    ".html": "html",
    ".htm": "html",
    ".md": "markdown",
    ".markdown": "markdown",
}

#: Refuse a table wider than this many grid columns after colspan/gridSpan
#: expansion. A hostile file must not turn one row into a million cells.
MAX_TABLE_COLUMNS = 512

_WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W = "{" + _WORD_NAMESPACE + "}"
_DOCX_HEADING_RE = re.compile(r"^(?:heading|titre|überschrift)[ \t_-]*([1-6])$", re.IGNORECASE)

#: Same 32 MiB bound as the text extractor: refuse a zip bomb before reading.
MAX_DOCX_XML_BYTES = 32 * 1024 * 1024

_MD_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_MD_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_MD_DELIMITER_CELL_RE = re.compile(r"^:?-+:?$")
_MD_CAPTION_RE = re.compile(r"^\s*(?:\*\*)?(?:table|tablo)\b", re.IGNORECASE)

_TABLE_CAPTION_RE = _MD_CAPTION_RE
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class TableRecord:
    """One table with its location and citation metadata.

    ``header`` is the header row when the format marked one, else ``None``.
    ``rows`` holds the body rows only; ``row_count`` counts the header when it
    exists, matching what a reader sees. ``col_count`` is the widest row after
    merged-cell expansion.
    """

    kind: str
    caption: str
    heading: str
    header: tuple[str, ...] | None
    rows: tuple[tuple[str, ...], ...]
    section_ordinal: int
    row_count: int
    col_count: int
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        if self.kind not in TABLE_KINDS:
            raise ValueError(f"unknown table kind: {self.kind!r}")
        if self.header is not None and not self.header:
            raise ValueError("header must be None or a non-empty row")
        if self.char_start < 0 or self.char_end < self.char_start:
            raise ValueError("table offsets must satisfy 0 <= char_start <= char_end")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "caption": self.caption,
            "heading": self.heading,
            "header": None if self.header is None else list(self.header),
            "rows": [list(row) for row in self.rows],
            "section_ordinal": self.section_ordinal,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


@dataclass(frozen=True, slots=True)
class TableProblem:
    """A table the extractor refused, with a human-readable reason."""

    kind: str
    reason: str
    char_start: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "reason": self.reason, "char_start": self.char_start}


@dataclass(frozen=True, slots=True)
class TableExtraction:
    """The tables found in one document plus every malformed-table reason."""

    tables: tuple[TableRecord, ...] = ()
    problems: tuple[TableProblem, ...] = ()

    @property
    def malformed_count(self) -> int:
        return len(self.problems)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tables": [table.to_dict() for table in self.tables],
            "problems": [problem.to_dict() for problem in self.problems],
        }


def flatten_table(table: TableRecord) -> str:
    """The table as searchable text: rows joined by newlines, cells by tabs.

    The header row, when present, comes first. An empty cell contributes an
    empty field, so the tab positions still describe the grid.
    """
    lines: list[str] = []
    if table.header is not None:
        lines.append("\t".join(table.header))
    lines.extend("\t".join(row) for row in table.rows)
    return "\n".join(lines)


def _record(
    *,
    kind: str,
    caption: str,
    heading: str,
    header: tuple[str, ...] | None,
    rows: tuple[tuple[str, ...], ...],
    section_ordinal: int,
    char_start: int,
    char_end: int,
) -> TableRecord:
    widths = [len(row) for row in rows]
    if header is not None:
        widths.append(len(header))
    col_count = max(widths, default=0)
    return TableRecord(
        kind=kind,
        caption=caption.strip(),
        heading=heading.strip(),
        header=header,
        rows=rows,
        section_ordinal=section_ordinal,
        row_count=len(rows) + (0 if header is None else 1),
        col_count=col_count,
        char_start=char_start,
        char_end=char_end,
    )


def _section_ordinal(heading: str, sections: Sequence[Section] | None) -> int:
    """Best-effort section lookup by heading path; ``0`` when it cannot match.

    Heading paths repeat in real documents. The first section with the exact
    path wins, which is deterministic and never invents a section.
    """
    if not sections:
        return 0
    for ordinal, section in enumerate(sections):
        if section.heading.strip() == heading.strip():
            return ordinal
    return 0


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for match in re.finditer("\n", text):
        starts.append(match.end())
    return starts


def _offset_at(starts: list[int], line: int, column: int, length: int) -> int:
    """``HTMLParser.getpos()`` (1-based line, 0-based column) to a char offset."""
    index = min(max(line - 1, 0), len(starts) - 1)
    return min(starts[index] + column, length)


# --------------------------------------------------------------------- Markdown


def _split_pipe_row(line: str) -> list[str]:
    """Split a pipe-table row, honouring ``\\|`` and leading/trailing pipes."""
    text = line.strip()
    text = text.removeprefix("|")
    if text.endswith("|") and not text.endswith("\\|"):
        text = text[:-1]
    cells: list[str] = []
    buffer: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text) and text[index + 1] == "|":
            buffer.append("|")
            index += 2
            continue
        if char == "|":
            cells.append("".join(buffer).strip())
            buffer = []
            index += 1
            continue
        buffer.append(char)
        index += 1
    cells.append("".join(buffer).strip())
    return cells


def _markdown_delimiter(line: str) -> list[str] | None:
    """The alignment cells of a delimiter row, or ``None`` when it is not one."""
    if "|" not in line and "-" not in line:
        return None
    cells = _split_pipe_row(line)
    if not cells or not all(_MD_DELIMITER_CELL_RE.match(cell) for cell in cells):
        return None
    return cells


def _extract_markdown(path: Path, sections: Sequence[Section] | None) -> TableExtraction:
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    starts = _line_starts(text)
    tables: list[TableRecord] = []
    problems: list[TableProblem] = []
    stack: list[tuple[int, str]] = []
    heading = ""
    in_fence = False

    def line_end(index: int) -> int:
        return starts[index] + len(lines[index])

    index = 0
    while index < len(lines):
        line = lines[index]
        if _MD_FENCE_RE.match(line):
            in_fence = not in_fence
            index += 1
            continue
        if in_fence:
            index += 1
            continue
        heading_match = _MD_HEADING_RE.match(line)
        if heading_match is not None:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            stack = [item for item in stack if item[0] < level]
            stack.append((level, title))
            heading = " > ".join(item[1] for item in stack)
            index += 1
            continue
        if "|" not in line or index + 1 >= len(lines):
            index += 1
            continue
        if "|" not in lines[index + 1]:
            # A pipe-free ``---`` line is a setext heading rule or a horizontal
            # rule, never a pipe-table delimiter.
            index += 1
            continue
        delimiter = _markdown_delimiter(lines[index + 1])
        if delimiter is None:
            index += 1
            continue
        header_cells = _split_pipe_row(line)
        if len(header_cells) != len(delimiter) or not header_cells:
            problems.append(
                TableProblem(
                    "markdown",
                    f"delimiter has {len(delimiter)} columns, header has {len(header_cells)}",
                    starts[index],
                )
            )
            index += 2
            continue
        if len(header_cells) > MAX_TABLE_COLUMNS:
            problems.append(
                TableProblem(
                    "markdown",
                    f"table has {len(header_cells)} columns, above the "
                    f"{MAX_TABLE_COLUMNS} column limit",
                    starts[index],
                )
            )
            index += 1
            continue

        body: list[tuple[str, ...]] = []
        cursor = index + 2
        while cursor < len(lines):
            candidate = lines[cursor]
            if (
                not candidate.strip()
                or "|" not in candidate
                or _MD_FENCE_RE.match(candidate)
                or _MD_HEADING_RE.match(candidate)
            ):
                break
            cells = _split_pipe_row(candidate)
            if len(cells) != len(header_cells):
                problems.append(
                    TableProblem(
                        "markdown",
                        f"row {cursor - index - 1} has {len(cells)} cells, expected "
                        f"{len(header_cells)}",
                        starts[cursor],
                    )
                )
                cells = (cells + [""] * len(header_cells))[: len(header_cells)]
            body.append(tuple(cells))
            cursor += 1

        caption = ""
        previous = index - 1
        while previous >= 0 and not lines[previous].strip():
            previous -= 1
        if previous >= 0 and _MD_CAPTION_RE.match(lines[previous]):
            caption = lines[previous].strip().strip("*").strip()
        tables.append(
            _record(
                kind="markdown",
                caption=caption,
                heading=heading,
                header=tuple(header_cells),
                rows=tuple(body),
                section_ordinal=_section_ordinal(heading, sections),
                char_start=starts[index],
                char_end=line_end(cursor - 1),
            )
        )
        index = cursor

    return TableExtraction(tables=tuple(tables), problems=tuple(problems))


# ------------------------------------------------------------------------- HTML


@dataclass(slots=True)
class _RawCell:
    text: list[str] = field(default_factory=list)
    colspan: int = 1
    rowspan: int = 1
    is_header: bool = False


@dataclass(slots=True)
class _RawRow:
    cells: list[_RawCell] = field(default_factory=list)
    in_thead: bool = False


@dataclass(slots=True)
class _RawTable:
    heading: str
    char_start: int
    rows: list[_RawRow] = field(default_factory=list)
    caption: list[str] = field(default_factory=list)
    caption_open: bool = False
    group_is_head: bool = False
    char_end: int = 0

    def current_row(self) -> _RawRow | None:
        return self.rows[-1] if self.rows else None


def _span_value(attrs: list[tuple[str, str | None]], name: str) -> int:
    for key, value in attrs:
        if key.lower() != name or value is None:
            continue
        try:
            number = int(str(value).strip())
        except ValueError:
            return 1
        return max(1, min(number, MAX_TABLE_COLUMNS))
    return 1


class _HtmlTableParser(HTMLParser):
    """Collect tables, captions, headings and cell spans from HTML text."""

    def __init__(self, line_starts: list[int], text_length: int) -> None:
        super().__init__(convert_charrefs=True)
        self._starts = line_starts
        self._text_length = text_length
        self._skip = 0
        self._heading_stack: list[tuple[int, str]] = []
        self._heading_level: int | None = None
        self._heading_parts: list[str] = []
        self._tables: list[_RawTable] = []
        self._open: list[_RawTable] = []
        self.problems: list[TableProblem] = []

    def _heading(self) -> str:
        return " > ".join(title for _, title in self._heading_stack)

    def _offset(self) -> int:
        line, column = self.getpos()
        return _offset_at(self._starts, line, column, self._text_length)

    def _close_cell(self, table: _RawTable) -> None:
        row = table.current_row()
        if row is None or not row.cells:
            return
        cell = row.cells[-1]
        cell.text = ["".join(cell.text)]

    def _close_row(self, table: _RawTable) -> None:
        self._close_cell(table)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template"}:
            self._skip += 1
            return
        if self._skip:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_level = int(tag[1])
            self._heading_parts = []
            return
        if tag in {"br", "hr"}:
            if self._open:
                row = self._open[-1].current_row()
                if row is not None and row.cells:
                    row.cells[-1].text.append(" ")
            return
        if tag == "table":
            table = _RawTable(heading=self._heading(), char_start=self._offset())
            self._open.append(table)
            self._tables.append(table)
            return
        if not self._open:
            return
        table = self._open[-1]
        if tag == "caption":
            table.caption_open = True
            table.caption = []
        elif tag in {"thead", "tbody", "tfoot"}:
            self._close_row(table)
            table.group_is_head = tag == "thead"
        elif tag == "tr":
            self._close_row(table)
            table.rows.append(_RawRow(in_thead=table.group_is_head))
        elif tag in {"td", "th"}:
            row = table.current_row()
            if row is None:
                row = _RawRow(in_thead=table.group_is_head)
                table.rows.append(row)
            self._close_cell(table)
            row.cells.append(
                _RawCell(
                    colspan=_span_value(attrs, "colspan"),
                    rowspan=_span_value(attrs, "rowspan"),
                    is_header=tag == "th",
                )
            )

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template"}:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self._heading_level is not None:
            title = _WHITESPACE_RE.sub(" ", "".join(self._heading_parts)).strip()
            level = self._heading_level
            self._heading_stack = [item for item in self._heading_stack if item[0] < level]
            if title:
                self._heading_stack.append((level, title))
            self._heading_level = None
            self._heading_parts = []
            return
        if not self._open:
            return
        table = self._open[-1]
        if tag == "caption":
            table.caption_open = False
        elif tag in {"td", "th"}:
            self._close_cell(table)
        elif tag == "tr":
            self._close_row(table)
        elif tag in {"thead", "tbody", "tfoot"}:
            self._close_row(table)
            table.group_is_head = False
        elif tag == "table":
            self._close_row(table)
            table.char_end = self._offset() + len("</table>")
            self._open.pop()

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._heading_level is not None:
            self._heading_parts.append(data)
            return
        if not self._open:
            return
        table = self._open[-1]
        if table.caption_open:
            table.caption.append(data)
            return
        row = table.current_row()
        if row is None or not row.cells:
            return
        row.cells[-1].text.append(data)

    def finish(self) -> tuple[list[_RawTable], list[TableProblem]]:
        for table in self._open:
            self._close_row(table)
            table.char_end = self._text_length
            self.problems.append(
                TableProblem("html", "table is not closed with </table>", table.char_start)
            )
        self._open.clear()
        return self._tables, self.problems


def _expand_html_rows(
    raw_rows: list[_RawRow],
) -> tuple[list[tuple[str, ...]], int]:
    """Expand colspan/rowspan into a square grid of strings.

    The covering cell keeps its text; every covered slot holds
    :data:`MERGED_CELL`. Missing trailing cells become ``""``.
    """
    grid: dict[tuple[int, int], str] = {}
    for row_index, row in enumerate(raw_rows):
        column = 0
        for cell in row.cells:
            while (row_index, column) in grid:
                column += 1
            text = _WHITESPACE_RE.sub(" ", "".join(cell.text)).strip()
            grid[(row_index, column)] = text
            for extra in range(1, cell.colspan):
                grid[(row_index, column + extra)] = MERGED_CELL
            for lower in range(row_index + 1, row_index + cell.rowspan):
                for extra in range(cell.colspan):
                    grid[(lower, column + extra)] = MERGED_CELL
            column += cell.colspan
    if not grid:
        return [], 0
    width = max(column for _, column in grid) + 1
    height = max(row for row, _ in grid) + 1
    rows = [tuple(grid.get((row, column), "") for column in range(width)) for row in range(height)]
    return rows, width


def _extract_html(path: Path, sections: Sequence[Section] | None) -> TableExtraction:
    text = path.read_text(encoding="utf-8-sig")
    parser = _HtmlTableParser(_line_starts(text), len(text))
    try:
        parser.feed(text)
        parser.close()
    except Exception as exc:  # noqa: BLE001 - a broken document must not crash ingestion
        return TableExtraction(problems=(TableProblem("html", f"{type(exc).__name__}: {exc}"),))
    raw_tables, problems = parser.finish()

    tables: list[TableRecord] = []
    collected: list[TableProblem] = list(problems)
    for raw in raw_tables:
        if not raw.rows:
            collected.append(TableProblem("html", "table has no rows", raw.char_start))
            continue
        header: tuple[str, ...] | None = None
        body_rows = raw.rows
        head_rows = [row for row in body_rows if row.in_thead]
        if head_rows:
            header_row = head_rows[0]
            body_rows = [row for row in body_rows if row is not header_row]
            expanded, _ = _expand_html_rows([header_row])
            header = expanded[0] if expanded else None
        elif (
            body_rows and body_rows[0].cells and all(cell.is_header for cell in body_rows[0].cells)
        ):
            expanded, _ = _expand_html_rows([body_rows[0]])
            header = expanded[0] if expanded else None
            body_rows = body_rows[1:]
        expanded_rows, _ = _expand_html_rows(body_rows)
        caption = _WHITESPACE_RE.sub(" ", "".join(raw.caption)).strip()
        tables.append(
            _record(
                kind="html",
                caption=caption,
                heading=raw.heading,
                header=header,
                rows=tuple(expanded_rows),
                section_ordinal=_section_ordinal(raw.heading, sections),
                char_start=raw.char_start,
                char_end=max(raw.char_start, raw.char_end),
            )
        )
    return TableExtraction(tables=tuple(tables), problems=tuple(collected))


# ------------------------------------------------------------------------- DOCX


def _docx_table_spans(payload: str) -> list[tuple[int, int]]:
    """Top-level ``<w:tbl>`` spans inside ``word/document.xml``, in order."""
    spans: list[tuple[int, int]] = []
    depth = 0
    start = 0
    for match in re.finditer(r"<w:tbl(?=[\s>])|</w:tbl>", payload):
        if payload.startswith("</", match.start()):
            depth -= 1
            if depth == 0:
                spans.append((start, match.end()))
        else:
            if depth == 0:
                start = match.start()
            depth += 1
    return spans


def _docx_heading_level(style: str) -> int | None:
    if not style:
        return None
    match = _DOCX_HEADING_RE.match(style.strip())
    if match is not None:
        return int(match.group(1))
    if style.strip().lower() in {"title", "subtitle"}:
        return 1
    return None


def _docx_paragraph_text(paragraph: ElementTree.Element) -> str:
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == f"{_W}t":
            parts.append(node.text or "")
        elif node.tag == f"{_W}tab" or node.tag in {f"{_W}br", f"{_W}cr"}:
            parts.append(" ")
    return _WHITESPACE_RE.sub(" ", "".join(parts)).strip()


def _docx_cell_texts(cell: ElementTree.Element) -> list[str]:
    """Paragraph texts of one cell, not descending into a nested table."""
    texts: list[str] = []
    for child in cell:
        if child.tag == f"{_W}tbl":
            continue
        if child.tag == f"{_W}p":
            text = _docx_paragraph_text(child)
            if text:
                texts.append(text)
        else:
            texts.extend(_docx_cell_texts(child))
    return texts


def _docx_cell(cell: ElementTree.Element) -> tuple[str, int, bool]:
    """``(text, grid_span, is_vertical_continuation)`` for one ``w:tc``."""
    properties = cell.find(f"{_W}tcPr")
    grid_span = 1
    continuation = False
    if properties is not None:
        span_node = properties.find(f"{_W}gridSpan")
        if span_node is not None:
            try:
                grid_span = max(1, int(span_node.get(f"{_W}val", "1")))
            except ValueError:
                grid_span = 1
        merge_node = properties.find(f"{_W}vMerge")
        if merge_node is not None:
            merge_value = (merge_node.get(f"{_W}val") or "").strip().lower()
            continuation = merge_value in {"", "continue"}
    grid_span = min(grid_span, MAX_TABLE_COLUMNS)
    text = " ".join(_docx_cell_texts(cell)).strip()
    if continuation:
        return MERGED_CELL, grid_span, True
    return text, grid_span, False


def _docx_row_is_header(row: ElementTree.Element) -> bool:
    properties = row.find(f"{_W}trPr")
    if properties is None:
        return False
    header = properties.find(f"{_W}tblHeader")
    if header is None:
        return False
    value = (header.get(f"{_W}val") or "true").strip().lower()
    return value not in {"0", "false", "off"}


def _docx_table_record(
    table: ElementTree.Element,
    *,
    heading: str,
    caption: str,
    sections: Sequence[Section] | None,
    char_start: int,
    char_end: int,
) -> TableRecord:
    rows_raw = table.findall(f"{_W}tr")
    header: tuple[str, ...] | None = None
    body_start = 0
    if rows_raw and _docx_row_is_header(rows_raw[0]):
        header = tuple(_docx_expand_row(rows_raw[0]))
        body_start = 1
    body = tuple(_docx_expand_row(row) for row in rows_raw[body_start:])
    widths = [len(row) for row in body]
    if header is not None:
        widths.append(len(header))
    width = max(widths, default=0)
    padded = tuple(tuple(list(row) + [""] * (width - len(row))) for row in body)
    padded_header = None if header is None else tuple(list(header) + [""] * (width - len(header)))
    return _record(
        kind="docx",
        caption=caption,
        heading=heading,
        header=padded_header,
        rows=padded,
        section_ordinal=_section_ordinal(heading, sections),
        char_start=char_start,
        char_end=char_end,
    )


def _docx_expand_row(row: ElementTree.Element) -> list[str]:
    cells: list[str] = []
    for cell in row.findall(f"{_W}tc"):
        text, grid_span, _ = _docx_cell(cell)
        cells.append(text)
        cells.extend([MERGED_CELL] * (grid_span - 1))
        if len(cells) > MAX_TABLE_COLUMNS:
            cells = cells[:MAX_TABLE_COLUMNS]
            break
    return cells


def _extract_docx(path: Path, sections: Sequence[Section] | None) -> TableExtraction:
    try:
        with zipfile.ZipFile(path) as archive:
            try:
                info = archive.getinfo("word/document.xml")
            except KeyError:
                return TableExtraction(
                    problems=(TableProblem("docx", "DOCX has no word/document.xml"),)
                )
            if info.file_size > MAX_DOCX_XML_BYTES:
                return TableExtraction(
                    problems=(
                        TableProblem(
                            "docx",
                            f"word/document.xml is {info.file_size} bytes, above the "
                            f"{MAX_DOCX_XML_BYTES} byte limit",
                        ),
                    )
                )
            payload_bytes = archive.read(info)
    except zipfile.BadZipFile as exc:
        return TableExtraction(problems=(TableProblem("docx", f"not a DOCX archive: {exc}"),))

    payload = payload_bytes.decode("utf-8", errors="replace")
    try:
        root = ElementTree.fromstring(payload_bytes)
    except ElementTree.ParseError as exc:
        return TableExtraction(problems=(TableProblem("docx", f"invalid document XML: {exc}"),))

    spans = _docx_table_spans(payload)
    tables: list[TableRecord] = []
    problems: list[TableProblem] = []
    stack: list[tuple[int, str]] = []
    heading = ""
    last_paragraph = ""

    def walk(container: ElementTree.Element) -> None:
        nonlocal heading, last_paragraph
        for child in container:
            if child.tag == f"{_W}p":
                style = ""
                properties = child.find(f"{_W}pPr")
                if properties is not None:
                    style_node = properties.find(f"{_W}pStyle")
                    if style_node is not None:
                        style = style_node.get(f"{_W}val", "")
                text = _docx_paragraph_text(child)
                if not text:
                    continue
                level = _docx_heading_level(style)
                if level is not None:
                    stack[:] = [item for item in stack if item[0] < level]
                    stack.append((level, text))
                    heading = " > ".join(item[1] for item in stack)
                    last_paragraph = ""
                else:
                    last_paragraph = text
            elif child.tag == f"{_W}tbl":
                index = len(tables)
                if index >= len(spans):
                    char_start = 0
                    char_end = 0
                else:
                    char_start, char_end = spans[index]
                caption = last_paragraph if _TABLE_CAPTION_RE.match(last_paragraph) else ""
                table = _docx_table_record(
                    child,
                    heading=heading,
                    caption=caption,
                    sections=sections,
                    char_start=char_start,
                    char_end=char_end,
                )
                if table.row_count == 0:
                    problems.append(TableProblem("docx", "table has no rows", char_start))
                else:
                    tables.append(table)
                last_paragraph = ""
            elif child.tag in {f"{_W}sdt", f"{_W}sdtContent"}:
                walk(child)

    body = root.find(f"{_W}body")
    if body is not None:
        walk(body)
    return TableExtraction(tables=tuple(tables), problems=tuple(problems))


# ---------------------------------------------------------------------- entry


def extract_tables(
    path: str | Path,
    sections: Sequence[Section] | None = None,
) -> TableExtraction:
    """Extract tables from ``path`` in document order.

    ``sections`` is the ingest output for the same file, when available; it is
    used only to resolve ``section_ordinal`` by exact heading path. A missing
    file raises; a malformed document returns problems instead of raising and
    never aborts the remaining tables.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError(f"path must be a str or Path, got {type(path).__name__}")
    source = Path(path)
    kind = _SUFFIX_KIND.get(source.suffix.lower())
    if kind is None:
        return TableExtraction()
    if source.is_dir():
        raise IsADirectoryError(f"not a document: {source}")
    if not source.exists():
        raise FileNotFoundError(f"source file does not exist: {source}")
    try:
        if kind == "markdown":
            return _extract_markdown(source, sections)
        if kind == "html":
            return _extract_html(source, sections)
        return _extract_docx(source, sections)
    except Exception as exc:  # noqa: BLE001 - untrusted input, report instead of crash
        return TableExtraction(problems=(TableProblem(kind, f"{type(exc).__name__}: {exc}"),))
