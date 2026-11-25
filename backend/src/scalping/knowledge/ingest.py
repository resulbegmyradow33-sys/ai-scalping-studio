"""Local document ingestion with provenance.

The ingestor copies the caller's file into ``<root>/sources/<sha256>/<name>``
before extracting anything, so a parser failure never loses the original. The
checksum identifies the document: importing the same bytes twice reuses the
stored copy and reports ``duplicate=True`` instead of writing a second one.

Every format except PDF uses only the Python standard library. PDF text
extraction is an optional extra: when ``pypdf`` is not installed the ingestor
reports ``status="unsupported"`` with the install hint instead of quietly
requiring a dependency. A parser that fails for another reason is recorded as
``status="extraction_failed"`` with the exception message; a record is always
returned for an existing file, never an empty silent drop.

Extracted text is inert data. Nothing in this module interprets it as a
strategy, a signal or an order.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol
from xml.etree import ElementTree

from scalping.knowledge.tables import (
    TableRecord,
    extract_tables,
    read_docx_payload,
)

__all__ = [
    "DocumentIngestionError",
    "DocumentIngestor",
    "DocumentRecord",
    "ExtractionError",
    "PdfExtraction",
    "Section",
    "TextExtractor",
    "UnsupportedFormat",
]

#: Directory below the ingestion root that holds preserved originals.
SOURCES_DIRNAME = "sources"

#: Recorded in ``status="unsupported"`` when the optional PDF parser is absent.
PDF_INSTALL_HINT = (
    "PDF text extraction needs the optional pypdf extra: run "
    "`uv sync --extra pdf` from the backend directory (or "
    "`pip install 'scalping[pdf]'`). The PDF is preserved as-is; it is never "
    "read as plain bytes and no text is invented."
)

#: Version of the extraction behaviour, recorded with every document. Bump it
#: when a parser change alters the text a document produces. Version 2 adds
#: structural table records; the paragraph text itself is unchanged.
PARSER_VERSION = "2"

_HASH_CHUNK_BYTES = 1024 * 1024

_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_HTML_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "title"})
_HTML_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "td",
        "th",
        "section",
        "article",
        "header",
        "footer",
        "blockquote",
        "pre",
        "table",
        "ul",
        "ol",
    }
)
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_DOCX_HEADING_RE = re.compile(r"^(?:heading|titre|überschrift)[ \t_-]*([1-6])$", re.IGNORECASE)

_WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W = "{" + _WORD_NAMESPACE + "}"


class DocumentIngestionError(Exception):
    """The source file cannot be ingested; nothing was recorded."""


class UnsupportedFormat(DocumentIngestionError):
    """No extractor is installed for this format."""

    def __init__(self, suffix: str, message: str) -> None:
        self.suffix = suffix
        super().__init__(message)


class ExtractionError(DocumentIngestionError):
    """A supported format could not be turned into text."""


@dataclass(frozen=True, slots=True)
class Section:
    """One logical part of a document, before offsets are assigned."""

    heading: str
    label: str
    text: str


@dataclass(frozen=True, slots=True)
class PdfExtraction:
    """Per-page PDF text plus the pages that carried none.

    ``sections`` holds one section per page that produced text; each label is
    ``page <n>`` with the 1-based physical page index. ``empty_pages`` names
    the pages whose text was blank: a scanned image or an empty page, never
    filled in with invented text.
    """

    sections: tuple[Section, ...]
    page_count: int
    empty_pages: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """Everything known about one imported document.

    ``doc_id`` is the SHA-256 of the original bytes and ``checksum`` repeats it
    for callers that prefer the explicit name. ``sections`` is empty when the
    status is ``extraction_failed`` or ``unsupported``.
    """

    doc_id: str
    checksum: str
    filename: str
    source_path: str
    stored_path: str
    size_bytes: int
    mtime: str
    format: str
    parser: str
    parser_version: str
    imported_at: str
    duplicate: bool
    status: str
    error: str | None = None
    sections: tuple[Section, ...] = field(default_factory=tuple)
    tables: tuple[TableRecord, ...] = field(default_factory=tuple)
    #: Non-fatal extraction notes, for example PDF pages that had no text.
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def extracted(self) -> bool:
        """True when text is available; failure records carry a reason instead."""
        return self.status == "indexed" and bool(self.sections)

    @property
    def sha256(self) -> str:
        """Alias for the checksum, for callers that name the algorithm."""
        return self.checksum

    @property
    def size(self) -> int:
        """Alias for ``size_bytes``."""
        return self.size_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "checksum": self.checksum,
            "filename": self.filename,
            "source_path": self.source_path,
            "stored_path": self.stored_path,
            "size_bytes": self.size_bytes,
            "mtime": self.mtime,
            "format": self.format,
            "parser": self.parser,
            "parser_version": self.parser_version,
            "imported_at": self.imported_at,
            "duplicate": self.duplicate,
            "status": self.status,
            "error": self.error,
            "section_count": len(self.sections),
            "table_count": len(self.tables),
            "warnings": list(self.warnings),
        }


class TextExtractor(Protocol):
    """A stdlib-only text extractor.

    ``suffixes`` names the file extensions it owns, each including the dot and
    lowercase. ``extract`` returns sections in reading order and may raise
    :class:`ExtractionError`; the ingestor turns that into a failed record.
    """

    name: str
    version: str
    suffixes: frozenset[str]

    def extract(self, path: Path) -> list[Section]: ...


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _normalise_text(raw: str) -> str:
    """Collapse horizontal runs, strip each line and drop blank lines."""
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in raw.splitlines()]
    return "\n".join(line for line in lines if line)


def _unique_sections(raw: list[Section]) -> list[Section]:
    """Drop empty sections and give every section a stable label."""
    sections: list[Section] = []
    for section in raw:
        text = section.text.strip()
        if not text:
            continue
        heading = section.heading.strip()
        label = section.label.strip()
        if not label:
            label = heading or f"section {len(sections) + 1}"
        sections.append(Section(heading=heading, label=label, text=text))
    return sections


def _read_text(path: Path) -> str:
    """UTF-8 text with a BOM tolerated; invalid bytes are an extraction failure."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ExtractionError(f"file is not valid UTF-8 text: {exc}") from exc


class _PlainTextExtractor:
    """UTF-8 text; form feeds become pages, otherwise the file is one section."""

    name = "stdlib-text"
    version = PARSER_VERSION
    suffixes = frozenset({".txt"})

    def extract(self, path: Path) -> list[Section]:
        text = _read_text(path)
        pages = text.split("\f")
        if len(pages) == 1:
            return _unique_sections([Section(heading="", label="section 1", text=text)])
        return _unique_sections(
            [
                Section(heading="", label=f"page {number}", text=page)
                for number, page in enumerate(pages, start=1)
            ]
        )


class _MarkdownExtractor:
    """ATX headings define sections; fenced code is not scanned for headings."""

    name = "stdlib-markdown"
    version = PARSER_VERSION
    suffixes = frozenset({".md", ".markdown"})

    def extract(self, path: Path) -> list[Section]:
        text = _read_text(path)
        sections: list[Section] = []
        body: list[str] = []
        stack: list[tuple[int, str]] = []
        titles: list[str] = []
        heading = ""
        in_fence = False

        def flush() -> None:
            value = "\n".join(body).strip()
            if value:
                sections.append(Section(heading=heading, label=heading, text=value))
            body.clear()

        for line in text.splitlines():
            if _FENCE_RE.match(line):
                in_fence = not in_fence
                body.append(line)
                continue
            match = None if in_fence else _MARKDOWN_HEADING_RE.match(line)
            if match is None:
                body.append(line)
                continue
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            titles.append(title)
            stack = [item for item in stack if item[0] < level]
            stack.append((level, title))
            heading = " > ".join(item[1] for item in stack)
        flush()
        found = _unique_sections(sections)
        if not found and titles:
            return [Section(heading="", label="section 1", text="\n".join(titles))]
        return found


class _HtmlTextParser(HTMLParser):
    """Collect visible text, splitting sections at ``h1``–``h6`` boundaries."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[Section] = []
        self._body: list[str] = []
        self._stack: list[tuple[int, str]] = []
        self._heading_level: int | None = None
        self._heading_parts: list[str] = []
        self.headings: list[str] = []
        self._skip = 0

    def _heading(self) -> str:
        return " > ".join(title for _, title in self._stack)

    def flush(self) -> None:
        text = _normalise_text("".join(self._body))
        if text:
            heading = self._heading()
            self.sections.append(Section(heading=heading, label=heading, text=text))
        self._body.clear()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _HTML_SKIP_TAGS:
            self._skip += 1
            return
        if self._skip:
            return
        if tag in _HEADING_TAGS:
            self.flush()
            self._heading_level = int(tag[1])
            self._heading_parts = []
            return
        if tag in _HTML_BLOCK_TAGS:
            self._body.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not self._skip and tag.lower() in {"br", "hr"}:
            self._body.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _HTML_SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag in _HEADING_TAGS and self._heading_level is not None:
            title = _normalise_text("".join(self._heading_parts))
            level = self._heading_level
            self._stack = [item for item in self._stack if item[0] < level]
            if title:
                self._stack.append((level, title))
                self.headings.append(title)
            self._heading_level = None
            self._heading_parts = []
            return
        if tag in _HTML_BLOCK_TAGS:
            self._body.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._heading_level is not None:
            self._heading_parts.append(data)
        else:
            self._body.append(data)


class _HtmlExtractor:
    """HTML via ``html.parser``; scripts, styles and templates are dropped."""

    name = "html.parser"
    version = PARSER_VERSION
    suffixes = frozenset({".html", ".htm"})

    def extract(self, path: Path) -> list[Section]:
        parser = _HtmlTextParser()
        parser.feed(_read_text(path))
        parser.close()
        parser.flush()
        found = _unique_sections(parser.sections)
        if not found and parser.headings:
            return [Section(heading="", label="section 1", text="\n".join(parser.headings))]
        return found


def _docx_heading_level(style: str) -> int | None:
    if not style:
        return None
    match = _DOCX_HEADING_RE.match(style.strip())
    if match is not None:
        return int(match.group(1))
    if style.strip().lower() in {"title", "subtitle"}:
        return 1
    return None


class _DocxExtractor:
    """DOCX via zipfile + ElementTree; paragraph text only, no styling."""

    name = "zipfile+ElementTree"
    version = PARSER_VERSION
    suffixes = frozenset({".docx"})

    def extract(self, path: Path) -> list[Section]:
        return self.extract_with_payload(path)[0]

    def extract_with_payload(self, path: Path) -> tuple[list[Section], bytes]:
        """Read ``word/document.xml`` once and return sections plus the payload.

        The ingestor passes the payload to table extraction so a DOCX import
        opens the archive a single time.
        """
        try:
            payload = read_docx_payload(path)
        except zipfile.BadZipFile as exc:
            raise ExtractionError(f"not a valid DOCX zip archive: {exc}") from exc
        except ValueError as exc:
            raise ExtractionError(str(exc)) from exc
        return self.sections_from_payload(payload), payload

    @staticmethod
    def sections_from_payload(payload: bytes) -> list[Section]:
        try:
            root = ElementTree.fromstring(payload)
        except ElementTree.ParseError as exc:
            raise ExtractionError(f"word/document.xml is not valid XML: {exc}") from exc

        sections: list[Section] = []
        body: list[str] = []
        stack: list[tuple[int, str]] = []
        heading_titles: list[str] = []
        heading = ""

        def flush() -> None:
            text = "\n\n".join(body).strip()
            if text:
                sections.append(Section(heading=heading, label=heading, text=text))
            body.clear()

        for paragraph in root.iter(f"{_W}p"):
            style = ""
            properties = paragraph.find(f"{_W}pPr")
            if properties is not None:
                style_node = properties.find(f"{_W}pStyle")
                if style_node is not None:
                    style = style_node.get(f"{_W}val", "")
            parts: list[str] = []
            for node in paragraph.iter():
                if node.tag == f"{_W}t":
                    parts.append(node.text or "")
                elif node.tag == f"{_W}tab":
                    parts.append("\t")
                elif node.tag in {f"{_W}br", f"{_W}cr"}:
                    parts.append("\n")
            text = "".join(parts).strip()
            if not text:
                continue
            level = _docx_heading_level(style)
            if level is not None:
                flush()
                stack = [item for item in stack if item[0] < level]
                stack.append((level, text))
                heading_titles.append(text)
                heading = " > ".join(item[1] for item in stack)
            else:
                body.append(text)
        flush()
        found = _unique_sections(sections)
        if not found and heading_titles:
            return [Section(heading="", label="section 1", text="\n".join(heading_titles))]
        return found


def _pypdf_available() -> bool:
    """Whether the optional PDF parser is importable, without importing it."""
    try:
        return importlib.util.find_spec("pypdf") is not None
    except (ImportError, ValueError):  # a broken or shadowed install is "absent"
        return False


class _PdfExtractor:
    """PDF text through the optional ``pypdf`` package, page by page.

    The import happens inside :meth:`extract_pdf`, so installation is the only
    thing that changes: the core install never imports ``pypdf`` and a missing
    package is an :class:`UnsupportedFormat`, not an ``ImportError``.
    """

    name = "pypdf"
    version = PARSER_VERSION
    suffixes = frozenset({".pdf"})

    def extract(self, path: Path) -> list[Section]:
        return list(self.extract_pdf(path).sections)

    def extract_pdf(self, path: Path) -> PdfExtraction:
        try:
            from pypdf import PdfReader
            from pypdf.errors import PdfReadError
        except ImportError as exc:  # pragma: no cover - guarded by _pypdf_available
            raise UnsupportedFormat(".pdf", PDF_INSTALL_HINT) from exc

        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as exc:  # pypdf's decrypt errors vary
                    raise ExtractionError("PDF is encrypted; decryption is not supported") from exc
                if reader.is_encrypted:
                    raise ExtractionError("PDF is encrypted; decryption is not supported")
            page_count = len(reader.pages)
            sections: list[Section] = []
            empty_pages: list[int] = []
            for number in range(1, page_count + 1):
                try:
                    raw = reader.pages[number - 1].extract_text() or ""
                except Exception as exc:  # one bad page must not hide the rest
                    raise ExtractionError(
                        f"PDF page {number} could not be read: {type(exc).__name__}: {exc}"
                    ) from exc
                text = _normalise_text(raw)
                if text:
                    sections.append(Section(heading="", label=f"page {number}", text=text))
                else:
                    empty_pages.append(number)
        except PdfReadError as exc:
            raise ExtractionError(f"not a readable PDF: {exc}") from exc

        if not sections:
            if page_count == 0:
                raise ExtractionError("PDF has no pages")
            listed = ", ".join(str(page) for page in empty_pages)
            raise ExtractionError(
                f"no extractable text in {page_count} PDF page(s); pages {listed} appear "
                "scanned or blank, and OCR is not implemented"
            )
        return PdfExtraction(tuple(sections), page_count, tuple(empty_pages))


_PDF_EXTRACTOR = _PdfExtractor()

_EXTRACTORS: tuple[TextExtractor, ...] = (
    _PlainTextExtractor(),
    _MarkdownExtractor(),
    _HtmlExtractor(),
    _DocxExtractor(),
    _PDF_EXTRACTOR,
)

#: Extension to recorded format name, including formats with no extractor yet.
_FORMATS: dict[str, str] = {
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".docx": "docx",
    ".pdf": "pdf",
}

#: Formats whose extracted tables are stored with the document.
_TABLE_FORMATS: frozenset[str] = frozenset({"docx", "html", "markdown"})


def _extractor_for(suffix: str) -> TextExtractor:
    if suffix == ".pdf" and not _pypdf_available():
        raise UnsupportedFormat(suffix, PDF_INSTALL_HINT)
    for extractor in _EXTRACTORS:
        if suffix in extractor.suffixes:
            return extractor
    raise UnsupportedFormat(suffix, f"no extractor for {suffix or 'a file without extension'}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source(path: str | Path) -> Path:
    """Resolve an existing regular file and refuse traversal in the given path.

    A ``..`` component is rejected even when the resolved file is legitimate:
    the source path can come from an API caller, and the preserved copy must
    not depend on a path that walks out of its declared location.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError(f"path must be a str or Path, got {type(path).__name__}")
    text = str(path)
    if "\x00" in text:
        raise ValueError("source path contains a NUL byte")
    given = Path(text)
    if any(part == ".." for part in given.parts):
        raise ValueError(f"source path must not contain '..': {text!r}")
    if not given.exists():
        raise FileNotFoundError(f"source file does not exist: {text}")
    resolved = given.resolve()
    if not resolved.is_file():
        raise DocumentIngestionError(f"source path is not a regular file: {resolved}")
    return resolved


def _safe_filename(name: str) -> str:
    """The preserved file keeps its name but must be a single path segment."""
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise DocumentIngestionError(f"unsafe source filename: {name!r}")
    if Path(name).name != name:
        raise DocumentIngestionError(f"source filename is not a single segment: {name!r}")
    return name


class DocumentIngestor:
    """Copy, checksum and extract local documents below ``root``.

    The object is stateless apart from ``root``; duplicate detection is the
    existence of ``sources/<checksum>/<name>`` on disk, so two ingestor
    instances agree without a database.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.sources = self.root / SOURCES_DIRNAME
        self.sources.mkdir(parents=True, exist_ok=True)

    def ingest(self, path: str | Path) -> DocumentRecord:
        """Preserve one file and return its provenance and extracted sections.

        An existing regular file always yields a record. Unsupported formats
        and parser failures are represented by ``status``/``error``; only a
        missing, non-regular or traversal-y path raises.
        """
        source = _validate_source(path)
        stat = source.stat()
        checksum = _sha256_file(source)
        filename = _safe_filename(source.name)
        stem = self.root / SOURCES_DIRNAME / checksum
        self._check_within_root(stem)
        existing = self._existing_copy(stem, checksum)
        if existing is None:
            stored = stem / filename
            self._preserve(source, stem, stored, checksum)
            duplicate = False
        else:
            # The checksum is the document identity: the same bytes under a
            # second name reuse the first copy instead of writing another one.
            stored = existing
            duplicate = True

        suffix = source.suffix.lower()
        document_format = _FORMATS.get(suffix, suffix.lstrip(".") or "unknown")
        imported_at = _now()
        status = "indexed"
        error: str | None = None
        sections: tuple[Section, ...] = ()
        tables: tuple[TableRecord, ...] = ()
        parser = "none"
        parser_version = PARSER_VERSION
        docx_payload: bytes | None = None
        warnings: tuple[str, ...] = ()

        try:
            extractor = _extractor_for(suffix)
            parser = extractor.name
            parser_version = extractor.version
            if isinstance(extractor, _DocxExtractor):
                extracted, docx_payload = extractor.extract_with_payload(stored)
                sections = tuple(extracted)
            elif isinstance(extractor, _PdfExtractor):
                pdf = extractor.extract_pdf(stored)
                sections = pdf.sections
                warnings = tuple(
                    f"PDF page {number} has no extractable text (scanned image or "
                    "blank page); OCR is not implemented"
                    for number in pdf.empty_pages
                )
            else:
                sections = tuple(extractor.extract(stored))
            if not sections:
                raise ExtractionError("no extractable text")
            if document_format in _TABLE_FORMATS:
                tables = extract_tables(stored, sections=sections, docx_payload=docx_payload).tables
        except UnsupportedFormat as exc:
            status = "unsupported"
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 - an untrusted file must not crash ingestion
            status = "extraction_failed"
            error = f"{type(exc).__name__}: {exc}"

        return DocumentRecord(
            doc_id=checksum,
            checksum=checksum,
            filename=filename,
            source_path=str(source),
            stored_path=str(stored),
            size_bytes=int(stat.st_size),
            mtime=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(timespec="seconds"),
            format=document_format,
            parser=parser,
            parser_version=parser_version,
            imported_at=imported_at,
            duplicate=duplicate,
            status=status,
            error=error,
            sections=sections,
            tables=tables,
            warnings=warnings,
        )

    def extract(self, path: str | Path) -> list[Section]:
        """Extract without copying; raises instead of returning a failed record.

        Useful for a caller that already owns the file. PDF raises
        :class:`UnsupportedFormat` only when the optional ``pypdf`` extra is
        absent; a readable PDF returns one section per page with text.
        """
        source = _validate_source(path)
        extractor = _extractor_for(source.suffix.lower())
        sections = extractor.extract(source)
        if not sections:
            raise ExtractionError("no extractable text")
        return sections

    def _existing_copy(self, directory: Path, checksum: str) -> Path | None:
        """A byte-identical copy already stored for this checksum, if any."""
        if not directory.is_dir():
            return None
        for candidate in sorted(directory.iterdir()):
            if not candidate.is_file() or candidate.name.startswith(".ingest-"):
                continue
            if _sha256_file(candidate) == checksum:
                return candidate
        return None

    def _check_within_root(self, target: Path) -> None:
        base = os.path.normpath(str(self.sources))
        candidate = os.path.normpath(str(target))
        if candidate != base and not candidate.startswith(base + os.sep):
            raise DocumentIngestionError(f"destination escapes {base}: {candidate}")

    def _preserve(self, source: Path, directory: Path, target: Path, checksum: str) -> None:
        """Copy atomically; the temp name lives in the final directory."""
        directory.mkdir(parents=True, exist_ok=True)
        self._check_within_root(target)
        descriptor, temp_name = tempfile.mkstemp(prefix=".ingest-", dir=directory)
        os.close(descriptor)
        temp = Path(temp_name)
        try:
            shutil.copyfile(source, temp)
            if _sha256_file(temp) != checksum:
                raise DocumentIngestionError(f"copy of {source} is not byte-identical")
            os.replace(temp, target)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
