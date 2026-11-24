"""Local SQLite index for extracted document text.

The index follows the registry conventions: one SQLite file, WAL, foreign keys
on, a ``schema_version`` table and additive migrations. It stores documents,
sections and chunks; search uses SQLite FTS5 when the interpreter provides it
and a documented ``LIKE`` fallback otherwise. ``search_backend`` reports which
one is active, so a result is never attributed to a search engine it did not
use.

Sections carry character offsets into the document text defined as
``"\\n\\n".join(section.text for section in sections)``. Chunk offsets are
relative to their section. Both invariants are kept by construction, so a
citation can be sliced back out without guessing.

The index is a library of source material. It never selects a strategy and has
no connection to the paper-trading or order code.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from scalping.knowledge.ingest import DocumentRecord

__all__ = [
    "SCHEMA_VERSION",
    "ChunkRecord",
    "DocumentSummary",
    "KnowledgeIndex",
    "KnowledgeIndexError",
    "SearchHit",
    "SectionRecord",
    "split_chunks",
]

SCHEMA_VERSION: int = 1

#: Chunk size in characters. Small enough for a local reader, large enough to
#: keep a paragraph's context together. No overlap: offsets stay exact.
CHUNK_CHARS = 800

SEARCH_BACKENDS: frozenset[str] = frozenset({"fts5", "like"})

_WAL_ATTEMPTS = 5
_WAL_BACKOFF_SECONDS = 0.05
_LIKE_ESCAPE = "\\"

_SCHEMA_VERSION_DDL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
)
"""

_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            checksum TEXT NOT NULL UNIQUE,
            filename TEXT NOT NULL,
            source_path TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
            mtime TEXT NOT NULL,
            format TEXT NOT NULL,
            parser TEXT NOT NULL,
            parser_version TEXT NOT NULL,
            imported_at TEXT NOT NULL,
            duplicate INTEGER NOT NULL DEFAULT 0 CHECK (duplicate IN (0, 1)),
            status TEXT NOT NULL CHECK (status IN ('indexed', 'extraction_failed', 'unsupported')),
            error TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS documents_imported_idx ON documents(imported_at, doc_id)",
        """
        CREATE TABLE IF NOT EXISTS sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            heading TEXT NOT NULL DEFAULT '',
            label TEXT NOT NULL DEFAULT '',
            text TEXT NOT NULL,
            char_start INTEGER NOT NULL CHECK (char_start >= 0),
            char_end INTEGER NOT NULL CHECK (char_end >= char_start),
            UNIQUE (doc_id, ordinal)
        )
        """,
        "CREATE INDEX IF NOT EXISTS sections_doc_idx ON sections(doc_id, ordinal)",
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            section_id INTEGER NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            text TEXT NOT NULL,
            token_count INTEGER NOT NULL CHECK (token_count >= 0),
            char_start INTEGER NOT NULL CHECK (char_start >= 0),
            char_end INTEGER NOT NULL CHECK (char_end >= char_start),
            UNIQUE (section_id, ordinal)
        )
        """,
        "CREATE INDEX IF NOT EXISTS chunks_section_idx ON chunks(section_id, ordinal)",
        """
        CREATE TABLE IF NOT EXISTS hypotheses (
            hypothesis_id TEXT PRIMARY KEY,
            statement TEXT NOT NULL,
            observed TEXT NOT NULL,
            proposed_condition TEXT NOT NULL,
            expected_effect TEXT NOT NULL,
            market_scope TEXT NOT NULL,
            timeframe_scope TEXT NOT NULL,
            assumptions_json TEXT NOT NULL DEFAULT '[]',
            falsification_test TEXT NOT NULL,
            status TEXT NOT NULL
                CHECK (status IN ('proposed', 'testing', 'supported', 'rejected')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS hypotheses_status_idx ON hypotheses(status, created_at)",
        """
        CREATE TABLE IF NOT EXISTS hypothesis_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hypothesis_id TEXT NOT NULL
                REFERENCES hypotheses(hypothesis_id) ON DELETE CASCADE,
            position INTEGER NOT NULL CHECK (position >= 0),
            doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE RESTRICT,
            section_ordinal INTEGER NOT NULL CHECK (section_ordinal >= 0),
            quote TEXT NOT NULL,
            UNIQUE (hypothesis_id, position)
        )
        """,
        (
            "CREATE INDEX IF NOT EXISTS hypothesis_sources_hyp_idx"
            " ON hypothesis_sources(hypothesis_id, position)"
        ),
        """
        CREATE TABLE IF NOT EXISTS hypothesis_experiments (
            hypothesis_id TEXT NOT NULL
                REFERENCES hypotheses(hypothesis_id) ON DELETE CASCADE,
            experiment_ref TEXT NOT NULL,
            position INTEGER NOT NULL CHECK (position >= 0),
            linked_at TEXT NOT NULL,
            PRIMARY KEY (hypothesis_id, experiment_ref)
        )
        """,
        (
            "CREATE INDEX IF NOT EXISTS hypothesis_experiments_hyp_idx"
            " ON hypothesis_experiments(hypothesis_id, position)"
        ),
    )
}

_FTS_SETUP: tuple[str, ...] = (
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        text,
        heading,
        tokenize = 'unicode61'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks
    BEGIN
        INSERT INTO chunks_fts(rowid, text, heading)
        VALUES (
            new.id,
            new.text,
            COALESCE((SELECT heading FROM sections WHERE id = new.section_id), '')
        );
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks
    BEGIN
        DELETE FROM chunks_fts WHERE rowid = old.id;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks
    BEGIN
        DELETE FROM chunks_fts WHERE rowid = old.id;
        INSERT INTO chunks_fts(rowid, text, heading)
        VALUES (
            new.id,
            new.text,
            COALESCE((SELECT heading FROM sections WHERE id = new.section_id), '')
        );
    END
    """,
)

_FTS_BACKFILL = """
INSERT INTO chunks_fts(rowid, text, heading)
SELECT c.id, c.text, COALESCE(s.heading, '')
FROM chunks AS c
JOIN sections AS s ON s.id = c.section_id
WHERE c.id NOT IN (SELECT rowid FROM chunks_fts)
"""


class KnowledgeIndexError(Exception):
    """The index rejected an operation; the database is unchanged."""


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    """The stored metadata of one document without its sections."""

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
    error: str | None

    def to_dict(self) -> dict[str, Any]:
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
        }


@dataclass(frozen=True, slots=True)
class SectionRecord:
    """One indexed section with document-level character offsets."""

    id: int
    doc_id: str
    ordinal: int
    heading: str
    label: str
    text: str
    char_start: int
    char_end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "ordinal": self.ordinal,
            "heading": self.heading,
            "label": self.label,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """One searchable chunk; offsets are relative to its section text."""

    id: int
    section_id: int
    ordinal: int
    text: str
    token_count: int
    char_start: int
    char_end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "section_id": self.section_id,
            "ordinal": self.ordinal,
            "text": self.text,
            "token_count": self.token_count,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A chunk that matched a query, with enough context to cite it."""

    doc_id: str
    filename: str
    section_id: int
    section_ordinal: int
    chunk_id: int
    chunk_ordinal: int
    heading: str
    text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "section_id": self.section_id,
            "section_ordinal": self.section_ordinal,
            "chunk_id": self.chunk_id,
            "chunk_ordinal": self.chunk_ordinal,
            "heading": self.heading,
            "text": self.text,
            "score": self.score,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a str, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    return text


def _require_non_negative(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{field} must be >= 0")
    return value


def split_chunks(text: str, *, max_chars: int = CHUNK_CHARS) -> list[tuple[str, int, int]]:
    """Split ``text`` into exact slices of at most ``max_chars`` characters.

    Breaks happen between words, so ``text[start:end]`` is the chunk and no
    character is invented or dropped. A single word longer than ``max_chars``
    is kept whole: cutting inside it would make the offsets lie.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    words = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
    if not words:
        return []
    chunks: list[tuple[str, int, int]] = []
    start, end = words[0]
    for word_start, word_end in words[1:]:
        if word_end - start > max_chars:
            chunks.append((text[start:end], start, end))
            start, end = word_start, word_end
        else:
            end = word_end
    chunks.append((text[start:end], start, end))
    return chunks


def _enable_wal(conn: sqlite3.Connection) -> None:
    row = conn.execute("PRAGMA journal_mode").fetchone()
    if row is not None and str(row[0]).lower() == "wal":
        return
    for attempt in range(1, _WAL_ATTEMPTS + 1):
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            return
        except sqlite3.OperationalError:
            if attempt == _WAL_ATTEMPTS:
                raise
            time.sleep(_WAL_BACKOFF_SECONDS * attempt)


def _fts_query(query: str) -> str:
    """Quote every term so user input cannot inject FTS5 operators."""
    terms = [term for term in re.split(r"[^\w]+", query, flags=re.UNICODE) if term]
    if not terms:
        raise ValueError(f"query has no searchable terms: {query!r}")
    return " ".join('"' + term.replace('"', '""') + '"' for term in terms)


def _like_pattern(query: str) -> str:
    escaped = (
        query.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
        .replace("%", _LIKE_ESCAPE + "%")
        .replace("_", _LIKE_ESCAPE + "_")
    )
    return f"%{escaped}%"


def _document_from_row(row: sqlite3.Row) -> DocumentSummary:
    return DocumentSummary(
        doc_id=str(row["doc_id"]),
        checksum=str(row["checksum"]),
        filename=str(row["filename"]),
        source_path=str(row["source_path"]),
        stored_path=str(row["stored_path"]),
        size_bytes=int(row["size_bytes"]),
        mtime=str(row["mtime"]),
        format=str(row["format"]),
        parser=str(row["parser"]),
        parser_version=str(row["parser_version"]),
        imported_at=str(row["imported_at"]),
        duplicate=bool(row["duplicate"]),
        status=str(row["status"]),
        error=None if row["error"] is None else str(row["error"]),
    )


def _section_from_row(row: sqlite3.Row) -> SectionRecord:
    return SectionRecord(
        id=int(row["id"]),
        doc_id=str(row["doc_id"]),
        ordinal=int(row["ordinal"]),
        heading=str(row["heading"]),
        label=str(row["label"]),
        text=str(row["text"]),
        char_start=int(row["char_start"]),
        char_end=int(row["char_end"]),
    )


def _chunk_from_row(row: sqlite3.Row) -> ChunkRecord:
    return ChunkRecord(
        id=int(row["id"]),
        section_id=int(row["section_id"]),
        ordinal=int(row["ordinal"]),
        text=str(row["text"]),
        token_count=int(row["token_count"]),
        char_start=int(row["char_start"]),
        char_end=int(row["char_end"]),
    )


class KnowledgeIndex:
    """One local SQLite file holding the document index and hypotheses.

    Not thread-safe: one connection belongs to the thread that opened it, which
    matches the Mac mini's one-heavy-job rule.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            self.path,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        try:
            _enable_wal(self._conn)
            self.init_schema()
            self._search_backend = self._setup_search(self._conn)
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------ lifecycle

    @property
    def connection(self) -> sqlite3.Connection:
        """The open connection; read-only use is expected outside this class."""
        return self._connection()

    @property
    def search_backend(self) -> str:
        """``"fts5"`` or ``"like"``, the backend the next search will use."""
        return self._search_backend

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise KnowledgeIndexError("knowledge index is closed")
        return self._conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """``BEGIN IMMEDIATE`` around a write, rolled back on any exception."""
        conn = self._connection()
        if conn.in_transaction:
            raise KnowledgeIndexError("nested knowledge index transaction")
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            raise KnowledgeIndexError(f"could not acquire the index write lock: {exc}") from exc
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()

    def _read_schema_version(self) -> int:
        try:
            row = self._connection().execute("SELECT MAX(version) FROM schema_version").fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc):
                raise
            return 0
        return 0 if row is None or row[0] is None else int(row[0])

    def init_schema(self) -> int:
        """Create tables or apply pending migrations; returns the version."""
        current = self._read_schema_version()
        if current > SCHEMA_VERSION:
            raise KnowledgeIndexError(
                f"database schema version {current} is newer than supported {SCHEMA_VERSION}"
            )
        if current == SCHEMA_VERSION:
            return current
        with self.transaction() as tx:
            tx.execute(_SCHEMA_VERSION_DDL)
            row = tx.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current = 0 if row is None or row[0] is None else int(row[0])
            if current > SCHEMA_VERSION:
                raise KnowledgeIndexError(
                    f"database schema version {current} is newer than supported {SCHEMA_VERSION}"
                )
            for version in sorted(_MIGRATIONS):
                if version <= current:
                    continue
                for statement in _MIGRATIONS[version]:
                    tx.execute(statement)
                tx.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (version, _now()),
                )
        return self.schema_version

    @property
    def schema_version(self) -> int:
        return self._read_schema_version()

    def close(self) -> None:
        conn = self._conn
        if conn is not None:
            self._conn = None
            conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------- search

    def _setup_search(self, conn: sqlite3.Connection) -> str:
        """Create the FTS5 table and triggers, or report the LIKE fallback.

        The virtual table is outside the versioned migrations because a Python
        build without FTS5 must still open the same database. It is created on
        demand, backfilled from existing chunks, and never claimed unless it was
        actually created.
        """
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks_fts'"
        ).fetchone()
        if table is None:
            try:
                conn.execute(_FTS_SETUP[0])
            except sqlite3.OperationalError:
                return "like"
        try:
            for statement in _FTS_SETUP[1:]:
                conn.execute(statement)
            conn.execute(_FTS_BACKFILL)
        except sqlite3.OperationalError as exc:
            if table is None:
                return "like"
            raise KnowledgeIndexError(f"FTS5 is present but unusable: {exc}") from exc
        return "fts5"

    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        """Search chunk text and headings; newest/first chunks win score ties.

        FTS5 ranks with ``bm25`` (lower is better, exposed as ``score``).
        The fallback is a case-insensitive ``LIKE`` over chunk text and section
        heading with ``score=0.0``; it is a substring match, not a token match.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a str, got {type(query).__name__}")
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        limit = _require_non_negative(limit, "limit")
        if limit < 1:
            raise ValueError("limit must be >= 1")
        conn = self._connection()
        if self._search_backend == "fts5":
            try:
                rows = conn.execute(
                    """
                    SELECT c.id AS chunk_id, c.ordinal AS chunk_ordinal, c.text AS text,
                           s.id AS section_id, s.ordinal AS section_ordinal,
                           s.heading AS heading, d.doc_id AS doc_id,
                           d.filename AS filename, bm25(chunks_fts) AS score
                    FROM chunks_fts
                    JOIN chunks AS c ON c.id = chunks_fts.rowid
                    JOIN sections AS s ON s.id = c.section_id
                    JOIN documents AS d ON d.doc_id = s.doc_id
                    WHERE chunks_fts MATCH ?
                    ORDER BY score, d.doc_id, s.ordinal, c.ordinal
                    LIMIT ?
                    """,
                    (_fts_query(query), limit),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                raise KnowledgeIndexError(f"FTS5 search failed: {exc}") from exc
        else:
            pattern = _like_pattern(query)
            rows = conn.execute(
                """
                SELECT c.id AS chunk_id, c.ordinal AS chunk_ordinal, c.text AS text,
                       s.id AS section_id, s.ordinal AS section_ordinal,
                       s.heading AS heading, d.doc_id AS doc_id,
                       d.filename AS filename, 0.0 AS score
                FROM chunks AS c
                JOIN sections AS s ON s.id = c.section_id
                JOIN documents AS d ON d.doc_id = s.doc_id
                WHERE c.text LIKE ? ESCAPE '\\' OR s.heading LIKE ? ESCAPE '\\'
                ORDER BY d.doc_id, s.ordinal, c.ordinal
                LIMIT ?
                """,
                (pattern, pattern, limit),
            ).fetchall()
        return [
            SearchHit(
                doc_id=str(row["doc_id"]),
                filename=str(row["filename"]),
                section_id=int(row["section_id"]),
                section_ordinal=int(row["section_ordinal"]),
                chunk_id=int(row["chunk_id"]),
                chunk_ordinal=int(row["chunk_ordinal"]),
                heading=str(row["heading"]),
                text=str(row["text"]),
                score=float(row["score"]),
            )
            for row in rows
        ]

    # ----------------------------------------------------------------- documents

    def add_document(self, record: DocumentRecord) -> DocumentSummary:
        """Store a record returned by the ingestor; a duplicate checksum is a no-op.

        ``doc_id`` must equal the checksum: the SHA-256 is the identity, so a
        caller cannot register two ids for the same bytes. Extraction failures
        are stored too, with their reason and no sections.
        """
        if not isinstance(record, DocumentRecord):
            raise TypeError("record must be a DocumentRecord")
        if record.doc_id != record.checksum:
            raise ValueError("doc_id must equal the checksum")
        existing = self.document(record.doc_id)
        if existing is not None:
            return existing
        params = (
            record.doc_id,
            record.checksum,
            record.filename,
            record.source_path,
            record.stored_path,
            record.size_bytes,
            record.mtime,
            record.format,
            record.parser,
            record.parser_version,
            record.imported_at,
            int(record.duplicate),
            record.status,
            record.error,
        )
        try:
            with self.transaction() as tx:
                tx.execute(
                    """
                    INSERT INTO documents (
                        doc_id, checksum, filename, source_path, stored_path,
                        size_bytes, mtime, format, parser, parser_version,
                        imported_at, duplicate, status, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    params,
                )
                if record.status == "indexed":
                    self._insert_sections(tx, record)
        except sqlite3.IntegrityError as exc:
            existing = self.document(record.doc_id)
            if existing is not None:
                return existing
            raise KnowledgeIndexError(f"document could not be stored: {exc}") from exc
        stored = self.document(record.doc_id)
        if stored is None:
            raise KnowledgeIndexError("document insert committed but the row is missing")
        return stored

    def _insert_sections(self, tx: sqlite3.Connection, record: DocumentRecord) -> None:
        text_start = 0
        for ordinal, section in enumerate(record.sections):
            text = section.text
            char_start = text_start
            char_end = char_start + len(text)
            cursor = tx.execute(
                """
                INSERT INTO sections (
                    doc_id, ordinal, heading, label, text, char_start, char_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.doc_id,
                    ordinal,
                    section.heading,
                    section.label,
                    text,
                    char_start,
                    char_end,
                ),
            )
            section_id = cursor.lastrowid
            if section_id is None:
                raise KnowledgeIndexError("section insert returned no id")
            for chunk_ordinal, (chunk_text, chunk_start, chunk_end) in enumerate(
                split_chunks(text)
            ):
                tx.execute(
                    """
                    INSERT INTO chunks (
                        section_id, ordinal, text, token_count, char_start, char_end
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(section_id),
                        chunk_ordinal,
                        chunk_text,
                        len(chunk_text.split()),
                        chunk_start,
                        chunk_end,
                    ),
                )
            text_start = char_end + 2  # the "\n\n" separator

    def document(self, doc_id: str) -> DocumentSummary | None:
        """Stored metadata for ``doc_id``, or ``None`` when unknown."""
        doc_id = _require_text(doc_id, "doc_id")
        row = (
            self._connection()
            .execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
            .fetchone()
        )
        return None if row is None else _document_from_row(row)

    def document_by_checksum(self, checksum: str) -> DocumentSummary | None:
        checksum = _require_text(checksum, "checksum")
        row = (
            self._connection()
            .execute("SELECT * FROM documents WHERE checksum = ?", (checksum,))
            .fetchone()
        )
        return None if row is None else _document_from_row(row)

    def list_documents(self, *, status: str | None = None) -> list[DocumentSummary]:
        """Documents oldest import first, with ``doc_id`` as the stable tie-break."""
        conn = self._connection()
        if status is None:
            rows = conn.execute("SELECT * FROM documents ORDER BY imported_at, doc_id").fetchall()
        else:
            status = _require_text(status, "status")
            rows = conn.execute(
                "SELECT * FROM documents WHERE status = ? ORDER BY imported_at, doc_id",
                (status,),
            ).fetchall()
        return [_document_from_row(row) for row in rows]

    def sections(self, doc_id: str) -> list[SectionRecord]:
        """Sections in document order; offsets cover the joined document text."""
        doc_id = _require_text(doc_id, "doc_id")
        rows = (
            self._connection()
            .execute("SELECT * FROM sections WHERE doc_id = ? ORDER BY ordinal", (doc_id,))
            .fetchall()
        )
        return [_section_from_row(row) for row in rows]

    def section(self, doc_id: str, ordinal: int) -> SectionRecord | None:
        """One section by document and ordinal, or ``None`` when absent.

        This is the read path citation checks use: a source ref names
        ``(doc_id, ordinal)`` and needs the section text to compare its quote
        against.
        """
        doc_id = _require_text(doc_id, "doc_id")
        ordinal = _require_non_negative(ordinal, "ordinal")
        row = (
            self._connection()
            .execute(
                "SELECT * FROM sections WHERE doc_id = ? AND ordinal = ?",
                (doc_id, ordinal),
            )
            .fetchone()
        )
        return None if row is None else _section_from_row(row)

    def chunks(self, section_id: int) -> list[ChunkRecord]:
        """Chunks of one section in order."""
        section_id = _require_non_negative(section_id, "section_id")
        rows = (
            self._connection()
            .execute("SELECT * FROM chunks WHERE section_id = ? ORDER BY ordinal", (section_id,))
            .fetchall()
        )
        return [_chunk_from_row(row) for row in rows]

    def document_text(self, doc_id: str) -> str:
        """The canonical text the section offsets refer to."""
        return "\n\n".join(section.text for section in self.sections(doc_id))

    # --------------------------------------------------------------- hypotheses

    def hypothesis_exists(self, hypothesis_id: str) -> bool:
        """Whether a hypothesis row exists; a read-only existence check."""
        hypothesis_id = _require_text(hypothesis_id, "hypothesis_id")
        row = (
            self._connection()
            .execute("SELECT 1 FROM hypotheses WHERE hypothesis_id = ?", (hypothesis_id,))
            .fetchone()
        )
        return row is not None

    def create_hypothesis(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to :func:`scalping.knowledge.hypothesis.create_hypothesis`."""
        from scalping.knowledge.hypothesis import create_hypothesis

        return create_hypothesis(self, *args, **kwargs)

    def get_hypothesis(self, hypothesis_id: str) -> Any:
        """Delegate to :func:`scalping.knowledge.hypothesis.get_hypothesis`."""
        from scalping.knowledge.hypothesis import get_hypothesis

        return get_hypothesis(self, hypothesis_id)

    def set_status(self, hypothesis_id: str, status: str) -> Any:
        """Delegate to :func:`scalping.knowledge.hypothesis.set_status`."""
        from scalping.knowledge.hypothesis import set_status

        return set_status(self, hypothesis_id, status)

    def link_experiment(self, hypothesis_id: str, experiment_ref: str) -> Any:
        """Delegate to :func:`scalping.knowledge.hypothesis.link_experiment`."""
        from scalping.knowledge.hypothesis import link_experiment

        return link_experiment(self, hypothesis_id, experiment_ref)

    def list_hypotheses(self, *, status: str | None = None) -> list[Any]:
        """Delegate to :func:`scalping.knowledge.hypothesis.list_hypotheses`."""
        from scalping.knowledge.hypothesis import list_hypotheses

        return list_hypotheses(self, status=status)
