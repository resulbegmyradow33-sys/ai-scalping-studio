"""Read-only autonomy status API.

The dashboard needs one honest answer to "what is the autonomous system doing
right now?" without becoming a second writer. Every route here is a ``GET``; no
request field selects a path and no handler creates, migrates or opens a job.

**No invented numbers.** Each section reports ``configured: false`` when its
store is absent. A file that exists but cannot be opened, or whose schema lacks
the columns a reader needs, reports ``configured: true, readable: false`` and
omits both items and counts. "The queue is empty" and "the queue could not be
read" are different answers and the API keeps them apart.

**Read-only connections.** SQLite stores are opened with a ``mode=ro`` URI and
``PRAGMA query_only = ON``. SQLite refuses to create a missing file, cannot
write through the connection and takes no write lock, so a ``GET`` never
migrates a schema and never races an orchestrator's ``BEGIN IMMEDIATE``. A
read-only connection still lets SQLite create or update ``-wal``/``-shm``
sidecars for a WAL database when the directory is writable; the database file
itself is never changed.

**Archive compatibility.** The discovery archive is read with plain SQL rather
than :class:`~scalping.discovery.archive.DiscoveryArchive`, because opening that
class runs schema migrations. A v1 archive without ``champions`` answers with an
empty champion list instead of an error or a migration; a champion table that
exists but lacks the required columns is unreadable rather than empty.

Nothing here accesses a locked holdout, exposes a filesystem path, starts a job
or reads a secret. Paths are configuration installed by the app factory.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from fastapi import APIRouter, Query, Request

from scalping.api.schemas import (
    AutonomyChampion,
    AutonomyChampionList,
    AutonomyDataStatus,
    AutonomyDiscoveries,
    AutonomyEventItem,
    AutonomyEventPage,
    AutonomyJobItem,
    AutonomyJobPage,
    AutonomyLastEvent,
    AutonomyOrchestratorStatus,
    AutonomyPaperStatus,
    AutonomyStatus,
)
from scalping.paper.ledger import LedgerError, PaperLedger

__all__ = ["EVENTS_DEFAULT", "EVENTS_MAX", "JOBS_DEFAULT", "JOBS_MAX", "router"]

router = APIRouter(prefix="/api/autonomy", tags=["autonomy"])

#: Recent history is a glance, not an export. The ceiling keeps one hostile
#: ``limit`` from materialising the whole queue.
JOBS_DEFAULT: Final[int] = 50
JOBS_MAX: Final[int] = 500
EVENTS_DEFAULT: Final[int] = 100
EVENTS_MAX: Final[int] = 500

_STATE_PAUSED: Final[str] = "orchestrator.paused"
_STATE_SAFE_MODE: Final[str] = "orchestrator.safe_mode"
_STATE_RESEARCH_DISABLED: Final[str] = "orchestrator.research_disabled"

#: Columns each reader actually touches. A v1 queue has no ``defer_count`` and
#: must not answer as an empty queue merely because the column is missing.
_JOBS_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "id",
        "kind",
        "status",
        "attempts",
        "defer_count",
        "priority",
        "created_at",
        "scheduled_at",
        "started_at",
        "finished_at",
    }
)
_EVENTS_COLUMNS: Final[frozenset[str]] = frozenset({"id", "job_id", "ts", "event"})
_STATE_COLUMNS: Final[frozenset[str]] = frozenset({"key", "value"})
_CHAMPION_COLUMNS: Final[frozenset[str]] = frozenset({"scope", "candidate_id", "promoted_at"})
_ARCHIVE_TABLES: Final[frozenset[str]] = frozenset(
    {"candidates", "trials", "decisions", "champions"}
)


# ------------------------------------------------------------------ connections


@contextmanager
def _read_only(path: Path) -> Iterator[sqlite3.Connection | None]:
    """A ``mode=ro`` connection, or ``None`` when there is no readable file.

    A missing path, a directory where a file was expected, a corrupt header and a
    permission failure all yield ``None``: the endpoint then reports the store as
    unreadable instead of turning a dashboard poll into a 500. The URI is built
    from server configuration only; no request value reaches this function.
    """
    if not path.is_file():
        yield None
        return
    conn: sqlite3.Connection | None = None
    try:
        target = path.resolve()
        uri = f"{target.as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA query_only = ON")
    except (OSError, ValueError, sqlite3.Error):
        if conn is not None:
            conn.close()
        yield None
        return
    try:
        yield conn
    finally:
        conn.close()


def _tables(conn: sqlite3.Connection) -> frozenset[str]:
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    except sqlite3.Error:
        return frozenset()
    return frozenset(str(row[0]) for row in rows)


def _columns(conn: sqlite3.Connection, table: str) -> frozenset[str]:
    """Column names for an internal table name; never a request value."""
    try:
        rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    except sqlite3.Error:
        return frozenset()
    return frozenset(str(row[1]) for row in rows)


def _count(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return 0 if row is None else int(row[0])


# ----------------------------------------------------------------------- paper


def _paper_status(path: Path) -> AutonomyPaperStatus:
    """Session and trade counts from the paper ledger, if it has a schema."""
    configured = path.is_file()
    if not configured:
        return AutonomyPaperStatus(configured=False, readable=False)
    try:
        book = PaperLedger.open_reader(path)
    except (FileNotFoundError, LedgerError, OSError, sqlite3.Error):
        return AutonomyPaperStatus(configured=True, readable=False)
    try:
        conn = book.connection
        conn.execute("PRAGMA query_only = ON")
        if not book.has_schema():
            return AutonomyPaperStatus(configured=True, readable=False)
        return AutonomyPaperStatus(
            configured=True,
            readable=True,
            sessions=_count(conn, "SELECT COUNT(*) FROM sessions"),
            open_positions=_count(conn, "SELECT COUNT(*) FROM positions WHERE status = 'open'"),
            closed_trades=_count(conn, "SELECT COUNT(*) FROM trades"),
        )
    except sqlite3.Error:
        return AutonomyPaperStatus(configured=True, readable=False)
    finally:
        book.close()


# ----------------------------------------------------------------- orchestrator


def _flag(conn: sqlite3.Connection, key: str) -> bool | None:
    """A persisted control flag, or ``None`` when the state table is absent.

    An existing store with no row for ``key`` answers ``False``: that is the
    orchestrator's own default, not an invented one. A payload that is not the
    expected JSON object also reads as off, matching
    :meth:`~scalping.orchestrator.core.Orchestrator._read_flag`.
    """
    row = conn.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
    if row is None:
        return False
    try:
        parsed = json.loads(str(row[0]))
    except ValueError:
        return False
    if not isinstance(parsed, dict):
        return False
    return bool(parsed.get("on", False))


def _last_event(conn: sqlite3.Connection, tables: frozenset[str]) -> AutonomyLastEvent | None:
    if "events" not in tables or not (_EVENTS_COLUMNS <= _columns(conn, "events")):
        return None
    row = conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        return None
    job_id = row["job_id"]
    return AutonomyLastEvent(
        id=int(row["id"]),
        ts=str(row["ts"]),
        event=str(row["event"]),
        job_id=None if job_id is None else int(job_id),
    )


def _orchestrator_status(path: Path) -> AutonomyOrchestratorStatus:
    configured = path.is_file()
    if not configured:
        return AutonomyOrchestratorStatus(configured=False, readable=False)
    with _read_only(path) as conn:
        if conn is None:
            return AutonomyOrchestratorStatus(configured=True, readable=False)
        try:
            tables = _tables(conn)
            if not (_JOBS_COLUMNS <= _columns(conn, "jobs")):
                return AutonomyOrchestratorStatus(configured=True, readable=False)
            counts = {"queued": 0, "running": 0, "failed": 0}
            for row in conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"):
                status = str(row["status"])
                if status in counts:
                    counts[status] = int(row["n"])
            flags: dict[str, bool | None] = {
                "paused": None,
                "safe_mode": None,
                "research_disabled": None,
            }
            if _STATE_COLUMNS <= _columns(conn, "state"):
                flags["paused"] = _flag(conn, _STATE_PAUSED)
                flags["safe_mode"] = _flag(conn, _STATE_SAFE_MODE)
                flags["research_disabled"] = _flag(conn, _STATE_RESEARCH_DISABLED)
            return AutonomyOrchestratorStatus(
                configured=True,
                readable=True,
                queued=counts["queued"],
                running=counts["running"],
                failed=counts["failed"],
                paused=flags["paused"],
                safe_mode=flags["safe_mode"],
                research_disabled=flags["research_disabled"],
                last_event=_last_event(conn, tables),
            )
        except sqlite3.Error:
            return AutonomyOrchestratorStatus(configured=True, readable=False)


# --------------------------------------------------------------------- discovery


def _candidate_metadata(canonical_json: object) -> tuple[str | None, str | None, str | None]:
    """``(mode, policy_id, policy_digest)`` from an archived payload, if present.

    The archive's candidate id is the sha256 of the canonical payload, so it is
    a real digest even when an older payload carries no metadata. ``mode`` falls
    back to ``dsl`` because every promoted candidate in this system is a DSL
    payload; a payload that names its own mode wins.
    """
    payload: dict[str, object] = {}
    if isinstance(canonical_json, str):
        try:
            parsed = json.loads(canonical_json)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            payload = parsed
    raw_mode = payload.get("mode")
    mode = raw_mode if isinstance(raw_mode, str) and raw_mode else "dsl"
    raw_policy = payload.get("policy_id")
    if not isinstance(raw_policy, str) or not raw_policy:
        raw_policy = payload.get("policy")
    policy_id = raw_policy if isinstance(raw_policy, str) and raw_policy else None
    raw_digest = payload.get("policy_digest")
    if not isinstance(raw_digest, str) or not raw_digest:
        raw_digest = payload.get("digest")
    policy_digest = raw_digest if isinstance(raw_digest, str) and raw_digest else None
    return mode, policy_id, policy_digest


def _champions(
    conn: sqlite3.Connection, tables: frozenset[str]
) -> tuple[bool, list[AutonomyChampion]]:
    """``(readable, items)`` for the champion table.

    An older archive with no ``champions`` table is readable and empty: nothing
    was promoted under that schema. A table that exists but lacks the required
    columns is unreadable, because an empty answer would be a lie.
    """
    if "champions" not in tables:
        return True, []
    if not (_CHAMPION_COLUMNS <= _columns(conn, "champions")):
        return False, []
    candidates_joinable = "candidates" in tables and {"candidate_id", "canonical_json"} <= _columns(
        conn, "candidates"
    )
    if candidates_joinable:
        rows = conn.execute(
            """
            SELECT c.scope, c.candidate_id, c.promoted_at, ca.canonical_json
            FROM champions AS c
            LEFT JOIN candidates AS ca ON ca.candidate_id = c.candidate_id
            ORDER BY c.scope
            """
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT scope, candidate_id, promoted_at, NULL AS canonical_json "
            "FROM champions ORDER BY scope"
        ).fetchall()
    items: list[AutonomyChampion] = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        mode, policy_id, policy_digest = _candidate_metadata(row["canonical_json"])
        items.append(
            AutonomyChampion(
                scope=str(row["scope"]),
                candidate_id=candidate_id,
                mode=mode,
                promoted_at=None if row["promoted_at"] is None else str(row["promoted_at"]),
                policy_id=policy_id,
                policy_digest=policy_digest or candidate_id,
            )
        )
    return True, items


def _archive_status(
    path: Path,
) -> tuple[bool, bool, bool, list[AutonomyChampion], AutonomyDiscoveries]:
    """Read both archive sections over one connection.

    Returns ``(configured, readable, champions_readable, champions, discoveries)``.
    Opening this file twice would double the I/O and could observe two different
    snapshots while a writer commits between the calls.
    """
    configured = path.is_file()
    empty = AutonomyDiscoveries(configured=configured, readable=False)
    if not configured:
        return False, False, False, [], empty
    with _read_only(path) as conn:
        if conn is None:
            return True, False, False, [], empty
        try:
            tables = _tables(conn)
            if not (tables & _ARCHIVE_TABLES):
                return True, False, False, [], empty
            champions_readable, champions = _champions(conn, tables)
            runs = _count(conn, "SELECT COUNT(*) FROM trials") if "trials" in tables else None
            decisions = (
                _count(conn, "SELECT COUNT(*) FROM decisions") if "decisions" in tables else None
            )
            discoveries = AutonomyDiscoveries(
                configured=True,
                readable=True,
                runs=runs,
                decisions=decisions,
            )
            return True, True, champions_readable, champions, discoveries
        except sqlite3.Error:
            return True, False, False, [], AutonomyDiscoveries(configured=True, readable=False)


# -------------------------------------------------------------------------- data


def _data_status(root: Path) -> AutonomyDataStatus:
    """Symbols with a ``<symbol>/<timeframe>/<year>/<month>.parquet`` partition.

    The glob is depth-bounded instead of recursive: a month partition is exactly
    three levels below the symbol, so ``rglob`` would walk unrelated files.
    """
    configured = root.is_dir()
    if not configured:
        return AutonomyDataStatus(configured=False, readable=False)
    symbols: list[str] = []
    try:
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name in {"imports", "raw"}:
                continue
            if next(entry.glob("*/*/*.parquet"), None) is not None:
                symbols.append(entry.name)
    except OSError:
        return AutonomyDataStatus(configured=True, readable=False)
    return AutonomyDataStatus(configured=True, readable=True, symbols=symbols)


# ------------------------------------------------------------------------ routes


@router.get("/status", response_model=AutonomyStatus, response_model_exclude_none=True)
def autonomy_status(request: Request) -> AutonomyStatus:
    """Aggregate real state; absent or unreadable stores carry no invented values."""
    _, _, champions_readable, champions, discoveries = _archive_status(
        Path(request.app.state.discovery_archive_path)
    )
    return AutonomyStatus(
        paper=_paper_status(Path(request.app.state.paper_ledger_path)),
        orchestrator=_orchestrator_status(Path(request.app.state.orchestrator_db_path)),
        champions=champions,
        champions_readable=champions_readable,
        discoveries=discoveries,
        data=_data_status(Path(request.app.state.data_root)),
    )


@router.get(
    "/champions",
    response_model=AutonomyChampionList,
    response_model_exclude_none=True,
)
def autonomy_champions(request: Request) -> AutonomyChampionList:
    """Champions per scope, read from the archive without migrating it."""
    configured, readable, champions_readable, items, _ = _archive_status(
        Path(request.app.state.discovery_archive_path)
    )
    available = readable and champions_readable
    return AutonomyChampionList(
        configured=configured,
        readable=available,
        items=items if available else None,
    )


@router.get("/jobs", response_model=AutonomyJobPage, response_model_exclude_none=True)
def autonomy_jobs(
    request: Request,
    limit: int = Query(default=JOBS_DEFAULT, ge=1, le=JOBS_MAX),
) -> AutonomyJobPage:
    """Newest jobs first; bounded so a dashboard never pages the whole queue."""
    path = Path(request.app.state.orchestrator_db_path)
    configured = path.is_file()
    with _read_only(path) as conn:
        if conn is None:
            return AutonomyJobPage(configured=configured, readable=False, limit=limit)
        try:
            if not (_JOBS_COLUMNS <= _columns(conn, "jobs")):
                return AutonomyJobPage(configured=True, readable=False, limit=limit)
            rows = conn.execute(
                """
                SELECT id, kind, status, attempts, defer_count, priority,
                       created_at, scheduled_at, started_at, finished_at
                FROM jobs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        except sqlite3.Error:
            return AutonomyJobPage(configured=configured, readable=False, limit=limit)
    items = [
        AutonomyJobItem(
            id=int(row["id"]),
            kind=str(row["kind"]),
            status=str(row["status"]),
            attempts=int(row["attempts"]),
            defer_count=int(row["defer_count"]),
            priority=int(row["priority"]),
            created_at=str(row["created_at"]),
            scheduled_at=str(row["scheduled_at"]),
            started_at=None if row["started_at"] is None else str(row["started_at"]),
            finished_at=None if row["finished_at"] is None else str(row["finished_at"]),
        )
        for row in rows
    ]
    return AutonomyJobPage(configured=configured, readable=True, items=items, limit=limit)


@router.get("/events", response_model=AutonomyEventPage, response_model_exclude_none=True)
def autonomy_events(
    request: Request,
    limit: int = Query(default=EVENTS_DEFAULT, ge=1, le=EVENTS_MAX),
) -> AutonomyEventPage:
    """Newest audit events first; the free-form ``detail`` column stays private."""
    path = Path(request.app.state.orchestrator_db_path)
    configured = path.is_file()
    with _read_only(path) as conn:
        if conn is None:
            return AutonomyEventPage(configured=configured, readable=False, limit=limit)
        try:
            if not (_EVENTS_COLUMNS <= _columns(conn, "events")):
                return AutonomyEventPage(configured=True, readable=False, limit=limit)
            rows = conn.execute(
                """
                SELECT id, job_id, ts, event
                FROM events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        except sqlite3.Error:
            return AutonomyEventPage(configured=configured, readable=False, limit=limit)
    items = [
        AutonomyEventItem(
            id=int(row["id"]),
            job_id=None if row["job_id"] is None else int(row["job_id"]),
            ts=str(row["ts"]),
            event=str(row["event"]),
        )
        for row in rows
    ]
    return AutonomyEventPage(configured=configured, readable=True, items=items, limit=limit)
