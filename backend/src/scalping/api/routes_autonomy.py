"""Read-only autonomy status API.

The dashboard needs one honest answer to "what is the autonomous system doing
right now?" without becoming a second writer. Every route here is a ``GET``; no
request field selects a path and no handler creates, migrates or opens a job.

**No invented numbers.** Each section reports ``configured: false`` when its
store is absent and omits the counts behind it. A missing paper ledger is not
"zero sessions"; it is a store that was never created. A store that exists but
has no schema yet (a writer is still creating it) also reports no counts.

**Read-only connections.** SQLite stores are opened with a ``mode=ro`` URI and
``PRAGMA query_only = ON``. SQLite refuses to create a missing file, cannot
write through the connection and takes no write lock, so a ``GET`` never
migrates a schema and never races an orchestrator's ``BEGIN IMMEDIATE``.

**Archive compatibility.** The discovery archive is read with plain SQL rather
than :class:`~scalping.discovery.archive.DiscoveryArchive`, because opening that
class runs schema migrations. A v1 archive without ``champions`` answers with an
empty champion list instead of an error, and a candidate payload is only
inspected for optional policy metadata when it parses as a JSON object.

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


# ------------------------------------------------------------------ connections


@contextmanager
def _read_only(path: Path) -> Iterator[sqlite3.Connection | None]:
    """A ``mode=ro`` connection, or ``None`` when there is no readable file.

    A missing path, a directory where a file was expected, a corrupt header and a
    permission failure all yield ``None``: the endpoint then reports the store as
    unconfigured instead of turning a dashboard poll into a 500. The URI is built
    from server configuration only; no request value reaches this function.
    """
    if not path.is_file():
        yield None
        return
    try:
        conn = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro",
            uri=True,
            isolation_level=None,
        )
    except sqlite3.Error:
        yield None
        return
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA query_only = ON")
        yield conn
    finally:
        conn.close()


def _tables(conn: sqlite3.Connection) -> frozenset[str]:
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    except sqlite3.Error:
        return frozenset()
    return frozenset(str(row[0]) for row in rows)


def _count(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return 0 if row is None else int(row[0])


def _exists(conn: sqlite3.Connection, table: str) -> bool:
    return table in _tables(conn)


# ----------------------------------------------------------------------- paper


def _paper_status(path: Path) -> AutonomyPaperStatus:
    """Session and trade counts from the paper ledger, if it has a schema."""
    if not path.is_file():
        return AutonomyPaperStatus(configured=False)
    try:
        book = PaperLedger.open_reader(path)
    except (FileNotFoundError, LedgerError, OSError, sqlite3.Error):
        return AutonomyPaperStatus(configured=True)
    try:
        if not book.has_schema():
            return AutonomyPaperStatus(configured=True)
        conn = book.connection
        return AutonomyPaperStatus(
            configured=True,
            sessions=_count(conn, "SELECT COUNT(*) FROM sessions"),
            open_positions=_count(conn, "SELECT COUNT(*) FROM positions WHERE status = 'open'"),
            closed_trades=_count(conn, "SELECT COUNT(*) FROM trades"),
        )
    except sqlite3.Error:
        return AutonomyPaperStatus(configured=True)
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
    if "events" not in tables:
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
    if not path.is_file():
        return AutonomyOrchestratorStatus(configured=False)
    with _read_only(path) as conn:
        if conn is None:
            return AutonomyOrchestratorStatus(configured=True)
        try:
            tables = _tables(conn)
            if "jobs" not in tables:
                return AutonomyOrchestratorStatus(configured=True)
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
            if "state" in tables:
                flags["paused"] = _flag(conn, _STATE_PAUSED)
                flags["safe_mode"] = _flag(conn, _STATE_SAFE_MODE)
                flags["research_disabled"] = _flag(conn, _STATE_RESEARCH_DISABLED)
            return AutonomyOrchestratorStatus(
                configured=True,
                queued=counts["queued"],
                running=counts["running"],
                failed=counts["failed"],
                paused=flags["paused"],
                safe_mode=flags["safe_mode"],
                research_disabled=flags["research_disabled"],
                last_event=_last_event(conn, tables),
            )
        except sqlite3.Error:
            return AutonomyOrchestratorStatus(configured=True)


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


def _champions(path: Path) -> tuple[bool, list[AutonomyChampion]]:
    """``(configured, items)``; an older archive with no champion table is empty."""
    if not path.is_file():
        return False, []
    with _read_only(path) as conn:
        if conn is None:
            return True, []
        try:
            tables = _tables(conn)
            if "champions" not in tables:
                return True, []
            if "candidates" in tables:
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
        except sqlite3.Error:
            return True, []
    items: list[AutonomyChampion] = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        mode, policy_id, policy_digest = _candidate_metadata(row["canonical_json"])
        items.append(
            AutonomyChampion(
                scope=str(row["scope"]),
                candidate_id=candidate_id,
                mode=mode,
                promoted_at=str(row["promoted_at"]),
                policy_id=policy_id,
                policy_digest=policy_digest or candidate_id,
            )
        )
    return True, items


def _discoveries(path: Path) -> AutonomyDiscoveries:
    if not path.is_file():
        return AutonomyDiscoveries(configured=False)
    with _read_only(path) as conn:
        if conn is None:
            return AutonomyDiscoveries(configured=True)
        try:
            tables = _tables(conn)
            runs = _count(conn, "SELECT COUNT(*) FROM trials") if "trials" in tables else None
            decisions = (
                _count(conn, "SELECT COUNT(*) FROM decisions") if "decisions" in tables else None
            )
            return AutonomyDiscoveries(
                configured=True,
                runs=runs,
                decisions=decisions,
            )
        except sqlite3.Error:
            return AutonomyDiscoveries(configured=True)


# -------------------------------------------------------------------------- data


def _data_status(root: Path) -> AutonomyDataStatus:
    """Symbols that have at least one timeframe partition on disk."""
    if not root.is_dir():
        return AutonomyDataStatus(configured=False)
    symbols: list[str] = []
    try:
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name in {"imports", "raw"}:
                continue
            if next(entry.rglob("*.parquet"), None) is not None:
                symbols.append(entry.name)
    except OSError:
        return AutonomyDataStatus(configured=True)
    return AutonomyDataStatus(configured=True, symbols=symbols)


# ------------------------------------------------------------------------ routes


@router.get("/status", response_model=AutonomyStatus, response_model_exclude_none=True)
def autonomy_status(request: Request) -> AutonomyStatus:
    """Aggregate real state; absent stores carry no counts."""
    _, champions = _champions(Path(request.app.state.discovery_archive_path))
    return AutonomyStatus(
        paper=_paper_status(Path(request.app.state.paper_ledger_path)),
        orchestrator=_orchestrator_status(Path(request.app.state.orchestrator_db_path)),
        champions=champions,
        discoveries=_discoveries(Path(request.app.state.discovery_archive_path)),
        data=_data_status(Path(request.app.state.data_root)),
    )


@router.get("/champions", response_model=AutonomyChampionList)
def autonomy_champions(request: Request) -> AutonomyChampionList:
    """Champions per scope, read from the archive without migrating it."""
    configured, items = _champions(Path(request.app.state.discovery_archive_path))
    return AutonomyChampionList(configured=configured, items=items)


@router.get("/jobs", response_model=AutonomyJobPage)
def autonomy_jobs(
    request: Request,
    limit: int = Query(default=JOBS_DEFAULT, ge=1, le=JOBS_MAX),
) -> AutonomyJobPage:
    """Newest jobs first; bounded so a dashboard never pages the whole queue."""
    path = Path(request.app.state.orchestrator_db_path)
    configured = path.is_file()
    with _read_only(path) as conn:
        if conn is None or not _exists(conn, "jobs"):
            return AutonomyJobPage(configured=configured, items=[], limit=limit)
        try:
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
            return AutonomyJobPage(configured=configured, items=[], limit=limit)
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
    return AutonomyJobPage(configured=configured, items=items, limit=limit)


@router.get("/events", response_model=AutonomyEventPage)
def autonomy_events(
    request: Request,
    limit: int = Query(default=EVENTS_DEFAULT, ge=1, le=EVENTS_MAX),
) -> AutonomyEventPage:
    """Newest audit events first; an absent queue is an empty page."""
    path = Path(request.app.state.orchestrator_db_path)
    configured = path.is_file()
    with _read_only(path) as conn:
        if conn is None or not _exists(conn, "events"):
            return AutonomyEventPage(configured=configured, items=[], limit=limit)
        try:
            rows = conn.execute(
                """
                SELECT id, job_id, ts, event, detail
                FROM events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        except sqlite3.Error:
            return AutonomyEventPage(configured=configured, items=[], limit=limit)
    items = [
        AutonomyEventItem(
            id=int(row["id"]),
            job_id=None if row["job_id"] is None else int(row["job_id"]),
            ts=str(row["ts"]),
            event=str(row["event"]),
            detail=None if row["detail"] is None else str(row["detail"]),
        )
        for row in rows
    ]
    return AutonomyEventPage(configured=configured, items=items, limit=limit)
