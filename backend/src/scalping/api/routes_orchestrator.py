"""Owner controls for the orchestrator: three flags, six transitions.

The autonomous loop already persists ``pause``, ``safe mode`` and
``disable research`` in the job store's ``state`` table and appends an event for
every change. This module exposes exactly those transitions over HTTP so the
owner can stop spending work from the panel without shell access. It does not
add a second control path: the request is applied through
:class:`~scalping.orchestrator.core.Orchestrator`, the same code the CLI and the
daemon read.

**Local-only assumption.** The API is served on loopback by the operator
(``uvicorn scalping.api.app:app``) and has no authentication layer. Every route
here is safe only under that assumption: a process that can reach the port can
pause or disable research. The project does not bind this app to a public
interface; if that changes, an authentication/authorization layer is required
before the control route is exposed. The CORS allowlist in
:mod:`scalping.api.app` is the only browser boundary and is not a substitute for
auth.

**Bounded writes.** ``POST`` opens the configured database for writing and
touches only ``state`` (the three flags) and ``events`` (the append-only audit
line). It never claims, enqueues, cancels or runs a job and never starts or stops
a daemon. ``GET`` is read-only, like ``/api/autonomy``: a missing database is
``configured: false`` and is not created.

**No request path.** The database path is ``app.state.orchestrator_db_path``,
installed by the app factory. The body carries an action and a reason only; an
unknown field is a ``422`` rather than a silently ignored control.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from fastapi import APIRouter, HTTPException, Request

from scalping.api.schemas import (
    OrchestratorControlAction,
    OrchestratorControlChange,
    OrchestratorControlRequest,
    OrchestratorControlState,
)
from scalping.orchestrator.core import (
    STATE_PAUSED,
    STATE_RESEARCH_DISABLED,
    STATE_SAFE_MODE,
    Orchestrator,
)
from scalping.orchestrator.store import SCHEMA_VERSION, JobStore, StoreError

__all__ = ["ACTOR_DEFAULT", "router"]

router = APIRouter(prefix="/api/orchestrator", tags=["orchestrator"])

#: Every control is local by construction. The API has no identity layer, so
#: this is an honest label for "the owner at this machine", not a claimed user.
ACTOR_DEFAULT: Final[str] = "local-owner"

#: The audit event written by :meth:`Orchestrator._set_flag` for each API
#: action, inverted so the newest audit line can be described to the client.
_EVENT_ACTIONS: Final[dict[str, OrchestratorControlAction]] = {
    "paused": "pause",
    "resumed": "resume",
    "safe_mode_on": "safe_mode_on",
    "safe_mode_off": "safe_mode_off",
    "research_disabled": "disable_research",
    "research_enabled": "enable_research",
}

_STATE_COLUMNS: Final[frozenset[str]] = frozenset({"key", "value"})
_CONTROL_EVENT_COLUMNS: Final[frozenset[str]] = frozenset({"id", "job_id", "ts", "event", "detail"})


# ------------------------------------------------------------------ connections


@contextmanager
def _read_only(path: Path) -> Iterator[sqlite3.Connection | None]:
    """A ``mode=ro`` connection, or ``None`` when the file cannot be read.

    Mirrors the autonomy reader: a missing file, a corrupt header or a
    permission failure yields ``None`` so a control poll cannot turn a local
    misconfiguration into a 500. The URI comes from server configuration only.
    """
    if not path.is_file():
        yield None
        return
    conn: sqlite3.Connection | None = None
    try:
        target = path.resolve()
        conn = sqlite3.connect(f"{target.as_uri()}?mode=ro", uri=True, isolation_level=None)
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
    try:
        rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    except sqlite3.Error:
        return frozenset()
    return frozenset(str(row[1]) for row in rows)


# --------------------------------------------------------------------- reading


def _flag(conn: sqlite3.Connection, key: str) -> bool:
    """A persisted flag; an absent row or a non-object payload reads as off.

    That is the orchestrator's own default, not an invented one.
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


def _last_change(
    conn: sqlite3.Connection, tables: frozenset[str]
) -> OrchestratorControlChange | None:
    """The newest control transition, read from the append-only audit log.

    ``detail`` is parsed only to recover the actor and reason. A pre-existing
    event from the CLI/daemon has no actor field; it is still a local owner
    action, so :data:`ACTOR_DEFAULT` is the honest answer.
    """
    if "events" not in tables or not (_CONTROL_EVENT_COLUMNS <= _columns(conn, "events")):
        return None
    placeholders = ", ".join("?" for _ in _EVENT_ACTIONS)
    row = conn.execute(
        f"""
        SELECT ts, event, detail FROM events
        WHERE job_id IS NULL AND event IN ({placeholders})
        ORDER BY id DESC
        LIMIT 1
        """,
        tuple(_EVENT_ACTIONS),
    ).fetchone()
    if row is None:
        return None
    action = _EVENT_ACTIONS.get(str(row["event"]))
    if action is None:  # pragma: no cover - the query already filters these
        return None
    actor = ACTOR_DEFAULT
    reason: str | None = None
    detail = row["detail"]
    if isinstance(detail, str):
        try:
            parsed = json.loads(detail)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            raw_actor = parsed.get("actor")
            if isinstance(raw_actor, str) and raw_actor:
                actor = raw_actor
            raw_reason = parsed.get("reason")
            if isinstance(raw_reason, str):
                reason = raw_reason
    return OrchestratorControlChange(
        action=action,
        actor=actor,
        reason=reason,
        at=str(row["ts"]),
    )


def _snapshot_from_connection(conn: sqlite3.Connection) -> OrchestratorControlState:
    tables = _tables(conn)
    if "state" not in tables or not (_STATE_COLUMNS <= _columns(conn, "state")):
        return OrchestratorControlState(configured=True, readable=False)
    return OrchestratorControlState(
        configured=True,
        readable=True,
        paused=_flag(conn, STATE_PAUSED),
        safe_mode=_flag(conn, STATE_SAFE_MODE),
        research_disabled=_flag(conn, STATE_RESEARCH_DISABLED),
        last_change=_last_change(conn, tables),
    )


# --------------------------------------------------------------------- writing


def _unavailable() -> HTTPException:
    """A 409 that never names the configured path."""
    return HTTPException(status_code=409, detail="orchestrator database is not available")


def _open_writable(request: Request) -> JobStore:
    """Open the configured job store for one short control write.

    A missing file is a ``409`` and is not created: the owner can control an
    orchestrator that exists, not conjure one from an HTTP request. A corrupt or
    locked database also answers ``409`` with a fixed message; the store's own
    error text can contain the absolute path and is not returned.
    """
    path = Path(request.app.state.orchestrator_db_path)
    if not path.is_file():
        raise _unavailable()
    # JobStore initialises/migrates its schema on construction. This endpoint
    # must not turn an empty, stale or unrelated SQLite file into an orchestrator
    # database as a side effect of a control request. Validate the existing
    # schema through a read-only connection before opening the writer.
    with _read_only(path) as conn:
        if conn is None:
            raise _unavailable()
        tables = _tables(conn)
        required = {"schema_version", "jobs", "events", "state"}
        if not required <= tables:
            raise _unavailable()
        try:
            row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        except sqlite3.Error as exc:
            raise _unavailable() from exc
        version = None if row is None else row[0]
        if version != SCHEMA_VERSION:
            raise _unavailable()
    try:
        return JobStore(path)
    except (OSError, ValueError, sqlite3.Error, StoreError) as exc:
        raise _unavailable() from exc


def _apply(
    orchestrator: Orchestrator,
    action: OrchestratorControlAction,
    reason: str | None,
    actor: str,
) -> None:
    if action == "pause":
        orchestrator.pause(reason, actor=actor)
    elif action == "resume":
        orchestrator.resume(actor=actor)
    elif action == "safe_mode_on":
        orchestrator.safe_mode(True, reason, actor=actor)
    elif action == "safe_mode_off":
        orchestrator.safe_mode(False, reason, actor=actor)
    elif action == "disable_research":
        orchestrator.disable_research(True, reason, actor=actor)
    else:
        orchestrator.disable_research(False, reason, actor=actor)


# ---------------------------------------------------------------------- routes


@router.get(
    "/control",
    response_model=OrchestratorControlState,
    response_model_exclude_none=True,
)
def orchestrator_control_state(request: Request) -> OrchestratorControlState:
    """Current persisted controls and the newest control transition."""
    path = Path(request.app.state.orchestrator_db_path)
    configured = path.is_file()
    if not configured:
        return OrchestratorControlState(configured=False, readable=False)
    with _read_only(path) as conn:
        if conn is None:
            return OrchestratorControlState(configured=True, readable=False)
        try:
            return _snapshot_from_connection(conn)
        except sqlite3.Error:
            return OrchestratorControlState(configured=True, readable=False)


@router.post(
    "/control",
    response_model=OrchestratorControlState,
    response_model_exclude_none=True,
)
def orchestrate_control(
    body: OrchestratorControlRequest, request: Request
) -> OrchestratorControlState:
    """Switch one persisted owner control; never touches the daemon or queue.

    ``409`` means there is no configured orchestrator database. ``422`` means the
    action is unknown, a required reason is missing, or the body has a field this
    contract does not accept.
    """
    store = _open_writable(request)
    try:
        orchestrator = Orchestrator(store)
        _apply(orchestrator, body.action, body.reason, ACTOR_DEFAULT)
        return _snapshot_from_connection(store.connection)
    except (sqlite3.Error, StoreError) as exc:
        raise HTTPException(
            status_code=409, detail="orchestrator control could not be written"
        ) from exc
    finally:
        store.close()
