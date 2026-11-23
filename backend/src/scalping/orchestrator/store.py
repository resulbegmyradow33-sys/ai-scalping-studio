"""Persistent job queue and append-only audit log for the learning orchestrator.

One local SQLite file holds every job and every state transition. A restart
reopens the same file, so an interrupted training run becomes an ``interrupted``
row instead of disappearing; the orchestrator decides whether it may be
requeued. The queue is deliberately dumb: it stores jobs, enforces the
dependency graph at claim time and writes an event for every change. *Which*
kind of job to create, what it does and whether a retry is worth it belong to
:mod:`scalping.orchestrator.core`.

State lives on disk, not in memory, because the acceptance criterion is restart
recovery, not throughput. ``BEGIN IMMEDIATE`` around every claim means two
processes cannot hand out the same job, and WAL lets a dashboard read while the
worker writes. Foreign keys are enforced on connections this module opens; a
raw connection created elsewhere has SQLite's default, which is off.

Timestamps are UTC with microseconds (``2026-01-01T00:00:00.000000+00:00``) and
ordering is TEXT ordering, so every stored stamp has the same width and a naive
string comparison is chronological. A caller may pass a ``datetime`` or an ISO
string in any offset; both are normalised before storage.

The ``events`` table is append-only by convention and by the absence of any
update or delete path in this module. It has a foreign key to ``jobs`` so a
typo cannot attach an event to a job that does not exist; control events about
the orchestrator as a whole carry a ``NULL`` job id.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Self

__all__ = [
    "BLOCKED_DEPENDENCY_STATUSES",
    "JOB_STATUSES",
    "SCHEMA_VERSION",
    "TERMINAL_STATUSES",
    "BlockedJob",
    "Event",
    "Job",
    "JobStateError",
    "JobStats",
    "JobStore",
    "Recovery",
    "StoreError",
    "UnknownJobError",
]

#: Schema version written by this module. A database with a newer version is
#: refused instead of being written with an older layout.
SCHEMA_VERSION: int = 1

#: Every state a job may hold. ``interrupted`` is terminal for a job whose
#: attempts are exhausted; a recoverable one is set back to ``queued``.
JOB_STATUSES: frozenset[str] = frozenset(
    {"queued", "running", "succeeded", "failed", "cancelled", "interrupted", "skipped"}
)

#: States a finished job stays in; no transition leaves them except back to
#: ``queued`` through an explicit recovery of a ``running`` row.
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"succeeded", "failed", "cancelled", "interrupted", "skipped"}
)

#: A dependency in one of these states can never succeed, so its dependents are
#: skipped rather than left queued forever.
BLOCKED_DEPENDENCY_STATUSES: frozenset[str] = frozenset(
    {"failed", "cancelled", "interrupted", "skipped"}
)

#: Switching a database to WAL needs a lock that ignores ``busy_timeout``;
#: retry briefly so two processes opening a fresh file do not race.
_WAL_ATTEMPTS: int = 5
_WAL_BACKOFF_SECONDS: float = 0.05

_SCHEMA_VERSION_DDL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
)
"""

#: Statements per version, applied in order and only once. A ``CHECK`` on
#: ``status`` is affordable here because the table is created with the column;
#: the registry had to migrate an existing table, which is why it used triggers.
_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            dedupe_key TEXT NOT NULL UNIQUE,
            priority INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'queued'
                CHECK (status IN ('queued', 'running', 'succeeded', 'failed',
                                  'cancelled', 'interrupted', 'skipped')),
            attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
            max_retries INTEGER NOT NULL DEFAULT 0 CHECK (max_retries >= 0),
            depends_on TEXT NOT NULL DEFAULT '[]',
            payload_json TEXT NOT NULL DEFAULT '{}',
            result_json TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            scheduled_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS jobs_claim_idx ON jobs(status, scheduled_at, id)",
        "CREATE INDEX IF NOT EXISTS jobs_kind_idx ON jobs(kind, id)",
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER REFERENCES jobs(id) ON DELETE RESTRICT,
            ts TEXT NOT NULL,
            event TEXT NOT NULL,
            detail TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS events_job_idx ON events(job_id, id)",
        """
        CREATE TABLE IF NOT EXISTS state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
    ),
}


class StoreError(Exception):
    """A queue operation was rejected; the database is unchanged."""


class UnknownJobError(StoreError):
    """The job id is not in the queue."""


class JobStateError(StoreError):
    """The job is in a state that does not allow the requested transition."""


@dataclass(frozen=True, slots=True)
class Job:
    """One queued unit of work and its current state."""

    id: int
    kind: str
    dedupe_key: str
    priority: int
    status: str
    attempts: int
    max_retries: int
    depends_on: tuple[int, ...]
    payload: Mapping[str, Any]
    result: Mapping[str, Any] | None
    error: str | None
    created_at: str
    scheduled_at: str
    started_at: str | None
    finished_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "dedupe_key": self.dedupe_key,
            "priority": self.priority,
            "status": self.status,
            "attempts": self.attempts,
            "max_retries": self.max_retries,
            "depends_on": list(self.depends_on),
            "payload": dict(self.payload),
            "result": None if self.result is None else dict(self.result),
            "error": self.error,
            "created_at": self.created_at,
            "scheduled_at": self.scheduled_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass(frozen=True, slots=True)
class Event:
    """One audit line: what happened to which job and why."""

    id: int
    job_id: int | None
    ts: str
    event: str
    detail: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "ts": self.ts,
            "event": self.event,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class BlockedJob:
    """A queued job whose dependency already failed, plus the culprit(s)."""

    job: Job
    blocking: tuple[tuple[int, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job": self.job.to_dict(),
            "blocking": [list(item) for item in self.blocking],
        }


@dataclass(frozen=True, slots=True)
class Recovery:
    """What ``recover_interrupted`` did with one job that was left running."""

    job: Job
    requeued: bool
    attempts: int

    def to_dict(self) -> dict[str, Any]:
        return {"job": self.job.to_dict(), "requeued": self.requeued, "attempts": self.attempts}


@dataclass(frozen=True, slots=True)
class JobStats:
    """Counts per status plus the queue-wide numbers a status page needs."""

    counts: Mapping[str, int]
    total: int
    events: int
    oldest_queued_at: str | None
    due: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "total": self.total,
            "events": self.events,
            "oldest_queued_at": self.oldest_queued_at,
            "due": self.due,
        }


def _now() -> str:
    return _format(datetime.now(UTC))


def _format(value: datetime) -> str:
    """UTC ISO-8601 with microseconds, so TEXT order equals time order."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _stamp(value: object | None, field: str = "timestamp") -> str:
    """Normalise a datetime or ISO string to the stored spelling."""
    if value is None:
        return _now()
    if isinstance(value, datetime):
        return _format(value)
    return _timestamp(value, field)


def _timestamp(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"{field} must be a datetime or an ISO-8601 str, got {type(value).__name__}"
        )
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp, got {text!r}") from exc
    return _format(parsed)


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a str, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    return text


def _require_int(value: object, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an int, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _require_delay(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("delay_seconds must be a number")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("delay_seconds must be finite and >= 0")
    return seconds


def _canonical_json(payload: Mapping[str, Any]) -> str:
    """Stable spelling per mapping; NaN fails here instead of on read."""
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _json_list(value: str, field: str) -> tuple[int, ...]:
    try:
        raw = json.loads(value)
    except ValueError as exc:
        raise StoreError(f"stored {field} is not valid JSON: {value!r}") from exc
    if not isinstance(raw, list):
        raise StoreError(f"stored {field} must be a JSON list, got {type(raw).__name__}")
    return tuple(int(item) for item in raw)


def _json_mapping(value: str | None, field: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    try:
        raw = json.loads(value)
    except ValueError as exc:
        raise StoreError(f"stored {field} is not valid JSON: {value!r}") from exc
    if not isinstance(raw, dict):
        raise StoreError(f"stored {field} must be a JSON object, got {type(raw).__name__}")
    result: dict[str, Any] = {}
    for key, item in raw.items():
        result[str(key)] = item
    return result


def _job_from_row(row: sqlite3.Row) -> Job:
    def optional(name: str) -> str | None:
        value = row[name]
        return None if value is None else str(value)

    payload = _json_mapping(str(row["payload_json"]), "payload_json") or {}
    return Job(
        id=int(row["id"]),
        kind=str(row["kind"]),
        dedupe_key=str(row["dedupe_key"]),
        priority=int(row["priority"]),
        status=str(row["status"]),
        attempts=int(row["attempts"]),
        max_retries=int(row["max_retries"]),
        depends_on=_json_list(str(row["depends_on"]), "depends_on"),
        payload=payload,
        result=_json_mapping(optional("result_json"), "result_json"),
        error=optional("error"),
        created_at=str(row["created_at"]),
        scheduled_at=str(row["scheduled_at"]),
        started_at=optional("started_at"),
        finished_at=optional("finished_at"),
    )


def _event_from_row(row: sqlite3.Row) -> Event:
    job_id = row["job_id"]
    detail = row["detail"]
    return Event(
        id=int(row["id"]),
        job_id=None if job_id is None else int(job_id),
        ts=str(row["ts"]),
        event=str(row["event"]),
        detail=None if detail is None else str(detail),
    )


def _enable_wal(conn: sqlite3.Connection) -> None:
    """Switch to WAL, retrying the exclusive lock SQLite wants for it.

    Reading the current mode first keeps an already-WAL file from asking for the
    exclusive lock on every open. Only a real switch can hit ``database is
    locked``; the retry covers two processes opening a fresh file at once.
    """
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


class JobStore:
    """A local SQLite job queue; the schema is created (or migrated) on open.

    Not thread-safe: one connection per thread, mirroring the registry and the
    Mac mini's "one heavy job" rule. ``connection`` is exposed for read queries
    and tests; writes belong to the methods so the audit log stays complete.
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
        # Connection-scoped pragmas: journal_mode persists in the file, the
        # others do not and must be repeated for every connection.
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        try:
            _enable_wal(self._conn)
            self.init_schema()
        except BaseException:
            self.close()
            raise

    # ---------------------------------------------------------------- lifecycle

    @property
    def connection(self) -> sqlite3.Connection:
        """The open connection; read-only use is expected outside the class."""
        return self._connection()

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise StoreError("job store is closed")
        return self._conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """``BEGIN IMMEDIATE`` around a write, rolled back on any exception.

        A lock timeout is part of the error contract: callers see
        :class:`StoreError`, not a raw ``OperationalError``.
        """
        conn = self._connection()
        if conn.in_transaction:
            raise StoreError("nested job store transaction")
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            raise StoreError(f"could not acquire the job store write lock: {exc}") from exc
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
        """Create missing tables or apply pending migrations; returns the version.

        Idempotent and cheap on an up-to-date database: the version is read in
        autocommit first, so a second reader never queues behind a writer. The
        write transaction is taken only when a migration is due, and the
        version is re-read inside it so two first opens cannot both apply it.
        """
        current = self._read_schema_version()
        if current > SCHEMA_VERSION:
            raise StoreError(
                f"database schema version {current} is newer than supported {SCHEMA_VERSION}"
            )
        if current == SCHEMA_VERSION:
            return current
        with self._transaction() as tx:
            tx.execute(_SCHEMA_VERSION_DDL)
            row = tx.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current = 0 if row is None or row[0] is None else int(row[0])
            if current > SCHEMA_VERSION:
                raise StoreError(
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
        """Close the connection; safe to call twice."""
        conn = self._conn
        if conn is not None:
            self._conn = None
            conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --------------------------------------------------------------------- jobs

    def enqueue(
        self,
        kind: str,
        *,
        dedupe_key: str,
        payload: Mapping[str, Any] | None = None,
        priority: int = 0,
        max_retries: int = 0,
        depends_on: Sequence[int] = (),
        scheduled_at: str | datetime | None = None,
        now: str | datetime | None = None,
    ) -> tuple[Job, bool]:
        """Add a job unless ``dedupe_key`` already exists.

        Returns ``(job, created)``. A duplicate returns the existing job
        untouched — same payload, same status — so a scanner can call this on
        every tick without opening a second copy of the work. Dependencies must
        already exist: a typo is rejected now, not discovered when the job never
        runs.
        """
        kind = _require_text(kind, "kind")
        dedupe_key = _require_text(dedupe_key, "dedupe_key")
        priority = _require_int(priority, "priority")
        max_retries = _require_int(max_retries, "max_retries", minimum=0)
        stamp = _stamp(now, "now")
        scheduled = stamp if scheduled_at is None else _stamp(scheduled_at, "scheduled_at")
        deps: list[int] = []
        for dep in depends_on:
            dep_id = _require_int(dep, "depends_on entry", minimum=1)
            if dep_id not in deps:
                deps.append(dep_id)
        payload_json = _canonical_json({} if payload is None else payload)

        with self._transaction() as tx:
            existing = tx.execute(
                "SELECT * FROM jobs WHERE dedupe_key = ?", (dedupe_key,)
            ).fetchone()
            if existing is not None:
                job = _job_from_row(existing)
                self._insert_event(
                    tx,
                    job.id,
                    "deduplicated",
                    _canonical_json({"dedupe_key": dedupe_key, "existing_job": job.id}),
                    stamp,
                )
                return job, False
            for dep_id in deps:
                known = tx.execute("SELECT 1 FROM jobs WHERE id = ?", (dep_id,)).fetchone()
                if known is None:
                    raise UnknownJobError(f"dependency job {dep_id} does not exist")
            cursor = tx.execute(
                """
                INSERT INTO jobs (
                    kind, dedupe_key, priority, status, attempts, max_retries,
                    depends_on, payload_json, created_at, scheduled_at
                ) VALUES (?, ?, ?, 'queued', 0, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    dedupe_key,
                    priority,
                    max_retries,
                    json.dumps(deps, separators=(",", ":")),
                    payload_json,
                    stamp,
                    scheduled,
                ),
            )
            job_id = cursor.lastrowid
            if job_id is None:
                raise StoreError("job insert returned no id")
            detail = _canonical_json(
                {
                    "kind": kind,
                    "priority": priority,
                    "max_retries": max_retries,
                    "depends_on": deps,
                    "scheduled_at": scheduled,
                    "dedupe_key": dedupe_key,
                }
            )
            self._insert_event(tx, int(job_id), "enqueued", detail, stamp)
            row = tx.execute("SELECT * FROM jobs WHERE id = ?", (int(job_id),)).fetchone()
            if row is None:
                raise StoreError(f"job {job_id} vanished after insert")
            return _job_from_row(row), True

    def get(self, job_id: int) -> Job | None:
        """The job, or ``None`` when the id is unknown."""
        job_id = _require_int(job_id, "job_id", minimum=1)
        row = self._connection().execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return None if row is None else _job_from_row(row)

    def list_jobs(
        self,
        *,
        status: str | None = None,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[Job]:
        """Jobs newest first, optionally filtered and capped.

        Newest first because the status page asks "what is running now"; the
        audit log stays oldest first.
        """
        clauses: list[str] = []
        params: list[object] = []
        if status is not None:
            if status not in JOB_STATUSES:
                raise ValueError(f"status must be one of {sorted(JOB_STATUSES)}, got {status!r}")
            clauses.append("status = ?")
            params.append(status)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(_require_text(kind, "kind"))
        if limit is not None:
            limit = _require_int(limit, "limit", minimum=1)
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        sql = f"SELECT * FROM jobs{where} ORDER BY id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._connection().execute(sql, params).fetchall()
        return [_job_from_row(row) for row in rows]

    def due(
        self,
        *,
        kinds: Sequence[str] = (),
        now: str | datetime | None = None,
    ) -> list[Job]:
        """Queued jobs whose schedule has arrived, in claim order (no deps check).

        Used for reporting ("is work waiting for the guard?") rather than for
        claiming, which is :meth:`claim_next`.
        """
        stamp = _stamp(now, "now")
        clauses = ["status = 'queued'", "scheduled_at <= ?"]
        params: list[object] = [stamp]
        wanted = [_require_text(kind, "kind") for kind in kinds]
        if wanted:
            clauses.append("kind IN (" + ", ".join("?" for _ in wanted) + ")")
            params.extend(wanted)
        rows = (
            self._connection()
            .execute(
                "SELECT * FROM jobs WHERE "
                + " AND ".join(clauses)
                + " ORDER BY priority DESC, scheduled_at, id",
                params,
            )
            .fetchall()
        )
        return [_job_from_row(row) for row in rows]

    def has_due(
        self,
        *,
        kinds: Sequence[str] = (),
        now: str | datetime | None = None,
    ) -> bool:
        return bool(self.due(kinds=kinds, now=now))

    def count_running(self, *, kinds: Sequence[str] = ()) -> int:
        """How many jobs are ``running``, optionally only certain kinds."""
        wanted = [_require_text(kind, "kind") for kind in kinds]
        sql = "SELECT COUNT(*) FROM jobs WHERE status = 'running'"
        params: list[object] = []
        if wanted:
            sql += " AND kind IN (" + ", ".join("?" for _ in wanted) + ")"
            params.extend(wanted)
        row = self._connection().execute(sql, params).fetchone()
        return 0 if row is None else int(row[0])

    def claim_next(
        self,
        *,
        now: str | datetime | None = None,
        heavy_kinds: Sequence[str] = (),
        exclude_kinds: Sequence[str] = (),
    ) -> Job | None:
        """Atomically move the best eligible queued job to ``running``.

        Eligibility: due (``scheduled_at <= now``), kind not excluded, every
        dependency ``succeeded``, and — when this kind is heavy — no other heavy
        job already running. Order: priority high first, then earliest schedule,
        then insertion order. The whole selection and update run inside one
        ``BEGIN IMMEDIATE``, so a second worker sees the row as running, never
        the same job twice.

        Dependencies in a terminal failure state do not block the scan: the job
        stays queued and :meth:`blocked_jobs` reports it, so the orchestrator can
        skip it with a reason instead of the queue silently starving.
        """
        stamp = _stamp(now, "now")
        heavy = frozenset(_require_text(kind, "heavy_kinds entry") for kind in heavy_kinds)
        excluded = frozenset(_require_text(kind, "exclude_kinds entry") for kind in exclude_kinds)
        with self._transaction() as tx:
            heavy_running = False
            if heavy:
                marks = ", ".join("?" for _ in heavy)
                row = tx.execute(
                    f"SELECT COUNT(*) FROM jobs WHERE status = 'running' AND kind IN ({marks})",
                    list(heavy),
                ).fetchone()
                heavy_running = row is not None and int(row[0]) > 0
            candidates = tx.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'queued' AND scheduled_at <= ?
                ORDER BY priority DESC, scheduled_at, id
                """,
                (stamp,),
            ).fetchall()
            chosen: Job | None = None
            for row in candidates:
                kind = str(row["kind"])
                if kind in excluded:
                    continue
                if kind in heavy and heavy_running:
                    continue
                deps = _json_list(str(row["depends_on"]), "depends_on")
                if not self._dependencies_succeeded(tx, deps):
                    continue
                attempts = int(row["attempts"]) + 1
                tx.execute(
                    """
                    UPDATE jobs
                    SET status = 'running', attempts = ?, started_at = ?, error = NULL
                    WHERE id = ?
                    """,
                    (attempts, stamp, int(row["id"])),
                )
                detail = _canonical_json(
                    {
                        "attempt": attempts,
                        "max_retries": int(row["max_retries"]),
                        "priority": int(row["priority"]),
                        "heavy": kind in heavy,
                        "depends_on": list(deps),
                    }
                )
                self._insert_event(tx, int(row["id"]), "started", detail, stamp)
                refreshed = tx.execute(
                    "SELECT * FROM jobs WHERE id = ?", (int(row["id"]),)
                ).fetchone()
                if refreshed is None:
                    raise StoreError(f"job {row['id']} vanished while being claimed")
                chosen = _job_from_row(refreshed)
                break
            return chosen

    def complete(
        self,
        job_id: int,
        result: Mapping[str, Any] | None = None,
        *,
        now: str | datetime | None = None,
    ) -> Job:
        """Mark a running job ``succeeded`` with its result document."""
        job_id = _require_int(job_id, "job_id", minimum=1)
        stamp = _stamp(now, "now")
        result_json = None if result is None else _canonical_json(result)
        with self._transaction() as tx:
            row = self._require_status(tx, job_id, {"running"}, "complete")
            tx.execute(
                """
                UPDATE jobs
                SET status = 'succeeded', finished_at = ?, result_json = ?, error = NULL
                WHERE id = ?
                """,
                (stamp, result_json, job_id),
            )
            detail = _canonical_json(
                {"attempt": int(row["attempts"]), "has_result": result is not None}
            )
            self._insert_event(tx, job_id, "succeeded", detail, stamp)
            return self._reload(tx, job_id)

    def fail(
        self,
        job_id: int,
        error: str,
        *,
        now: str | datetime | None = None,
    ) -> Job:
        """Mark a running job ``failed`` for good; no retry follows."""
        job_id = _require_int(job_id, "job_id", minimum=1)
        error = _require_text(error, "error")
        stamp = _stamp(now, "now")
        with self._transaction() as tx:
            row = self._require_status(tx, job_id, {"running"}, "fail")
            tx.execute(
                "UPDATE jobs SET status = 'failed', finished_at = ?, error = ? WHERE id = ?",
                (stamp, error, job_id),
            )
            detail = _canonical_json(
                {
                    "attempt": int(row["attempts"]),
                    "max_retries": int(row["max_retries"]),
                    "error": error,
                }
            )
            self._insert_event(tx, job_id, "failed", detail, stamp)
            return self._reload(tx, job_id)

    def retry(
        self,
        job_id: int,
        error: str,
        *,
        delay_seconds: float,
        now: str | datetime | None = None,
    ) -> Job:
        """Put a running job back in the queue after ``delay_seconds``.

        Allowed only while a retry remains (``attempts <= max_retries``). The
        policy that decides the delay lives in the orchestrator; the store
        enforces the bound so a bug cannot loop a job forever.
        """
        job_id = _require_int(job_id, "job_id", minimum=1)
        error = _require_text(error, "error")
        delay = _require_delay(delay_seconds)
        stamp = _stamp(now, "now")
        scheduled = _format(datetime.fromisoformat(stamp) + timedelta(seconds=delay))
        with self._transaction() as tx:
            row = self._require_status(tx, job_id, {"running"}, "retry")
            attempts = int(row["attempts"])
            max_retries = int(row["max_retries"])
            if attempts > max_retries:
                raise JobStateError(
                    f"job {job_id} has used {attempts} attempt(s) of {max_retries + 1}; "
                    "fail it instead of retrying"
                )
            tx.execute(
                """
                UPDATE jobs
                SET status = 'queued', scheduled_at = ?, started_at = NULL,
                    finished_at = NULL, error = ?
                WHERE id = ?
                """,
                (scheduled, error, job_id),
            )
            detail = _canonical_json(
                {
                    "attempt": attempts,
                    "max_retries": max_retries,
                    "delay_seconds": delay,
                    "scheduled_at": scheduled,
                    "error": error,
                }
            )
            self._insert_event(tx, job_id, "retry", detail, stamp)
            return self._reload(tx, job_id)

    def defer(
        self,
        job_id: int,
        reason: str,
        *,
        delay_seconds: float,
        now: str | datetime | None = None,
    ) -> Job:
        """Return a claimed job to the queue without spending an attempt.

        For a refusal that happened before the handler ran — a resource guard saying
        "not now" — the job did no work, so ``attempts`` goes back by one and the
        job keeps its full retry budget. Retrying instead would let a busy machine
        exhaust every attempt without a single execution.
        """
        job_id = _require_int(job_id, "job_id", minimum=1)
        reason = _require_text(reason, "reason")
        delay = _require_delay(delay_seconds)
        stamp = _stamp(now, "now")
        scheduled = _format(datetime.fromisoformat(stamp) + timedelta(seconds=delay))
        with self._transaction() as tx:
            row = self._require_status(tx, job_id, {"running"}, "defer")
            attempts = int(row["attempts"])
            if attempts < 1:
                raise JobStateError(f"job {job_id} was never claimed; nothing to defer")
            tx.execute(
                """
                UPDATE jobs
                SET status = 'queued', attempts = ?, scheduled_at = ?, started_at = NULL,
                    finished_at = NULL, error = ?
                WHERE id = ?
                """,
                (attempts - 1, scheduled, reason, job_id),
            )
            detail = _canonical_json(
                {
                    "attempts_spent": 0,
                    "claimed_attempt": attempts,
                    "delay_seconds": delay,
                    "scheduled_at": scheduled,
                    "reason": reason,
                }
            )
            self._insert_event(tx, job_id, "deferred", detail, stamp)
            return self._reload(tx, job_id)

    def cancel(
        self,
        job_id: int,
        reason: str | None = None,
        *,
        now: str | datetime | None = None,
    ) -> Job:
        """Cancel a queued or running job; a running handler must check itself.

        Cancelling a running job changes its row immediately but cannot stop the
        code already executing. The orchestrator compares the row after the
        handler returns and discards a result that arrived for a cancelled job.
        """
        job_id = _require_int(job_id, "job_id", minimum=1)
        stamp = _stamp(now, "now")
        text = None if reason is None else _require_text(reason, "reason")
        with self._transaction() as tx:
            row = self._require_job(tx, job_id)
            previous = str(row["status"])
            if previous == "cancelled":
                return _job_from_row(row)
            if previous in TERMINAL_STATUSES:
                raise JobStateError(f"job {job_id} is {previous}; it cannot be cancelled")
            error = None if text is None else f"cancelled: {text}"
            tx.execute(
                "UPDATE jobs SET status = 'cancelled', finished_at = ?, error = ? WHERE id = ?",
                (stamp, error, job_id),
            )
            detail = _canonical_json({"previous_status": previous, "reason": text})
            self._insert_event(tx, job_id, "cancelled", detail, stamp)
            return self._reload(tx, job_id)

    def skip(
        self,
        job_id: int,
        reason: str,
        *,
        now: str | datetime | None = None,
    ) -> Job:
        """Skip a queued job because it can never run; the reason is stored."""
        job_id = _require_int(job_id, "job_id", minimum=1)
        reason = _require_text(reason, "reason")
        stamp = _stamp(now, "now")
        with self._transaction() as tx:
            self._require_status(tx, job_id, {"queued"}, "skip")
            error = f"skipped: {reason}"
            tx.execute(
                "UPDATE jobs SET status = 'skipped', finished_at = ?, error = ? WHERE id = ?",
                (stamp, error, job_id),
            )
            self._insert_event(tx, job_id, "skipped", _canonical_json({"reason": reason}), stamp)
            return self._reload(tx, job_id)

    def blocked_jobs(self) -> list[BlockedJob]:
        """Queued jobs with at least one dependency that can never succeed.

        Scheduled time is ignored on purpose: a dependency failure is terminal,
        so waiting for the child's nominal start time would only delay the
        explanation.
        """
        conn = self._connection()
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY priority DESC, scheduled_at, id"
        ).fetchall()
        blocked: list[BlockedJob] = []
        for row in rows:
            job = _job_from_row(row)
            if not job.depends_on:
                continue
            marks = ", ".join("?" for _ in job.depends_on)
            dep_rows = conn.execute(
                f"SELECT id, status FROM jobs WHERE id IN ({marks})", list(job.depends_on)
            ).fetchall()
            statuses = {int(dep["id"]): str(dep["status"]) for dep in dep_rows}
            blocking: list[tuple[int, str]] = []
            for dep_id in job.depends_on:
                status = statuses.get(dep_id)
                if status is None:
                    blocking.append((dep_id, "missing"))
                elif status in BLOCKED_DEPENDENCY_STATUSES:
                    blocking.append((dep_id, status))
            if blocking:
                blocked.append(BlockedJob(job=job, blocking=tuple(blocking)))
        return blocked

    # ----------------------------------------------------------------- recovery

    def recover_interrupted(self, *, now: str | datetime | None = None) -> list[Recovery]:
        """Turn every ``running`` row into ``interrupted``; requeue if allowed.

        A job left running when the process died has no worker and never will.
        Attempts already spent decide its fate: with a retry left it goes back
        to ``queued`` due immediately, otherwise it stays ``interrupted`` so the
        history shows a job that ran but never finished.
        """
        stamp = _stamp(now, "now")
        recovered: list[Recovery] = []
        with self._transaction() as tx:
            rows = tx.execute("SELECT * FROM jobs WHERE status = 'running' ORDER BY id").fetchall()
            for row in rows:
                job_id = int(row["id"])
                attempts = int(row["attempts"])
                max_retries = int(row["max_retries"])
                tx.execute(
                    "UPDATE jobs SET status = 'interrupted', finished_at = ?, error = ? "
                    "WHERE id = ?",
                    (stamp, "interrupted: process restarted", job_id),
                )
                requeued = attempts <= max_retries
                detail = _canonical_json(
                    {
                        "attempts": attempts,
                        "max_retries": max_retries,
                        "requeued": requeued,
                        "reason": "process restarted while running",
                    }
                )
                self._insert_event(tx, job_id, "interrupted", detail, stamp)
                if requeued:
                    tx.execute(
                        """
                        UPDATE jobs
                        SET status = 'queued', scheduled_at = ?, started_at = NULL,
                            finished_at = NULL
                        WHERE id = ?
                        """,
                        (stamp, job_id),
                    )
                    self._insert_event(tx, job_id, "requeued", detail, stamp)
                recovered.append(
                    Recovery(job=self._reload(tx, job_id), requeued=requeued, attempts=attempts)
                )
        return recovered

    # ------------------------------------------------------------------- audit

    def append_event(
        self,
        event: str,
        *,
        job_id: int | None = None,
        detail: str | None = None,
        now: str | datetime | None = None,
    ) -> Event:
        """Append an audit line; control events pass ``job_id=None``."""
        event = _require_text(event, "event")
        if job_id is not None:
            job_id = _require_int(job_id, "job_id", minimum=1)
        if detail is not None:
            detail = _require_text(detail, "detail")
        stamp = _stamp(now, "now")
        with self._transaction() as tx:
            if job_id is not None:
                self._require_job(tx, job_id)
            event_id = self._insert_event(tx, job_id, event, detail, stamp)
            row = tx.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
            if row is None:
                raise StoreError(f"event {event_id} vanished after insert")
            return _event_from_row(row)

    def events(
        self,
        job_id: int | None = None,
        *,
        since_id: int | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """Audit lines oldest first, optionally one job's or only newer ones."""
        clauses: list[str] = []
        params: list[object] = []
        if job_id is not None:
            clauses.append("job_id = ?")
            params.append(_require_int(job_id, "job_id", minimum=1))
        if since_id is not None:
            clauses.append("id > ?")
            params.append(_require_int(since_id, "since_id", minimum=0))
        if limit is not None:
            limit = _require_int(limit, "limit", minimum=1)
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        sql = f"SELECT * FROM events{where} ORDER BY id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._connection().execute(sql, params).fetchall()
        return [_event_from_row(row) for row in rows]

    # ------------------------------------------------------------------- state

    def get_state(self, key: str) -> str | None:
        """A small persisted value (control flags, trigger cursors)."""
        key = _require_text(key, "key")
        row = self._connection().execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_state(
        self,
        key: str,
        value: str,
        *,
        now: str | datetime | None = None,
    ) -> None:
        """Upsert a persisted value; the orchestrator's controls live here."""
        key = _require_text(key, "key")
        value = _require_text(value, "value")
        stamp = _stamp(now, "now")
        with self._transaction() as tx:
            tx.execute(
                """
                INSERT INTO state (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                              updated_at = excluded.updated_at
                """,
                (key, value, stamp),
            )

    def clear_state(self, key: str) -> None:
        key = _require_text(key, "key")
        with self._transaction() as tx:
            tx.execute("DELETE FROM state WHERE key = ?", (key,))

    # ------------------------------------------------------------------- stats

    def stats(self, *, now: str | datetime | None = None) -> JobStats:
        """Counts per status, queue total, audit size and the due backlog."""
        stamp = _stamp(now, "now")
        conn = self._connection()
        counts = {status: 0 for status in sorted(JOB_STATUSES)}
        for row in conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"):
            counts[str(row["status"])] = int(row["n"])
        total = sum(counts.values())
        event_row = conn.execute("SELECT COUNT(*) FROM events").fetchone()
        events = 0 if event_row is None else int(event_row[0])
        oldest_row = conn.execute(
            "SELECT MIN(created_at) FROM jobs WHERE status = 'queued'"
        ).fetchone()
        oldest = None if oldest_row is None or oldest_row[0] is None else str(oldest_row[0])
        due_row = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status = 'queued' AND scheduled_at <= ?",
            (stamp,),
        ).fetchone()
        due = 0 if due_row is None else int(due_row[0])
        return JobStats(
            counts=counts,
            total=total,
            events=events,
            oldest_queued_at=oldest,
            due=due,
        )

    # ------------------------------------------------------------------ helpers

    def _require_job(self, tx: sqlite3.Connection, job_id: int) -> sqlite3.Row:
        row = tx.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise UnknownJobError(f"job {job_id} does not exist")
        return row

    def _require_status(
        self,
        tx: sqlite3.Connection,
        job_id: int,
        allowed: set[str],
        action: str,
    ) -> sqlite3.Row:
        row = self._require_job(tx, job_id)
        status = str(row["status"])
        if status not in allowed:
            raise JobStateError(
                f"job {job_id} is {status}; only {sorted(allowed)} can be {action}ed"
            )
        return row

    def _reload(self, tx: sqlite3.Connection, job_id: int) -> Job:
        row = tx.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise StoreError(f"job {job_id} disappeared")
        return _job_from_row(row)

    def _dependencies_succeeded(self, tx: sqlite3.Connection, deps: tuple[int, ...]) -> bool:
        if not deps:
            return True
        marks = ", ".join("?" for _ in deps)
        rows = tx.execute(
            f"SELECT id, status FROM jobs WHERE id IN ({marks})", list(deps)
        ).fetchall()
        statuses = {int(row["id"]): str(row["status"]) for row in rows}
        return all(statuses.get(dep) == "succeeded" for dep in deps)

    def _insert_event(
        self,
        tx: sqlite3.Connection,
        job_id: int | None,
        event: str,
        detail: str | None,
        stamp: str,
    ) -> int:
        cursor = tx.execute(
            "INSERT INTO events (job_id, ts, event, detail) VALUES (?, ?, ?, ?)",
            (job_id, stamp, event, detail),
        )
        event_id = cursor.lastrowid
        if event_id is None:
            raise StoreError("event insert returned no id")
        return int(event_id)
