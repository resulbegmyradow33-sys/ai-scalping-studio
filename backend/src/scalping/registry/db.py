"""SQLite registry for datasets, models, promotions and jobs.

One local file holds the metadata; the artefacts themselves (Parquet datasets,
model checkpoints) stay on disk and the registry stores their paths and
digests. The champion is derived from the newest row of ``promotions`` in a
scope, so the audit trail and the live pointer cannot disagree: promoting a
model and recording why it won is a single insert.

Promotion and rollback run inside ``BEGIN IMMEDIATE`` transactions. A second
process either waits or sees the finished transaction, never a half-written
state where no model is champion or two are. Foreign keys are enforced on
connections this module opens (``PRAGMA foreign_keys = ON``); a raw connection
created elsewhere has SQLite's default, which is off.

The registry stores ``digest`` and ``path`` but does not verify them. Checking
that a Parquet file or checkpoint still matches its digest before training or
serving is the caller's job; a registry row is a claim, not a checksum.

``synchronous = NORMAL`` in WAL mode keeps the database consistent across a
process crash, but an OS crash or power loss can drop the newest committed
transactions. The registry accepts that trade: the artefacts are the expensive
part, and a lost registration can be repeated, while a corrupt database cannot
be read at all.

A connection belongs to the thread that opened it (sqlite3's default). Create
one registry per thread instead of sharing one; the Mac mini's "one heavy job"
rule means that is rarely more than one.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

__all__ = [
    "JOB_STATUSES",
    "SCHEMA_VERSION",
    "Champion",
    "DatasetRecord",
    "DuplicateError",
    "JobRecord",
    "ModelRecord",
    "ModelRegistry",
    "PromotionRecord",
    "RegistryError",
    "UnknownDatasetError",
    "UnknownModelError",
]

#: Schema version written by this module. A database with a newer version is
#: refused instead of being silently written with an older layout.
SCHEMA_VERSION: int = 2

#: Job states the scheduler understands. ``record_job`` keeps accepting free
#: text for forward compatibility; ``update_job`` enforces this set so a typo
#: cannot silently park a job in a state nothing polls for.
JOB_STATUSES: frozenset[str] = frozenset({"pending", "running", "succeeded", "failed", "cancelled"})

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

#: Statements per version, applied in order and only once.
_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            digest TEXT NOT NULL,
            path TEXT NOT NULL,
            rows INTEGER NOT NULL CHECK (rows >= 0),
            first_ts TEXT,
            last_ts TEXT,
            synthetic INTEGER NOT NULL DEFAULT 0 CHECK (synthetic IN (0, 1)),
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE RESTRICT,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            metrics_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS models_dataset_idx ON models(dataset_id)",
        """
        CREATE TABLE IF NOT EXISTS promotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL REFERENCES models(model_id) ON DELETE RESTRICT,
            previous_model_id TEXT REFERENCES models(model_id) ON DELETE RESTRICT,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS promotions_model_idx ON promotions(model_id)",
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            detail TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS jobs_kind_idx ON jobs(kind, id)",
    ),
    2: (
        "ALTER TABLE promotions ADD COLUMN scope TEXT NOT NULL DEFAULT 'global'",
        "CREATE INDEX IF NOT EXISTS promotions_scope_idx ON promotions(scope, id)",
        # A CHECK constraint would need a full table rebuild, and rebuilding
        # ``models`` while ``promotions`` references it is worse than the two
        # triggers below, which enforce the same thing without touching rows.
        """
        CREATE TRIGGER IF NOT EXISTS models_metrics_json_insert
        BEFORE INSERT ON models
        FOR EACH ROW WHEN json_valid(NEW.metrics_json) = 0
        BEGIN
            SELECT RAISE(ABORT, 'models.metrics_json must be valid JSON');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS models_metrics_json_update
        BEFORE UPDATE OF metrics_json ON models
        FOR EACH ROW WHEN json_valid(NEW.metrics_json) = 0
        BEGIN
            SELECT RAISE(ABORT, 'models.metrics_json must be valid JSON');
        END
        """,
    ),
}


class RegistryError(Exception):
    """A registry operation was rejected; the database is unchanged."""


class DuplicateError(RegistryError):
    """The dataset or model id is already registered."""


class UnknownModelError(RegistryError):
    """The model id is not registered."""


class UnknownDatasetError(RegistryError):
    """The dataset id is not registered."""


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """One registered dataset: what it is, where it is and how to verify it."""

    dataset_id: str
    symbol: str
    timeframe: str
    digest: str
    path: str
    rows: int
    first_ts: str | None
    last_ts: str | None
    synthetic: bool
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "digest": self.digest,
            "path": self.path,
            "rows": self.rows,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "synthetic": self.synthetic,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class ModelRecord:
    """One trained model and the dataset it was fitted on."""

    model_id: str
    dataset_id: str
    name: str
    path: str
    metrics_json: str
    created_at: str

    @property
    def metrics(self) -> dict[str, Any]:
        """The stored metrics as a fresh dict; callers may mutate it freely."""
        try:
            data = json.loads(self.metrics_json)
        except json.JSONDecodeError as exc:
            raise RegistryError(f"model {self.model_id!r} has invalid metrics JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise RegistryError(f"model {self.model_id!r} has non-object metrics")
        return data

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "path": self.path,
            "metrics": self.metrics,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class PromotionRecord:
    """One champion change, with the model it replaced (``None`` for the first)."""

    id: int
    model_id: str
    previous_model_id: str | None
    reason: str
    created_at: str
    scope: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_id": self.model_id,
            "previous_model_id": self.previous_model_id,
            "reason": self.reason,
            "created_at": self.created_at,
            "scope": self.scope,
        }


@dataclass(frozen=True, slots=True)
class Champion:
    """The current champion and the promotion that put it there.

    One call gives a dashboard the model row and the audit fields, instead of a
    champion lookup followed by a history query that could disagree if another
    process promotes in between.
    """

    model: ModelRecord
    promotion_id: int
    promoted_at: str
    reason: str
    scope: str
    previous_model_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "promotion_id": self.promotion_id,
            "promoted_at": self.promoted_at,
            "reason": self.reason,
            "scope": self.scope,
            "previous_model_id": self.previous_model_id,
        }


@dataclass(frozen=True, slots=True)
class JobRecord:
    """One unit of work; ``detail`` is free text, usually a JSON document."""

    id: int
    kind: str
    status: str
    started_at: str | None
    finished_at: str | None
    detail: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "detail": self.detail,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _timestamp(value: object, field: str) -> str:
    """Parse an ISO-8601 timestamp and spell it in UTC.

    Ordering is TEXT ordering, so ``+03:00`` and ``+00:00`` rows would otherwise
    sort by their local digits. A timestamp without an offset is taken as UTC;
    anything that is not ISO-8601 is rejected instead of silently stored.
    """
    text = _require_text(value, field)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp, got {text!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat(timespec="seconds")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    """Stable spelling per mapping, and NaN fails here instead of on read."""
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _require_text(value: object, field: str) -> str:
    """Reject non-text and blank values; surrounding whitespace is dropped."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a str, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    return text


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field)


def _require_path(value: object, field: str) -> str:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{field} must be a str or Path, got {type(value).__name__}")
    text = str(value)
    if not text.strip():
        raise ValueError(f"{field} must not be empty")
    return text


def _require_count(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{field} must be >= 0")
    return value


def _dataset_from_row(row: sqlite3.Row) -> DatasetRecord:
    return DatasetRecord(
        dataset_id=str(row["dataset_id"]),
        symbol=str(row["symbol"]),
        timeframe=str(row["timeframe"]),
        digest=str(row["digest"]),
        path=str(row["path"]),
        rows=int(row["rows"]),
        first_ts=None if row["first_ts"] is None else str(row["first_ts"]),
        last_ts=None if row["last_ts"] is None else str(row["last_ts"]),
        synthetic=bool(row["synthetic"]),
        created_at=str(row["created_at"]),
    )


def _model_from_row(row: sqlite3.Row) -> ModelRecord:
    return ModelRecord(
        model_id=str(row["model_id"]),
        dataset_id=str(row["dataset_id"]),
        name=str(row["name"]),
        path=str(row["path"]),
        metrics_json=str(row["metrics_json"]),
        created_at=str(row["created_at"]),
    )


def _promotion_from_row(row: sqlite3.Row) -> PromotionRecord:
    previous = row["previous_model_id"]
    return PromotionRecord(
        id=int(row["id"]),
        model_id=str(row["model_id"]),
        previous_model_id=None if previous is None else str(previous),
        reason=str(row["reason"]),
        created_at=str(row["created_at"]),
        scope=str(row["scope"]),
    )


def _job_from_row(row: sqlite3.Row) -> JobRecord:
    def optional(name: str) -> str | None:
        value = row[name]
        return None if value is None else str(value)

    return JobRecord(
        id=int(row["id"]),
        kind=str(row["kind"]),
        status=str(row["status"]),
        started_at=optional("started_at"),
        finished_at=optional("finished_at"),
        detail=optional("detail"),
    )


def _enable_wal(conn: sqlite3.Connection) -> None:
    """Switch to WAL, retrying the exclusive lock SQLite wants for it.

    Reading the current mode first is a plain read, so an already-WAL database
    never asks for the exclusive lock on every open. Only a real switch can hit
    ``database is locked``; the retry covers two processes opening a fresh file
    at the same moment, and the final attempt re-raises.
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


class ModelRegistry:
    """A local SQLite registry; the schema is created (or migrated) on open.

    The class is not thread-safe and expects one writer at a time, which is the
    Mac mini's "one heavy job" rule anyway. ``connection`` exposes the raw
    handle for read queries and tests; writes belong to the methods.
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

    # ------------------------------------------------------------------ lifecycle

    @property
    def connection(self) -> sqlite3.Connection:
        """The open connection; read-only use is expected outside the class."""
        return self._connection()

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RegistryError("registry is closed")
        return self._conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """``BEGIN IMMEDIATE`` around a write, rolled back on any exception.

        A lock timeout is part of the documented error contract: callers see
        :class:`RegistryError`, not a raw ``OperationalError``.
        """
        conn = self._connection()
        if conn.in_transaction:
            raise RegistryError("nested registry transaction")
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            raise RegistryError(f"could not acquire the registry write lock: {exc}") from exc
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()

    def _read_schema_version(self) -> int:
        """Read the version without a write lock; 0 when the table is absent."""
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
        autocommit first, so opening a second reader never queues behind a
        writer. The write transaction is taken only when a migration is due,
        and the version is re-read inside it so two first opens cannot both
        apply the same migration.
        """
        current = self._read_schema_version()
        if current > SCHEMA_VERSION:
            raise RegistryError(
                f"database schema version {current} is newer than supported {SCHEMA_VERSION}"
            )
        if current == SCHEMA_VERSION:
            return current
        with self._transaction() as tx:
            tx.execute(_SCHEMA_VERSION_DDL)
            row = tx.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current = 0 if row is None or row[0] is None else int(row[0])
            if current > SCHEMA_VERSION:
                raise RegistryError(
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

    # ------------------------------------------------------------------- datasets

    def register_dataset(
        self,
        dataset_id: str,
        *,
        symbol: str,
        timeframe: str,
        digest: str,
        path: str | Path,
        rows: int,
        first_ts: str | None = None,
        last_ts: str | None = None,
        synthetic: bool = False,
        created_at: str | None = None,
    ) -> DatasetRecord:
        """Record a built dataset; a duplicate id raises :class:`DuplicateError`."""
        dataset_id = _require_text(dataset_id, "dataset_id")
        symbol = _require_text(symbol, "symbol")
        timeframe = _require_text(timeframe, "timeframe")
        digest = _require_text(digest, "digest")
        dataset_path = _require_path(path, "path")
        _require_count(rows, "rows")
        first = _optional_text(first_ts, "first_ts")
        last = _optional_text(last_ts, "last_ts")
        if not isinstance(synthetic, bool):
            raise TypeError(f"synthetic must be a bool, got {type(synthetic).__name__}")
        stamp = _now() if created_at is None else _timestamp(created_at, "created_at")

        params = (
            dataset_id,
            symbol,
            timeframe,
            digest,
            dataset_path,
            rows,
            first,
            last,
            int(synthetic),
            stamp,
        )
        try:
            with self._transaction() as tx:
                tx.execute(
                    """
                    INSERT INTO datasets (
                        dataset_id, symbol, timeframe, digest, path, rows,
                        first_ts, last_ts, synthetic, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    params,
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateError(f"dataset {dataset_id!r} is already registered") from exc
        return DatasetRecord(
            dataset_id=dataset_id,
            symbol=symbol,
            timeframe=timeframe,
            digest=digest,
            path=dataset_path,
            rows=rows,
            first_ts=first,
            last_ts=last,
            synthetic=synthetic,
            created_at=stamp,
        )

    def dataset(self, dataset_id: str) -> DatasetRecord | None:
        """The registered dataset, or ``None`` when the id is unknown."""
        dataset_id = _require_text(dataset_id, "dataset_id")
        row = (
            self._connection()
            .execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,))
            .fetchone()
        )
        return None if row is None else _dataset_from_row(row)

    # --------------------------------------------------------------------- models

    def register_model(
        self,
        model_id: str,
        *,
        dataset_id: str,
        name: str,
        path: str | Path,
        metrics: Mapping[str, Any] | None = None,
        created_at: str | None = None,
    ) -> ModelRecord:
        """Record a trained model; the dataset must already be registered."""
        model_id = _require_text(model_id, "model_id")
        dataset_id = _require_text(dataset_id, "dataset_id")
        name = _require_text(name, "name")
        model_path = _require_path(path, "path")
        metrics_json = _canonical_json({} if metrics is None else metrics)
        stamp = _now() if created_at is None else _timestamp(created_at, "created_at")

        try:
            with self._transaction() as tx:
                known = tx.execute(
                    "SELECT 1 FROM datasets WHERE dataset_id = ?", (dataset_id,)
                ).fetchone()
                if known is None:
                    raise UnknownDatasetError(f"unknown dataset {dataset_id!r}")
                tx.execute(
                    """
                    INSERT INTO models (
                        model_id, dataset_id, name, path, metrics_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (model_id, dataset_id, name, model_path, metrics_json, stamp),
                )
        except sqlite3.IntegrityError as exc:
            message = str(exc)
            if "metrics_json" in message:
                raise RegistryError(f"model {model_id!r} metrics must be valid JSON") from exc
            if "FOREIGN KEY" in message:
                raise UnknownDatasetError(f"unknown dataset {dataset_id!r}") from exc
            raise DuplicateError(f"model {model_id!r} is already registered") from exc
        return ModelRecord(
            model_id=model_id,
            dataset_id=dataset_id,
            name=name,
            path=model_path,
            metrics_json=metrics_json,
            created_at=stamp,
        )

    def list_models(self, dataset_id: str | None = None) -> list[ModelRecord]:
        """Every model, oldest first, optionally only those of one dataset.

        ``created_at`` is stored in UTC, so TEXT ordering matches chronological
        ordering regardless of the caller's original offset. Models registered
        in the same second fall back to ``model_id`` for a stable order.
        """
        conn = self._connection()
        if dataset_id is None:
            rows = conn.execute("SELECT * FROM models ORDER BY created_at, model_id").fetchall()
        else:
            dataset_id = _require_text(dataset_id, "dataset_id")
            rows = conn.execute(
                "SELECT * FROM models WHERE dataset_id = ? ORDER BY created_at, model_id",
                (dataset_id,),
            ).fetchall()
        return [_model_from_row(row) for row in rows]

    # ---------------------------------------------------------- champion changes

    def champion_promotion(self, *, scope: str = "global") -> Champion | None:
        """The current champion plus the promotion that made it, or ``None``.

        The newest promotion is selected without a join, then its model is
        looked up on its own: an inner join would silently return the previous
        champion if the newest model row were missing.
        """
        scope = _require_text(scope, "scope")
        conn = self._connection()
        row = conn.execute(
            "SELECT * FROM promotions WHERE scope = ? ORDER BY id DESC LIMIT 1",
            (scope,),
        ).fetchone()
        if row is None:
            return None
        promotion = _promotion_from_row(row)
        model_row = conn.execute(
            "SELECT * FROM models WHERE model_id = ?", (promotion.model_id,)
        ).fetchone()
        if model_row is None:
            raise RegistryError(f"champion model {promotion.model_id!r} is missing")
        return Champion(
            model=_model_from_row(model_row),
            promotion_id=promotion.id,
            promoted_at=promotion.created_at,
            reason=promotion.reason,
            scope=promotion.scope,
            previous_model_id=promotion.previous_model_id,
        )

    def champion(self, *, scope: str = "global") -> ModelRecord | None:
        """The current champion model in ``scope``, or ``None`` before the first."""
        champion = self.champion_promotion(scope=scope)
        return None if champion is None else champion.model

    def promote(self, model_id: str, reason: str, *, scope: str = "global") -> PromotionRecord:
        """Make ``model_id`` the champion of ``scope``, recording its predecessor.

        Promoting the current champion again is rejected rather than duplicated:
        the audit trail should say when the champion actually changed. Scopes
        (a symbol, a timeframe, a strategy) keep independent champions.
        """
        model_id = _require_text(model_id, "model_id")
        reason = _require_text(reason, "reason")
        scope = _require_text(scope, "scope")
        try:
            with self._transaction() as tx:
                known = tx.execute(
                    "SELECT 1 FROM models WHERE model_id = ?", (model_id,)
                ).fetchone()
                if known is None:
                    raise UnknownModelError(f"model {model_id!r} is not registered")
                previous_row = tx.execute(
                    "SELECT model_id FROM promotions WHERE scope = ? ORDER BY id DESC LIMIT 1",
                    (scope,),
                ).fetchone()
                previous = None if previous_row is None else str(previous_row["model_id"])
                if previous is not None:
                    still_there = tx.execute(
                        "SELECT 1 FROM models WHERE model_id = ?", (previous,)
                    ).fetchone()
                    if still_there is None:
                        raise RegistryError(f"champion model {previous!r} is missing")
                if previous == model_id:
                    raise RegistryError(f"model {model_id!r} is already the champion")
                stamp = _now()
                cursor = tx.execute(
                    """
                    INSERT INTO promotions (model_id, previous_model_id, reason, created_at, scope)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (model_id, previous, reason, stamp, scope),
                )
                promotion_id = cursor.lastrowid
                if promotion_id is None:
                    raise RegistryError("promotion insert returned no id")
                record = PromotionRecord(
                    id=int(promotion_id),
                    model_id=model_id,
                    previous_model_id=previous,
                    reason=reason,
                    created_at=stamp,
                    scope=scope,
                )
        except sqlite3.IntegrityError as exc:
            raise RegistryError(f"promotion failed: {exc}") from exc
        return record

    def rollback(self, to_model_id: str, reason: str, *, scope: str = "global") -> PromotionRecord:
        """Restore an earlier champion of ``scope``, recording the one it replaces.

        The target must have been champion in that scope before: a model that
        never won is promoted, not rolled back to, and mixing the two would
        make the audit trail lie about what happened.
        """
        to_model_id = _require_text(to_model_id, "to_model_id")
        reason = _require_text(reason, "reason")
        scope = _require_text(scope, "scope")
        try:
            with self._transaction() as tx:
                known = tx.execute(
                    "SELECT 1 FROM models WHERE model_id = ?", (to_model_id,)
                ).fetchone()
                if known is None:
                    raise UnknownModelError(f"model {to_model_id!r} is not registered")
                previous_row = tx.execute(
                    "SELECT model_id FROM promotions WHERE scope = ? ORDER BY id DESC LIMIT 1",
                    (scope,),
                ).fetchone()
                if previous_row is None:
                    raise RegistryError("no champion to roll back from")
                previous = str(previous_row["model_id"])
                if previous == to_model_id:
                    raise RegistryError(f"model {to_model_id!r} is already the champion")
                was_champion = tx.execute(
                    "SELECT 1 FROM promotions WHERE model_id = ? AND scope = ? LIMIT 1",
                    (to_model_id, scope),
                ).fetchone()
                if was_champion is None:
                    raise RegistryError(
                        f"model {to_model_id!r} was never the champion; use promote instead"
                    )
                stamp = _now()
                cursor = tx.execute(
                    """
                    INSERT INTO promotions (model_id, previous_model_id, reason, created_at, scope)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (to_model_id, previous, reason, stamp, scope),
                )
                promotion_id = cursor.lastrowid
                if promotion_id is None:
                    raise RegistryError("promotion insert returned no id")
                record = PromotionRecord(
                    id=int(promotion_id),
                    model_id=to_model_id,
                    previous_model_id=previous,
                    reason=reason,
                    created_at=stamp,
                    scope=scope,
                )
        except sqlite3.IntegrityError as exc:
            raise RegistryError(f"rollback failed: {exc}") from exc
        return record

    def list_promotions(
        self, model_id: str | None = None, *, scope: str | None = None
    ) -> list[PromotionRecord]:
        """Champion changes, oldest first, optionally filtered by model or scope."""
        conn = self._connection()
        clauses: list[str] = []
        params: list[object] = []
        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(_require_text(model_id, "model_id"))
        if scope is not None:
            clauses.append("scope = ?")
            params.append(_require_text(scope, "scope"))
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        rows = conn.execute(f"SELECT * FROM promotions{where} ORDER BY id", params).fetchall()
        return [_promotion_from_row(row) for row in rows]

    # ----------------------------------------------------------------------- jobs

    def record_job(
        self,
        kind: str,
        status: str = "pending",
        *,
        started_at: str | None = None,
        finished_at: str | None = None,
        detail: str | None = None,
    ) -> JobRecord:
        """Append a job row; ``detail`` is free text, usually a JSON document."""
        kind = _require_text(kind, "kind")
        status = _require_text(status, "status")
        started = _optional_text(started_at, "started_at")
        finished = _optional_text(finished_at, "finished_at")
        if detail is not None:
            detail = _require_text(detail, "detail")
        with self._transaction() as tx:
            cursor = tx.execute(
                """
                INSERT INTO jobs (kind, status, started_at, finished_at, detail)
                VALUES (?, ?, ?, ?, ?)
                """,
                (kind, status, started, finished, detail),
            )
            job_id = cursor.lastrowid
            if job_id is None:
                raise RegistryError("job insert returned no id")
            record = JobRecord(
                id=int(job_id),
                kind=kind,
                status=status,
                started_at=started,
                finished_at=finished,
                detail=detail,
            )
        return record

    def list_jobs(
        self,
        kind: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[JobRecord]:
        """Jobs newest first, optionally filtered by kind/status and capped.

        Newest first because the scheduler and dashboard ask "what is running
        now"; the audit chain in ``promotions`` stays oldest first.
        """
        clauses: list[str] = []
        params: list[object] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(_require_text(kind, "kind"))
        if status is not None:
            clauses.append("status = ?")
            params.append(_require_text(status, "status"))
        if limit is not None:
            limit = _require_count(limit, "limit")
            if limit < 1:
                raise ValueError("limit must be >= 1")
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        sql = f"SELECT * FROM jobs{where} ORDER BY id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._connection().execute(sql, params).fetchall()
        return [_job_from_row(row) for row in rows]

    def update_job(
        self,
        job_id: int,
        status: str,
        *,
        finished_at: str | None = None,
        detail: str | None = None,
    ) -> JobRecord:
        """Change a job's state; ``None`` leaves ``finished_at``/``detail`` as is.

        ``status`` is checked against :data:`JOB_STATUSES` here (unlike
        ``record_job``, which accepts free text) so a typo cannot park a job in
        a state the scheduler never polls. An unknown id raises
        :class:`RegistryError`.
        """
        job_id = _require_count(job_id, "job_id")
        status = _require_text(status, "status")
        if status not in JOB_STATUSES:
            raise ValueError(f"status must be one of {sorted(JOB_STATUSES)}, got {status!r}")
        with self._transaction() as tx:
            row = tx.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise RegistryError(f"job {job_id} is not registered")
            if finished_at is None:
                finished = None if row["finished_at"] is None else str(row["finished_at"])
            else:
                finished = _optional_text(finished_at, "finished_at")
            stored_detail = None if row["detail"] is None else str(row["detail"])
            new_detail = stored_detail if detail is None else _require_text(detail, "detail")
            tx.execute(
                "UPDATE jobs SET status = ?, finished_at = ?, detail = ? WHERE id = ?",
                (status, finished, new_detail, job_id),
            )
            record = JobRecord(
                id=job_id,
                kind=str(row["kind"]),
                status=status,
                started_at=None if row["started_at"] is None else str(row["started_at"]),
                finished_at=finished,
                detail=new_detail,
            )
        return record
