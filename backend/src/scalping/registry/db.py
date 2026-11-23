"""SQLite registry for datasets, models, promotions and jobs.

One local file holds the metadata; the artefacts themselves (Parquet datasets,
model checkpoints) stay on disk and the registry stores their paths and
digests. The champion is derived from the newest row of ``promotions``, so the
audit trail and the live pointer cannot disagree: promoting a model and
recording why it won is a single insert.

Promotion and rollback run inside ``BEGIN IMMEDIATE`` transactions. A second
process either waits or sees the finished transaction, never a half-written
state where no model is champion or two are. Foreign keys are enabled on every
connection, so a model cannot point at a dataset that does not exist.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

__all__ = [
    "SCHEMA_VERSION",
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
SCHEMA_VERSION: int = 1

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
        data = json.loads(self.metrics_json)
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_id": self.model_id,
            "previous_model_id": self.previous_model_id,
            "reason": self.reason,
            "created_at": self.created_at,
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
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value


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
        # other three do not and must be repeated for every connection.
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        try:
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
        """``BEGIN IMMEDIATE`` around a write, rolled back on any exception."""
        conn = self._connection()
        if conn.in_transaction:
            raise RegistryError("nested registry transaction")
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()

    def init_schema(self) -> int:
        """Create missing tables or apply pending migrations; returns the version.

        Idempotent: a second call on a current database executes the
        ``CREATE ... IF NOT EXISTS`` statements again and writes no version row.
        """
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
        row = self._connection().execute("SELECT MAX(version) FROM schema_version").fetchone()
        return 0 if row is None or row[0] is None else int(row[0])

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
        _require_text(dataset_id, "dataset_id")
        _require_text(symbol, "symbol")
        _require_text(timeframe, "timeframe")
        _require_text(digest, "digest")
        dataset_path = _require_path(path, "path")
        _require_count(rows, "rows")
        first = _optional_text(first_ts, "first_ts")
        last = _optional_text(last_ts, "last_ts")
        if not isinstance(synthetic, bool):
            raise TypeError(f"synthetic must be a bool, got {type(synthetic).__name__}")
        stamp = _now() if created_at is None else _require_text(created_at, "created_at")

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
        _require_text(dataset_id, "dataset_id")
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
        _require_text(model_id, "model_id")
        _require_text(dataset_id, "dataset_id")
        _require_text(name, "name")
        model_path = _require_path(path, "path")
        metrics_json = _canonical_json({} if metrics is None else metrics)
        stamp = _now() if created_at is None else _require_text(created_at, "created_at")

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
            if "FOREIGN KEY" in str(exc):
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
        """Every model, oldest first, optionally only those of one dataset."""
        conn = self._connection()
        if dataset_id is None:
            rows = conn.execute("SELECT * FROM models ORDER BY created_at, model_id").fetchall()
        else:
            _require_text(dataset_id, "dataset_id")
            rows = conn.execute(
                "SELECT * FROM models WHERE dataset_id = ? ORDER BY created_at, model_id",
                (dataset_id,),
            ).fetchall()
        return [_model_from_row(row) for row in rows]

    # ---------------------------------------------------------- champion changes

    def champion(self) -> ModelRecord | None:
        """The current champion, or ``None`` before the first promotion."""
        row = (
            self._connection()
            .execute(
                """
                SELECT m.* FROM promotions AS p
                JOIN models AS m ON m.model_id = p.model_id
                ORDER BY p.id DESC
                LIMIT 1
                """
            )
            .fetchone()
        )
        return None if row is None else _model_from_row(row)

    def promote(self, model_id: str, reason: str) -> PromotionRecord:
        """Make ``model_id`` the champion, recording the model it replaced.

        Promoting the current champion again is rejected rather than duplicated:
        the audit trail should say when the champion actually changed.
        """
        _require_text(model_id, "model_id")
        _require_text(reason, "reason")
        with self._transaction() as tx:
            known = tx.execute("SELECT 1 FROM models WHERE model_id = ?", (model_id,)).fetchone()
            if known is None:
                raise UnknownModelError(f"model {model_id!r} is not registered")
            previous_row = tx.execute(
                "SELECT model_id FROM promotions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            previous = None if previous_row is None else str(previous_row["model_id"])
            if previous == model_id:
                raise RegistryError(f"model {model_id!r} is already the champion")
            stamp = _now()
            cursor = tx.execute(
                """
                INSERT INTO promotions (model_id, previous_model_id, reason, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (model_id, previous, reason, stamp),
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
            )
        return record

    def rollback(self, to_model_id: str, reason: str) -> PromotionRecord:
        """Restore an earlier champion, recording the one it replaces.

        The target must have been champion before: a model that never won is
        promoted, not rolled back to, and mixing the two would make the audit
        trail lie about what happened.
        """
        _require_text(to_model_id, "to_model_id")
        _require_text(reason, "reason")
        with self._transaction() as tx:
            known = tx.execute("SELECT 1 FROM models WHERE model_id = ?", (to_model_id,)).fetchone()
            if known is None:
                raise UnknownModelError(f"model {to_model_id!r} is not registered")
            previous_row = tx.execute(
                "SELECT model_id FROM promotions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if previous_row is None:
                raise RegistryError("no champion to roll back from")
            previous = str(previous_row["model_id"])
            if previous == to_model_id:
                raise RegistryError(f"model {to_model_id!r} is already the champion")
            was_champion = tx.execute(
                "SELECT 1 FROM promotions WHERE model_id = ? LIMIT 1", (to_model_id,)
            ).fetchone()
            if was_champion is None:
                raise RegistryError(
                    f"model {to_model_id!r} was never the champion; use promote instead"
                )
            stamp = _now()
            cursor = tx.execute(
                """
                INSERT INTO promotions (model_id, previous_model_id, reason, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (to_model_id, previous, reason, stamp),
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
            )
        return record

    def list_promotions(self, model_id: str | None = None) -> list[PromotionRecord]:
        """Champion changes, oldest first, optionally only those of one model."""
        conn = self._connection()
        if model_id is None:
            rows = conn.execute("SELECT * FROM promotions ORDER BY id").fetchall()
        else:
            _require_text(model_id, "model_id")
            rows = conn.execute(
                "SELECT * FROM promotions WHERE model_id = ? ORDER BY id",
                (model_id,),
            ).fetchall()
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
        _require_text(kind, "kind")
        _require_text(status, "status")
        started = _optional_text(started_at, "started_at")
        finished = _optional_text(finished_at, "finished_at")
        if detail is not None:
            _require_text(detail, "detail")
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
