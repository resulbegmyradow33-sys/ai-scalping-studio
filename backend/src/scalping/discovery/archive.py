"""SQLite archive for discovery candidates, trials, lineage and decisions.

The archive is the evidence log of the strategy search. It stores the source
representation (canonical JSON), who derived it from whom, every evaluation with
its dataset, seed, code version and metrics, and the decisions taken afterwards.
It does not evaluate anything and it does not decide anything; a search run
writes trials here and a champion gate writes decisions.

Identity is the candidate hash. ``add_candidate`` deduplicates by that hash and
returns the existing row, so a repeated proposal is not an error and does not
create a second identity for the same strategy. The id is verified against the
stored canonical JSON: a candidate whose ``canonical_hash()`` disagrees with
``canonical_json(canonical_dict())`` is rejected, because an id that does not
describe its own payload makes every later lookup a lie.

The ``trials`` column on ``candidates`` is a cached counter maintained in the
same transaction as each insert. ``trial_count`` and ``best`` count rows in
``trials`` instead, so the database has one source of truth: the evidence.

Retention differs from the model registry on purpose. Registry artefacts live on
disk and a row can be re-registered; a discovery trial cannot be reconstructed
from a path. Deleting a candidate that has children or trials is refused by the
foreign keys (``ON DELETE RESTRICT``) rather than cascading away the lineage.

WAL, ``busy_timeout``, ``synchronous = NORMAL`` and connection-scoped foreign
keys follow :mod:`scalping.registry.db`. A connection belongs to the thread that
opened it; the Mac mini's "one heavy job" rule means one writer at a time.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from scalping.discovery.protocols import Candidate, TrialResult, canonical_digest, canonical_json

__all__ = [
    "DECISIONS",
    "SCHEMA_VERSION",
    "ArchiveError",
    "CandidateRecord",
    "DecisionRecord",
    "DiscoveryArchive",
    "TrialRecord",
    "UnknownCandidateError",
]

#: Schema version written by this module. A database from a newer module is
#: refused instead of being written with an older layout.
SCHEMA_VERSION: int = 1

#: Decisions a champion gate may record. ``no_trade`` and ``no_change`` are
#: first-class outcomes: rejecting every candidate is a valid result.
DECISIONS: frozenset[str] = frozenset({"promoted", "rejected", "no_change", "no_trade"})

#: Metric names are used to build a JSON path, so they stay in a conservative
#: alphabet (no quotes, no backslashes, no whitespace).
_METRIC_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]*")

_WAL_ATTEMPTS: int = 5
_WAL_BACKOFF_SECONDS: float = 0.05

_SCHEMA_VERSION_DDL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
)
"""

_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id TEXT PRIMARY KEY,
            canonical_json TEXT NOT NULL,
            complexity INTEGER NOT NULL CHECK (complexity >= 0),
            created_at TEXT NOT NULL,
            parent_a TEXT REFERENCES candidates(candidate_id) ON DELETE RESTRICT,
            parent_b TEXT REFERENCES candidates(candidate_id) ON DELETE RESTRICT,
            mutation TEXT,
            code_version TEXT NOT NULL,
            trials INTEGER NOT NULL DEFAULT 0 CHECK (trials >= 0)
        )
        """,
        "CREATE INDEX IF NOT EXISTS candidates_created_idx ON candidates(created_at)",
        """
        CREATE TABLE IF NOT EXISTS trials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE RESTRICT,
            dataset_id TEXT NOT NULL,
            code_version TEXT NOT NULL,
            seed INTEGER NOT NULL CHECK (seed >= 0),
            metrics_json TEXT NOT NULL,
            accepted INTEGER NOT NULL CHECK (accepted IN (0, 1)),
            rejections_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS trials_candidate_idx ON trials(candidate_id, id)",
        "CREATE INDEX IF NOT EXISTS trials_dataset_idx ON trials(dataset_id)",
        """
        CREATE TABLE IF NOT EXISTS lineage (
            candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE RESTRICT,
            parent_id TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE RESTRICT,
            relation TEXT NOT NULL,
            PRIMARY KEY (candidate_id, parent_id, relation)
        )
        """,
        "CREATE INDEX IF NOT EXISTS lineage_parent_idx ON lineage(parent_id)",
        """
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE RESTRICT,
            decision TEXT NOT NULL
                CHECK (decision IN ('promoted', 'rejected', 'no_change', 'no_trade')),
            reason TEXT NOT NULL,
            champion_id TEXT REFERENCES candidates(candidate_id) ON DELETE RESTRICT,
            created_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS decisions_candidate_idx ON decisions(candidate_id, id)",
        # The archive writes JSON it built itself, but a raw connection can
        # insert anything. A CHECK cannot use json_valid on an existing table
        # without a rebuild, so two triggers do the same job.
        """
        CREATE TRIGGER IF NOT EXISTS candidates_canonical_json_insert
        BEFORE INSERT ON candidates
        FOR EACH ROW WHEN json_valid(NEW.canonical_json) = 0
            OR json_type(NEW.canonical_json) <> 'object'
        BEGIN
            SELECT RAISE(ABORT, 'candidates.canonical_json must be a JSON object');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trials_metrics_json_insert
        BEFORE INSERT ON trials
        FOR EACH ROW WHEN json_valid(NEW.metrics_json) = 0
            OR json_type(NEW.metrics_json) <> 'object'
        BEGIN
            SELECT RAISE(ABORT, 'trials.metrics_json must be a JSON object');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trials_rejections_json_insert
        BEFORE INSERT ON trials
        FOR EACH ROW WHEN json_valid(NEW.rejections_json) = 0
            OR json_type(NEW.rejections_json) <> 'array'
        BEGIN
            SELECT RAISE(ABORT, 'trials.rejections_json must be a JSON array');
        END
        """,
    ),
}


class ArchiveError(Exception):
    """A discovery archive operation was rejected; the database is unchanged."""


class UnknownCandidateError(ArchiveError):
    """The candidate id is not in the archive."""


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """One archived strategy representation and its derivation."""

    candidate_id: str
    canonical_json: str
    complexity: int
    created_at: str
    parent_a: str | None
    parent_b: str | None
    mutation: str | None
    code_version: str
    trials: int

    @property
    def canonical(self) -> dict[str, Any]:
        """The stored representation as a fresh dict; callers may mutate it."""
        try:
            data = json.loads(self.canonical_json)
        except json.JSONDecodeError as exc:  # pragma: no cover - triggers prevent this
            raise ArchiveError(f"candidate {self.candidate_id!r} has invalid JSON: {exc}") from exc
        if not isinstance(data, dict):  # pragma: no cover - triggers prevent this
            raise ArchiveError(f"candidate {self.candidate_id!r} is not a JSON object")
        return data

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "canonical": self.canonical,
            "complexity": self.complexity,
            "created_at": self.created_at,
            "parent_a": self.parent_a,
            "parent_b": self.parent_b,
            "mutation": self.mutation,
            "code_version": self.code_version,
            "trials": self.trials,
        }


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One evaluation: what was run, on which dataset, with which seed."""

    id: int
    candidate_id: str
    dataset_id: str
    code_version: str
    seed: int
    metrics_json: str
    accepted: bool
    rejections_json: str
    created_at: str

    @property
    def metrics(self) -> dict[str, float]:
        data = json.loads(self.metrics_json)
        if not isinstance(data, dict):  # pragma: no cover - triggers prevent this
            raise ArchiveError(f"trial {self.id} has non-object metrics")
        return {str(key): float(value) for key, value in data.items()}

    @property
    def rejections(self) -> tuple[str, ...]:
        data = json.loads(self.rejections_json)
        if not isinstance(data, list):  # pragma: no cover - triggers prevent this
            raise ArchiveError(f"trial {self.id} has non-array rejections")
        return tuple(str(reason) for reason in data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "candidate_id": self.candidate_id,
            "dataset_id": self.dataset_id,
            "code_version": self.code_version,
            "seed": self.seed,
            "metrics": self.metrics,
            "accepted": self.accepted,
            "rejections": list(self.rejections),
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """One champion-gate decision about one candidate."""

    id: int
    candidate_id: str
    decision: str
    reason: str
    champion_id: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "candidate_id": self.candidate_id,
            "decision": self.decision,
            "reason": self.reason,
            "champion_id": self.champion_id,
            "created_at": self.created_at,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _timestamp(value: object, field: str) -> str:
    """Parse an ISO-8601 timestamp and store it in UTC (TEXT ordering)."""
    text = _require_text(value, field)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp, got {text!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat(timespec="seconds")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return canonical_json(payload)


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a str, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    return text


def _optional_text(value: object, field: str) -> str | None:
    return None if value is None else _require_text(value, field)


def _require_count(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{field} must be >= 0")
    return value


def _require_metric(value: object) -> str:
    metric = _require_text(value, "metric")
    if _METRIC_PATTERN.fullmatch(metric) is None:
        raise ValueError(
            f"metric {metric!r} must match {_METRIC_PATTERN.pattern}; it is used as a JSON path"
        )
    return metric


def _candidate_from_row(row: sqlite3.Row) -> CandidateRecord:
    parent_a = row["parent_a"]
    parent_b = row["parent_b"]
    mutation = row["mutation"]
    return CandidateRecord(
        candidate_id=str(row["candidate_id"]),
        canonical_json=str(row["canonical_json"]),
        complexity=int(row["complexity"]),
        created_at=str(row["created_at"]),
        parent_a=None if parent_a is None else str(parent_a),
        parent_b=None if parent_b is None else str(parent_b),
        mutation=None if mutation is None else str(mutation),
        code_version=str(row["code_version"]),
        trials=int(row["trials"]),
    )


def _trial_from_row(row: sqlite3.Row) -> TrialRecord:
    return TrialRecord(
        id=int(row["id"]),
        candidate_id=str(row["candidate_id"]),
        dataset_id=str(row["dataset_id"]),
        code_version=str(row["code_version"]),
        seed=int(row["seed"]),
        metrics_json=str(row["metrics_json"]),
        accepted=bool(row["accepted"]),
        rejections_json=str(row["rejections_json"]),
        created_at=str(row["created_at"]),
    )


def _decision_from_row(row: sqlite3.Row) -> DecisionRecord:
    champion = row["champion_id"]
    return DecisionRecord(
        id=int(row["id"]),
        candidate_id=str(row["candidate_id"]),
        decision=str(row["decision"]),
        reason=str(row["reason"]),
        champion_id=None if champion is None else str(champion),
        created_at=str(row["created_at"]),
    )


def _enable_wal(conn: sqlite3.Connection) -> None:
    """Switch to WAL, retrying the exclusive lock it needs on a fresh file."""
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


class DiscoveryArchive:
    """A local SQLite archive; the schema is created (or migrated) on open.

    Not thread-safe, one writer at a time. ``connection`` exposes the handle for
    read queries and tests; writes belong to the methods.
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
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------ lifecycle

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection()

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise ArchiveError("archive is closed")
        return self._conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connection()
        if conn.in_transaction:
            raise ArchiveError("nested archive transaction")
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            raise ArchiveError(f"could not acquire the archive write lock: {exc}") from exc
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
        """Create missing tables or apply pending migrations; idempotent."""
        current = self._read_schema_version()
        if current > SCHEMA_VERSION:
            raise ArchiveError(
                f"database schema version {current} is newer than supported {SCHEMA_VERSION}"
            )
        if current == SCHEMA_VERSION:
            return current
        with self._transaction() as tx:
            tx.execute(_SCHEMA_VERSION_DDL)
            row = tx.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current = 0 if row is None or row[0] is None else int(row[0])
            if current > SCHEMA_VERSION:
                raise ArchiveError(
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

    # ---------------------------------------------------------------- candidates

    def add_candidate(
        self,
        candidate: Candidate,
        *,
        code_version: str,
        parent_a: str | None = None,
        parent_b: str | None = None,
        mutation: str | None = None,
        created_at: str | None = None,
    ) -> CandidateRecord:
        """Archive ``candidate`` and return its row, deduplicating by hash.

        A repeated candidate returns the existing row instead of raising: the
        search may propose the same strategy twice, and that is not a second
        identity. The canonical JSON of the stored row and the incoming
        candidate must agree, otherwise the id means nothing and the insert is
        refused. Lineage is written on first registration only; a later
        re-proposal does not add edges, so the recorded derivation stays the one
        that actually produced the candidate.
        """
        if not isinstance(candidate, Candidate):
            raise TypeError(
                "candidate must provide canonical_dict(), canonical_hash() and complexity()"
            )
        payload = candidate.canonical_dict()
        if not isinstance(payload, dict):
            raise ArchiveError(f"canonical_dict must return a dict, got {type(payload).__name__}")
        try:
            canonical = _canonical_json(payload)
        except (TypeError, ValueError) as exc:
            raise ArchiveError(f"candidate payload is not canonical JSON: {exc}") from exc
        digest = _require_text(candidate.canonical_hash(), "canonical_hash")
        if digest != canonical_digest(payload):
            raise ArchiveError(
                "canonical_hash does not match sha256 of canonical_dict(); "
                "the candidate identity is inconsistent"
            )
        complexity = _require_count(candidate.complexity(), "complexity")
        parent_a = _optional_text(parent_a, "parent_a")
        parent_b = _optional_text(parent_b, "parent_b")
        mutation = _optional_text(mutation, "mutation")
        code_version = _require_text(code_version, "code_version")

        existing = self.candidate(digest)
        if existing is not None:
            if existing.canonical_json != canonical or existing.complexity != complexity:
                raise ArchiveError(
                    f"candidate {digest!r} exists with a different payload; "
                    "an id collision cannot be merged"
                )
            return existing

        stamp = _now() if created_at is None else _timestamp(created_at, "created_at")
        if parent_b is not None and parent_a is None:
            raise ValueError("parent_b requires parent_a")

        try:
            with self._transaction() as tx:
                for parent in (parent_a, parent_b):
                    if parent is None:
                        continue
                    known = tx.execute(
                        "SELECT 1 FROM candidates WHERE candidate_id = ?", (parent,)
                    ).fetchone()
                    if known is None:
                        raise ArchiveError(f"unknown parent {parent!r}")
                tx.execute(
                    """
                    INSERT INTO candidates (
                        candidate_id, canonical_json, complexity, created_at,
                        parent_a, parent_b, mutation, code_version, trials
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        digest,
                        canonical,
                        complexity,
                        stamp,
                        parent_a,
                        parent_b,
                        mutation,
                        code_version,
                    ),
                )
                relation = "crossover" if parent_b is not None else "mutation"
                for parent in (parent_a, parent_b):
                    if parent is None:
                        continue
                    tx.execute(
                        """
                        INSERT OR IGNORE INTO lineage (candidate_id, parent_id, relation)
                        VALUES (?, ?, ?)
                        """,
                        (digest, parent, relation),
                    )
        except sqlite3.IntegrityError as exc:
            message = str(exc)
            if "canonical_json" in message:
                raise ArchiveError(f"candidate {digest!r} payload is not valid JSON") from exc
            raise ArchiveError(f"candidate {digest!r} could not be archived: {exc}") from exc

        record = self.candidate(digest)
        if record is None:  # pragma: no cover - the insert just committed
            raise ArchiveError(f"candidate {digest!r} disappeared after insert")
        return record

    def candidate(self, candidate_id: str) -> CandidateRecord | None:
        """The archived candidate, or ``None`` when the id is unknown."""
        candidate_id = _require_text(candidate_id, "candidate_id")
        row = (
            self._connection()
            .execute("SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,))
            .fetchone()
        )
        return None if row is None else _candidate_from_row(row)

    def dedupe(self, candidate: Candidate) -> CandidateRecord | None:
        """The existing row for this candidate's hash, or ``None`` if unseen."""
        digest = _require_text(candidate.canonical_hash(), "canonical_hash")
        return self.candidate(digest)

    def lineage(self, candidate_id: str) -> tuple[tuple[str, str], ...]:
        """Direct ``(parent_id, relation)`` edges of one candidate."""
        candidate_id = _require_text(candidate_id, "candidate_id")
        if self.candidate(candidate_id) is None:
            raise UnknownCandidateError(f"unknown candidate {candidate_id!r}")
        rows = (
            self._connection()
            .execute(
                "SELECT parent_id, relation FROM lineage WHERE candidate_id = ? ORDER BY parent_id, relation",
                (candidate_id,),
            )
            .fetchall()
        )
        return tuple((str(row["parent_id"]), str(row["relation"])) for row in rows)

    def depth(self, candidate_id: str) -> int:
        """Number of generations between ``candidate_id`` and a root (0 for random)."""
        candidate_id = _require_text(candidate_id, "candidate_id")
        if self.candidate(candidate_id) is None:
            raise UnknownCandidateError(f"unknown candidate {candidate_id!r}")
        row = (
            self._connection()
            .execute(
                """
                WITH RECURSIVE ancestors(id, depth) AS (
                    SELECT candidate_id, 0 FROM candidates WHERE candidate_id = ?
                    UNION ALL
                    SELECT l.parent_id, a.depth + 1
                    FROM lineage l JOIN ancestors a ON l.candidate_id = a.id
                )
                SELECT MAX(depth) FROM ancestors
                """,
                (candidate_id,),
            )
            .fetchone()
        )
        return 0 if row is None or row[0] is None else int(row[0])

    # -------------------------------------------------------------------- trials

    def record_trial(
        self,
        candidate_id: str,
        result: TrialResult,
        *,
        dataset_id: str,
        code_version: str,
        seed: int,
    ) -> TrialRecord:
        """Append one evaluation and bump the cached counter on its candidate."""
        candidate_id = _require_text(candidate_id, "candidate_id")
        if not isinstance(result, TrialResult):
            raise TypeError(f"result must be a TrialResult, got {type(result).__name__}")
        dataset_id = _require_text(dataset_id, "dataset_id")
        code_version = _require_text(code_version, "code_version")
        seed = _require_count(seed, "seed")
        metrics_json = _canonical_json(result.metrics)
        rejections_json = json.dumps(
            list(result.rejections), ensure_ascii=True, separators=(",", ":")
        )
        stamp = _now()

        trial_id: int | None = None
        try:
            with self._transaction() as tx:
                row = tx.execute(
                    "SELECT 1 FROM candidates WHERE candidate_id = ?", (candidate_id,)
                ).fetchone()
                if row is None:
                    raise UnknownCandidateError(f"unknown candidate {candidate_id!r}")
                cursor = tx.execute(
                    """
                    INSERT INTO trials (
                        candidate_id, dataset_id, code_version, seed,
                        metrics_json, accepted, rejections_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate_id,
                        dataset_id,
                        code_version,
                        seed,
                        metrics_json,
                        int(result.accepted),
                        rejections_json,
                        stamp,
                    ),
                )
                trial_id = cursor.lastrowid
                if trial_id is None:  # pragma: no cover - sqlite guarantees a rowid
                    raise ArchiveError("trial insert returned no id")
                tx.execute(
                    "UPDATE candidates SET trials = trials + 1 WHERE candidate_id = ?",
                    (candidate_id,),
                )
        except sqlite3.IntegrityError as exc:
            raise ArchiveError(f"trial for {candidate_id!r} was rejected: {exc}") from exc

        if trial_id is None:  # pragma: no cover - the insert just committed
            raise ArchiveError("trial insert returned no id")
        return TrialRecord(
            id=int(trial_id),
            candidate_id=candidate_id,
            dataset_id=dataset_id,
            code_version=code_version,
            seed=seed,
            metrics_json=metrics_json,
            accepted=result.accepted,
            rejections_json=rejections_json,
            created_at=stamp,
        )

    def history(self, candidate_id: str) -> tuple[TrialRecord, ...]:
        """Every trial of one candidate, oldest first."""
        candidate_id = _require_text(candidate_id, "candidate_id")
        if self.candidate(candidate_id) is None:
            raise UnknownCandidateError(f"unknown candidate {candidate_id!r}")
        rows = (
            self._connection()
            .execute("SELECT * FROM trials WHERE candidate_id = ? ORDER BY id", (candidate_id,))
            .fetchall()
        )
        return tuple(_trial_from_row(row) for row in rows)

    def trial_count(self, candidate_id: str) -> int:
        """Number of recorded trials for one candidate (rows, not the cache)."""
        candidate_id = _require_text(candidate_id, "candidate_id")
        if self.candidate(candidate_id) is None:
            raise UnknownCandidateError(f"unknown candidate {candidate_id!r}")
        row = (
            self._connection()
            .execute("SELECT COUNT(*) FROM trials WHERE candidate_id = ?", (candidate_id,))
            .fetchone()
        )
        return 0 if row is None else int(row[0])

    def has_trial(
        self, candidate_id: str, dataset_id: str, *, code_version: str | None = None
    ) -> bool:
        """Whether this candidate was already evaluated on this dataset.

        With ``code_version`` the question is narrower: has this exact code been
        run on this candidate and dataset? A fixed evaluator bug is a new
        experiment, and the search should be allowed to pay for it, while an
        unchanged code version would only reproduce the same rows.
        """
        candidate_id = _require_text(candidate_id, "candidate_id")
        dataset_id = _require_text(dataset_id, "dataset_id")
        if code_version is None:
            row = (
                self._connection()
                .execute(
                    "SELECT 1 FROM trials WHERE candidate_id = ? AND dataset_id = ? LIMIT 1",
                    (candidate_id, dataset_id),
                )
                .fetchone()
            )
        else:
            version = _require_text(code_version, "code_version")
            row = (
                self._connection()
                .execute(
                    """
                    SELECT 1 FROM trials
                    WHERE candidate_id = ? AND dataset_id = ? AND code_version = ?
                    LIMIT 1
                    """,
                    (candidate_id, dataset_id, version),
                )
                .fetchone()
            )
        return row is not None

    def best(
        self,
        metric: str,
        minimum_trials: int = 1,
        *,
        accepted_only: bool = True,
        minimize: bool = False,
    ) -> CandidateRecord | None:
        """Highest-scoring candidate with at least ``minimum_trials`` trials.

        The score of a candidate is the best (maximum) value of ``metric`` over
        its qualifying trials; trials whose metrics lack the key are ignored. A
        candidate with no qualifying value never wins. Ties break by
        ``candidate_id`` so two archives built from the same run agree.
        """
        metric = _require_metric(metric)
        minimum = _require_count(minimum_trials, "minimum_trials")
        if minimum < 1:
            raise ValueError("minimum_trials must be >= 1")
        path = f'$."{metric}"'  # the metric was validated, so quoting is unambiguous
        direction = "ASC" if minimize else "DESC"
        row = (
            self._connection()
            .execute(
                f"""
                WITH scored AS (
                    SELECT candidate_id,
                           MAX(json_extract(metrics_json, ?)) AS value,
                           COUNT(*) AS trials
                    FROM trials
                    WHERE (? = 0 OR accepted = 1)
                    GROUP BY candidate_id
                )
                SELECT c.* FROM candidates c
                JOIN scored s ON s.candidate_id = c.candidate_id
                WHERE s.value IS NOT NULL AND s.trials >= ?
                ORDER BY s.value {direction}, c.candidate_id ASC
                LIMIT 1
                """,
                (path, 1 if accepted_only else 0, minimum),
            )
            .fetchone()
        )
        return None if row is None else _candidate_from_row(row)

    # ----------------------------------------------------------------- decisions

    def record_decision(
        self,
        candidate_id: str,
        decision: str,
        reason: str,
        *,
        champion_id: str | None = None,
    ) -> DecisionRecord:
        """Append a champion-gate decision; ``no_trade``/``no_change`` are valid."""
        candidate_id = _require_text(candidate_id, "candidate_id")
        decision = _require_text(decision, "decision")
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {sorted(DECISIONS)}, got {decision!r}")
        reason = _require_text(reason, "reason")
        champion_id = _optional_text(champion_id, "champion_id")
        stamp = _now()

        decision_id: int | None = None
        try:
            with self._transaction() as tx:
                for known_id in (candidate_id, champion_id):
                    if known_id is None:
                        continue
                    row = tx.execute(
                        "SELECT 1 FROM candidates WHERE candidate_id = ?", (known_id,)
                    ).fetchone()
                    if row is None:
                        raise UnknownCandidateError(f"unknown candidate {known_id!r}")
                cursor = tx.execute(
                    """
                    INSERT INTO decisions (candidate_id, decision, reason, champion_id, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (candidate_id, decision, reason, champion_id, stamp),
                )
                decision_id = cursor.lastrowid
                if decision_id is None:  # pragma: no cover - sqlite guarantees a rowid
                    raise ArchiveError("decision insert returned no id")
        except sqlite3.IntegrityError as exc:
            raise ArchiveError(f"decision for {candidate_id!r} was rejected: {exc}") from exc

        if decision_id is None:  # pragma: no cover - the insert just committed
            raise ArchiveError("decision insert returned no id")
        return DecisionRecord(
            id=int(decision_id),
            candidate_id=candidate_id,
            decision=decision,
            reason=reason,
            champion_id=champion_id,
            created_at=stamp,
        )

    def decisions(self, candidate_id: str | None = None) -> tuple[DecisionRecord, ...]:
        """Decisions oldest first, optionally only those about one candidate."""
        conn = self._connection()
        if candidate_id is None:
            rows = conn.execute("SELECT * FROM decisions ORDER BY id").fetchall()
        else:
            candidate_id = _require_text(candidate_id, "candidate_id")
            rows = conn.execute(
                "SELECT * FROM decisions WHERE candidate_id = ? ORDER BY id",
                (candidate_id,),
            ).fetchall()
        return tuple(_decision_from_row(row) for row in rows)
