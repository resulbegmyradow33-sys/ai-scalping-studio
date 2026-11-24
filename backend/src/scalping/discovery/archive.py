"""SQLite archive for discovery candidates, trials, lineage and decisions.

The archive is the evidence log of the strategy search. It stores the source
representation (canonical JSON), who derived it from whom, every evaluation with
its dataset, seed, code version and metrics, and the decisions taken afterwards.
It does not evaluate anything and it does not decide anything; a search run
writes trials here and a champion gate writes decisions.

Schema version 2 adds strategy champion governance. ``champion_promotions`` is
the append-only history of every promotion and rollback in a scope, ``champions``
holds exactly one current pointer per scope, and ``decisions`` gained a JSON
``detail_json`` and the ``rolled_back`` outcome. The pointer and the history are
written in one ``BEGIN IMMEDIATE`` transaction, so the live champion and the
audit trail cannot disagree.

A scope is a free-text key, conventionally ``<symbol>:<timeframe>``. A trial
"belongs to" a scope when every alphanumeric token of the scope appears in its
``dataset_id``: scope ``XAUUSD:M5`` matches ``validation:XAUUSD:M5:abcd1234`` and
not ``validation:XAUUSD:M15:abcd1234``. That rule lets the archive answer "was
this candidate evaluated on the scope's dataset?" without a second table, and it
is a naming convention, not a guarantee: a dataset id that omits the timeframe
will not qualify.

Two promotion paths exist and they are not equivalent. ``promote_evidence`` is
the production path: it requires a named/versioned promotion-policy decision with
an evidence checksum, stores no metric and records the policy identity in the
audit detail. ``promote_exploratory`` is the older one-metric comparison kept for
research and ablations; it must never select the paper champion.

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
from typing import Any, Final, Self

from scalping.discovery.protocols import Candidate, TrialResult, canonical_digest, canonical_json

__all__ = [
    "DECISIONS",
    "DEFAULT_SCOPE",
    "MANUAL_OVERRIDE_POLICY_ID",
    "SCHEMA_VERSION",
    "ArchiveError",
    "CandidateRecord",
    "ChampionComparison",
    "ChampionPromotion",
    "ChampionRecord",
    "DecisionRecord",
    "DiscoveryArchive",
    "TrialRecord",
    "UnknownCandidateError",
]

#: Schema version written by this module. A database from a newer module is
#: refused instead of being written with an older layout.
SCHEMA_VERSION: int = 2

#: Default scope key. A real deployment passes ``<symbol>:<timeframe>``
#: (:func:`scalping.discovery.champion.scope_for` builds it); this literal keeps
#: the column and the read methods usable without inventing a symbol.
DEFAULT_SCOPE: str = "symbol:timeframe"

#: Reserved policy id for a recorded manual override. An override is a human
#: record, not policy validation, so it can never be stored on an evidence
#: promotion.
MANUAL_OVERRIDE_POLICY_ID: Final[str] = "manual-override"

#: Decisions a champion gate may record. ``no_trade`` and ``no_change`` are
#: first-class outcomes: rejecting every candidate is a valid result.
#: ``rolled_back`` is written only by :meth:`DiscoveryArchive.rollback`.
DECISIONS: frozenset[str] = frozenset(
    {"promoted", "rejected", "no_change", "no_trade", "rolled_back"}
)


#: Metric names are used to build a JSON path, so they stay in a conservative
#: alphabet (no quotes, no backslashes, no whitespace).
_METRIC_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]*")

#: Scope and dataset ids are free text; ``xauusd:m5`` and ``XAUUSD-M5-2026``
#: both split into symbol/period tokens for scope matching.
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


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
    2: (
        # One row per promotion or rollback, append-only: the audit trail of the
        # scope. ``champions`` points at the newest row, so history and the live
        # pointer are written in the same transaction and cannot disagree.
        """
        CREATE TABLE IF NOT EXISTS champion_promotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL,
            kind TEXT NOT NULL CHECK (kind IN ('promoted', 'rolled_back')),
            candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE RESTRICT,
            previous_candidate_id TEXT REFERENCES candidates(candidate_id) ON DELETE RESTRICT,
            metric TEXT,
            value REAL,
            previous_value REAL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS champion_promotions_scope_idx
        ON champion_promotions(scope, id)
        """,
        # The primary key is what makes "exactly one champion per scope" a
        # database fact instead of a convention. Scopes keep independent
        # champions, so two symbols never replace each other.
        """
        CREATE TABLE IF NOT EXISTS champions (
            scope TEXT NOT NULL PRIMARY KEY,
            candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE RESTRICT,
            promotion_id INTEGER NOT NULL
                REFERENCES champion_promotions(id) ON DELETE RESTRICT,
            promoted_at TEXT NOT NULL,
            reason TEXT NOT NULL
        )
        """,
        # ``decisions`` must accept ``rolled_back`` and carry the numbers a
        # promotion used. A CHECK cannot be altered in place, and rebuilding the
        # table is safe here because nothing references ``decisions``.
        """
        CREATE TABLE decisions_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE RESTRICT,
            decision TEXT NOT NULL
                CHECK (decision IN
                    ('promoted', 'rejected', 'no_change', 'no_trade', 'rolled_back')),
            reason TEXT NOT NULL,
            champion_id TEXT REFERENCES candidates(candidate_id) ON DELETE RESTRICT,
            created_at TEXT NOT NULL,
            detail_json TEXT
        )
        """,
        """
        INSERT INTO decisions_v2 (id, candidate_id, decision, reason, champion_id, created_at)
        SELECT id, candidate_id, decision, reason, champion_id, created_at FROM decisions
        """,
        "DROP TABLE decisions",
        "ALTER TABLE decisions_v2 RENAME TO decisions",
        "CREATE INDEX IF NOT EXISTS decisions_candidate_idx ON decisions(candidate_id, id)",
        """
        CREATE TRIGGER IF NOT EXISTS decisions_detail_json_insert
        BEFORE INSERT ON decisions
        FOR EACH ROW WHEN NEW.detail_json IS NOT NULL AND (
            json_valid(NEW.detail_json) = 0
            OR json_type(NEW.detail_json) <> 'object')
        BEGIN
            SELECT RAISE(ABORT, 'decisions.detail_json must be a JSON object or NULL');
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
    detail_json: str | None = None

    @property
    def detail(self) -> dict[str, Any] | None:
        """The numbers the decision used, or ``None`` when none were recorded."""
        if self.detail_json is None:
            return None
        try:
            data = json.loads(self.detail_json)
        except json.JSONDecodeError as exc:  # pragma: no cover - triggers prevent this
            raise ArchiveError(f"decision {self.id} has invalid detail JSON: {exc}") from exc
        if not isinstance(data, dict):  # pragma: no cover - triggers prevent this
            raise ArchiveError(f"decision {self.id} has a non-object detail")
        return data

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "candidate_id": self.candidate_id,
            "decision": self.decision,
            "reason": self.reason,
            "champion_id": self.champion_id,
            "created_at": self.created_at,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class ChampionPromotion:
    """One historical champion change in one scope, promotion or rollback."""

    id: int
    scope: str
    kind: str
    candidate_id: str
    previous_candidate_id: str | None
    metric: str | None
    value: float | None
    previous_value: float | None
    reason: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope,
            "kind": self.kind,
            "candidate_id": self.candidate_id,
            "previous_candidate_id": self.previous_candidate_id,
            "metric": self.metric,
            "value": self.value,
            "previous_value": self.previous_value,
            "reason": self.reason,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class ChampionRecord:
    """The current champion of a scope, its spec and the promotion that made it.

    ``metrics`` is the newest accepted trial of the candidate on the scope's
    dataset, so "the champion's current evidence" is one read and not a
    reconstruction. ``candidate.canonical`` is the canonical spec payload the
    paper runner validates.
    """

    scope: str
    candidate: CandidateRecord
    metrics: dict[str, float]
    promotion_id: int
    promoted_at: str
    reason: str
    previous_candidate_id: str | None

    @property
    def canonical(self) -> dict[str, Any]:
        """The stored canonical spec payload, ready for ``StrategySpec.from_dict``."""
        return self.candidate.canonical

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "candidate": self.candidate.to_dict(),
            "canonical": self.canonical,
            "metrics": dict(self.metrics),
            "promotion_id": self.promotion_id,
            "promoted_at": self.promoted_at,
            "reason": self.reason,
            "previous_candidate_id": self.previous_candidate_id,
        }


@dataclass(frozen=True, slots=True)
class ChampionComparison:
    """One candidate against the current champion on one metric.

    ``better`` is strict (``candidate > champion``) and false when either side
    has no measured value: a missing number is not a win. Higher is better; a
    lower-is-better metric (drawdown) must not be promoted through this helper.
    """

    scope: str
    metric: str
    candidate_id: str
    candidate_value: float | None
    champion_id: str | None
    champion_value: float | None
    delta: float | None
    better: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "metric": self.metric,
            "candidate_id": self.candidate_id,
            "candidate_value": self.candidate_value,
            "champion_id": self.champion_id,
            "champion_value": self.champion_value,
            "delta": self.delta,
            "better": self.better,
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


def _require_digest(value: object, field: str) -> str:
    """A lowercase sha256 hex digest; identity fields must not be free text."""
    text = _require_text(value, field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase sha256 hex digest, got {text!r}")
    return text


def _scope_tokens(scope: str) -> frozenset[str]:
    """Lowercase alphanumeric tokens of a scope or dataset id.

    ``XAUUSD:M5`` -> ``{"xauusd", "m5"}``. Both scope matching and dataset
    matching use this splitter, so the rule has one spelling.
    """
    return frozenset(token for token in _TOKEN_SPLIT.split(scope.lower()) if token)


def _require_scope(value: object) -> str:
    scope = _require_text(value, "scope")
    if not _scope_tokens(scope):
        raise ValueError(f"scope {scope!r} must contain at least one alphanumeric token")
    return scope


def _scope_trials(
    conn: sqlite3.Connection, candidate_id: str, tokens: frozenset[str]
) -> list[TrialRecord]:
    """This candidate's trials whose dataset id names every scope token.

    Newest first. A trial on another symbol, timeframe or split is not evidence
    for this scope, and neither is a trial whose dataset id forgot the token.
    """
    rows = conn.execute(
        "SELECT * FROM trials WHERE candidate_id = ? ORDER BY id DESC", (candidate_id,)
    ).fetchall()
    return [
        trial
        for trial in (_trial_from_row(row) for row in rows)
        if tokens <= _scope_tokens(trial.dataset_id)
    ]


def _latest_accepted(
    conn: sqlite3.Connection, candidate_id: str, tokens: frozenset[str]
) -> TrialRecord | None:
    """The newest accepted trial on the scope's dataset, if any."""
    for trial in _scope_trials(conn, candidate_id, tokens):
        if trial.accepted:
            return trial
    return None


def _metric_value(
    conn: sqlite3.Connection, candidate_id: str, tokens: frozenset[str], metric: str
) -> float | None:
    """The metric of the newest accepted trial that publishes it, or ``None``.

    Only accepted trials count: the evaluator's verdict is the evidence gate, and
    a number from a gate-rejected run is not a claim the policy may act on.
    """
    for trial in _scope_trials(conn, candidate_id, tokens):
        if not trial.accepted:
            continue
        metrics = trial.metrics
        if metric in metrics:
            return metrics[metric]
    return None


def _promotion_reason(
    *,
    reason: str,
    forced: bool,
    metric: str,
    value: float | None,
    previous_id: str | None,
    previous_value: float | None,
) -> str:
    """The stored reason: the human sentence plus the numbers that were compared.

    A forced promotion starts with ``forced:`` and keeps the human reason, so the
    audit trail says that the policy was overridden instead of hiding it.
    """
    value_text = "unmeasured" if value is None else f"{value:g}"
    if previous_id is None:
        detail = f"first promotion; {metric} {value_text}"
    else:
        previous_text = "unmeasured" if previous_value is None else f"{previous_value:g}"
        detail = f"{metric} {value_text} vs {previous_text} of {previous_id[:12]}"
    prefix = "forced: " if forced else ""
    return f"{prefix}{reason}; {detail}"


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
    detail = row["detail_json"]
    return DecisionRecord(
        id=int(row["id"]),
        candidate_id=str(row["candidate_id"]),
        decision=str(row["decision"]),
        reason=str(row["reason"]),
        champion_id=None if champion is None else str(champion),
        created_at=str(row["created_at"]),
        detail_json=None if detail is None else str(detail),
    )


def _promotion_from_row(row: sqlite3.Row) -> ChampionPromotion:
    previous = row["previous_candidate_id"]
    metric = row["metric"]
    value = row["value"]
    previous_value = row["previous_value"]
    return ChampionPromotion(
        id=int(row["id"]),
        scope=str(row["scope"]),
        kind=str(row["kind"]),
        candidate_id=str(row["candidate_id"]),
        previous_candidate_id=None if previous is None else str(previous),
        metric=None if metric is None else str(metric),
        value=None if value is None else float(value),
        previous_value=None if previous_value is None else float(previous_value),
        reason=str(row["reason"]),
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
        detail: Mapping[str, Any] | None = None,
    ) -> DecisionRecord:
        """Append a champion-gate decision; ``no_trade``/``no_change`` are valid.

        ``detail`` is the JSON-safe evidence the decision used (metric names and
        values). It is stored canonically, so a later reader compares numbers
        instead of parsing a sentence.
        """
        candidate_id = _require_text(candidate_id, "candidate_id")
        decision = _require_text(decision, "decision")
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {sorted(DECISIONS)}, got {decision!r}")
        reason = _require_text(reason, "reason")
        champion_id = _optional_text(champion_id, "champion_id")
        detail_json = None if detail is None else self._detail_json(detail)
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
                    INSERT INTO decisions (
                        candidate_id, decision, reason, champion_id, created_at, detail_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (candidate_id, decision, reason, champion_id, stamp, detail_json),
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
            detail_json=detail_json,
        )

    @staticmethod
    def _detail_json(detail: Mapping[str, Any]) -> str:
        if not isinstance(detail, Mapping):
            raise TypeError(f"detail must be a mapping, got {type(detail).__name__}")
        try:
            return _canonical_json(detail)
        except (TypeError, ValueError) as exc:
            raise ArchiveError(f"decision detail is not canonical JSON: {exc}") from exc

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

    # ------------------------------------------------------- champion governance

    def promote_exploratory(
        self,
        candidate_id: str,
        *,
        scope: str = DEFAULT_SCOPE,
        reason: str,
        force: bool = False,
        metric: str = "expectancy_r",
    ) -> ChampionRecord:
        """Metric-only promotion for **exploratory** research; not a production gate.

        This is the pre-policy path kept for ablations and research notebooks. It
        compares one metric of the newest accepted trial, so it cannot see
        walk-forward folds, cost stress, regime coverage or a confidence
        interval. It must never be used to select the paper champion: production
        promotion goes through :meth:`promote_evidence`, which requires a
        :class:`~scalping.discovery.promotion.PromotionDecision` from a named,
        versioned, fail-closed policy.

        Refusals leave the champion and the audit trail unchanged. An unknown
        candidate raises :class:`UnknownCandidateError`; a candidate with no trial
        on the scope's dataset, a candidate with no *accepted* trial there, and —
        unless ``force`` — a candidate that does not strictly beat the current
        champion raise :class:`ArchiveError`.

        ``force`` overrides the comparison only. The evidence gate is not
        forceable: a rejected candidate cannot be promoted by any flag. A forced
        promotion requires a reason and its stored reason starts with
        ``forced:``.

        The metric is read from the newest accepted trial of each side that
        publishes it. Missing numbers are not wins: an unmeasured candidate, or
        an unmeasured champion to compare against, is refused rather than
        treated as zero. Only ``metric`` is compared; this is a one-metric
        policy, not a scorecard.
        """
        candidate_id = _require_text(candidate_id, "candidate_id")
        scope = _require_scope(scope)
        if not isinstance(force, bool):
            raise TypeError(f"force must be a bool, got {type(force).__name__}")
        if not isinstance(reason, str):
            raise TypeError(f"reason must be a str, got {type(reason).__name__}")
        reason = reason.strip()
        if not reason:
            raise ValueError(
                "a forced promotion requires a non-empty reason"
                if force
                else "reason must not be empty"
            )
        metric = _require_metric(metric)
        tokens = _scope_tokens(scope)
        stamp = _now()

        record: ChampionRecord | None = None
        try:
            with self._transaction() as tx:
                known = tx.execute(
                    "SELECT 1 FROM candidates WHERE candidate_id = ?", (candidate_id,)
                ).fetchone()
                if known is None:
                    raise UnknownCandidateError(f"unknown candidate {candidate_id!r}")

                trials = _scope_trials(tx, candidate_id, tokens)
                if not trials:
                    raise ArchiveError(
                        f"candidate {candidate_id!r} has no trial on the dataset of scope "
                        f"{scope!r}; evaluate it on that dataset before promoting it"
                    )
                latest = _latest_accepted(tx, candidate_id, tokens)
                if latest is None:
                    rejections = sorted({item for trial in trials for item in trial.rejections})
                    detail = f": {'; '.join(rejections)}" if rejections else ""
                    raise ArchiveError(
                        f"candidate {candidate_id!r} was rejected by the gate on scope "
                        f"{scope!r}{detail}"
                    )
                value = _metric_value(tx, candidate_id, tokens, metric)

                row = tx.execute("SELECT * FROM champions WHERE scope = ?", (scope,)).fetchone()
                previous_id = None if row is None else str(row["candidate_id"])
                if previous_id == candidate_id:
                    raise ArchiveError(
                        f"candidate {candidate_id!r} is already the champion of {scope!r}"
                    )
                previous_value = (
                    None if previous_id is None else _metric_value(tx, previous_id, tokens, metric)
                )

                if not force:
                    if value is None:
                        raise ArchiveError(
                            f"the accepted trial of {candidate_id!r} does not publish "
                            f"{metric!r}; a promotion needs a measured value"
                        )
                    if previous_id is not None:
                        if previous_value is None:
                            raise ArchiveError(
                                f"champion {previous_id!r} has no {metric!r} on scope "
                                f"{scope!r}; refusing a comparison without a baseline"
                            )
                        if not value > previous_value:
                            raise ArchiveError(
                                f"{metric} {value:g} of {candidate_id!r} does not strictly "
                                f"beat {previous_value:g} of champion {previous_id!r}"
                            )

                reason_text = _promotion_reason(
                    reason=reason,
                    forced=force,
                    metric=metric,
                    value=value,
                    previous_id=previous_id,
                    previous_value=previous_value,
                )
                cursor = tx.execute(
                    """
                    INSERT INTO champion_promotions (
                        scope, kind, candidate_id, previous_candidate_id,
                        metric, value, previous_value, reason, created_at
                    ) VALUES (?, 'promoted', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scope,
                        candidate_id,
                        previous_id,
                        metric,
                        value,
                        previous_value,
                        reason_text,
                        stamp,
                    ),
                )
                promotion_id = cursor.lastrowid
                if promotion_id is None:  # pragma: no cover - sqlite guarantees a rowid
                    raise ArchiveError("promotion insert returned no id")
                tx.execute(
                    """
                    INSERT INTO champions (scope, candidate_id, promotion_id, promoted_at, reason)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(scope) DO UPDATE SET
                        candidate_id = excluded.candidate_id,
                        promotion_id = excluded.promotion_id,
                        promoted_at = excluded.promoted_at,
                        reason = excluded.reason
                    """,
                    (scope, candidate_id, promotion_id, stamp, reason_text),
                )
                tx.execute(
                    """
                    INSERT INTO decisions (
                        candidate_id, decision, reason, champion_id, created_at, detail_json
                    ) VALUES (?, 'promoted', ?, ?, ?, ?)
                    """,
                    (
                        candidate_id,
                        reason_text,
                        candidate_id,
                        stamp,
                        self._detail_json(
                            {
                                "scope": scope,
                                "metric": metric,
                                "value": value,
                                "previous_candidate_id": previous_id,
                                "previous_value": previous_value,
                                "forced": force,
                                "promotion_id": int(promotion_id),
                            }
                        ),
                    ),
                )
                candidate_row = tx.execute(
                    "SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)
                ).fetchone()
                if candidate_row is None:  # pragma: no cover - checked above
                    raise ArchiveError(f"candidate {candidate_id!r} disappeared")
                record = ChampionRecord(
                    scope=scope,
                    candidate=_candidate_from_row(candidate_row),
                    metrics=dict(latest.metrics),
                    promotion_id=int(promotion_id),
                    promoted_at=stamp,
                    reason=reason_text,
                    previous_candidate_id=previous_id,
                )
        except sqlite3.IntegrityError as exc:
            raise ArchiveError(f"promotion of {candidate_id!r} failed: {exc}") from exc

        if record is None:  # pragma: no cover - the insert just committed
            raise ArchiveError("promotion record is missing")
        return record

    def promote_evidence(
        self,
        candidate_id: str,
        *,
        scope: str = DEFAULT_SCOPE,
        reason: str,
        policy_id: str,
        policy_version: int,
        evidence_checksum: str,
        dataset_id: str,
        dataset_checksum: str,
        settings_digest: str,
        candidate_spec_hash: str,
        champion_checksum: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> ChampionRecord:
        """Promote through a promotion-policy decision, not through a metric.

        This is the production path. The caller has already evaluated a
        :class:`~scalping.discovery.promotion.PromotionDecision`; this method
        stores the governance facts an auditor needs and refuses to invent any:

        * ``candidate_spec_hash`` must equal ``candidate_id``, because the id is
          the canonical hash of the spec the evidence was built for.
        * ``policy_id`` must be a named policy other than the reserved
          manual-override id, and ``policy_version`` must be positive.
        * ``evidence_checksum`` must be a sha256 hex digest; the decision that
          carried it is the only reason this promotion exists.
        * ``dataset_checksum`` and ``settings_digest`` are required digests: a
          promotion must name the exact data build and execution settings it
          was measured under.
        * every scope token must appear in ``dataset_id``, the same convention
          trials use, so evidence from another symbol/timeframe cannot promote
          here.

        The promotion row stores no metric (``metric``/``value`` are NULL) and
        the decision's ``detail_json`` records the policy id/version, the
        evidence checksum, the champion checksum, the dataset id/checksum, the
        settings digest and the candidate spec hash. That makes an evidence
        promotion distinguishable from an exploratory metric-only promotion by
        inspection, not by convention.

        This is the archive primitive. Call
        :func:`scalping.discovery.promotion.promote_with_evidence` in production
        code: it owns the ``PromotionDecision`` type check and the
        spec-hash/candidate-id link, so a caller cannot skip the policy by
        reaching this method with hand-written strings.

        No force flag exists. A rejected decision cannot be forced through this
        method; a manual override is a separate record and is not accepted here.
        """
        candidate_id = _require_text(candidate_id, "candidate_id")
        scope = _require_scope(scope)
        reason = _require_text(reason, "reason")
        policy_id = _require_text(policy_id, "policy_id")
        if policy_id == MANUAL_OVERRIDE_POLICY_ID:
            raise ArchiveError(
                "a manual override is not policy validation; it cannot promote evidence"
            )
        if isinstance(policy_version, bool) or not isinstance(policy_version, int):
            raise TypeError(f"policy_version must be an int, got {type(policy_version).__name__}")
        if policy_version < 1:
            raise ValueError(f"policy_version must be >= 1, got {policy_version}")
        evidence_checksum = _require_digest(evidence_checksum, "evidence_checksum")
        dataset_checksum = _require_digest(dataset_checksum, "dataset_checksum")
        settings_digest = _require_digest(settings_digest, "settings_digest")
        champion_checksum = (
            None
            if champion_checksum is None
            else _require_digest(champion_checksum, "champion_checksum")
        )
        candidate_spec_hash = _require_digest(candidate_spec_hash, "candidate_spec_hash")
        if candidate_spec_hash != candidate_id:
            raise ArchiveError(
                f"evidence was built for spec {candidate_spec_hash[:12]}, "
                f"not candidate {candidate_id[:12]}"
            )
        dataset_id = _require_text(dataset_id, "dataset_id")
        if not _scope_tokens(scope) <= _scope_tokens(dataset_id):
            raise ArchiveError(
                f"dataset {dataset_id!r} does not name scope {scope!r}; "
                "evidence from another market cannot promote here"
            )
        if detail is not None and not isinstance(detail, Mapping):
            raise TypeError(f"detail must be a mapping, got {type(detail).__name__}")
        tokens = _scope_tokens(scope)
        stamp = _now()
        reason_text = (
            f"{reason}; evidence {evidence_checksum[:12]} under {policy_id} v{policy_version}"
        )

        record: ChampionRecord | None = None
        try:
            with self._transaction() as tx:
                candidate_row = tx.execute(
                    "SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)
                ).fetchone()
                if candidate_row is None:
                    raise UnknownCandidateError(f"unknown candidate {candidate_id!r}")

                row = tx.execute("SELECT * FROM champions WHERE scope = ?", (scope,)).fetchone()
                previous_id = None if row is None else str(row["candidate_id"])
                if previous_id == candidate_id:
                    raise ArchiveError(
                        f"candidate {candidate_id!r} is already the champion of {scope!r}"
                    )

                cursor = tx.execute(
                    """
                    INSERT INTO champion_promotions (
                        scope, kind, candidate_id, previous_candidate_id,
                        metric, value, previous_value, reason, created_at
                    ) VALUES (?, 'promoted', ?, ?, NULL, NULL, NULL, ?, ?)
                    """,
                    (scope, candidate_id, previous_id, reason_text, stamp),
                )
                promotion_id = cursor.lastrowid
                if promotion_id is None:  # pragma: no cover - sqlite guarantees a rowid
                    raise ArchiveError("promotion insert returned no id")
                tx.execute(
                    """
                    INSERT INTO champions (scope, candidate_id, promotion_id, promoted_at, reason)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(scope) DO UPDATE SET
                        candidate_id = excluded.candidate_id,
                        promotion_id = excluded.promotion_id,
                        promoted_at = excluded.promoted_at,
                        reason = excluded.reason
                    """,
                    (scope, candidate_id, promotion_id, stamp, reason_text),
                )

                payload = dict(detail or {})
                payload.update(
                    {
                        "scope": scope,
                        "mode": "evidence",
                        "policy_id": policy_id,
                        "policy_version": policy_version,
                        "evidence_checksum": evidence_checksum,
                        "champion_checksum": champion_checksum,
                        "candidate_spec_hash": candidate_spec_hash,
                        "dataset_id": dataset_id,
                        "dataset_checksum": dataset_checksum,
                        "settings_digest": settings_digest,
                        "previous_candidate_id": previous_id,
                        "promotion_id": int(promotion_id),
                    }
                )
                tx.execute(
                    """
                    INSERT INTO decisions (
                        candidate_id, decision, reason, champion_id, created_at, detail_json
                    ) VALUES (?, 'promoted', ?, ?, ?, ?)
                    """,
                    (candidate_id, reason_text, candidate_id, stamp, self._detail_json(payload)),
                )

                latest = _latest_accepted(tx, candidate_id, tokens)
                record = ChampionRecord(
                    scope=scope,
                    candidate=_candidate_from_row(candidate_row),
                    metrics={} if latest is None else latest.metrics,
                    promotion_id=int(promotion_id),
                    promoted_at=stamp,
                    reason=reason_text,
                    previous_candidate_id=previous_id,
                )
        except sqlite3.IntegrityError as exc:
            raise ArchiveError(f"evidence promotion of {candidate_id!r} failed: {exc}") from exc

        if record is None:  # pragma: no cover - the insert just committed
            raise ArchiveError("promotion record is missing")
        return record

    def champion(self, scope: str = DEFAULT_SCOPE) -> ChampionRecord | None:
        """The current champion of ``scope``, or ``None`` before the first
        promotion."""
        scope = _require_scope(scope)
        conn = self._connection()
        row = conn.execute("SELECT * FROM champions WHERE scope = ?", (scope,)).fetchone()
        if row is None:
            return None
        candidate_id = str(row["candidate_id"])
        candidate = self.candidate(candidate_id)
        if candidate is None:  # pragma: no cover - the foreign key prevents this
            raise ArchiveError(f"champion candidate {candidate_id!r} is missing")
        latest = _latest_accepted(conn, candidate_id, _scope_tokens(scope))
        promotion = conn.execute(
            "SELECT previous_candidate_id FROM champion_promotions WHERE id = ?",
            (int(row["promotion_id"]),),
        ).fetchone()
        if promotion is None:  # pragma: no cover - the foreign key prevents this
            raise ArchiveError(f"champion promotion {row['promotion_id']!r} is missing")
        previous = promotion["previous_candidate_id"]
        return ChampionRecord(
            scope=scope,
            candidate=candidate,
            metrics={} if latest is None else latest.metrics,
            promotion_id=int(row["promotion_id"]),
            promoted_at=str(row["promoted_at"]),
            reason=str(row["reason"]),
            previous_candidate_id=None if previous is None else str(previous),
        )

    def rollback(
        self, scope: str = DEFAULT_SCOPE, *, to_candidate_id: str, reason: str
    ) -> ChampionRecord:
        """Restore a candidate that was champion of ``scope`` before.

        Only a previous champion of the same scope can be restored: a candidate
        that never won is promoted, not rolled back to, and mixing the two would
        make the audit trail lie about what happened. The replaced champion stays
        in ``champion_promotions`` and a ``rolled_back`` decision is appended.
        """
        scope = _require_scope(scope)
        to_candidate_id = _require_text(to_candidate_id, "to_candidate_id")
        reason = _require_text(reason, "reason")
        tokens = _scope_tokens(scope)
        stamp = _now()

        record: ChampionRecord | None = None
        try:
            with self._transaction() as tx:
                known = tx.execute(
                    "SELECT 1 FROM candidates WHERE candidate_id = ?", (to_candidate_id,)
                ).fetchone()
                if known is None:
                    raise UnknownCandidateError(f"unknown candidate {to_candidate_id!r}")
                row = tx.execute("SELECT * FROM champions WHERE scope = ?", (scope,)).fetchone()
                if row is None:
                    raise ArchiveError(f"scope {scope!r} has no champion to roll back from")
                current = str(row["candidate_id"])
                if current == to_candidate_id:
                    raise ArchiveError(
                        f"candidate {to_candidate_id!r} is already the champion of {scope!r}"
                    )
                was_champion = tx.execute(
                    "SELECT 1 FROM champion_promotions WHERE scope = ? AND candidate_id = ? LIMIT 1",
                    (scope, to_candidate_id),
                ).fetchone()
                if was_champion is None:
                    raise ArchiveError(
                        f"candidate {to_candidate_id!r} was never the champion of {scope!r}; "
                        "promote it instead"
                    )

                reason_text = f"{reason}; rolled back from {current[:12]}"
                cursor = tx.execute(
                    """
                    INSERT INTO champion_promotions (
                        scope, kind, candidate_id, previous_candidate_id,
                        metric, value, previous_value, reason, created_at
                    ) VALUES (?, 'rolled_back', ?, ?, NULL, NULL, NULL, ?, ?)
                    """,
                    (scope, to_candidate_id, current, reason_text, stamp),
                )
                promotion_id = cursor.lastrowid
                if promotion_id is None:  # pragma: no cover - sqlite guarantees a rowid
                    raise ArchiveError("rollback insert returned no id")
                tx.execute(
                    """
                    INSERT INTO champions (scope, candidate_id, promotion_id, promoted_at, reason)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(scope) DO UPDATE SET
                        candidate_id = excluded.candidate_id,
                        promotion_id = excluded.promotion_id,
                        promoted_at = excluded.promoted_at,
                        reason = excluded.reason
                    """,
                    (scope, to_candidate_id, promotion_id, stamp, reason_text),
                )
                tx.execute(
                    """
                    INSERT INTO decisions (
                        candidate_id, decision, reason, champion_id, created_at, detail_json
                    ) VALUES (?, 'rolled_back', ?, ?, ?, ?)
                    """,
                    (
                        to_candidate_id,
                        reason_text,
                        to_candidate_id,
                        stamp,
                        self._detail_json(
                            {
                                "scope": scope,
                                "previous_candidate_id": current,
                                "promotion_id": int(promotion_id),
                            }
                        ),
                    ),
                )
                candidate_row = tx.execute(
                    "SELECT * FROM candidates WHERE candidate_id = ?", (to_candidate_id,)
                ).fetchone()
                if candidate_row is None:  # pragma: no cover - checked above
                    raise ArchiveError(f"candidate {to_candidate_id!r} disappeared")
                latest = _latest_accepted(tx, to_candidate_id, tokens)
                record = ChampionRecord(
                    scope=scope,
                    candidate=_candidate_from_row(candidate_row),
                    metrics={} if latest is None else latest.metrics,
                    promotion_id=int(promotion_id),
                    promoted_at=stamp,
                    reason=reason_text,
                    previous_candidate_id=current,
                )
        except sqlite3.IntegrityError as exc:
            raise ArchiveError(f"rollback to {to_candidate_id!r} failed: {exc}") from exc

        if record is None:  # pragma: no cover - the insert just committed
            raise ArchiveError("rollback record is missing")
        return record

    def champion_history(self, scope: str = DEFAULT_SCOPE) -> tuple[ChampionPromotion, ...]:
        """Every champion change of ``scope``, oldest first, rollbacks included."""
        scope = _require_scope(scope)
        rows = (
            self._connection()
            .execute("SELECT * FROM champion_promotions WHERE scope = ? ORDER BY id", (scope,))
            .fetchall()
        )
        return tuple(_promotion_from_row(row) for row in rows)

    def compare_to_champion(
        self,
        candidate_id: str,
        scope: str = DEFAULT_SCOPE,
        metric: str = "expectancy_r",
    ) -> ChampionComparison:
        """This candidate's ``metric`` against the champion's, without promoting.

        A comparison is a read: it refuses an unknown candidate and reports
        missing values as ``None`` with ``better=False``. Lower-is-better metrics
        are not supported here; promoting on one would invert the rule.
        """
        candidate_id = _require_text(candidate_id, "candidate_id")
        scope = _require_scope(scope)
        metric = _require_metric(metric)
        conn = self._connection()
        if self.candidate(candidate_id) is None:
            raise UnknownCandidateError(f"unknown candidate {candidate_id!r}")
        tokens = _scope_tokens(scope)

        champion_id: str | None = None
        champion_value: float | None = None
        row = conn.execute(
            "SELECT candidate_id FROM champions WHERE scope = ?", (scope,)
        ).fetchone()
        if row is not None:
            champion_id = str(row["candidate_id"])
            champion_value = _metric_value(conn, champion_id, tokens, metric)
        candidate_value = _metric_value(conn, candidate_id, tokens, metric)
        delta = (
            None
            if candidate_value is None or champion_value is None
            else candidate_value - champion_value
        )
        better = (
            candidate_value is not None
            and champion_value is not None
            and candidate_value > champion_value
        )
        return ChampionComparison(
            scope=scope,
            metric=metric,
            candidate_id=candidate_id,
            candidate_value=candidate_value,
            champion_id=champion_id,
            champion_value=champion_value,
            delta=delta,
            better=better,
        )
