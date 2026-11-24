"""One-shot locked final-holdout protocol with a durable access ledger.

The final test split is the only genuinely unseen sample the project owns. It is
worth exactly one evaluation: after a candidate has been selected, measured on
walk-forward folds and accepted by the promotion policy, the holdout can be
unsealed once. Re-running the same candidate, a different candidate or a
"small fix" on the same dataset turns the holdout into another validation slice
and destroys its independence.

This module enforces that protocol in three layers:

* :class:`HoldoutLedger` is a small SQLite ledger. A holdout is keyed by
  ``(dataset_id, dataset_checksum)`` and starts ``sealed``. The append-only
  ``access_log`` records every grant and every refusal with actor, time and
  reason.
* :meth:`HoldoutLedger.unlock` accepts only an **accepted**
  :class:`~scalping.discovery.promotion.PromotionDecision` whose dataset,
  checksum, spec, code version and settings digest match the sealed holdout. It
  atomically marks the row ``spent`` and returns a single-use token.
* :func:`evaluate_holdout` consumes that token, slices the manifest's test rows,
  runs the DSL candidate **once** through the existing
  :class:`~scalping.discovery.evaluate.DslEvaluator` with the configured costs and
  stress multipliers, and writes an immutable canonical result into
  ``holdouts.result_json``. There is no code path that recomputes it.

A refused attempt never reads the test split: the ledger only knows identities,
and the feature frame is not touched until the token has been validated and the
evaluation has been claimed. ``attach_holdout`` returns a **new**
:class:`~scalping.discovery.evidence.CandidateEvidence` with a ``locked_oos``
section; the frozen input bundle is left untouched. The promotion policy already
recognises that section through ``require_locked_oos``.

Honest limits
-------------
One holdout result is one sample. The protocol prevents reuse and records who
spent it; it does not add multiple-testing correction, turn one pass into a
profitability claim, or replace paper trading. A crash after the evaluation is
claimed leaves the holdout spent without a result: that is the one-shot rule,
not a bug to retry around.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import sqlite3
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, NoReturn, Self

import numpy as np

from scalping.config import Settings
from scalping.datasets.builder import DatasetBundle
from scalping.discovery.evaluate import DslEvaluator
from scalping.discovery.evidence import (
    CandidateEvidence,
    EvidenceError,
    RegimeEvidence,
    StressEvidence,
    bootstrap_ci,
    per_trade_regimes,
    regime_labels,
    regime_rows,
    settings_digest,
    slice_frame,
    verify_evidence,
)
from scalping.discovery.promotion import (
    PromotionDecision,
    PromotionPolicy,
    is_sealed_decision,
)
from scalping.discovery.protocols import canonical_json
from scalping.features.build import FeatureFrame
from scalping.regime.strategy import UNKNOWN_REGIME
from scalping.strategy.dsl import DslValidationError, StrategySpec

__all__ = [
    "HOLDOUT_SCHEMA_VERSION",
    "AccessEntry",
    "HoldoutConfig",
    "HoldoutError",
    "HoldoutLedger",
    "HoldoutRecord",
    "HoldoutSpec",
    "HoldoutSpent",
    "HoldoutToken",
    "LockedHoldoutResult",
    "attach_holdout",
    "evaluate_holdout",
    "verify_holdout_result",
]

#: Layout version of :meth:`LockedHoldoutResult.to_dict`; version 2 records the
#: sealed decision's policy digest alongside the id/version.
HOLDOUT_SCHEMA_VERSION: Final[int] = 2

_WAL_ATTEMPTS: Final[int] = 5
_WAL_BACKOFF_SECONDS: Final[float] = 0.05

_SCHEMA_VERSION_DDL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
)
"""

_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """
        CREATE TABLE IF NOT EXISTS holdouts (
            dataset_id TEXT NOT NULL,
            dataset_checksum TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('sealed', 'spent')),
            spec_hash TEXT NOT NULL,
            code_version TEXT NOT NULL,
            settings_digest TEXT NOT NULL,
            policy_id TEXT,
            policy_version INTEGER,
            policy_digest TEXT,
            evidence_checksum TEXT,
            decision_json TEXT,
            sealed_at TEXT NOT NULL,
            sealed_by TEXT NOT NULL,
            spent_at TEXT,
            unlocked_by TEXT,
            unlock_token TEXT,
            evaluation_started_at TEXT,
            result_json TEXT,
            result_checksum TEXT,
            PRIMARY KEY (dataset_id, dataset_checksum)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS access_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            action TEXT NOT NULL,
            outcome TEXT NOT NULL CHECK (outcome IN ('granted', 'refused')),
            dataset_id TEXT,
            dataset_checksum TEXT,
            actor TEXT NOT NULL,
            reason TEXT NOT NULL,
            detail TEXT
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS access_log_dataset
        ON access_log (dataset_id, dataset_checksum, id)
        """,
        """
        CREATE TRIGGER IF NOT EXISTS access_log_no_update
        BEFORE UPDATE ON access_log
        BEGIN
            SELECT RAISE(ABORT, 'access_log is append-only');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS access_log_no_delete
        BEFORE DELETE ON access_log
        BEGIN
            SELECT RAISE(ABORT, 'access_log is append-only');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS holdouts_no_reopen
        BEFORE UPDATE OF status ON holdouts
        WHEN OLD.status = 'spent' AND NEW.status <> 'spent'
        BEGIN
            SELECT RAISE(ABORT, 'a spent holdout cannot be reopened');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS holdouts_result_immutable
        BEFORE UPDATE OF result_json ON holdouts
        WHEN OLD.result_json IS NOT NULL AND NEW.result_json IS NOT OLD.result_json
        BEGIN
            SELECT RAISE(ABORT, 'a holdout result cannot be replaced');
        END
        """,
    ),
}


class HoldoutError(ValueError):
    """A refused holdout operation; ``code`` is the machine-readable reason."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class HoldoutSpent(HoldoutError):
    """The ``(dataset_id, dataset_checksum)`` holdout has already been spent."""

    def __init__(self, message: str) -> None:
        super().__init__("holdout_spent", message)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _finite(value: object, *, where: str) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise HoldoutError("non_finite_holdout", f"{where} is not finite ({number!r})")
    return number


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


# --- immutable identity and results -------------------------------------------


@dataclass(frozen=True, slots=True)
class HoldoutSpec:
    """The identity sealed before any test row is read.

    Every field is part of what makes this holdout unique. A later unlock must
    present a promotion decision with the same dataset, checksum, spec, code
    version and settings digest; the symbol/timeframe are checked against the
    evaluation config as well.
    """

    dataset_id: str
    dataset_checksum: str
    symbol: str
    timeframe: str
    spec_hash: str
    code_version: str
    settings_digest: str

    def __post_init__(self) -> None:
        for name in (
            "dataset_id",
            "dataset_checksum",
            "symbol",
            "timeframe",
            "spec_hash",
            "code_version",
            "settings_digest",
        ):
            _text(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class AccessEntry:
    """One append-only access-log row."""

    id: int
    ts: str
    action: str
    outcome: str
    dataset_id: str | None
    dataset_checksum: str | None
    actor: str
    reason: str
    detail: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HoldoutToken:
    """The single-use capability returned by :meth:`HoldoutLedger.unlock`.

    ``token`` is the secret. The remaining fields repeat the locked identity so
    :func:`evaluate_holdout` can reject a token presented for the wrong holdout
    before the database is queried.
    """

    token: str
    dataset_id: str
    dataset_checksum: str
    symbol: str
    timeframe: str
    spec_hash: str
    code_version: str
    settings_digest: str
    policy_id: str
    policy_version: int
    policy_digest: str
    evidence_checksum: str

    def __post_init__(self) -> None:
        for name in (
            "token",
            "dataset_id",
            "dataset_checksum",
            "symbol",
            "timeframe",
            "spec_hash",
            "code_version",
            "settings_digest",
            "policy_id",
            "policy_digest",
            "evidence_checksum",
        ):
            _text(getattr(self, name), name)
        if isinstance(self.policy_version, bool) or not isinstance(self.policy_version, int):
            raise TypeError("policy_version must be an int")
        if self.policy_version < 1:
            raise ValueError("policy_version must be >= 1")


@dataclass(frozen=True, slots=True)
class HoldoutRecord:
    """One row of the ``holdouts`` table."""

    dataset_id: str
    dataset_checksum: str
    symbol: str
    timeframe: str
    status: str
    spec_hash: str
    code_version: str
    settings_digest: str
    policy_id: str | None
    policy_version: int | None
    policy_digest: str | None
    evidence_checksum: str | None
    decision_json: str | None
    sealed_at: str
    sealed_by: str
    spent_at: str | None
    unlocked_by: str | None
    unlock_token: str | None
    evaluation_started_at: str | None
    result_json: str | None
    result_checksum: str | None

    @property
    def result(self) -> LockedHoldoutResult | None:
        """Parse the stored result, or ``None`` while the holdout is unspent."""
        if self.result_json is None:
            return None
        return LockedHoldoutResult.from_dict(json.loads(self.result_json))


@dataclass(frozen=True, slots=True)
class LockedHoldoutResult:
    """The single, immutable result of spending one locked holdout.

    The checksum covers :meth:`to_dict` without ``checksum`` and ``created_at``,
    so re-serialising the same numbers is stable while the creation stamp may
    differ. Identity fields bind the result to the dataset build, candidate,
    code version, execution settings and the promotion evidence that unlocked
    it.
    """

    dataset_id: str
    dataset_checksum: str
    symbol: str
    timeframe: str
    spec_hash: str
    spec_version: int
    complexity: int
    code_version: str
    settings_digest: str
    policy_id: str
    policy_version: int
    policy_digest: str
    evidence_checksum: str
    test_start_row: int
    test_end_row: int
    test_rows: int
    test_start_ts: str
    test_end_ts: str
    trades: int
    net_profit: float
    profit_factor: float
    win_rate: float
    expectancy_r: float
    total_r: float
    max_drawdown: float
    regimes: tuple[RegimeEvidence, ...]
    stress: tuple[StressEvidence, ...]
    expectancy_r_ci95_low: float
    expectancy_r_ci95_high: float
    total_r_ci95_low: float
    total_r_ci95_high: float
    bootstrap_samples: int
    bootstrap_confidence: float
    bootstrap_method: str
    regime_coverage: float
    checksum: str
    created_at: str
    schema_version: int = HOLDOUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "dataset_id",
            "dataset_checksum",
            "symbol",
            "timeframe",
            "spec_hash",
            "code_version",
            "settings_digest",
            "policy_id",
            "policy_digest",
            "evidence_checksum",
            "test_start_ts",
            "test_end_ts",
            "created_at",
        ):
            _text(getattr(self, name), name)
        for name in ("spec_version", "complexity", "test_start_row", "test_end_row"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("test_rows", "trades", "bootstrap_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in (
            "net_profit",
            "profit_factor",
            "win_rate",
            "expectancy_r",
            "total_r",
            "max_drawdown",
            "expectancy_r_ci95_low",
            "expectancy_r_ci95_high",
            "total_r_ci95_low",
            "total_r_ci95_high",
            "bootstrap_confidence",
            "regime_coverage",
        ):
            number = float(getattr(self, name))
            if not math.isfinite(number):
                raise HoldoutError("non_finite_holdout", f"{name} is not finite")
        if isinstance(self.policy_version, bool) or not isinstance(self.policy_version, int):
            raise TypeError("policy_version must be an int")
        if self.policy_version < 1:
            raise ValueError("policy_version must be >= 1")
        if not isinstance(self.regimes, tuple):
            object.__setattr__(self, "regimes", tuple(self.regimes))
        if not isinstance(self.stress, tuple):
            object.__setattr__(self, "stress", tuple(self.stress))
        if self.schema_version != HOLDOUT_SCHEMA_VERSION:
            raise HoldoutError(
                "schema_mismatch",
                f"holdout result schema {self.schema_version} is not {HOLDOUT_SCHEMA_VERSION}",
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LockedHoldoutResult:
        data = dict(payload)
        data["regimes"] = tuple(
            row if isinstance(row, RegimeEvidence) else RegimeEvidence(**row)
            for row in data.get("regimes", ())
        )
        data["stress"] = tuple(
            row if isinstance(row, StressEvidence) else StressEvidence(**row)
            for row in data.get("stress", ())
        )
        return cls(**data)


def _result_checksum(result: LockedHoldoutResult) -> str:
    body = result.to_dict()
    body.pop("checksum", None)
    body.pop("created_at", None)
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


# --- evaluation configuration -------------------------------------------------


@dataclass(frozen=True, slots=True)
class HoldoutConfig:
    """Inputs the one-shot evaluator needs beyond the sealed identity.

    ``features`` is required only when the data argument is a
    :class:`DatasetBundle`: the bundle carries labels and split rows, not the
    OHLC/spread/ATR arrays the simulator prices fills with. When the caller
    already holds the exact test slice as a :class:`FeatureFrame`, that frame is
    used directly and ``features`` may stay ``None``.

    ``cost_stress`` must contain at least one multiplier above ``1.0``. A
    no-op stress would let a result claim cost resilience it never measured.
    """

    code_version: str
    symbol: str = "XAUUSD"
    timeframe: str = "M1"
    window: int = 60
    cost_stress: tuple[float, ...] = (1.5, 2.0)
    bootstrap_samples: int = 1000
    bootstrap_confidence: float = 0.95
    seed: int = 0
    settings: Settings | None = None
    features: FeatureFrame | None = None
    dataset_id: str | None = None
    dataset_checksum: str | None = None

    def __post_init__(self) -> None:
        for name in ("code_version", "symbol", "timeframe"):
            _text(getattr(self, name), name)
        for name in ("window", "bootstrap_samples", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if self.bootstrap_samples < 1:
            raise ValueError(f"bootstrap_samples must be >= 1, got {self.bootstrap_samples}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if (
            not math.isfinite(float(self.bootstrap_confidence))
            or not 0.0 < float(self.bootstrap_confidence) < 1.0
        ):
            raise ValueError("bootstrap_confidence must be in (0, 1)")
        if self.settings is not None and not isinstance(self.settings, Settings):
            raise TypeError("settings must be a Settings or None")
        if self.features is not None and not isinstance(self.features, FeatureFrame):
            raise TypeError("features must be a FeatureFrame or None")
        for name in ("dataset_id", "dataset_checksum"):
            value = getattr(self, name)
            if value is not None:
                _text(value, name)

        multipliers: list[float] = []
        for raw in self.cost_stress:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(f"cost_stress multipliers must be numbers, got {raw!r}")
            number = float(raw)
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"cost_stress multipliers must be finite and > 0, got {raw!r}")
            multipliers.append(number)
        labels = [f"x{value:g}" for value in multipliers]
        if len(set(labels)) != len(labels):
            raise ValueError("cost_stress multipliers must be distinct")
        if not multipliers or not any(value > 1.0 for value in multipliers):
            raise ValueError("cost_stress must contain at least one multiplier above 1.0")
        object.__setattr__(self, "cost_stress", tuple(multipliers))


# --- ledger -------------------------------------------------------------------


def _row_to_record(row: sqlite3.Row) -> HoldoutRecord:
    return HoldoutRecord(
        dataset_id=str(row["dataset_id"]),
        dataset_checksum=str(row["dataset_checksum"]),
        symbol=str(row["symbol"]),
        timeframe=str(row["timeframe"]),
        status=str(row["status"]),
        spec_hash=str(row["spec_hash"]),
        code_version=str(row["code_version"]),
        settings_digest=str(row["settings_digest"]),
        policy_id=None if row["policy_id"] is None else str(row["policy_id"]),
        policy_version=None if row["policy_version"] is None else int(row["policy_version"]),
        policy_digest=None if row["policy_digest"] is None else str(row["policy_digest"]),
        evidence_checksum=(
            None if row["evidence_checksum"] is None else str(row["evidence_checksum"])
        ),
        decision_json=None if row["decision_json"] is None else str(row["decision_json"]),
        sealed_at=str(row["sealed_at"]),
        sealed_by=str(row["sealed_by"]),
        spent_at=None if row["spent_at"] is None else str(row["spent_at"]),
        unlocked_by=None if row["unlocked_by"] is None else str(row["unlocked_by"]),
        unlock_token=None if row["unlock_token"] is None else str(row["unlock_token"]),
        evaluation_started_at=(
            None if row["evaluation_started_at"] is None else str(row["evaluation_started_at"])
        ),
        result_json=None if row["result_json"] is None else str(row["result_json"]),
        result_checksum=(None if row["result_checksum"] is None else str(row["result_checksum"])),
    )


def _row_to_access(row: sqlite3.Row) -> AccessEntry:
    return AccessEntry(
        id=int(row["id"]),
        ts=str(row["ts"]),
        action=str(row["action"]),
        outcome=str(row["outcome"]),
        dataset_id=None if row["dataset_id"] is None else str(row["dataset_id"]),
        dataset_checksum=(
            None if row["dataset_checksum"] is None else str(row["dataset_checksum"])
        ),
        actor=str(row["actor"]),
        reason=str(row["reason"]),
        detail=None if row["detail"] is None else str(row["detail"]),
    )


class HoldoutLedger:
    """A local SQLite ledger for sealed, spent and refused holdouts.

    Not thread-safe: a connection belongs to the thread that opened it, matching
    :class:`~scalping.discovery.archive.DiscoveryArchive`. Concurrent processes
    each open their own ledger; the conditional unlock update is what makes one
    of them the winner. WAL, ``busy_timeout``, ``synchronous = NORMAL`` and
    connection-scoped foreign keys follow the registry/archive conventions.
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

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection()

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise HoldoutError("holdout_ledger_closed", "the holdout ledger is closed")
        return self._conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connection()
        if conn.in_transaction:
            raise HoldoutError("holdout_ledger_busy", "nested holdout transaction")
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            raise HoldoutError(
                "holdout_ledger_busy", f"could not acquire the ledger write lock: {exc}"
            ) from exc
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
        if current > HOLDOUT_SCHEMA_VERSION:
            raise HoldoutError(
                "holdout_schema_too_new",
                f"ledger schema {current} is newer than supported {HOLDOUT_SCHEMA_VERSION}",
            )
        if current == HOLDOUT_SCHEMA_VERSION:
            return current
        with self._transaction() as tx:
            tx.execute(_SCHEMA_VERSION_DDL)
            row = tx.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current = 0 if row is None or row[0] is None else int(row[0])
            if current > HOLDOUT_SCHEMA_VERSION:
                raise HoldoutError(
                    "holdout_schema_too_new",
                    f"ledger schema {current} is newer than supported {HOLDOUT_SCHEMA_VERSION}",
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

    # --- access log -------------------------------------------------------

    def _append_access(
        self,
        tx: sqlite3.Connection,
        *,
        action: str,
        outcome: str,
        actor: str,
        reason: str,
        dataset_id: str | None,
        dataset_checksum: str | None,
        detail: str | None = None,
    ) -> None:
        tx.execute(
            """
            INSERT INTO access_log
                (ts, action, outcome, dataset_id, dataset_checksum, actor, reason, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                action,
                outcome,
                dataset_id,
                dataset_checksum,
                actor,
                reason,
                detail,
            ),
        )

    def _log(
        self,
        *,
        action: str,
        outcome: str,
        actor: str,
        reason: str,
        dataset_id: str | None,
        dataset_checksum: str | None,
        detail: str | None = None,
    ) -> None:
        with self._transaction() as tx:
            self._append_access(
                tx,
                action=action,
                outcome=outcome,
                actor=actor,
                reason=reason,
                dataset_id=dataset_id,
                dataset_checksum=dataset_checksum,
                detail=detail,
            )

    def _refuse(
        self,
        *,
        action: str,
        actor: str,
        reason: str,
        dataset_id: str | None,
        dataset_checksum: str | None,
        detail: str | None = None,
        error_type: type[HoldoutError] = HoldoutError,
    ) -> NoReturn:
        self._log(
            action=action,
            outcome="refused",
            actor=actor,
            reason=reason,
            dataset_id=dataset_id,
            dataset_checksum=dataset_checksum,
            detail=detail,
        )
        if error_type is HoldoutSpent:
            raise HoldoutSpent(detail or reason)
        raise error_type(reason, detail or reason)

    def access_entries(
        self,
        *,
        dataset_id: str | None = None,
        dataset_checksum: str | None = None,
    ) -> tuple[AccessEntry, ...]:
        """Append-only log, oldest first, optionally filtered to one holdout."""
        clauses: list[str] = []
        params: list[Any] = []
        if dataset_id is not None:
            clauses.append("dataset_id = ?")
            params.append(dataset_id)
        if dataset_checksum is not None:
            clauses.append("dataset_checksum = ?")
            params.append(dataset_checksum)
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        rows = (
            self._connection()
            .execute(f"SELECT * FROM access_log{where} ORDER BY id", tuple(params))
            .fetchall()
        )
        return tuple(_row_to_access(row) for row in rows)

    # --- reads ------------------------------------------------------------

    def get(self, dataset_id: str, dataset_checksum: str) -> HoldoutRecord | None:
        row = (
            self._connection()
            .execute(
                "SELECT * FROM holdouts WHERE dataset_id = ? AND dataset_checksum = ?",
                (dataset_id, dataset_checksum),
            )
            .fetchone()
        )
        return None if row is None else _row_to_record(row)

    def all_holdouts(self) -> tuple[HoldoutRecord, ...]:
        rows = (
            self._connection()
            .execute("SELECT * FROM holdouts ORDER BY sealed_at, dataset_id, dataset_checksum")
            .fetchall()
        )
        return tuple(_row_to_record(row) for row in rows)

    def _find_by_token(self, digest: str) -> HoldoutRecord | None:
        row = (
            self._connection()
            .execute(
                "SELECT * FROM holdouts WHERE unlock_token = ?",
                (digest,),
            )
            .fetchone()
        )
        return None if row is None else _row_to_record(row)

    # --- seal / unlock ----------------------------------------------------

    def seal(
        self,
        holdout: HoldoutSpec,
        *,
        actor: str = "system",
        reason: str = "final holdout sealed",
        sealed_at: str | None = None,
    ) -> HoldoutRecord:
        """Register a sealed holdout; an identical repeat is idempotent.

        A conflicting seal for the same ``(dataset_id, dataset_checksum)`` is
        refused: the key identifies one dataset build, and a later candidate
        must not silently replace the identity that was sealed first. A new
        dataset version/checksum is a new key and therefore a new seal.
        """
        if not isinstance(holdout, HoldoutSpec):
            raise TypeError(f"holdout must be a HoldoutSpec, got {type(holdout).__name__}")
        actor = _text(actor, "actor")
        reason = _text(reason, "reason")
        stamp = _now() if sealed_at is None else _text(sealed_at, "sealed_at")

        existing = self.get(holdout.dataset_id, holdout.dataset_checksum)
        if existing is not None:
            if (
                existing.symbol,
                existing.timeframe,
                existing.spec_hash,
                existing.code_version,
                existing.settings_digest,
            ) == (
                holdout.symbol,
                holdout.timeframe,
                holdout.spec_hash,
                holdout.code_version,
                holdout.settings_digest,
            ):
                self._log(
                    action="seal",
                    outcome="granted",
                    actor=actor,
                    reason=reason,
                    dataset_id=holdout.dataset_id,
                    dataset_checksum=holdout.dataset_checksum,
                    detail="already sealed with the same identity",
                )
                return existing
            self._refuse(
                action="seal",
                actor=actor,
                reason="holdout_conflict",
                dataset_id=holdout.dataset_id,
                dataset_checksum=holdout.dataset_checksum,
                detail="a different identity is already sealed for this dataset build",
            )

        try:
            with self._transaction() as tx:
                tx.execute(
                    """
                    INSERT INTO holdouts
                        (dataset_id, dataset_checksum, symbol, timeframe, status,
                         spec_hash, code_version, settings_digest, sealed_at, sealed_by)
                    VALUES (?, ?, ?, ?, 'sealed', ?, ?, ?, ?, ?)
                    """,
                    (
                        holdout.dataset_id,
                        holdout.dataset_checksum,
                        holdout.symbol,
                        holdout.timeframe,
                        holdout.spec_hash,
                        holdout.code_version,
                        holdout.settings_digest,
                        stamp,
                        actor,
                    ),
                )
                self._append_access(
                    tx,
                    action="seal",
                    outcome="granted",
                    actor=actor,
                    reason=reason,
                    dataset_id=holdout.dataset_id,
                    dataset_checksum=holdout.dataset_checksum,
                    detail=None,
                )
        except sqlite3.IntegrityError:
            # Another process won the insert between our read and write. Re-read
            # and apply the same idempotent/conflict rule.
            concurrent = self.get(holdout.dataset_id, holdout.dataset_checksum)
            if concurrent is not None and (
                concurrent.spec_hash,
                concurrent.code_version,
                concurrent.settings_digest,
            ) == (holdout.spec_hash, holdout.code_version, holdout.settings_digest):
                return concurrent
            self._refuse(
                action="seal",
                actor=actor,
                reason="holdout_conflict",
                dataset_id=holdout.dataset_id,
                dataset_checksum=holdout.dataset_checksum,
                detail="a concurrent different identity is sealed for this dataset build",
            )

        record = self.get(holdout.dataset_id, holdout.dataset_checksum)
        if record is None:  # pragma: no cover - the insert above must have written it
            raise HoldoutError("holdout_not_sealed", "seal insert did not persist")
        return record

    def unlock(
        self,
        holdout: HoldoutSpec,
        *,
        decision: PromotionDecision,
        policy: PromotionPolicy,
        actor: str = "system",
        reason: str = "promotion evidence accepted",
    ) -> HoldoutToken:
        """Spend the holdout once, if and only if the evidence gate passed.

        The checks run in this order: the row must exist and still be sealed
        (a spent row refuses with :class:`HoldoutSpent` even for a different
        spec), the decision must be an accepted :class:`PromotionDecision` from
        the supplied policy version, and its dataset/checksum/spec/code/settings
        identity must match the sealed holdout. The status transition is one
        conditional ``UPDATE``; concurrent callers therefore produce exactly one
        winner.
        """
        if not isinstance(holdout, HoldoutSpec):
            raise TypeError(f"holdout must be a HoldoutSpec, got {type(holdout).__name__}")
        actor = _text(actor, "actor")
        reason = _text(reason, "reason")
        if not isinstance(decision, PromotionDecision):
            self._refuse(
                action="unlock",
                actor=actor,
                reason="invalid_decision",
                dataset_id=holdout.dataset_id,
                dataset_checksum=holdout.dataset_checksum,
                detail=f"expected PromotionDecision, got {type(decision).__name__}",
            )
        if not is_sealed_decision(decision):
            self._refuse(
                action="unlock",
                actor=actor,
                reason="invalid_decision",
                dataset_id=holdout.dataset_id,
                dataset_checksum=holdout.dataset_checksum,
                detail="the decision is not sealed by evaluate_promotion",
            )
        if not isinstance(policy, PromotionPolicy):
            self._refuse(
                action="unlock",
                actor=actor,
                reason="invalid_policy",
                dataset_id=holdout.dataset_id,
                dataset_checksum=holdout.dataset_checksum,
                detail=f"expected PromotionPolicy, got {type(policy).__name__}",
            )

        record = self.get(holdout.dataset_id, holdout.dataset_checksum)
        if record is None:
            self._refuse(
                action="unlock",
                actor=actor,
                reason="holdout_not_sealed",
                dataset_id=holdout.dataset_id,
                dataset_checksum=holdout.dataset_checksum,
                detail="no sealed holdout exists for this dataset build",
            )
        if record.status == "spent":
            self._refuse(
                action="unlock",
                actor=actor,
                reason="holdout_spent",
                dataset_id=record.dataset_id,
                dataset_checksum=record.dataset_checksum,
                detail=(
                    f"holdout was already spent at {record.spent_at}; a second spec cannot reuse it"
                ),
                error_type=HoldoutSpent,
            )

        if (
            record.symbol,
            record.timeframe,
            record.spec_hash,
            record.code_version,
            record.settings_digest,
        ) != (
            holdout.symbol,
            holdout.timeframe,
            holdout.spec_hash,
            holdout.code_version,
            holdout.settings_digest,
        ):
            self._refuse(
                action="unlock",
                actor=actor,
                reason="holdout_identity_mismatch",
                dataset_id=record.dataset_id,
                dataset_checksum=record.dataset_checksum,
                detail="the requested spec/code/settings do not match the sealed identity",
            )

        if not decision.accepted:
            self._refuse(
                action="unlock",
                actor=actor,
                reason="decision_rejected",
                dataset_id=record.dataset_id,
                dataset_checksum=record.dataset_checksum,
                detail=f"policy {decision.policy_id} v{decision.version}: {decision.summary}",
            )
        if decision.policy_id != policy.policy_id or decision.version != policy.version:
            self._refuse(
                action="unlock",
                actor=actor,
                reason="decision_policy_mismatch",
                dataset_id=record.dataset_id,
                dataset_checksum=record.dataset_checksum,
                detail=(
                    f"decision is {decision.policy_id} v{decision.version}; "
                    f"caller supplied {policy.policy_id} v{policy.version}"
                ),
            )
        if decision.policy_digest != policy.digest:
            self._refuse(
                action="unlock",
                actor=actor,
                reason="decision_policy_mismatch",
                dataset_id=record.dataset_id,
                dataset_checksum=record.dataset_checksum,
                detail=(
                    "the decision was sealed by a different policy payload; "
                    "re-evaluate it with the policy being applied"
                ),
            )

        expected = (
            holdout.dataset_id,
            holdout.dataset_checksum,
            holdout.spec_hash,
            holdout.code_version,
            holdout.settings_digest,
        )
        actual = (
            decision.dataset_id,
            decision.dataset_checksum,
            decision.candidate_spec_hash,
            decision.code_version,
            decision.settings_digest,
        )
        if actual != expected:
            self._refuse(
                action="unlock",
                actor=actor,
                reason="decision_identity_mismatch",
                dataset_id=record.dataset_id,
                dataset_checksum=record.dataset_checksum,
                detail=(
                    "decision identity does not match the sealed holdout; "
                    f"expected {expected!r}, got {actual!r}"
                ),
            )
        if (
            not isinstance(decision.evidence_checksum, str)
            or not decision.evidence_checksum.strip()
        ):
            self._refuse(
                action="unlock",
                actor=actor,
                reason="decision_identity_mismatch",
                dataset_id=record.dataset_id,
                dataset_checksum=record.dataset_checksum,
                detail="an accepted decision has no usable evidence checksum",
            )

        try:
            decision_payload = canonical_json(decision.to_dict())
        except (TypeError, ValueError) as exc:
            self._refuse(
                action="unlock",
                actor=actor,
                reason="invalid_decision",
                dataset_id=record.dataset_id,
                dataset_checksum=record.dataset_checksum,
                detail=f"the decision payload is not canonical JSON: {exc}",
            )

        token = secrets.token_urlsafe(32)
        digest = _token_digest(token)
        unlock_token = HoldoutToken(
            token=token,
            dataset_id=record.dataset_id,
            dataset_checksum=record.dataset_checksum,
            symbol=record.symbol,
            timeframe=record.timeframe,
            spec_hash=record.spec_hash,
            code_version=record.code_version,
            settings_digest=record.settings_digest,
            policy_id=decision.policy_id,
            policy_version=decision.version,
            policy_digest=decision.policy_digest,
            evidence_checksum=decision.evidence_checksum,
        )
        stamp = _now()
        with self._transaction() as tx:
            cursor = tx.execute(
                """
                UPDATE holdouts
                   SET status = 'spent',
                       policy_id = ?,
                       policy_version = ?,
                       policy_digest = ?,
                       evidence_checksum = ?,
                       decision_json = ?,
                       spent_at = ?,
                       unlocked_by = ?,
                       unlock_token = ?
                 WHERE dataset_id = ?
                   AND dataset_checksum = ?
                   AND status = 'sealed'
                   AND result_json IS NULL
                """,
                (
                    decision.policy_id,
                    decision.version,
                    decision.policy_digest,
                    decision.evidence_checksum,
                    decision_payload,
                    stamp,
                    actor,
                    digest,
                    record.dataset_id,
                    record.dataset_checksum,
                ),
            )
            won = cursor.rowcount == 1
            if won:
                self._append_access(
                    tx,
                    action="unlock",
                    outcome="granted",
                    actor=actor,
                    reason=reason,
                    dataset_id=record.dataset_id,
                    dataset_checksum=record.dataset_checksum,
                    detail=f"policy {decision.policy_id} v{decision.version}",
                )

        if not won:
            fresh = self.get(record.dataset_id, record.dataset_checksum)
            if fresh is not None and fresh.status == "spent":
                self._refuse(
                    action="unlock",
                    actor=actor,
                    reason="holdout_spent",
                    dataset_id=record.dataset_id,
                    dataset_checksum=record.dataset_checksum,
                    detail="a concurrent unlock already spent this holdout",
                    error_type=HoldoutSpent,
                )
            self._refuse(
                action="unlock",
                actor=actor,
                reason="holdout_unlock_conflict",
                dataset_id=record.dataset_id,
                dataset_checksum=record.dataset_checksum,
                detail="the holdout could not be transitioned from sealed to spent",
            )

        return unlock_token

    # --- evaluation claim / result persistence ----------------------------

    def _begin_evaluation(self, record: HoldoutRecord, digest: str, *, actor: str) -> None:
        """Atomically claim the one evaluation slot for a spent holdout."""
        stamp = _now()
        with self._transaction() as tx:
            cursor = tx.execute(
                """
                UPDATE holdouts
                   SET evaluation_started_at = ?
                 WHERE dataset_id = ?
                   AND dataset_checksum = ?
                   AND status = 'spent'
                   AND result_json IS NULL
                   AND evaluation_started_at IS NULL
                   AND unlock_token = ?
                """,
                (stamp, record.dataset_id, record.dataset_checksum, digest),
            )
            won = cursor.rowcount == 1
        if won:
            return
        fresh = self.get(record.dataset_id, record.dataset_checksum)
        if fresh is not None and fresh.result_json is not None:
            self._refuse(
                action="evaluate",
                actor=actor,
                reason="holdout_result_exists",
                dataset_id=record.dataset_id,
                dataset_checksum=record.dataset_checksum,
                detail="the one-shot result is already stored and cannot be recomputed",
            )
        if fresh is not None and fresh.unlock_token != digest:
            self._refuse(
                action="evaluate",
                actor=actor,
                reason="holdout_invalid_token",
                dataset_id=record.dataset_id,
                dataset_checksum=record.dataset_checksum,
                detail="the token does not belong to this holdout or was already consumed",
            )
        if fresh is not None and fresh.evaluation_started_at is not None:
            self._refuse(
                action="evaluate",
                actor=actor,
                reason="holdout_evaluation_in_progress",
                dataset_id=record.dataset_id,
                dataset_checksum=record.dataset_checksum,
                detail="another evaluation already claimed this holdout",
            )
        self._refuse(
            action="evaluate",
            actor=actor,
            reason="holdout_evaluation_unavailable",
            dataset_id=record.dataset_id,
            dataset_checksum=record.dataset_checksum,
            detail="the holdout is not spent or is not available for evaluation",
        )

    def _invalidate_token(self, record: HoldoutRecord, digest: str) -> None:
        """Burn the token after a failed evaluation.

        The holdout is already spent at this point. Clearing the token means a
        retry cannot look like an in-progress evaluation forever: it is refused
        as an invalid token, and the result stays absent. That is the one-shot
        rule applied to failure as well as success.
        """
        with self._transaction() as tx:
            tx.execute(
                """
                UPDATE holdouts
                   SET unlock_token = NULL
                 WHERE dataset_id = ?
                   AND dataset_checksum = ?
                   AND result_json IS NULL
                   AND unlock_token = ?
                """,
                (record.dataset_id, record.dataset_checksum, digest),
            )

    def _save_result(self, record: HoldoutRecord, digest: str, result: LockedHoldoutResult) -> None:
        payload = canonical_json(result.to_dict())
        with self._transaction() as tx:
            cursor = tx.execute(
                """
                UPDATE holdouts
                   SET result_json = ?,
                       result_checksum = ?,
                       unlock_token = NULL
                 WHERE dataset_id = ?
                   AND dataset_checksum = ?
                   AND result_json IS NULL
                   AND evaluation_started_at IS NOT NULL
                   AND unlock_token = ?
                """,
                (
                    payload,
                    result.checksum,
                    record.dataset_id,
                    record.dataset_checksum,
                    digest,
                ),
            )
            if cursor.rowcount != 1:
                raise HoldoutError(
                    "holdout_result_conflict",
                    "another writer stored a result or the evaluation claim was lost",
                )
            self._append_access(
                tx,
                action="evaluate",
                outcome="granted",
                actor=record.unlocked_by or "system",
                reason="locked holdout evaluated once",
                dataset_id=record.dataset_id,
                dataset_checksum=record.dataset_checksum,
                detail=result.checksum,
            )


# --- evaluation ---------------------------------------------------------------


def _evaluator_alias(record: HoldoutRecord) -> str:
    """A neutral dataset id for the evaluator.

    ``DslEvaluator`` refuses ids that name a test/holdout split, because search
    must not reach the final slice. This module is the one caller allowed to read
    it, so it passes an alias that carries no holdout token. The alias is
    provenance only; the arithmetic reads the frame.
    """
    return f"oos-eval:{record.dataset_checksum[:16]}"


def _resolve_token(
    ledger: HoldoutLedger,
    unlock_token: HoldoutToken | str,
) -> tuple[HoldoutRecord, str]:
    if isinstance(unlock_token, HoldoutToken):
        digest = _token_digest(unlock_token.token)
        record = ledger.get(unlock_token.dataset_id, unlock_token.dataset_checksum)
        if record is None:
            raise HoldoutError(
                "holdout_invalid_token",
                "the token names a dataset build with no holdout row",
            )
        if record.result_json is not None:
            raise HoldoutError(
                "holdout_result_exists",
                "the one-shot result is already stored and cannot be recomputed",
            )
        if record.unlock_token != digest:
            raise HoldoutError(
                "holdout_invalid_token",
                "the token does not belong to this holdout or was already consumed",
            )
        if (
            record.symbol,
            record.timeframe,
            record.spec_hash,
            record.code_version,
            record.settings_digest,
        ) != (
            unlock_token.symbol,
            unlock_token.timeframe,
            unlock_token.spec_hash,
            unlock_token.code_version,
            unlock_token.settings_digest,
        ):
            raise HoldoutError(
                "holdout_token_identity_mismatch",
                "the token identity does not match the stored holdout",
            )
        return record, digest
    if isinstance(unlock_token, str):
        digest = _token_digest(unlock_token)
        record = ledger._find_by_token(digest)
        if record is None:
            raise HoldoutError("holdout_invalid_token", "no holdout matches this token")
        if record.result_json is not None:
            raise HoldoutError(
                "holdout_result_exists",
                "the one-shot result is already stored and cannot be recomputed",
            )
        return record, digest
    raise TypeError(
        f"unlock_token must be a HoldoutToken or str, got {type(unlock_token).__name__}"
    )


def _test_frame(
    data: DatasetBundle | FeatureFrame,
    config: HoldoutConfig,
    record: HoldoutRecord,
) -> tuple[FeatureFrame, int, int, int]:
    """Return the locked test slice plus its row offsets within the source frame.

    For a :class:`DatasetBundle`, the manifest's absolute test range is mapped
    through ``features.valid_from`` and the supplied frame must be the bundle's
    source frame. For a bare :class:`FeatureFrame`, the frame *is* the locked
    slice; the token authorises it, and the caller is responsible for not handing
    over validation rows.
    """
    if isinstance(data, DatasetBundle):
        manifest = data.manifest
        if manifest.dataset_id != record.dataset_id:
            raise HoldoutError(
                "holdout_dataset_mismatch",
                f"bundle is {manifest.dataset_id!r}, holdout is {record.dataset_id!r}",
            )
        if manifest.checksum != record.dataset_checksum:
            raise HoldoutError(
                "holdout_dataset_checksum_mismatch",
                f"bundle checksum is {manifest.checksum!r}, holdout is {record.dataset_checksum!r}",
            )
        features = config.features
        if features is None:
            raise HoldoutError(
                "missing_features",
                "a DatasetBundle needs config.features (the frame it was built from)",
            )
        offset = int(features.valid_from)
        expected_rows = int(features.X.shape[0]) - offset - int(manifest.horizon)
        if expected_rows != int(data.ts.shape[0]):
            raise HoldoutError(
                "frame_mismatch",
                (
                    f"features has {features.X.shape[0]} rows from valid_from={offset} "
                    f"and horizon={manifest.horizon}; expected {expected_rows} bundle rows, "
                    f"got {data.ts.shape[0]}"
                ),
            )
        if not np.array_equal(
            np.asarray(data.ts),
            np.asarray(features.ts[offset : offset + int(data.ts.shape[0])]),
        ):
            raise HoldoutError("frame_mismatch", "the bundle timestamps do not match the frame")
        start = int(manifest.test.start)
        end = int(manifest.test.end)
        if start < 0 or end > int(features.X.shape[0]) or end <= start:
            raise HoldoutError(
                "holdout_geometry",
                f"test range [{start}, {end}) is outside a {features.X.shape[0]}-row frame",
            )
        try:
            frame = slice_frame(features, start, end)
        except EvidenceError as exc:
            raise HoldoutError("holdout_geometry", str(exc)) from exc
        return frame, start - offset, end - offset, offset

    if isinstance(data, FeatureFrame):
        if config.dataset_id is not None and config.dataset_id != record.dataset_id:
            raise HoldoutError(
                "holdout_dataset_mismatch",
                f"configured dataset {config.dataset_id!r} is not {record.dataset_id!r}",
            )
        if (
            config.dataset_checksum is not None
            and config.dataset_checksum != record.dataset_checksum
        ):
            raise HoldoutError(
                "holdout_dataset_checksum_mismatch",
                f"configured checksum {config.dataset_checksum!r} is not "
                f"{record.dataset_checksum!r}",
            )
        if int(data.X.shape[0]) <= 0:
            raise HoldoutError("holdout_geometry", "the locked test frame is empty")
        return data, 0, int(data.X.shape[0]), 0

    raise TypeError(f"data must be a DatasetBundle or FeatureFrame, got {type(data).__name__}")


def _stress_rows(
    evaluator: DslEvaluator,
    spec: StrategySpec,
    multipliers: Sequence[float],
    *,
    base: Any,
) -> tuple[StressEvidence, ...]:
    rows: list[StressEvidence] = []
    for multiplier in multipliers:
        if multiplier == 1.0:
            metrics = base
        else:
            try:
                metrics = evaluator.simulate(spec, multiplier=multiplier).metrics
            except DslValidationError as exc:  # pragma: no cover - spec validated first
                raise HoldoutError("invalid_spec", str(exc)) from exc
        rows.append(
            StressEvidence(
                multiplier=float(multiplier),
                trades=int(metrics.trades),
                net_profit=_finite(metrics.net_profit, where=f"stress x{multiplier:g} net_profit"),
                profit_factor=_finite(
                    metrics.profit_factor, where=f"stress x{multiplier:g} profit_factor"
                ),
                max_drawdown=_finite(
                    metrics.max_drawdown, where=f"stress x{multiplier:g} max_drawdown"
                ),
                expectancy_r=_finite(
                    metrics.expectancy_r, where=f"stress x{multiplier:g} expectancy_r"
                ),
            )
        )
    return tuple(rows)


def evaluate_holdout(
    data: DatasetBundle | FeatureFrame,
    spec: StrategySpec,
    config: HoldoutConfig,
    *,
    ledger: HoldoutLedger,
    unlock_token: HoldoutToken | str,
) -> LockedHoldoutResult:
    """Run the candidate exactly once on the locked final test split.

    The token is validated and the evaluation slot is claimed before the test
    rows are sliced. A refused token, a spent holdout or an existing result
    therefore cannot read the frame. The result is stored inside the ledger
    transaction; a failure after the claim leaves the holdout spent with no
    result, which is the one-shot rule rather than a retry opportunity.
    """
    if not isinstance(ledger, HoldoutLedger):
        raise TypeError(f"ledger must be a HoldoutLedger, got {type(ledger).__name__}")
    if not isinstance(spec, StrategySpec):
        raise TypeError(f"spec must be a StrategySpec, got {type(spec).__name__}")
    if not isinstance(config, HoldoutConfig):
        raise TypeError(f"config must be a HoldoutConfig, got {type(config).__name__}")
    if not isinstance(data, (DatasetBundle, FeatureFrame)):
        raise TypeError(f"data must be a DatasetBundle or FeatureFrame, got {type(data).__name__}")

    try:
        record, digest = _resolve_token(ledger, unlock_token)
    except HoldoutError as exc:
        identity = unlock_token if isinstance(unlock_token, HoldoutToken) else None
        ledger._log(
            action="evaluate",
            outcome="refused",
            actor="system",
            reason=exc.code,
            dataset_id=None if identity is None else identity.dataset_id,
            dataset_checksum=None if identity is None else identity.dataset_checksum,
            detail=str(exc),
        )
        raise
    actor = record.unlocked_by or "system"

    try:
        spec_hash = spec.canonical_hash()
        spec_version = int(spec.spec_version)
        complexity = int(spec.complexity())
    except DslValidationError as exc:
        raise HoldoutError("invalid_spec", str(exc)) from exc

    settings = Settings() if config.settings is None else config.settings
    try:
        digest_settings = settings_digest(settings)
    except (TypeError, ValueError) as exc:
        raise HoldoutError("invalid_settings", str(exc)) from exc

    expected = (
        record.spec_hash,
        record.code_version,
        record.settings_digest,
        record.symbol,
        record.timeframe,
    )
    actual = (
        spec_hash,
        config.code_version,
        digest_settings,
        config.symbol,
        config.timeframe,
    )
    if actual != expected:
        ledger._refuse(
            action="evaluate",
            actor=actor,
            reason="holdout_identity_mismatch",
            dataset_id=record.dataset_id,
            dataset_checksum=record.dataset_checksum,
            detail=f"expected {expected!r}, got {actual!r}",
        )

    if isinstance(data, DatasetBundle):
        if config.dataset_id is not None and config.dataset_id != record.dataset_id:
            ledger._refuse(
                action="evaluate",
                actor=actor,
                reason="holdout_dataset_mismatch",
                dataset_id=record.dataset_id,
                dataset_checksum=record.dataset_checksum,
                detail=f"configured dataset {config.dataset_id!r}",
            )
        if (
            config.dataset_checksum is not None
            and config.dataset_checksum != record.dataset_checksum
        ):
            ledger._refuse(
                action="evaluate",
                actor=actor,
                reason="holdout_dataset_checksum_mismatch",
                dataset_id=record.dataset_id,
                dataset_checksum=record.dataset_checksum,
                detail=f"configured checksum {config.dataset_checksum!r}",
            )

    # Optional preflight that touches only shapes, never test-row values. A wrong
    # bundle identity is a caller error; it should not silently burn the holdout.
    if isinstance(data, DatasetBundle):
        if data.manifest.dataset_id != record.dataset_id:
            ledger._refuse(
                action="evaluate",
                actor=actor,
                reason="holdout_dataset_mismatch",
                dataset_id=record.dataset_id,
                dataset_checksum=record.dataset_checksum,
                detail=(
                    f"bundle is {data.manifest.dataset_id!r}, holdout is {record.dataset_id!r}"
                ),
            )
        if data.manifest.checksum != record.dataset_checksum:
            ledger._refuse(
                action="evaluate",
                actor=actor,
                reason="holdout_dataset_checksum_mismatch",
                dataset_id=record.dataset_id,
                dataset_checksum=record.dataset_checksum,
                detail=(
                    f"bundle checksum is {data.manifest.checksum!r}, "
                    f"holdout is {record.dataset_checksum!r}"
                ),
            )

    ledger._begin_evaluation(record, digest, actor=actor)

    try:
        test_frame, start_row, end_row, _offset = _test_frame(data, config, record)
        if int(test_frame.X.shape[0]) <= config.window:
            raise HoldoutError(
                "holdout_too_small",
                (
                    f"the locked test split has {test_frame.X.shape[0]} rows; "
                    f"a {config.window}-bar window needs more"
                ),
            )
        evaluator = DslEvaluator(
            test_frame,
            settings,
            symbol=config.symbol,
            timeframe=config.timeframe,
            dataset_id=_evaluator_alias(record),
            window=config.window,
            stress=(),
            max_steps=int(test_frame.X.shape[0]),
        )
        base = evaluator.simulate(spec)
        metrics = base.metrics
        trades = int(metrics.trades)
        if trades != len(base.trades):
            raise HoldoutError(
                "inconsistent_holdout",
                f"metrics report {trades} trades but {len(base.trades)} were returned",
            )
        if trades < 0:  # pragma: no cover - metrics never returns a negative count
            raise HoldoutError("inconsistent_holdout", "negative trade count")

        net_profit = _finite(metrics.net_profit, where="net_profit")
        profit_factor = _finite(metrics.profit_factor, where="profit_factor")
        win_rate = _finite(metrics.win_rate, where="win_rate")
        expectancy_r = _finite(metrics.expectancy_r, where="expectancy_r")
        max_drawdown = _finite(metrics.max_drawdown, where="max_drawdown")

        labels = regime_labels(test_frame)
        if labels.is_empty():
            raise HoldoutError(
                "missing_regime_evidence",
                "the rule engine produced no regime labels for the test split",
            )
        attached = per_trade_regimes(base.trades, labels)
        regimes = regime_rows(attached)
        known = sum(row.trades for row in regimes if row.regime != UNKNOWN_REGIME)
        regime_coverage = _finite((known / trades) if trades else 0.0, where="regime_coverage")

        stress = _stress_rows(
            evaluator,
            spec,
            config.cost_stress,
            base=metrics,
        )

        r_multiples = [
            _finite(trade.r_multiple, where=f"trade {index} r_multiple")
            for index, trade in enumerate(base.trades)
        ]
        if r_multiples:
            total_r = _finite(sum(r_multiples), where="total_r")
            pooled_expectancy = _finite(
                sum(r_multiples) / len(r_multiples), where="pooled expectancy_r"
            )
            expectancy_ci = bootstrap_ci(
                r_multiples,
                samples=config.bootstrap_samples,
                seed=config.seed,
                confidence=config.bootstrap_confidence,
                statistic="mean",
            )
            total_ci = bootstrap_ci(
                r_multiples,
                samples=config.bootstrap_samples,
                seed=config.seed,
                confidence=config.bootstrap_confidence,
                statistic="sum",
            )
            method = "iid-bootstrap"
        else:
            total_r = 0.0
            pooled_expectancy = 0.0
            expectancy_ci = (0.0, 0.0)
            total_ci = (0.0, 0.0)
            method = "not_applicable_no_trades"

        ts = np.asarray(test_frame.ts)
        result = LockedHoldoutResult(
            dataset_id=record.dataset_id,
            dataset_checksum=record.dataset_checksum,
            symbol=config.symbol,
            timeframe=config.timeframe,
            spec_hash=spec_hash,
            spec_version=spec_version,
            complexity=complexity,
            code_version=config.code_version,
            settings_digest=digest_settings,
            policy_id=record.policy_id or "unknown",
            policy_version=int(record.policy_version or 0),
            policy_digest=record.policy_digest or "",
            evidence_checksum=record.evidence_checksum or "",
            test_start_row=int(start_row),
            test_end_row=int(end_row),
            test_rows=int(test_frame.X.shape[0]),
            test_start_ts=_iso_ts(ts[0]),
            test_end_ts=_iso_ts(ts[-1]),
            trades=trades,
            net_profit=net_profit,
            profit_factor=profit_factor,
            win_rate=win_rate,
            expectancy_r=pooled_expectancy if trades else expectancy_r,
            total_r=total_r,
            max_drawdown=max_drawdown,
            regimes=regimes,
            stress=stress,
            expectancy_r_ci95_low=expectancy_ci[0],
            expectancy_r_ci95_high=expectancy_ci[1],
            total_r_ci95_low=total_ci[0],
            total_r_ci95_high=total_ci[1],
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_confidence=float(config.bootstrap_confidence),
            bootstrap_method=method,
            regime_coverage=regime_coverage,
            checksum="",
            created_at=_now(),
        )
        result = replace(result, checksum=_result_checksum(result))
    except HoldoutError as exc:
        try:
            ledger._invalidate_token(record, digest)
        except HoldoutError:  # pragma: no cover - the original refusal is more useful
            pass
        ledger._log(
            action="evaluate",
            outcome="refused",
            actor=actor,
            reason=exc.code,
            dataset_id=record.dataset_id,
            dataset_checksum=record.dataset_checksum,
            detail=str(exc),
        )
        raise
    except Exception as exc:
        try:
            ledger._invalidate_token(record, digest)
        except HoldoutError:  # pragma: no cover - the original failure is more useful
            pass
        ledger._log(
            action="evaluate",
            outcome="refused",
            actor=actor,
            reason="holdout_evaluation_failed",
            dataset_id=record.dataset_id,
            dataset_checksum=record.dataset_checksum,
            detail=f"{type(exc).__name__}: {exc}",
        )
        raise HoldoutError("holdout_evaluation_failed", str(exc)) from exc

    try:
        ledger._save_result(record, digest, result)
    except HoldoutError as exc:
        try:
            ledger._invalidate_token(record, digest)
        except HoldoutError:  # pragma: no cover - the original failure is more useful
            pass
        ledger._log(
            action="evaluate",
            outcome="refused",
            actor=actor,
            reason=exc.code,
            dataset_id=record.dataset_id,
            dataset_checksum=record.dataset_checksum,
            detail=str(exc),
        )
        raise
    return result


def _iso_ts(stamp: Any) -> str:
    value = np.datetime64(stamp, "us").astype(datetime)
    return value.replace(tzinfo=UTC).isoformat()


# --- verification / evidence attachment ---------------------------------------


def verify_holdout_result(
    result: LockedHoldoutResult,
    evidence: CandidateEvidence,
) -> None:
    """Reject a stale, mismatched or edited holdout result.

    Identity is checked first (dataset id/checksum, spec, code version,
    settings, symbol and timeframe), then the result's own canonical checksum.
    If the evidence already carries a ``locked_oos`` section, it must name the
    same result checksum. This function raises; a caller must not swallow the
    error and use the numbers anyway.
    """
    if not isinstance(result, LockedHoldoutResult):
        raise TypeError(f"result must be a LockedHoldoutResult, got {type(result).__name__}")
    if not isinstance(evidence, CandidateEvidence):
        raise TypeError(f"evidence must be a CandidateEvidence, got {type(evidence).__name__}")
    if result.schema_version != HOLDOUT_SCHEMA_VERSION:
        raise HoldoutError(
            "schema_mismatch",
            f"holdout result schema {result.schema_version} is not {HOLDOUT_SCHEMA_VERSION}",
        )

    checks = (
        ("stale_dataset", result.dataset_id, evidence.dataset_id),
        ("stale_dataset_checksum", result.dataset_checksum, evidence.dataset_checksum),
        ("stale_spec", result.spec_hash, evidence.spec_hash),
        ("stale_code_version", result.code_version, evidence.code_version),
        ("stale_settings", result.settings_digest, evidence.settings_digest),
        ("stale_symbol", result.symbol, evidence.symbol),
        ("stale_timeframe", result.timeframe, evidence.timeframe),
    )
    for code, actual, expected in checks:
        if actual != expected:
            raise HoldoutError(code, f"holdout result has {actual!r}, evidence has {expected!r}")

    expected_checksum = _result_checksum(result)
    if result.checksum != expected_checksum:
        raise HoldoutError(
            "checksum_mismatch",
            f"holdout result checksum {result.checksum!r} does not match {expected_checksum!r}",
        )

    if evidence.locked_oos is not None:
        section = evidence.locked_oos
        if not isinstance(section, Mapping):
            raise HoldoutError("locked_oos_mismatch", "the evidence holdout section is not a map")
        if section.get("checksum") != result.checksum:
            raise HoldoutError(
                "locked_oos_mismatch",
                "the evidence holdout section names a different result checksum",
            )


def attach_holdout(
    evidence: CandidateEvidence,
    result: LockedHoldoutResult,
) -> CandidateEvidence:
    """Return a **new** evidence bundle carrying the locked-holdout section.

    The input bundle is frozen and is never mutated. The result must match the
    evidence identity and must have been unlocked for this exact bundle
    (``result.evidence_checksum == evidence.checksum``), which prevents attaching
    a holdout result to a different candidate that happens to share a dataset.
    Attaching the same result to an already-attached bundle is idempotent.
    """
    if not isinstance(evidence, CandidateEvidence):
        raise TypeError(f"evidence must be a CandidateEvidence, got {type(evidence).__name__}")
    if not isinstance(result, LockedHoldoutResult):
        raise TypeError(f"result must be a LockedHoldoutResult, got {type(result).__name__}")
    verify_holdout_result(result, evidence)

    if evidence.locked_oos is not None:
        section = evidence.locked_oos
        if isinstance(section, Mapping) and section.get("checksum") == result.checksum:
            return evidence
        raise HoldoutError(
            "locked_oos_conflict",
            "the evidence already carries a different locked-holdout section",
        )
    if result.evidence_checksum != evidence.checksum:
        raise HoldoutError(
            "evidence_checksum_mismatch",
            (
                f"the holdout was unlocked by evidence {result.evidence_checksum!r}, "
                f"not {evidence.checksum!r}"
            ),
        )

    section = result.to_dict()
    payload = evidence.to_dict()
    payload["locked_oos"] = section
    body = dict(payload)
    body.pop("checksum", None)
    body.pop("created_at", None)
    checksum = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    attached = replace(evidence, locked_oos=section, checksum=checksum)
    verify_evidence(
        attached,
        dataset_id=attached.dataset_id,
        dataset_checksum=attached.dataset_checksum,
        spec_hash=attached.spec_hash,
        code_version=attached.code_version,
        settings_digest=attached.settings_digest,
    )
    return attached
