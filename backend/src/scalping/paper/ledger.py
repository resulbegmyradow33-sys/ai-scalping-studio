"""Persistent, restart-safe ledger for paper trading.

The ledger is the durable record of one paper session: the signals a policy
emitted, the orders they became, the fills those orders received, the trades
that closed and the equity curve that ties them together. It is written with the
standard library's :mod:`sqlite3` only; there is no ORM and no network access.

Conventions (the same ones the SQLite model registry uses):

* WAL journal, ``synchronous=NORMAL``, ``foreign_keys=ON`` and a 5 s busy
  timeout on every connection;
* a ``schema_version`` table and a migration dict, so ``init_schema`` is
  idempotent and a database written by a newer version is refused rather than
  silently downgraded;
* writes run inside ``BEGIN IMMEDIATE``; a failed write leaves no partial row;
* every row belongs to a session through a foreign key with ``ON DELETE
  CASCADE``, so a session cannot own orphans.

Money and prices are stored as ``REAL``. The broker already embeds spread and
slippage in the fill prices, so a trade's ``pnl`` is fill-to-fill minus the
round-turn commission; the ledger stores the number the broker produced and
never re-derives it. ``fills.commission`` and ``trades.pnl`` are therefore
consistent by construction, and the tests check that arithmetic.

The open position is stored explicitly in ``positions`` rather than inferred
from fills. Fills alone cannot reconstruct SL/TP, and ``resume`` must return the
state needed to continue, not a guess. A partial unique index enforces at most
one open position per session, matching the broker's single-position model.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Final, Self

from scalping.types import Side

__all__ = [
    "SCHEMA_VERSION",
    "DuplicateError",
    "EquityPoint",
    "FillRecord",
    "LedgerError",
    "OrderRecord",
    "PaperLedger",
    "PositionRecord",
    "ResumeState",
    "SessionRecord",
    "SignalRecord",
    "TradeRecord",
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
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            strategy TEXT NOT NULL,
            model_version TEXT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            status TEXT NOT NULL CHECK (status IN ('running', 'closed')),
            config_json TEXT NOT NULL DEFAULT '{}'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS signals (
            signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            ts TEXT NOT NULL,
            action TEXT NOT NULL,
            reason TEXT,
            model_version TEXT,
            probabilities_json TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS signals_session_idx ON signals(session_id, signal_id)",
        """
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            signal_id INTEGER REFERENCES signals(signal_id) ON DELETE SET NULL,
            ts TEXT NOT NULL,
            side TEXT NOT NULL CHECK (side IN ('long', 'short')),
            lot REAL NOT NULL CHECK (lot >= 0),
            requested_price REAL,
            status TEXT NOT NULL CHECK (status IN ('pending', 'filled', 'rejected')),
            rejection_reason TEXT,
            CHECK (status <> 'rejected' OR rejection_reason IS NOT NULL)
        )
        """,
        "CREATE INDEX IF NOT EXISTS orders_session_idx ON orders(session_id, order_id)",
        """
        CREATE TABLE IF NOT EXISTS fills (
            fill_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
            ts TEXT NOT NULL,
            price REAL NOT NULL,
            spread REAL NOT NULL DEFAULT 0,
            slippage REAL NOT NULL DEFAULT 0,
            commission REAL NOT NULL DEFAULT 0,
            lot REAL NOT NULL CHECK (lot > 0)
        )
        """,
        "CREATE INDEX IF NOT EXISTS fills_order_idx ON fills(order_id, fill_id)",
        """
        CREATE TABLE IF NOT EXISTS positions (
            position_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            order_id INTEGER REFERENCES orders(order_id) ON DELETE SET NULL,
            side TEXT NOT NULL CHECK (side IN ('long', 'short')),
            lot REAL NOT NULL CHECK (lot > 0),
            entry_ts TEXT NOT NULL,
            entry_price REAL NOT NULL,
            sl REAL NOT NULL,
            tp REAL NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
            closed_ts TEXT,
            closed_price REAL
        )
        """,
        # One open position per session: the same rule the broker enforces.
        (
            "CREATE UNIQUE INDEX IF NOT EXISTS positions_one_open_idx"
            " ON positions(session_id) WHERE status = 'open'"
        ),
        """
        CREATE TABLE IF NOT EXISTS trades (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            side TEXT NOT NULL CHECK (side IN ('long', 'short')),
            entry_ts TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_ts TEXT NOT NULL,
            exit_price REAL NOT NULL,
            lot REAL NOT NULL CHECK (lot > 0),
            pnl REAL NOT NULL,
            r_multiple REAL NOT NULL,
            exit_reason TEXT NOT NULL,
            CHECK (exit_ts >= entry_ts)
        )
        """,
        "CREATE INDEX IF NOT EXISTS trades_session_idx ON trades(session_id, trade_id)",
        """
        CREATE TABLE IF NOT EXISTS equity_curve (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            ts TEXT NOT NULL,
            balance REAL NOT NULL,
            equity REAL NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS equity_session_idx ON equity_curve(session_id, id)",
    ),
}

#: Order states a caller may write.
_ORDER_STATUSES: Final[frozenset[str]] = frozenset({"pending", "filled", "rejected"})

#: Paginated reads. The table and ordering are chosen here, never by a caller,
#: so the only request-derived values that reach SQL are the bound parameters.
_PAGE_SQL: Final[dict[str, str]] = {
    "sessions": (
        "SELECT * FROM sessions ORDER BY started_at DESC, session_id DESC LIMIT ? OFFSET ?"
    ),
    "positions": (
        "SELECT * FROM positions WHERE session_id = ? ORDER BY position_id LIMIT ? OFFSET ?"
    ),
    "orders": "SELECT * FROM orders WHERE session_id = ? ORDER BY order_id LIMIT ? OFFSET ?",
    "trades": "SELECT * FROM trades WHERE session_id = ? ORDER BY trade_id LIMIT ? OFFSET ?",
    "equity": "SELECT * FROM equity_curve WHERE session_id = ? ORDER BY id DESC LIMIT ? OFFSET ?",
}

#: The matching totals, so a page can say how many rows exist behind it.
_COUNT_SQL: Final[dict[str, str]] = {
    "sessions": "SELECT COUNT(*) FROM sessions",
    "positions": "SELECT COUNT(*) FROM positions WHERE session_id = ?",
    "orders": "SELECT COUNT(*) FROM orders WHERE session_id = ?",
    "trades": "SELECT COUNT(*) FROM trades WHERE session_id = ?",
    "equity": "SELECT COUNT(*) FROM equity_curve WHERE session_id = ?",
}


class LedgerError(Exception):
    """A ledger operation was rejected; the database is unchanged."""


class DuplicateError(LedgerError):
    """The session id is already recorded."""


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """One paper session and the configuration it ran with."""

    session_id: str
    symbol: str
    timeframe: str
    strategy: str
    model_version: str | None
    started_at: datetime
    ended_at: datetime | None
    status: str
    config_json: str

    @property
    def config(self) -> dict[str, Any]:
        """The stored configuration as a fresh dict; callers may mutate it."""
        data = json.loads(self.config_json)
        if not isinstance(data, dict):
            raise LedgerError(f"session {self.session_id!r} has non-object config")
        return data


@dataclass(frozen=True, slots=True)
class SignalRecord:
    """One policy decision, with the gate reason when it was blocked."""

    signal_id: int
    session_id: str
    ts: datetime
    action: str
    reason: str | None
    model_version: str | None
    probabilities_json: str | None


@dataclass(frozen=True, slots=True)
class OrderRecord:
    """One order attempt; a rejection always carries its reason."""

    order_id: int
    session_id: str
    signal_id: int | None
    ts: datetime
    side: str
    lot: float
    requested_price: float | None
    status: str
    rejection_reason: str | None


@dataclass(frozen=True, slots=True)
class FillRecord:
    """One fill against an order."""

    fill_id: int
    order_id: int
    ts: datetime
    price: float
    spread: float
    slippage: float
    commission: float
    lot: float


@dataclass(frozen=True, slots=True)
class PositionRecord:
    """One position lifecycle; ``closed_ts`` is set when the trade settles."""

    position_id: int
    session_id: str
    order_id: int | None
    side: str
    lot: float
    entry_ts: datetime
    entry_price: float
    sl: float
    tp: float
    status: str
    closed_ts: datetime | None
    closed_price: float | None


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """One closed trade, exactly as the broker settled it."""

    trade_id: int
    session_id: str
    side: str
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    lot: float
    pnl: float
    r_multiple: float
    exit_reason: str


@dataclass(frozen=True, slots=True)
class EquityPoint:
    """One mark of the account after a step."""

    id: int
    session_id: str
    ts: datetime
    balance: float
    equity: float


@dataclass(frozen=True, slots=True)
class ResumeState:
    """Everything a runner needs to continue a session.

    ``balance``/``equity`` are ``None`` until the first equity mark is written.
    ``pending_orders`` is empty for the current market-order runner; the column
    exists for the future limit-order MT5 adapter, and the ledger reports what
    is there instead of inventing a queue.
    """

    session_id: str
    status: str
    balance: float | None
    equity: float | None
    position: PositionRecord | None
    last_bar_ts: datetime | None
    pending_orders: tuple[OrderRecord, ...]


def _now() -> datetime:
    return datetime.now(UTC)


def _stamp(value: datetime, field: str = "ts") -> str:
    """Normalise an aware datetime to a sortable UTC ISO-8601 string."""
    if not isinstance(value, datetime):
        raise TypeError(f"{field} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware, got {value!r}")
    return value.astimezone(UTC).isoformat()


def _parse(value: object, field: str = "ts") -> datetime:
    if not isinstance(value, str):
        raise LedgerError(f"stored {field} is not text: {value!r}")
    return datetime.fromisoformat(value)


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


def _require_finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number, got {type(value).__name__}")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def _normalize_side(value: object) -> str:
    if isinstance(value, Side):
        if value is Side.FLAT:
            raise ValueError("side must be LONG or SHORT, not FLAT")
        return "long" if value is Side.LONG else "short"
    text = _require_text(value, "side").lower()
    if text not in {"long", "short"}:
        raise ValueError(f"side must be 'long' or 'short', got {value!r}")
    return text


def _page_args(limit: object, offset: object) -> tuple[int, int]:
    """Validate a page request; the API bounds these too, this is the last gate."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError(f"limit must be an int, got {type(limit).__name__}")
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError(f"offset must be an int, got {type(offset).__name__}")
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit!r}")
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset!r}")
    return limit, offset


def _session_from_row(row: sqlite3.Row) -> SessionRecord:
    ended = row["ended_at"]
    model_version = row["model_version"]
    return SessionRecord(
        session_id=str(row["session_id"]),
        symbol=str(row["symbol"]),
        timeframe=str(row["timeframe"]),
        strategy=str(row["strategy"]),
        model_version=None if model_version is None else str(model_version),
        started_at=_parse(row["started_at"], "started_at"),
        ended_at=None if ended is None else _parse(ended, "ended_at"),
        status=str(row["status"]),
        config_json=str(row["config_json"]),
    )


def _signal_from_row(row: sqlite3.Row) -> SignalRecord:
    reason = row["reason"]
    model_version = row["model_version"]
    probabilities = row["probabilities_json"]
    return SignalRecord(
        signal_id=int(row["signal_id"]),
        session_id=str(row["session_id"]),
        ts=_parse(row["ts"]),
        action=str(row["action"]),
        reason=None if reason is None else str(reason),
        model_version=None if model_version is None else str(model_version),
        probabilities_json=None if probabilities is None else str(probabilities),
    )


def _order_from_row(row: sqlite3.Row) -> OrderRecord:
    signal_id = row["signal_id"]
    requested = row["requested_price"]
    rejection = row["rejection_reason"]
    return OrderRecord(
        order_id=int(row["order_id"]),
        session_id=str(row["session_id"]),
        signal_id=None if signal_id is None else int(signal_id),
        ts=_parse(row["ts"]),
        side=str(row["side"]),
        lot=float(row["lot"]),
        requested_price=None if requested is None else float(requested),
        status=str(row["status"]),
        rejection_reason=None if rejection is None else str(rejection),
    )


def _fill_from_row(row: sqlite3.Row) -> FillRecord:
    return FillRecord(
        fill_id=int(row["fill_id"]),
        order_id=int(row["order_id"]),
        ts=_parse(row["ts"]),
        price=float(row["price"]),
        spread=float(row["spread"]),
        slippage=float(row["slippage"]),
        commission=float(row["commission"]),
        lot=float(row["lot"]),
    )


def _position_from_row(row: sqlite3.Row) -> PositionRecord:
    order_id = row["order_id"]
    closed_ts = row["closed_ts"]
    closed_price = row["closed_price"]
    return PositionRecord(
        position_id=int(row["position_id"]),
        session_id=str(row["session_id"]),
        order_id=None if order_id is None else int(order_id),
        side=str(row["side"]),
        lot=float(row["lot"]),
        entry_ts=_parse(row["entry_ts"], "entry_ts"),
        entry_price=float(row["entry_price"]),
        sl=float(row["sl"]),
        tp=float(row["tp"]),
        status=str(row["status"]),
        closed_ts=None if closed_ts is None else _parse(closed_ts, "closed_ts"),
        closed_price=None if closed_price is None else float(closed_price),
    )


def _trade_from_row(row: sqlite3.Row) -> TradeRecord:
    return TradeRecord(
        trade_id=int(row["trade_id"]),
        session_id=str(row["session_id"]),
        side=str(row["side"]),
        entry_ts=_parse(row["entry_ts"], "entry_ts"),
        entry_price=float(row["entry_price"]),
        exit_ts=_parse(row["exit_ts"], "exit_ts"),
        exit_price=float(row["exit_price"]),
        lot=float(row["lot"]),
        pnl=float(row["pnl"]),
        r_multiple=float(row["r_multiple"]),
        exit_reason=str(row["exit_reason"]),
    )


def _equity_from_row(row: sqlite3.Row) -> EquityPoint:
    return EquityPoint(
        id=int(row["id"]),
        session_id=str(row["session_id"]),
        ts=_parse(row["ts"]),
        balance=float(row["balance"]),
        equity=float(row["equity"]),
    )


class PaperLedger:
    """A local SQLite ledger; the schema is created (or migrated) on open.

    The class is not thread-safe and expects one writer at a time. The raw
    ``connection`` is exposed for read queries and tests; writes belong to the
    methods so that every write is transactional.
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

    @classmethod
    def open_reader(cls, path: str | Path) -> PaperLedger:
        """Open an existing ledger for reading only.

        The connection is a ``mode=ro`` URI: SQLite refuses to create the file,
        cannot write through it and will not take a write lock. No schema
        migration runs and the journal mode is left alone, so a reader never
        changes the file and never blocks the session's writer (WAL readers see
        the last committed snapshot). A missing file raises :class:`FileNotFoundError`
        and an unreadable one raises :class:`LedgerError`; neither is a write.

        The class is still not thread-safe; open one reader per thread/request
        and close it when the read is done.
        """
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"no ledger at {str(target)!r}")
        try:
            conn = sqlite3.connect(
                f"{target.resolve().as_uri()}?mode=ro",
                uri=True,
                isolation_level=None,
            )
        except sqlite3.Error as exc:
            raise LedgerError(f"cannot open ledger {str(target)!r} for reading: {exc}") from exc
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        reader = cls.__new__(cls)
        reader.path = str(target)
        reader._conn = conn
        return reader

    @property
    def connection(self) -> sqlite3.Connection:
        """The open connection; read-only use is expected outside the class."""
        return self._connection()

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise LedgerError("ledger is closed")
        return self._conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """``BEGIN IMMEDIATE`` around a write, rolled back on any exception."""
        conn = self._connection()
        if conn.in_transaction:
            raise LedgerError("nested ledger transaction")
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
                raise LedgerError(
                    f"database schema version {current} is newer than supported {SCHEMA_VERSION}"
                )
            for version in sorted(_MIGRATIONS):
                if version <= current:
                    continue
                for statement in _MIGRATIONS[version]:
                    tx.execute(statement)
                tx.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (version, _now().isoformat()),
                )
        return self.schema_version

    @property
    def schema_version(self) -> int:
        row = self._connection().execute("SELECT MAX(version) FROM schema_version").fetchone()
        return 0 if row is None or row[0] is None else int(row[0])

    def has_schema(self) -> bool:
        """Whether the sessions table exists; a reader sees ``False`` on a stray file.

        A GET that lands while the first writer is still inside ``init_schema``
        can see an empty database. The API treats that as "no ledger yet"
        instead of turning a race into a 500.
        """
        row = (
            self._connection()
            .execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sessions'")
            .fetchone()
        )
        return row is not None

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

    # ------------------------------------------------------------------- sessions

    def open_session(
        self,
        session_id: str,
        *,
        symbol: str,
        timeframe: str,
        strategy: str,
        model_version: str | None = None,
        started_at: datetime | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> SessionRecord:
        """Record a new running session; a duplicate id raises :class:`DuplicateError`."""
        _require_text(session_id, "session_id")
        _require_text(symbol, "symbol")
        _require_text(timeframe, "timeframe")
        _require_text(strategy, "strategy")
        version = _optional_text(model_version, "model_version")
        stamp = _stamp(_now() if started_at is None else started_at, "started_at")
        config_json = _canonical_json({} if config is None else config)

        try:
            with self._transaction() as tx:
                tx.execute(
                    """
                    INSERT INTO sessions (
                        session_id, symbol, timeframe, strategy, model_version,
                        started_at, ended_at, status, config_json
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL, 'running', ?)
                    """,
                    (session_id, symbol, timeframe, strategy, version, stamp, config_json),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateError(f"session {session_id!r} is already recorded") from exc
        return SessionRecord(
            session_id=session_id,
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy,
            model_version=version,
            started_at=_parse(stamp, "started_at"),
            ended_at=None,
            status="running",
            config_json=config_json,
        )

    def close_session(
        self,
        session_id: str,
        *,
        ended_at: datetime | None = None,
    ) -> SessionRecord:
        """Mark a session closed; closing an already closed session is a no-op."""
        _require_text(session_id, "session_id")
        stamp = _stamp(_now() if ended_at is None else ended_at, "ended_at")
        with self._transaction() as tx:
            known = tx.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if known is None:
                raise LedgerError(f"unknown session {session_id!r}")
            tx.execute(
                """
                UPDATE sessions SET status = 'closed', ended_at = ?
                WHERE session_id = ? AND status = 'running'
                """,
                (stamp, session_id),
            )
        record = self.session(session_id)
        if record is None:  # pragma: no cover - the row was just checked
            raise LedgerError(f"session {session_id!r} disappeared during close")
        return record

    def session(self, session_id: str) -> SessionRecord | None:
        """The session, or ``None`` when the id is unknown."""
        _require_text(session_id, "session_id")
        row = (
            self._connection()
            .execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            .fetchone()
        )
        return None if row is None else _session_from_row(row)

    def session_page(self, *, limit: int, offset: int = 0) -> tuple[list[SessionRecord], int]:
        """One page of sessions, newest start first, with the total count."""
        rows, total = self._page("sessions", None, limit, offset)
        return [_session_from_row(row) for row in rows], total

    # -------------------------------------------------------------------- signals

    def record_signal(
        self,
        session_id: str,
        ts: datetime,
        action: str,
        *,
        reason: str | None = None,
        model_version: str | None = None,
        probabilities: Mapping[str, float] | None = None,
    ) -> int:
        """Append one decision; ``reason`` carries the gate that blocked it."""
        _require_text(session_id, "session_id")
        stamp = _stamp(ts)
        name = _require_text(action, "action")
        why = _optional_text(reason, "reason")
        version = _optional_text(model_version, "model_version")
        probabilities_json = None if probabilities is None else _canonical_json(probabilities)
        with self._transaction() as tx:
            self._require_session(tx, session_id)
            cursor = tx.execute(
                """
                INSERT INTO signals (
                    session_id, ts, action, reason, model_version, probabilities_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, stamp, name, why, version, probabilities_json),
            )
            signal_id = cursor.lastrowid
            if signal_id is None:  # pragma: no cover - sqlite always returns it
                raise LedgerError("signal insert returned no id")
        return int(signal_id)

    def signals(self, session_id: str) -> list[SignalRecord]:
        """Every decision of the session, oldest first."""
        _require_text(session_id, "session_id")
        rows = (
            self._connection()
            .execute(
                "SELECT * FROM signals WHERE session_id = ? ORDER BY signal_id",
                (session_id,),
            )
            .fetchall()
        )
        return [_signal_from_row(row) for row in rows]

    # --------------------------------------------------------------------- orders

    def record_order(
        self,
        session_id: str,
        *,
        signal_id: int | None,
        ts: datetime,
        side: Side | str,
        lot: float,
        requested_price: float | None = None,
        status: str = "filled",
        rejection_reason: str | None = None,
    ) -> int:
        """Append one order attempt.

        A rejected order must carry a reason (the schema enforces it), so a
        rejection can never be silently dropped. ``lot`` is the requested size;
        it is ``0`` for an entry the risk gate refused before sizing.
        """
        _require_text(session_id, "session_id")
        stamp = _stamp(ts)
        direction = _normalize_side(side)
        size = _require_finite(lot, "lot")
        if size < 0:
            raise ValueError(f"lot must be >= 0, got {lot!r}")
        if status not in _ORDER_STATUSES:
            raise ValueError(f"status must be one of {sorted(_ORDER_STATUSES)}, got {status!r}")
        price = (
            None if requested_price is None else _require_finite(requested_price, "requested_price")
        )
        reason = _optional_text(rejection_reason, "rejection_reason")
        if status == "rejected" and reason is None:
            raise ValueError("a rejected order must carry a rejection_reason")
        if signal_id is not None and (
            isinstance(signal_id, bool) or not isinstance(signal_id, int)
        ):
            raise TypeError(f"signal_id must be an int, got {type(signal_id).__name__}")

        with self._transaction() as tx:
            self._require_session(tx, session_id)
            if signal_id is not None:
                known = tx.execute(
                    "SELECT 1 FROM signals WHERE signal_id = ? AND session_id = ?",
                    (signal_id, session_id),
                ).fetchone()
                if known is None:
                    raise LedgerError(f"unknown signal {signal_id!r} for session {session_id!r}")
            cursor = tx.execute(
                """
                INSERT INTO orders (
                    session_id, signal_id, ts, side, lot, requested_price,
                    status, rejection_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, signal_id, stamp, direction, size, price, status, reason),
            )
            order_id = cursor.lastrowid
            if order_id is None:  # pragma: no cover - sqlite always returns it
                raise LedgerError("order insert returned no id")
        return int(order_id)

    def orders(self, session_id: str) -> list[OrderRecord]:
        """Every order of the session, oldest first."""
        _require_text(session_id, "session_id")
        rows = (
            self._connection()
            .execute(
                "SELECT * FROM orders WHERE session_id = ? ORDER BY order_id",
                (session_id,),
            )
            .fetchall()
        )
        return [_order_from_row(row) for row in rows]

    def order_page(
        self, session_id: str, *, limit: int, offset: int = 0
    ) -> tuple[list[OrderRecord], int]:
        """One page of orders, oldest first, with the session's order count."""
        rows, total = self._page("orders", session_id, limit, offset)
        return [_order_from_row(row) for row in rows], total

    # ---------------------------------------------------------------------- fills

    def record_fill(
        self,
        order_id: int,
        *,
        ts: datetime,
        price: float,
        lot: float,
        spread: float = 0.0,
        slippage: float = 0.0,
        commission: float = 0.0,
        sl: float | None = None,
        tp: float | None = None,
    ) -> int:
        """Append a fill; with ``sl`` and ``tp`` it also opens a position.

        An entry fill is the only fill that opens a position, so the two writes
        share one transaction: the ledger can never hold a fill whose position
        row was lost. Exit fills are not written here; the broker settles exits
        as trades, and ``record_trade`` closes the position.
        """
        if isinstance(order_id, bool) or not isinstance(order_id, int):
            raise TypeError(f"order_id must be an int, got {type(order_id).__name__}")
        stamp = _stamp(ts)
        fill_price = _require_finite(price, "price")
        size = _require_finite(lot, "lot")
        if size <= 0:
            raise ValueError(f"lot must be > 0, got {lot!r}")
        spread_value = _require_finite(spread, "spread")
        slippage_value = _require_finite(slippage, "slippage")
        commission_value = _require_finite(commission, "commission")
        if sl is None and tp is None:
            stop = take = None
        elif sl is None or tp is None:
            raise ValueError("sl and tp must be provided together")
        else:
            stop = _require_finite(sl, "sl")
            take = _require_finite(tp, "tp")

        with self._transaction() as tx:
            order = tx.execute(
                "SELECT session_id, side, status FROM orders WHERE order_id = ?", (order_id,)
            ).fetchone()
            if order is None:
                raise LedgerError(f"unknown order {order_id!r}")
            if str(order["status"]) != "filled":
                raise LedgerError(
                    f"order {order_id!r} is {str(order['status'])!r}; only a filled order"
                    " can receive a fill"
                )
            cursor = tx.execute(
                """
                INSERT INTO fills (
                    order_id, ts, price, spread, slippage, commission, lot
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    stamp,
                    fill_price,
                    spread_value,
                    slippage_value,
                    commission_value,
                    size,
                ),
            )
            fill_id = cursor.lastrowid
            if fill_id is None:  # pragma: no cover - sqlite always returns it
                raise LedgerError("fill insert returned no id")
            if stop is not None and take is not None:
                try:
                    tx.execute(
                        """
                        INSERT INTO positions (
                            session_id, order_id, side, lot, entry_ts, entry_price,
                            sl, tp, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')
                        """,
                        (
                            str(order["session_id"]),
                            order_id,
                            str(order["side"]),
                            size,
                            stamp,
                            fill_price,
                            stop,
                            take,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise LedgerError(
                        f"session {str(order['session_id'])!r} already has an open position"
                    ) from exc
        return int(fill_id)

    def fills(self, session_id: str) -> list[FillRecord]:
        """Every fill of the session, oldest first."""
        _require_text(session_id, "session_id")
        rows = (
            self._connection()
            .execute(
                """
                SELECT f.* FROM fills AS f
                JOIN orders AS o ON o.order_id = f.order_id
                WHERE o.session_id = ?
                ORDER BY f.fill_id
                """,
                (session_id,),
            )
            .fetchall()
        )
        return [_fill_from_row(row) for row in rows]

    # ------------------------------------------------------------------ positions

    def positions(self, session_id: str) -> list[PositionRecord]:
        """Every position of the session, oldest first."""
        _require_text(session_id, "session_id")
        rows = (
            self._connection()
            .execute(
                "SELECT * FROM positions WHERE session_id = ? ORDER BY position_id",
                (session_id,),
            )
            .fetchall()
        )
        return [_position_from_row(row) for row in rows]

    def position_page(
        self, session_id: str, *, limit: int, offset: int = 0
    ) -> tuple[list[PositionRecord], int]:
        """One page of positions, oldest first, with the session's position count."""
        rows, total = self._page("positions", session_id, limit, offset)
        return [_position_from_row(row) for row in rows], total

    # --------------------------------------------------------------------- trades

    def record_trade(
        self,
        session_id: str,
        *,
        side: Side | str,
        entry_ts: datetime,
        entry_price: float,
        exit_ts: datetime,
        exit_price: float,
        lot: float,
        pnl: float,
        r_multiple: float,
        exit_reason: str,
    ) -> int:
        """Append a closed trade and close the matching open position, if any.

        The values are the broker's; the ledger does not recompute P&L. The
        position is matched on ``side`` and ``entry_ts`` because a session holds
        one position at a time.
        """
        _require_text(session_id, "session_id")
        direction = _normalize_side(side)
        entry_stamp = _stamp(entry_ts, "entry_ts")
        exit_stamp = _stamp(exit_ts, "exit_ts")
        if _parse(exit_stamp, "exit_ts") < _parse(entry_stamp, "entry_ts"):
            raise ValueError(f"exit_ts {exit_ts!r} is before entry_ts {entry_ts!r}")
        entry_value = _require_finite(entry_price, "entry_price")
        exit_value = _require_finite(exit_price, "exit_price")
        size = _require_finite(lot, "lot")
        if size <= 0:
            raise ValueError(f"lot must be > 0, got {lot!r}")
        profit = _require_finite(pnl, "pnl")
        r_value = _require_finite(r_multiple, "r_multiple")
        reason = _require_text(exit_reason, "exit_reason")

        with self._transaction() as tx:
            self._require_session(tx, session_id)
            cursor = tx.execute(
                """
                INSERT INTO trades (
                    session_id, side, entry_ts, entry_price, exit_ts, exit_price,
                    lot, pnl, r_multiple, exit_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    direction,
                    entry_stamp,
                    entry_value,
                    exit_stamp,
                    exit_value,
                    size,
                    profit,
                    r_value,
                    reason,
                ),
            )
            trade_id = cursor.lastrowid
            if trade_id is None:  # pragma: no cover - sqlite always returns it
                raise LedgerError("trade insert returned no id")
            tx.execute(
                """
                UPDATE positions SET status = 'closed', closed_ts = ?, closed_price = ?
                WHERE position_id = (
                    SELECT position_id FROM positions
                    WHERE session_id = ? AND status = 'open'
                      AND side = ? AND entry_ts = ?
                    ORDER BY position_id DESC LIMIT 1
                )
                """,
                (exit_stamp, exit_value, session_id, direction, entry_stamp),
            )
        return int(trade_id)

    def trades(self, session_id: str) -> list[TradeRecord]:
        """Every closed trade of the session, oldest first."""
        _require_text(session_id, "session_id")
        rows = (
            self._connection()
            .execute(
                "SELECT * FROM trades WHERE session_id = ? ORDER BY trade_id",
                (session_id,),
            )
            .fetchall()
        )
        return [_trade_from_row(row) for row in rows]

    def trade_page(
        self, session_id: str, *, limit: int, offset: int = 0
    ) -> tuple[list[TradeRecord], int]:
        """One page of closed trades, oldest first, with the session's trade count."""
        rows, total = self._page("trades", session_id, limit, offset)
        return [_trade_from_row(row) for row in rows], total

    def count_trades(self, session_id: str) -> int:
        """How many trades the session closed; cheaper than reading them all."""
        _require_text(session_id, "session_id")
        row = (
            self._connection()
            .execute("SELECT COUNT(*) FROM trades WHERE session_id = ?", (session_id,))
            .fetchone()
        )
        return 0 if row is None else int(row[0])

    # --------------------------------------------------------------------- equity

    def record_equity(self, session_id: str, ts: datetime, balance: float, equity: float) -> int:
        """Append one equity mark; the curve is append-only, never updated."""
        _require_text(session_id, "session_id")
        stamp = _stamp(ts)
        balance_value = _require_finite(balance, "balance")
        equity_value = _require_finite(equity, "equity")
        with self._transaction() as tx:
            self._require_session(tx, session_id)
            cursor = tx.execute(
                """
                INSERT INTO equity_curve (session_id, ts, balance, equity)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, stamp, balance_value, equity_value),
            )
            mark_id = cursor.lastrowid
            if mark_id is None:  # pragma: no cover - sqlite always returns it
                raise LedgerError("equity insert returned no id")
        return int(mark_id)

    def equity(self, session_id: str) -> list[EquityPoint]:
        """Every equity mark of the session, oldest first."""
        _require_text(session_id, "session_id")
        rows = (
            self._connection()
            .execute(
                "SELECT * FROM equity_curve WHERE session_id = ? ORDER BY id",
                (session_id,),
            )
            .fetchall()
        )
        return [_equity_from_row(row) for row in rows]

    def equity_page(
        self, session_id: str, *, limit: int, offset: int = 0
    ) -> tuple[list[EquityPoint], int]:
        """The most recent equity window, in chronological order, and the total.

        ``offset`` pages backwards from the newest mark: ``offset=0`` is the
        tail of the curve, which is what a live chart draws. The rows inside the
        window are oldest-first so a caller can plot them without reversing.
        """
        rows, total = self._page("equity", session_id, limit, offset)
        return [_equity_from_row(row) for row in reversed(rows)], total

    # --------------------------------------------------------------------- resume

    def resume(self, session_id: str) -> ResumeState:
        """The state needed to continue ``session_id``, read from the ledger.

        ``last_bar_ts`` is the furthest bar the ledger has seen, over signals
        and equity marks. ``balance``/``equity`` are ``None`` before the first
        mark. Nothing here is inferred beyond those two aggregates: the open
        position is the row ``positions`` says is open, and pending orders are
        the rows ``orders`` says are pending.
        """
        record = self.session(session_id)
        if record is None:
            raise LedgerError(f"unknown session {session_id!r}")
        conn = self._connection()

        last_equity = conn.execute(
            """
            SELECT ts, balance, equity FROM equity_curve
            WHERE session_id = ? ORDER BY id DESC LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        last_signal = conn.execute(
            "SELECT MAX(ts) AS ts FROM signals WHERE session_id = ?", (session_id,)
        ).fetchone()
        last_ts: datetime | None = None
        for row in (last_signal, last_equity):
            if row is None or row["ts"] is None:
                continue
            candidate = _parse(row["ts"])
            if last_ts is None or candidate > last_ts:
                last_ts = candidate

        position_row = conn.execute(
            "SELECT * FROM positions WHERE session_id = ? AND status = 'open'"
            " ORDER BY position_id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        pending = tuple(
            _order_from_row(row)
            for row in conn.execute(
                "SELECT * FROM orders WHERE session_id = ? AND status = 'pending'"
                " ORDER BY order_id",
                (session_id,),
            ).fetchall()
        )
        return ResumeState(
            session_id=session_id,
            status=record.status,
            balance=None if last_equity is None else float(last_equity["balance"]),
            equity=None if last_equity is None else float(last_equity["equity"]),
            position=None if position_row is None else _position_from_row(position_row),
            last_bar_ts=last_ts,
            pending_orders=pending,
        )

    # -------------------------------------------------------------------- internal

    def _page(
        self,
        key: str,
        session_id: str | None,
        limit: int,
        offset: int,
    ) -> tuple[list[sqlite3.Row], int]:
        """Run one whitelisted page query and its total; no caller SQL."""
        size, start = _page_args(limit, offset)
        conn = self._connection()
        if session_id is None:
            rows = conn.execute(_PAGE_SQL[key], (size, start)).fetchall()
            total_row = conn.execute(_COUNT_SQL[key]).fetchone()
        else:
            _require_text(session_id, "session_id")
            rows = conn.execute(_PAGE_SQL[key], (session_id, size, start)).fetchall()
            total_row = conn.execute(_COUNT_SQL[key], (session_id,)).fetchone()
        total = 0 if total_row is None else int(total_row[0])
        return rows, total

    @staticmethod
    def _require_session(tx: sqlite3.Connection, session_id: str) -> None:
        known = tx.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if known is None:
            raise LedgerError(f"unknown session {session_id!r}")
