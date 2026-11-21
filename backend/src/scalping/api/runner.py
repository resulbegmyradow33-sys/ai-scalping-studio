"""Sequenced replay loop behind the WebSocket feed.

The runner owns the simulator, the message sequence counter, a short session id
and a ring buffer of the last :data:`RING_SIZE` messages. Connections are
*cursors* into that ring: each WebSocket has a ``last_sent`` sequence number, a
``session_id`` it is synced to, and an ``awaiting_status`` flag. A pump task
(living in the connection's own event loop) advances the simulator when it is
due and then sends every buffered message after the cursor.

Contract, stated explicitly because it is unusual:

* **A run advances only while a socket is connected.** ``start`` installs the
  simulator and flips ``playing``; there is no global driver task. Each
  connected socket's pump performs the steps, taking the runner lock and a
  wall-clock due time so that N clients still advance at most one step per
  ``1/speed`` interval. The app factory is exercised by ``TestClient``, which
  runs every HTTP request in a fresh event loop, so a task created by
  ``POST /api/sim/start`` would be cancelled as soon as that request finished.
  Run the API with a single uvicorn worker (all state lives in ``app.state``).
* **``speed = 0`` is as fast as possible** (``asyncio.sleep(0)``); otherwise a
  step is due every ``1/speed`` seconds.
* **``seq`` restarts at 1 for every start/reset.** ``status`` is the only
  frame with ``seq = 0``; it carries ``state``, ``symbol``, ``session_id`` and
  ``head_seq``. A resume from a sequence above the head is clamped and answered
  with a fresh ``status`` handshake so the client can reset its gate.
* **``stop`` sets a ``threading.Event``** and ``start`` clears it. A stopped
  (or paused, or finished) pump does not poll: it blocks on a wake event with a
  timeout until start/resume. A new start/resume sets that wake event, so the
  latency is a state change, not a poll interval.
* **Control actions announce themselves.** ``stop``/``pause``/``resume``/
  ``reset`` mark every live connection's cursor dirty, and the next pump sends a
  fresh ``status`` frame, so the panel never keeps showing ``running`` after the
  run stopped.

v1 has no agent, so the replay policy is a fixed five-bar momentum rule: open
with the momentum sign when flat, close when it flips. It is deterministic, it
exercises the trade/risk message paths, and the strategy label on every message
is the single place the future policy hook replaces.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Final

import numpy as np

from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.env.simulator import Simulator
from scalping.features.build import FeatureFrame, build_features
from scalping.types import Position, Side, Trade

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SYMBOL",
    "MIN_BARS",
    "RING_SIZE",
    "TIMEFRAME",
    "NoDataError",
    "RunnerError",
    "SimConnection",
    "SimulationRunner",
]

#: Stored timeframe the runner replays.
TIMEFRAME: Final[str] = "M1"

#: Number of messages kept for a reconnecting client.
RING_SIZE: Final[int] = 500

#: Symbol served when the caller does not name one.
DEFAULT_SYMBOL: Final[str] = "XAUUSD"

#: Label recorded on trades and ``decision`` messages.
STRATEGY: Final[str] = "momentum"

#: Bars required before the simulator has its first decision bar.
#: ``Simulator`` starts at ``valid_from + WINDOW - 1`` = 60 + 59 = 119.
MIN_BARS: Final[int] = 120

#: Longest sleep before state is re-checked (keeps pause/stop responsive).
_MAX_DELAY: Final[float] = 0.1

#: How long an idle pump blocks on the wake event before re-checking state.
_IDLE_TIMEOUT: Final[float] = 0.25

#: Messages sent per pump iteration; the remainder waits for the next tick.
_DRAIN_LIMIT: Final[int] = 200

#: Bars used for the replay momentum signal.
_MOMENTUM_BARS: Final[int] = 5

_HOLD: Final[int] = 0
_BUY: Final[int] = 1
_SELL: Final[int] = 2
_CLOSE: Final[int] = 3


class RunnerError(RuntimeError):
    """A start/reset request cannot be served as asked."""


class NoDataError(RunnerError):
    """The store holds no bars, so there is nothing to replay."""


class SimConnection:
    """One WebSocket's cursor into the shared message ring."""

    __slots__ = ("awaiting_status", "last_sent", "session_id")

    def __init__(self, last_sent: int, session_id: str | None) -> None:
        self.last_sent = last_sent
        self.session_id = session_id
        self.awaiting_status = False


def _iso(stamp: datetime) -> str:
    """Render a stored UTC timestamp for JSON."""
    return (stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=UTC)).isoformat()


def _ts(features: FeatureFrame, index: int) -> str:
    stamp = np.datetime64(features.ts[index], "us").astype(datetime)
    return _iso(stamp.replace(tzinfo=UTC))


def _bar_payload(features: FeatureFrame, index: int) -> dict[str, Any]:
    open_, high, low, close = (float(value) for value in features.ohlc[index])
    return {
        "ts": _ts(features, index),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }


def _position_payload(position: Position) -> dict[str, Any]:
    return {
        "side": int(position.side),
        "lot": float(position.lot),
        "entry_price": float(position.entry_price),
        "entry_ts": _iso(position.entry_ts),
        "sl": float(position.sl),
        "tp": float(position.tp),
        "strategy": position.strategy,
    }


def _trade_payload(trade: Trade) -> dict[str, Any]:
    return {
        "entry_ts": _iso(trade.entry_ts),
        "exit_ts": _iso(trade.exit_ts),
        "side": int(trade.side),
        "lot": float(trade.lot),
        "entry": float(trade.entry),
        "exit": float(trade.exit),
        "sl": float(trade.sl),
        "tp": float(trade.tp),
        "pnl": float(trade.pnl),
        "costs": float(trade.costs),
        "r_multiple": float(trade.r_multiple),
        "reason": trade.reason,
        "strategy": trade.strategy,
    }


class SimulationRunner:
    """Replay one symbol's stored M1 bars with a monotonic message sequence."""

    def __init__(
        self,
        store: ParquetStore,
        settings: Settings,
        symbol: str = DEFAULT_SYMBOL,
        ring_size: int = RING_SIZE,
    ) -> None:
        if ring_size <= 0:
            raise ValueError("ring_size must be positive")
        self._store = store
        self._settings = settings
        self._symbol = symbol
        self._ring: deque[dict[str, Any]] = deque(maxlen=ring_size)
        self._lock = threading.RLock()
        self._seq = 0
        self._session_id: str | None = None
        self._features: FeatureFrame | None = None
        self._sim: Simulator | None = None
        self._side: Side | None = None
        self._playing = False
        self._state = "idle"
        self._speed = 1.0
        self._next_due = 0.0
        # Live cursor per WebSocket; control actions mark every cursor dirty so
        # its next pump announces the new state instead of leaving the panel
        # showing "running".
        self._connections: set[SimConnection] = set()
        # Stop is a state flag; wake is what idle pumps block on. A set
        # ``threading.Event`` returns from wait() immediately, so waiting on
        # ``_stop`` itself could never idle a stopped pump.
        self._stop = threading.Event()
        self._wake = threading.Event()

    # --- state -----------------------------------------------------------
    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    @property
    def session_id(self) -> str | None:
        with self._lock:
            return self._session_id

    @property
    def playing(self) -> bool:
        with self._lock:
            return self._playing

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def buffered_size(self) -> int:
        """Messages currently held for reconnects (bounded by the ring size)."""
        with self._lock:
            return len(self._ring)

    def _status_locked(self, head_seq: int | None = None) -> dict[str, Any]:
        """Status frame; ``head_seq`` defaults to the stream head."""
        return {
            "seq": 0,
            "type": "status",
            "payload": {
                "state": self._state,
                "symbol": self._symbol,
                "session_id": self._session_id,
                "head_seq": self._seq if head_seq is None else head_seq,
            },
        }

    def status_message(self) -> dict[str, Any]:
        """The ``status`` frame every connection receives first (``seq`` 0)."""
        with self._lock:
            return self._status_locked()

    # --- connection cursors ---------------------------------------------
    def connect(self) -> SimConnection:
        """A cursor that receives messages emitted after the current sequence."""
        with self._lock:
            conn = SimConnection(self._seq, self._session_id)
            self._connections.add(conn)
            return conn

    def disconnect(self, conn: SimConnection) -> None:
        """Forget a cursor; a closed socket must never be marked dirty again."""
        with self._lock:
            self._connections.discard(conn)

    def _announce_locked(self) -> None:
        """Mark every live cursor for a status handshake on its next pump."""
        for conn in self._connections:
            conn.awaiting_status = True

    def resume(self, conn: SimConnection, seq: int) -> None:
        """Replay buffered messages with ``seq`` greater than ``seq``.

        A sequence above the current head cannot be honoured: the cursor is
        clamped to the head and the next pump sends a fresh ``status``
        handshake with the true ``head_seq`` (and the session it belongs to).
        """
        with self._lock:
            if seq > self._seq:
                conn.last_sent = self._seq
                conn.awaiting_status = True
            else:
                conn.last_sent = max(0, int(seq))
        self._wake.set()

    def _sync(self, conn: SimConnection) -> dict[str, Any] | None:
        """Return a status frame when this cursor must re-handshake."""
        with self._lock:
            if conn.session_id != self._session_id and (
                conn.last_sent > 0 or conn.awaiting_status
            ):
                # A start/reset rewound the sequence under a live client, or a
                # control action asked for a status across that rewind. The
                # cursor goes back to 0, and the handshake advertises that same
                # 0: a client using head_seq as its resume gate must not skip
                # the visible start of the new session. The real head is not
                # relevant, because this cursor will be sent every event.
                conn.awaiting_status = False
                conn.session_id = self._session_id
                conn.last_sent = 0
                return self._status_locked(head_seq=0)
            if conn.awaiting_status:
                conn.awaiting_status = False
                conn.last_sent = min(conn.last_sent, self._seq)
                return self._status_locked()
            conn.session_id = self._session_id
            return None

    # --- lifecycle -------------------------------------------------------
    def _load_features(self) -> FeatureFrame:
        frame = self._store.read(self._symbol, TIMEFRAME)
        if frame.is_empty():
            raise NoDataError(
                f"no data for {self._symbol} ({TIMEFRAME}); download bars before starting"
            )
        features = build_features(frame, self._settings.costs)
        if features.X.shape[0] < MIN_BARS:
            raise RunnerError(f"not enough bars: need {MIN_BARS}")
        return features

    def start(self, speed: float, mode: str = "replay") -> str:
        """Build a simulator from stored bars and mark the run playing.

        A run advances only while at least one WebSocket pump is connected;
        this call installs the state and rewinds ``seq`` to the new session.
        """
        if mode != "replay":
            raise ValueError(f"unsupported mode {mode!r}")
        if not speed >= 0.0:
            raise ValueError("speed must be non-negative")
        features = self._load_features()
        sim = Simulator(features, self._settings, strategy=STRATEGY)
        session_id = uuid.uuid4().hex[:8]
        with self._lock:
            self._features = features
            self._sim = sim
            self._side = None
            self._speed = float(speed)
            self._playing = True
            self._state = "running"
            self._session_id = session_id
            self._seq = 0
            self._next_due = 0.0
            self._ring.clear()
        self._stop.clear()
        self._wake.set()
        return session_id

    def control(self, action: str) -> bool:
        """Apply a control action; ``False`` when there is no run to act on."""
        with self._lock:
            if action == "stop":
                self._playing = False
                self._stop.set()
                self._state = "stopped" if self._sim is not None else "idle"
                self._announce_locked()
                self._wake.set()
                return True
            if action == "pause":
                if self._sim is None:
                    return False
                self._playing = False
                self._state = "paused"
                self._announce_locked()
                self._wake.set()
                return True
            if action == "resume":
                if self._sim is None or self._state == "finished":
                    return False
                self._playing = True
                self._state = "running"
                self._next_due = 0.0
                self._stop.clear()
                self._announce_locked()
                self._wake.set()
                return True
            if action == "reset":
                self._reset_locked()
                self._stop.set()
                self._announce_locked()
                self._wake.set()
                return True
            raise ValueError(f"unknown control action {action!r}")

    def _reset_locked(self) -> None:
        """Restart the episode from the beginning; the caller holds the lock."""
        features = self._features if self._features is not None else self._load_features()
        self._features = features
        self._sim = Simulator(features, self._settings, strategy=STRATEGY)
        self._side = None
        self._playing = False
        self._state = "idle"
        self._session_id = uuid.uuid4().hex[:8]
        self._seq = 0
        self._ring.clear()

    def fail(self, reason: str = "error") -> None:
        """Stop stepping and queue an ``episode_end`` for an internal failure."""
        self._fail(reason)

    def _fail(self, reason: str) -> None:
        with self._lock:
            if self._state == "error":
                return
            self._playing = False
            self._state = "error"
            self._emit("episode_end", {"reason": reason})

    # --- pumping ----------------------------------------------------------
    async def pump(
        self,
        conn: SimConnection,
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Advance when due and send this cursor's buffered messages.

        Idle states (stopped, paused, never started, finished) block on the
        wake event instead of polling; start/resume set it.
        """
        while True:
            self._wake.clear()
            announcement = self._sync(conn)
            if announcement is not None:
                await send(announcement)
            if not self.playing or self._stop.is_set():
                batch = self.drain(conn)
                for message in batch:
                    await send(message)
                if len(batch) >= _DRAIN_LIMIT:
                    continue
                await asyncio.to_thread(self._wake.wait, _IDLE_TIMEOUT)
                continue
            delay = self._advance()
            batch = self.drain(conn)
            for message in batch:
                await send(message)
            await asyncio.sleep(0.0 if len(batch) >= _DRAIN_LIMIT else delay)

    def _advance(self) -> float:
        """Step the simulator if the schedule says it is due; return the sleep."""
        try:
            with self._lock:
                if not self._playing or self._sim is None:
                    return _IDLE_TIMEOUT
                now = time.monotonic()
                if now < self._next_due:
                    return min(self._next_due - now, _MAX_DELAY)
                interval = 0.0 if self._speed <= 0.0 else 1.0 / self._speed
                self._next_due = now + interval
                self._step_locked()
                return interval
        except Exception:
            # An unexpected step failure must not be swallowed by the pump
            # task; end the episode visibly instead of freezing the stream.
            logger.exception("simulator step failed")
            self._fail("error")
            return _IDLE_TIMEOUT

    def drain(self, conn: SimConnection) -> list[dict[str, Any]]:
        """Take up to a batch of messages after this cursor, advancing it."""
        with self._lock:
            batch = [message for message in self._ring if message["seq"] > conn.last_sent]
            batch = batch[:_DRAIN_LIMIT]
            if batch:
                conn.last_sent = batch[-1]["seq"]
            return batch

    # --- one step ---------------------------------------------------------
    def _step_locked(self) -> None:
        """Step the simulator and emit its messages; the caller holds the lock."""
        sim, features = self._sim, self._features
        if sim is None or features is None:  # pragma: no cover - guarded by _advance
            return
        index = sim.bar_index
        action = self._action(index, features)
        result = sim.step(action)
        info = result.info

        self._emit("bar", _bar_payload(features, index))
        self._emit(
            "decision",
            {"action": int(action), "strategy": sim.strategy, "blocked": info["blocked"]},
        )
        position: Position | None = info["position"]
        opened: Position | None = info["opened"]
        trade: Trade | None = info["trade"]
        if opened is not None:
            # ``opened`` survives a same-step stop-out, so trade_open is never
            # missing before a trade_close.
            self._emit("trade_open", _position_payload(opened))
        if trade is not None:
            self._emit("trade_close", _trade_payload(trade))
        if info["blocked"]:
            self._emit("risk_blocked", {"reason": info["blocked"]})
        self._emit("equity", {"equity": float(info["equity"]), "balance": float(sim.balance)})
        self._side = None if position is None else position.side

        if result.terminated or result.truncated:
            self._playing = False
            self._state = "finished"
            self._emit("episode_end", {"reason": info["reason"] or "finished"})

    def _action(self, index: int, features: FeatureFrame) -> int:
        if index < _MOMENTUM_BARS:
            return _HOLD
        close = float(features.ohlc[index, 3])
        base = float(features.ohlc[index - _MOMENTUM_BARS, 3])
        if base <= 0.0:
            return _HOLD
        momentum = close / base - 1.0
        if self._side is None:
            if momentum > 0.0:
                return _BUY
            if momentum < 0.0:
                return _SELL
            return _HOLD
        long = self._side is Side.LONG
        if (long and momentum < 0.0) or (not long and momentum > 0.0):
            return _CLOSE
        return _HOLD

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        self._seq += 1
        self._ring.append({"seq": self._seq, "type": kind, "payload": payload})
