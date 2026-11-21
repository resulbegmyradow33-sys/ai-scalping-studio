"""Sequenced replay loop behind the WebSocket feed.

The runner owns the simulator, the message sequence counter and a ring buffer of
the last :data:`RING_SIZE` messages. Connections are *cursors* into that ring:
each WebSocket has a ``last_sent`` sequence number, and a pump task (living in
the connection's own event loop) advances the simulator when it is due and then
sends every buffered message after the cursor.

There is deliberately no loop-bound "driver" task. The app factory is exercised
by ``TestClient``, which runs every HTTP request in a fresh event loop, so a
task created by ``POST /api/sim/start`` would be cancelled as soon as that
request finishes. Instead ``start`` only installs the simulator and flips a
playing flag; the connected socket's pump does the stepping under a threading
lock and a wall-clock due time keeps the rate at ``speed`` even with several
clients pumping concurrently.

v1 has no agent, so the replay policy is a fixed five-bar momentum rule: open
with the momentum sign when flat, close when it flips. It is deterministic, it
exercises the trade/risk message paths, and the strategy label on every message
is the single place the future policy hook replaces.
"""

from __future__ import annotations

import asyncio
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

__all__ = ["NoDataError", "SimConnection", "SimulationRunner"]

#: Stored timeframe the runner replays.
TIMEFRAME: Final[str] = "M1"

#: Number of messages kept for a reconnecting client.
RING_SIZE: Final[int] = 500

#: Symbol served when the caller does not name one.
DEFAULT_SYMBOL: Final[str] = "XAUUSD"

#: Label recorded on trades and ``decision`` messages.
STRATEGY: Final[str] = "momentum"

#: Poll interval while idle/paused, so a start or resume is seen quickly.
_IDLE_DELAY: Final[float] = 0.05

#: Longest sleep before state is re-checked (keeps pause/stop responsive).
_MAX_DELAY: Final[float] = 0.1

#: Messages sent per pump iteration; the remainder waits for the next tick.
_DRAIN_LIMIT: Final[int] = 200

#: Bars used for the replay momentum signal.
_MOMENTUM_BARS: Final[int] = 5

_HOLD: Final[int] = 0
_BUY: Final[int] = 1
_SELL: Final[int] = 2
_CLOSE: Final[int] = 3


class NoDataError(LookupError):
    """The store holds no bars, so there is nothing to replay."""


class SimConnection:
    """One WebSocket's cursor into the shared message ring."""

    __slots__ = ("last_sent",)

    def __init__(self, last_sent: int) -> None:
        self.last_sent = last_sent


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
        self._features: FeatureFrame | None = None
        self._sim: Simulator | None = None
        self._side: Side | None = None
        self._playing = False
        self._state = "idle"
        self._speed = 1.0
        self._session_id: str | None = None
        self._next_due = 0.0

    # --- state -----------------------------------------------------------
    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    @property
    def playing(self) -> bool:
        with self._lock:
            return self._playing

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def session_id(self) -> str | None:
        with self._lock:
            return self._session_id

    def status_message(self) -> dict[str, Any]:
        """The ``status`` frame every connection receives first (``seq`` 0)."""
        with self._lock:
            return {
                "seq": 0,
                "type": "status",
                "payload": {"state": self._state, "symbol": self._symbol},
            }

    # --- connection cursors ---------------------------------------------
    def connect(self) -> SimConnection:
        """A cursor that receives messages emitted after the current sequence."""
        return SimConnection(self.seq)

    def resume(self, conn: SimConnection, seq: int) -> None:
        """Replay buffered messages with ``seq`` greater than ``seq``."""
        with self._lock:
            conn.last_sent = max(0, min(int(seq), self._seq))

    # --- lifecycle -------------------------------------------------------
    def start(self, speed: float, mode: str = "replay") -> str:
        """Build a simulator from stored bars and mark the run playing."""
        if mode != "replay":
            raise ValueError(f"unsupported mode {mode!r}")
        if not speed >= 0.0:
            raise ValueError("speed must be non-negative")
        frame = self._store.read(self._symbol, TIMEFRAME)
        if frame.is_empty():
            raise NoDataError(
                f"no data for {self._symbol} ({TIMEFRAME}); download bars before starting"
            )
        features = build_features(frame)
        sim = Simulator(features, self._settings, strategy=STRATEGY)
        session_id = uuid.uuid4().hex
        with self._lock:
            self._features = features
            self._sim = sim
            self._side = None
            self._speed = float(speed)
            self._playing = True
            self._state = "running"
            self._session_id = session_id
            self._next_due = 0.0
            self._ring.clear()
        return session_id

    def control(self, action: str) -> bool:
        """Apply a control action; ``False`` when there is no run to act on."""
        with self._lock:
            if action == "stop":
                self._playing = False
                self._state = "stopped" if self._sim is not None else "idle"
                return True
            if action == "pause":
                if self._sim is None:
                    return False
                self._playing = False
                self._state = "paused"
                return True
            if action == "resume":
                if self._sim is None or self._state == "finished":
                    return False
                self._playing = True
                self._state = "running"
                self._next_due = 0.0
                return True
            if action == "reset":
                self._reset_locked()
                return True
            raise ValueError(f"unknown control action {action!r}")

    def _reset_locked(self) -> None:
        """Restart the episode from the beginning; the caller holds the lock."""
        features = self._features
        if features is None:
            frame = self._store.read(self._symbol, TIMEFRAME)
            if frame.is_empty():
                raise NoDataError(
                    f"no data for {self._symbol} ({TIMEFRAME}); download bars before resetting"
                )
            features = build_features(frame)
            self._features = features
        self._sim = Simulator(features, self._settings, strategy=STRATEGY)
        self._side = None
        self._playing = False
        self._state = "idle"
        self._session_id = None
        self._ring.clear()

    # --- pumping ----------------------------------------------------------
    async def pump(
        self,
        conn: SimConnection,
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Advance when due and send this cursor's buffered messages."""
        while True:
            delay = self._advance()
            batch = self._drain(conn)
            for message in batch:
                await send(message)
            await asyncio.sleep(0.0 if len(batch) >= _DRAIN_LIMIT else delay)

    def _advance(self) -> float:
        """Step the simulator if the schedule says it is due; return the sleep."""
        with self._lock:
            if not self._playing or self._sim is None:
                return _IDLE_DELAY
            now = time.monotonic()
            if now < self._next_due:
                return min(self._next_due - now, _MAX_DELAY)
            interval = 0.0 if self._speed <= 0.0 else 1.0 / self._speed
            self._next_due = now + interval
            self._step_locked()
            return interval

    def _drain(self, conn: SimConnection) -> list[dict[str, Any]]:
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
        previous = self._side
        result = sim.step(action)
        info = result.info

        self._emit("bar", _bar_payload(features, index))
        self._emit(
            "decision",
            {"action": int(action), "strategy": sim.strategy, "blocked": info["blocked"]},
        )
        position: Position | None = info["position"]
        trade: Trade | None = info["trade"]
        if previous is None and position is not None:
            self._emit("trade_open", _position_payload(position))
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
