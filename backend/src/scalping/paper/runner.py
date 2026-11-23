"""Deterministic paper-trading session on top of :class:`Simulator`.

The runner is deliberately thin. It owns no fill logic: every price, P&L and
equity number comes from :class:`~scalping.env.simulator.Simulator`, which in
turn delegates fills to :class:`~scalping.env.broker.PaperBroker`. The runner's
job is to call ``step`` with the policy's action and write what happened to the
:class:`~scalping.paper.ledger.PaperLedger`.

Why replay instead of restoring a simulator
-------------------------------------------
The simulator is a pure function of ``(feature frame, settings, policy actions,
seed)``. The ledger records the actions. ``resume`` therefore rebuilds the
simulator from scratch and replays the recorded actions without writing, then
checks the replayed balance and open position against the ledger. If they
disagree, it raises :class:`ResumeError` instead of continuing on invented
state. This is slower than a serialised snapshot and much harder to get wrong:
there is no private simulator state to copy, and a divergence shows up as a
loud error on the next start.

The replay is only as deterministic as the policy's recorded actions: a
non-deterministic policy is harmless, because the ledger's actions are what is
replayed, not the policy.

No real-money path
------------------
This module imports no network client and holds no credentials. ``PaperSession``
can only write to a local SQLite file through :class:`PaperLedger`; there is no
code path that reaches a broker, an exchange or an MT5 terminal. The future
MT5 demo adapter is a separate, explicitly enabled module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import isfinite
from typing import Any, Final

import numpy as np

from scalping.config import Settings
from scalping.env.simulator import WINDOW, Simulator
from scalping.features.build import FeatureFrame
from scalping.paper.ledger import (
    LedgerError,
    PaperLedger,
    ResumeState,
    SessionRecord,
)
from scalping.types import Side

__all__ = [
    "ACTION_CODES",
    "ACTION_NAMES",
    "PaperRunResult",
    "PaperSession",
    "ResumeError",
]

#: Policy: an observation from the simulator, one action code back.
Policy = Callable[[np.ndarray], int]

#: Action code -> stored name. The simulator's contract is 0/1/2/3.
ACTION_NAMES: Final[dict[int, str]] = {0: "hold", 1: "buy", 2: "sell", 3: "close"}

#: Stored name -> action code, used to replay recorded signals.
ACTION_CODES: Final[dict[str, int]] = {name: code for code, name in ACTION_NAMES.items()}

_BUY: Final[int] = 1
_SELL: Final[int] = 2

#: Price tolerance when reconciling a replayed position against the ledger.
_PRICE_EPS: Final[float] = 1e-9


class ResumeError(LedgerError):
    """The ledger and a fresh simulator disagree; continuing would invent state."""


@dataclass(frozen=True, slots=True)
class PaperRunResult:
    """Outcome of one :meth:`PaperSession.run` call."""

    steps: int
    balance: float
    equity: float
    last_bar_ts: datetime | None
    done: bool
    reason: str | None


class PaperSession:
    """Drive one :class:`Simulator` and persist every step to a ledger.

    ``session_id`` identifies the paper run. Calling :meth:`run` on a session
    that is already in the ledger first replays the recorded actions, verifies
    the account against the stored state and only then appends new rows, so a
    process restart never duplicates history.

    ``started_at`` is injectable so two replays can produce byte-identical
    ledgers; production callers leave it out and the ledger stamps ``now``.
    """

    def __init__(
        self,
        ledger: PaperLedger,
        features: FeatureFrame,
        settings: Settings,
        policy: Policy,
        *,
        session_id: str,
        symbol: str = "XAUUSD",
        timeframe: str = "M1",
        strategy: str = "manual",
        model_version: str | None = None,
        config: Mapping[str, Any] | None = None,
        started_at: datetime | None = None,
        seed: int = 0,
        max_steps: int | None = None,
        window: int = WINDOW,
    ) -> None:
        if not callable(policy):
            raise TypeError(f"policy must be callable, got {type(policy).__name__}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")
        self.ledger = ledger
        self.features = features
        self.settings = settings
        self.policy = policy
        self.session_id = session_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy = strategy
        self.model_version = model_version
        self.started_at = started_at
        self.seed = seed
        self._config = self._build_config(config, window, max_steps)
        self.sim = Simulator(
            features,
            settings,
            strategy=strategy,
            max_steps=max_steps,
            window=window,
            seed=seed,
        )
        self._obs = self.sim.reset(seed)
        self._ready = False
        self._closed = False
        self._done = False
        self._reason: str | None = None
        self._balance = float(settings.contract.initial_balance)
        self._equity = float(settings.contract.initial_balance)
        self._last_bar_ts: datetime | None = None

    # ---------------------------------------------------------------- lifecycle

    def resume(self) -> ResumeState:
        """Create or attach to the ledger session and reconcile the simulator.

        Returns the ledger's :class:`ResumeState`. On an existing session the
        recorded actions are replayed first; a mismatch in balance or open
        position raises :class:`ResumeError`.
        """
        if self._ready:
            return self.ledger.resume(self.session_id)
        existing = self.ledger.session(self.session_id)
        if existing is None:
            self._open()
            self._ready = True
            return self.ledger.resume(self.session_id)
        if existing.status != "running":
            raise ResumeError(f"session {self.session_id!r} is {existing.status}, not running")
        self._replay()
        state = self.ledger.resume(self.session_id)
        self._verify(state)
        self._ready = True
        return state

    def run(self, bars: int) -> PaperRunResult:
        """Advance at most ``bars`` decision bars and persist each one.

        One bar is one ``Simulator.step``: the policy acts on the current
        decision bar and the simulator prices the consequences. The call stops
        early when the episode terminates or truncates.
        """
        if isinstance(bars, bool) or not isinstance(bars, int):
            raise TypeError(f"bars must be an int, got {type(bars).__name__}")
        if bars < 0:
            raise ValueError(f"bars must be >= 0, got {bars!r}")
        if self._closed:
            raise LedgerError(f"session {self.session_id!r} is closed")
        if not self._ready:
            self.resume()
        if self.sim.bar_index >= len(self.features.ts):
            # The frame is shorter than the observation window; the simulator
            # already reported "no_bars" and there is no bar to record.
            self._done = True
            self._reason = "no_bars"
            return PaperRunResult(
                steps=0,
                balance=self._balance,
                equity=self._equity,
                last_bar_ts=self._last_bar_ts,
                done=True,
                reason=self._reason,
            )

        steps = 0
        while steps < bars and not self._done:
            self._step()
            steps += 1
        return PaperRunResult(
            steps=steps,
            balance=self._balance,
            equity=self._equity,
            last_bar_ts=self._last_bar_ts,
            done=self._done,
            reason=self._reason,
        )

    def close(self) -> SessionRecord:
        """Close the ledger session; further :meth:`run` calls are refused."""
        if not self._ready:
            self.resume()
        record = self.ledger.close_session(self.session_id)
        self._closed = True
        return record

    # -------------------------------------------------------------------- internals

    def _build_config(
        self,
        config: Mapping[str, Any] | None,
        window: int,
        max_steps: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract": asdict(self.settings.contract),
            "risk": asdict(self.settings.risk),
            "costs": asdict(self.settings.costs),
            "seed": self.seed,
            "window": window,
            "max_steps": max_steps,
            "policy": _policy_label(self.policy),
        }
        if config is not None:
            payload["extra"] = dict(config)
        return payload

    def _open(self) -> SessionRecord:
        return self.ledger.open_session(
            self.session_id,
            symbol=self.symbol,
            timeframe=self.timeframe,
            strategy=self.strategy,
            model_version=self.model_version,
            started_at=self.started_at,
            config=self._config,
        )

    def _replay(self) -> None:
        """Re-run the recorded actions without writing; rebuild simulator state."""
        signals = self.ledger.signals(self.session_id)
        if not signals:
            return
        for signal in signals:
            code = ACTION_CODES.get(signal.action)
            if code is None:
                raise ResumeError(
                    f"session {self.session_id!r} holds unknown action {signal.action!r}"
                )
            result = self.sim.step(code)
            self._obs = result.obs
            self._balance = float(self.sim.balance)
            self._equity = float(result.info["equity"])
            self._last_bar_ts = signal.ts
            self._reason = result.info["reason"]
            self._done = result.terminated or result.truncated

    def _verify(self, state: ResumeState) -> None:
        """The replayed account must match the ledger's last mark exactly."""
        if state.balance is not None and not _close(self.sim.balance, state.balance):
            raise ResumeError(
                f"session {self.session_id!r}: replay balance {self.sim.balance!r}"
                f" != ledger {state.balance!r}"
            )
        replayed = self.sim.broker.position
        stored = state.position
        if (replayed is None) != (stored is None):
            raise ResumeError(
                f"session {self.session_id!r}: replay position {replayed!r} != ledger {stored!r}"
            )
        if replayed is not None and stored is not None:
            side = "long" if replayed.side is Side.LONG else "short"
            if (
                side != stored.side
                or not _close(replayed.lot, stored.lot)
                or not _close(replayed.entry_price, stored.entry_price)
            ):
                raise ResumeError(
                    f"session {self.session_id!r}: replay position {replayed!r}"
                    f" != ledger {stored!r}"
                )

    def _step(self) -> None:
        index = self.sim.bar_index
        ts = self._decision_ts(index)
        bid, ask, spread = self._quote(index)
        action = int(self.policy(self._obs))
        if action not in ACTION_NAMES:
            raise ValueError(f"policy returned {action!r}; expected 0, 1, 2 or 3")

        result = self.sim.step(action)
        blocked = result.info["blocked"]
        signal_id = self.ledger.record_signal(
            self.session_id,
            ts,
            ACTION_NAMES[action],
            reason=None if blocked is None else str(blocked),
            model_version=self.model_version,
        )

        opened = result.info["opened"]
        if action in (_BUY, _SELL):
            side = Side.LONG if action == _BUY else Side.SHORT
            requested = ask if side is Side.LONG else bid
            if opened is None:
                self.ledger.record_order(
                    self.session_id,
                    signal_id=signal_id,
                    ts=ts,
                    side=side,
                    lot=0.0,
                    requested_price=requested,
                    status="rejected",
                    rejection_reason=str(blocked or "unknown"),
                )
            else:
                order_id = self.ledger.record_order(
                    self.session_id,
                    signal_id=signal_id,
                    ts=ts,
                    side=side,
                    lot=opened.lot,
                    requested_price=requested,
                    status="filled",
                )
                self.ledger.record_fill(
                    order_id,
                    ts=ts,
                    price=opened.entry_price,
                    lot=opened.lot,
                    spread=spread,
                    slippage=abs(opened.entry_price - requested),
                    commission=self.settings.costs.commission_per_lot * opened.lot,
                    sl=opened.sl,
                    tp=opened.tp,
                )

        trade = result.info["trade"]
        if trade is not None:
            self.ledger.record_trade(
                self.session_id,
                side=trade.side,
                entry_ts=trade.entry_ts,
                entry_price=trade.entry,
                exit_ts=trade.exit_ts,
                exit_price=trade.exit,
                lot=trade.lot,
                pnl=trade.pnl,
                r_multiple=trade.r_multiple,
                exit_reason=trade.reason,
            )

        self._balance = float(self.sim.balance)
        self._equity = float(result.info["equity"])
        self._last_bar_ts = self._decision_ts(self.sim.bar_index)
        self.ledger.record_equity(self.session_id, self._last_bar_ts, self._balance, self._equity)
        self._obs = result.obs
        self._done = result.terminated or result.truncated
        self._reason = result.info["reason"]

    def _decision_ts(self, index: int) -> datetime:
        stamp = np.datetime64(self.features.ts[index], "us").astype(datetime)
        return stamp.replace(tzinfo=UTC)

    def _quote(self, index: int) -> tuple[float, float, float]:
        """``(bid, ask, spread)`` of the decision bar, matching the simulator."""
        bid = float(self.features.ohlc[index, 3])
        spread = float(self.features.spread[index])
        if not isfinite(spread) or spread < 0:
            spread = 0.0
        return bid, bid + spread, spread


def _policy_label(policy: Policy) -> str:
    return getattr(policy, "__qualname__", None) or type(policy).__qualname__


def _close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= _PRICE_EPS * max(1.0, abs(float(right)))
