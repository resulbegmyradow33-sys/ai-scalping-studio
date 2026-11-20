"""Deterministic accounting loop: risk decision, broker fill, equity, reward.

Timeline of one ``step`` at bar ``i`` (the *decision bar*):

1. Roll the 17:00 session anchor if bar ``i`` starts a new one. The anchor is
   the marked equity at the start of the step, so the daily loss gate always
   measures the session that is about to trade.
2. Apply the action at bar ``i``'s closing quote. An entry fills at the
   adverse side plus ATR slippage; Holds are free.
3. Advance to bar ``i + 1`` and let the broker evaluate stops and targets on
   that bar, then force-close a position that has reached
   ``RiskConfig.max_bars_in_trade``.
4. Mark equity, compute the reward from the step's equity delta and build the
   observation from rows ``i + 1 - 59 .. i + 1``.

The entry bar itself is only marked, never tested for SL/TP: its high and low
printed before the close the entry was filled at, so testing them would apply
past price action to a position opened afterwards. Exits are therefore always
evaluated on the first bar priced *after* the fill.

The simulator, not the broker, owns the balance, equity and day anchor; the
broker only reports fills and the unrealized PnL of its one position.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from math import isfinite
from typing import Any, Final, NamedTuple

import numpy as np

from scalping.config import Settings
from scalping.env.broker import PaperBroker
from scalping.env.risk import RiskEngine
from scalping.features.build import FeatureFrame
from scalping.types import AccountState, Bar, Position, Side, Trade

__all__ = ["MAX_STEPS", "WINDOW", "Simulator", "StepResult"]

#: Bars of ``X`` in one observation window.
WINDOW: Final[int] = 60

#: Hard episode length. A run truncates once this many ``step`` calls happened.
MAX_STEPS: Final[int] = 20_000

#: Position features appended after the flattened window.
POSITION_FEATURES: Final[int] = 5

#: Normaliser of ``bars_in_trade`` in the position features (v1 holding limit).
_BARS_NORM: Final[float] = 120.0

#: Fraction of the initial equity that may be lost before the episode ends.
_TERMINATE_FRACTION: Final[float] = 0.8

#: Episode drawdown allowance before the reward is penalised.
_DD_LIMIT: Final[float] = 0.03

#: Reward shaping weights.
_DD_PENALTY: Final[float] = 0.5
_OPEN_PENALTY: Final[float] = 0.02
_RISK_BONUS: Final[float] = 0.05
_REWARD_CLIP: Final[float] = 5.0

_HOLD: Final[int] = 0
_BUY: Final[int] = 1
_SELL: Final[int] = 2
_CLOSE: Final[int] = 3
_ACTIONS: Final[frozenset[int]] = frozenset({_HOLD, _BUY, _SELL, _CLOSE})


class StepResult(NamedTuple):
    """One environment transition; iterable as ``(obs, reward, term, trunc, info)``."""

    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class Simulator:
    """Single-account backtest loop over a :class:`FeatureFrame`.

    ``start``/``end`` bound the usable bar range (``end`` exclusive). The first
    decision bar is ``max(start, valid_from + WINDOW - 1)``; with fewer than
    ``WINDOW`` usable bars ``reset`` returns a zero observation and the first
    ``step`` truncates with ``info["reason"] == "no_bars"``.

    The reward is ``Δequity / initial * 100
    - 0.5 * max(0, drawdown - 0.03) - 0.02 * opened + 0.05 * risk_ok`` clipped
    to ``[-5, 5]``, where ``drawdown`` is the fractional drop from the episode
    peak, ``opened`` is 1 when the step opened a position and ``risk_ok`` is 1
    when an entry intent passed the risk gate. The episode terminates once
    equity reaches 80% of the initial balance and truncates at ``MAX_STEPS`` or
    when the data ends. ``info`` carries ``equity``, ``position``, ``trade``
    (closed this step), ``blocked``, ``strategy`` and ``reason``.
    """

    def __init__(
        self,
        features: FeatureFrame,
        settings: Settings,
        start: int = 0,
        end: int | None = None,
        strategy: str = "manual",
    ) -> None:
        self.features = features
        self.settings = settings
        self.risk = RiskEngine(settings.risk, settings.contract)
        self.strategy = strategy
        self._initial = float(settings.contract.initial_balance)
        self._n_features = int(features.X.shape[1])
        self.obs_dim = WINDOW * self._n_features + POSITION_FEATURES
        self._last_index = (len(features.X) if end is None else min(int(end), len(features.X))) - 1
        self._first_index = max(int(start), features.valid_from + WINDOW - 1, WINDOW - 1)

        # Episode state, fully established by reset().
        self.balance = self._initial
        self.day_start_equity = self._initial
        self.session: date | None = None
        self.steps = 0
        self.seed: int | None = None
        self.broker = PaperBroker(settings.contract, settings.costs, 0.0)
        self._i = self._first_index
        self._peak = self._initial
        self._equity_before = self._initial
        self._equity_value = self._initial
        self._last_obs = np.zeros(self.obs_dim, dtype=np.float32)
        self._done = False
        self._terminated = False
        self._no_bars = False
        self._reason: str | None = None
        self.reset()

    # --- public state ------------------------------------------------------

    @property
    def bar_index(self) -> int:
        """Index of the decision bar the next ``step`` will act on."""
        return self._i

    # --- episode -----------------------------------------------------------

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Restart the episode and return the first observation."""
        self.seed = seed
        self._i = self._first_index
        self.steps = 0
        self.balance = self._initial
        self.day_start_equity = self._initial
        self._peak = self._initial
        self._equity_before = self._initial
        self._equity_value = self._initial
        self._done = False
        self._terminated = False
        self._reason = None
        self.session = None
        self._no_bars = not self._has_window()
        self.broker = PaperBroker(
            self.settings.contract,
            self.settings.costs,
            self._atr(self._i) if not self._no_bars else 0.0,
        )
        if self._no_bars:
            self._done = True
            self._reason = "no_bars"
            self._last_obs = np.zeros(self.obs_dim, dtype=np.float32)
            return self._last_obs.copy()
        self.session = self.risk.session_anchor(self._ts(self._i))
        self._last_obs = self._observe()
        self._equity_value = self._equity()
        return self._last_obs.copy()

    def step(self, action: int) -> StepResult:
        """Apply ``action`` at the current bar and advance one bar."""
        if self._done:
            return StepResult(
                self._last_obs.copy(),
                0.0,
                self._terminated,
                not self._terminated,
                self._info(self._equity_value, None, None),
            )
        action = int(action)
        if action not in _ACTIONS:
            raise ValueError(f"action must be 0, 1, 2 or 3, got {action!r}")

        bar = self._bar(self._i)
        equity_now = self._equity()
        session = self.risk.session_anchor(bar.ts)
        if session != self.session:
            self.day_start_equity = equity_now
            self.session = session
        self._equity_before = equity_now

        trade, opened, risk_ok, blocked = self._apply(action, bar)

        if self._i >= self._last_index:
            # The fill was the last thing this data set can price.
            equity = self._equity()
            reason = "data_end"
        else:
            self._i += 1
            self.steps += 1
            next_bar = self._bar(self._i)
            closed = self.broker.update(next_bar)
            if closed is not None:
                trade = closed
                self.balance += closed.pnl
            if self.broker.position is not None:
                account = self.broker.state(self.balance, self.day_start_equity)
                if self.risk.must_force_close(account.bars_in_trade):
                    trade = self._close(next_bar, "timeout")
            equity = self._equity()
            reason = None

        self._peak = max(self._peak, equity)
        drawdown = (self._peak - equity) / self._peak if self._peak > 0 else 0.0
        reward = (equity - self._equity_before) / self._initial * 100.0
        reward -= _DD_PENALTY * max(0.0, drawdown - _DD_LIMIT)
        reward -= _OPEN_PENALTY * opened
        reward += _RISK_BONUS * risk_ok
        reward = float(np.clip(reward, -_REWARD_CLIP, _REWARD_CLIP))

        self._terminated = equity <= _TERMINATE_FRACTION * self._initial
        truncated = False
        if self._terminated:
            reason = "equity_depleted"
        elif self.steps >= MAX_STEPS:
            truncated = True
            reason = "max_steps"
        elif reason == "data_end":
            truncated = True
        self._done = self._terminated or truncated
        self._reason = reason
        self._equity_value = equity

        self._last_obs = self._observe()
        return StepResult(
            self._last_obs.copy(),
            reward,
            self._terminated,
            truncated,
            self._info(equity, trade, blocked),
        )

    # --- test hooks --------------------------------------------------------

    def _force_equity(self, value: float) -> None:
        """Test-only hook: set the balance so ``equity == value`` while flat.

        Any open position is closed at the current bar and its PnL discarded,
        so the next step starts from exactly ``value``. Never call this from a
        strategy, backtest or training run.
        """
        if not isfinite(value) or value <= 0:
            raise ValueError(f"value must be finite and > 0, got {value!r}")
        if self.broker.position is not None:
            self.broker.close(self._bar(self._i), "force_equity")
        self.balance = float(value)
        self._equity_value = float(value)

    # --- actions -----------------------------------------------------------

    def _apply(self, action: int, bar: Bar) -> tuple[Trade | None, float, float, str | None]:
        """Execute ``action`` at ``bar``; return ``(trade, opened, risk_ok, blocked)``."""
        trade: Trade | None = None
        blocked: str | None = None
        opened = 0.0
        risk_ok = 0.0

        if action in (_BUY, _SELL):
            account = self.broker.state(self.balance, self.day_start_equity)
            allowed, reason = self.risk.allow_new(account, bar.ts)
            if not allowed:
                blocked = reason
            elif self.broker.position is not None:  # only reachable with max_positions > 1
                blocked = "position_open"
            else:
                atr = self._atr(self._i)
                if atr <= 0:
                    blocked = "no_atr"
                else:
                    side = Side.LONG if action == _BUY else Side.SHORT
                    self.broker.atr = atr
                    slippage = self.settings.costs.slippage_atr_frac * atr
                    entry = (
                        bar.ask_close + slippage if side is Side.LONG else bar.bid_close - slippage
                    )
                    sl, tp = self.risk.sl_tp(entry, side, atr)
                    sizing = self.risk.size(account.equity, entry, sl)
                    if sizing.lot <= 0:
                        blocked = sizing.reason or "lot_too_small"
                    else:
                        self.broker.open(
                            side,
                            sizing.lot,
                            bar.ts,
                            bar.bid_close,
                            bar.ask_close,
                            sl,
                            tp,
                            self.strategy,
                        )
                        opened = 1.0
                        risk_ok = 1.0
        elif action == _CLOSE:
            if self.broker.position is None:
                blocked = "no_position"
            else:
                trade = self._close(bar, "manual")

        return trade, opened, risk_ok, blocked

    def _close(self, bar: Bar, reason: str) -> Trade:
        trade = self.broker.close(bar, reason)
        self.balance += trade.pnl
        return trade

    # --- observation and accounting ---------------------------------------

    def _observe(self) -> np.ndarray:
        """Flattened window of rows ``i-59 .. i`` plus the five position features."""
        start = self._i - WINDOW + 1
        window = self.features.X[start : self._i + 1].reshape(-1)
        flat = window.astype(np.float32, copy=True)
        account = self.broker.state(self.balance, self.day_start_equity)
        position = account.position
        if position is None:
            extras = (0.0, 0.0, 0.0, self._day_pnl_pct(account), 0.0)
        else:
            unrealized = account.equity - account.balance
            extras = (
                float(int(position.side)),
                account.bars_in_trade / _BARS_NORM,
                unrealized / account.balance if account.balance > 0 else 0.0,
                self._day_pnl_pct(account),
                self._risk_used(position, account.equity),
            )
        return np.concatenate((flat, np.asarray(extras, dtype=np.float32)))

    @staticmethod
    def _day_pnl_pct(account: AccountState) -> float:
        if account.day_start_equity <= 0:
            return 0.0
        return (account.equity - account.day_start_equity) / account.day_start_equity

    def _risk_used(self, position: Position, equity: float) -> float:
        """Fraction of equity currently at risk at the stop."""
        if equity <= 0:
            return 0.0
        distance = abs(position.entry_price - position.sl)
        return distance * self.settings.contract.size * position.lot / equity

    def _equity(self) -> float:
        return self.broker.state(self.balance, self.day_start_equity).equity

    def _info(self, equity: float, trade: Trade | None, blocked: str | None) -> dict[str, Any]:
        return {
            "equity": equity,
            "position": self.broker.position,
            "trade": trade,
            "blocked": blocked,
            "strategy": self.strategy,
            "reason": self._reason,
        }

    # --- bars --------------------------------------------------------------

    def _has_window(self) -> bool:
        return self._first_index <= self._last_index and self._first_index >= WINDOW - 1

    def _ts(self, index: int) -> datetime:
        stamp = np.datetime64(self.features.ts[index], "us").astype(datetime)
        return stamp.replace(tzinfo=UTC)

    def _atr(self, index: int) -> float:
        if 0 <= index < len(self.features.atr):
            value = float(self.features.atr[index])
            if isfinite(value) and value > 0:
                return value
        return 0.0

    def _bar(self, index: int) -> Bar:
        open_, high, low, close = (float(value) for value in self.features.ohlc[index])
        spread = float(self.features.spread[index])
        if not isfinite(spread) or spread < 0:
            spread = 0.0
        return Bar(
            ts=self._ts(index),
            open=open_,
            high=high,
            low=low,
            close=close,
            bid_close=close,
            ask_close=close + spread,
            spread=spread,
            volume=0,
        )
