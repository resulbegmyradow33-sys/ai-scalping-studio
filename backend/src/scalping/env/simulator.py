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

Strategy context
----------------
With a ``strategy_selector`` the observation gains six values at the end: a
one-hot of the arm chosen for the bar the observation describes plus that arm's
signal strength on the same bar. The bandit's context matrix and the strength
matrix are built once, in ``__init__``; a step only indexes the current row, so
the labelling cost is never paid per bar. ``reset`` selects for the first
decision bar and every ``step`` selects for the bar of the observation it
returns, so the policy always sees the strategy of the decision it is about to
make — exactly one ``select`` per decision bar. ``info["strategy"]`` is the arm
of the observation's bar; the label an entry carries is the arm chosen for the
bar the entry fills on (the arm the policy saw when it decided to act).

``labels`` is an optional :func:`~scalping.strategy.labeling.label_events` frame
for the context matrix's success columns; pass ``label_events(df, settings)`` so
those labels are priced with the same account the episode runs, instead of the
module's default :class:`~scalping.config.Settings`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from math import isfinite
from typing import Any, Final, NamedTuple, Protocol, runtime_checkable

import numpy as np
import polars as pl

from scalping.config import Settings
from scalping.env.broker import PaperBroker
from scalping.env.risk import RiskEngine
from scalping.features.build import FeatureFrame
from scalping.strategy.signals import ALL_STRATEGIES
from scalping.types import AccountState, Bar, Position, Side, Trade

__all__ = ["MAX_STEPS", "WINDOW", "ArmSelector", "Simulator", "StepResult"]


@runtime_checkable
class ArmSelector(Protocol):
    """A bandit-like selector: one call returns the arm for a context row."""

    def select(self, context: np.ndarray) -> int: ...


#: Bars of ``X`` in one observation window.
WINDOW: Final[int] = 60

#: Hard episode length. A run truncates once this many ``step`` calls happened.
MAX_STEPS: Final[int] = 20_000

#: Position features appended after the flattened window.
POSITION_FEATURES: Final[int] = 5

#: Strategy context appended when a selector is present: one-hot plus strength.
_STRATEGY_CONTEXT: Final[int] = len(ALL_STRATEGIES) + 1

#: Normaliser of ``bars_in_trade`` in the position features (v1 holding limit).
_BARS_NORM: Final[float] = 120.0

#: Fraction of the initial equity that may be lost before the episode ends.
_TERMINATE_FRACTION: Final[float] = 0.8

#: Episode drawdown allowance before the reward is penalised.
_DD_LIMIT: Final[float] = 0.03

#: Reward shaping weights (spec 2026-11-20, reward rebalance d48fc51).
#: ``_RISK_BONUS`` is a one-time entry credit, never a per-bar payment.
_DD_PENALTY: Final[float] = 50.0
_OPEN_PENALTY: Final[float] = 0.05
_RISK_BONUS: Final[float] = 0.02
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
    - 50 * max(0, excess - prev_excess) - 0.05 * opened + 0.02 * risk_ok``
    clipped to ``[-5, 5]``, where ``drawdown`` is the fractional drop from the
    episode peak, ``excess = max(0, drawdown - 0.03)`` and ``prev_excess`` is the
    highest excess already charged this episode. The drawdown term is therefore
    a one-time charge for reaching a **new** depth, not a per-bar tax that would
    make "never trade" the best policy. ``opened`` is 1.0 only on the step that
    opens a position, and ``risk_ok`` is a one-time 1.0 on that same
    gate-approved step (flat, hold and close steps earn no shaping bonus). The
    episode terminates once equity reaches 80% of the initial balance and
    truncates at ``max_steps`` (the constructor override, default
    :data:`MAX_STEPS`) or when the data ends; an open position is
    force-closed with reason ``"liquidation"`` when equity depletion terminates
    the episode, and billed through a ``"data_end"`` or ``"max_steps"`` quote
    close before truncating, so an episode never ends with an open position.
    ``info`` carries ``equity``, ``position``, ``opened`` (the position this
    step created, or ``None``), ``trade`` (closed this step), ``blocked``,
    ``strategy`` and ``reason``. With a ``strategy_selector``, ``strategy`` is
    the arm chosen for the bar the returned observation describes; otherwise it
    is the constructor's ``strategy`` argument.

    ``random_start`` makes every ``reset`` pick a new start uniformly from the
    range that still leaves ``max_steps`` bars (the last episode's start is at
    ``last_index - max_steps``); the RNG is seeded once at construction, so a
    reset stream is reproducible and a fixed seed replays the same starts. The
    default keeps the historical single start at ``start``. ``episode_start``
    reports the bar index the current episode began on.
    """

    def __init__(
        self,
        features: FeatureFrame,
        settings: Settings,
        start: int = 0,
        end: int | None = None,
        strategy: str = "manual",
        strategy_selector: Callable[[np.ndarray], int] | ArmSelector | None = None,
        labels: pl.DataFrame | None = None,
        max_steps: int | None = None,
        random_start: bool = False,
        seed: int = 0,
    ) -> None:
        self.features = features
        self.settings = settings
        self.max_steps = MAX_STEPS if max_steps is None else int(max_steps)
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")
        self.random_start = bool(random_start)
        self.rng_seed = int(seed)
        self._rng = np.random.default_rng(self.rng_seed)
        self.risk = RiskEngine(settings.risk, settings.contract)
        self.strategy = strategy
        self._base_strategy = strategy
        # ``LinUCB`` is called through ``select``; a bare callable is used as is.
        self._strategy_selector: Callable[[np.ndarray], int] | None
        if strategy_selector is None:
            self._strategy_selector = None
        elif isinstance(strategy_selector, ArmSelector):
            self._strategy_selector = strategy_selector.select
        elif callable(strategy_selector):
            self._strategy_selector = strategy_selector
        else:
            raise TypeError(
                "strategy_selector must be callable or expose select(context), "
                f"got {strategy_selector!r}"
            )
        self._arm = -1
        self._arm_strength = 0.0
        self._initial = float(settings.contract.initial_balance)
        self._n_features = int(features.X.shape[1])
        self.obs_dim = WINDOW * self._n_features + POSITION_FEATURES
        if self._strategy_selector is not None:
            self.obs_dim += _STRATEGY_CONTEXT
        self._last_index = (len(features.X) if end is None else min(int(end), len(features.X))) - 1
        self._first_index = max(int(start), features.valid_from + WINDOW - 1, WINDOW - 1)

        # Both matrices are frame-wide but causal for the rows they expose, so
        # building them once is equivalent to rebuilding a row per step without
        # the per-step labelling cost.
        self._strategy_context: np.ndarray | None = None
        self._strategy_strengths: np.ndarray | None = None
        if strategy_selector is not None:
            # Local import: labeling imports the env package (broker, risk), so a
            # module-level import here would break `import scalping.strategy.labeling`
            # while ``scalping.env`` is still initialising this module.
            from scalping.strategy.labeling import context_matrix, strength_matrix

            self._strategy_context = context_matrix(features, labels=labels)
            self._strategy_strengths = strength_matrix(features)

        # Episode state, fully established by reset().
        self.balance = self._initial
        self.day_start_equity = self._initial
        self.session: date | None = None
        self.steps = 0
        self.seed: int | None = None
        self.broker = PaperBroker(settings.contract, settings.costs, 0.0)
        self._i = self._first_index
        self._episode_start = self._first_index
        self._peak = self._initial
        self._prev_excess = 0.0
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

    @property
    def episode_start(self) -> int:
        """Index of the bar the current episode started from."""
        return self._episode_start

    # --- episode -----------------------------------------------------------

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Restart the episode and return the first observation."""
        self.seed = seed
        self._i = self._first_index
        if self.random_start and self._last_index >= self._first_index:
            # Keep a full horizon when the frame allows one; the last possible
            # start still leaves exactly ``max_steps`` bars to step through.
            upper = max(self._first_index, self._last_index - self.max_steps)
            self._i = int(self._rng.integers(self._first_index, upper + 1))
        self._episode_start = self._i
        self.steps = 0
        self.balance = self._initial
        self.day_start_equity = self._initial
        self._peak = self._initial
        self._prev_excess = 0.0
        self._equity_before = self._initial
        self._equity_value = self._initial
        self._done = False
        self._terminated = False
        self._reason = None
        self.session = None
        self.strategy = self._base_strategy
        self._arm = -1
        self._arm_strength = 0.0
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
        self._choose_strategy()
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
        decision_bar = self._i

        bar = self._bar(self._i)
        self.broker.atr = self._atr(self._i)
        equity_now = self._equity()
        session = self.risk.session_anchor(bar.ts)
        if session != self.session:
            self.day_start_equity = equity_now
            self.session = session
        self._equity_before = equity_now

        trade, opened, blocked = self._apply(action, bar)

        if self._i >= self._last_index:
            # No next bar to price the exit: bill an open position at this one.
            if self.broker.position is not None:
                trade = self._close(bar, "data_end")
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

        if self.broker.position is not None and equity <= _TERMINATE_FRACTION * self._initial:
            # Equity depletion with a live position: book the exit at this bar
            # so the episode never ends with an unaccounted trade. The close is
            # a quote exit, priced like the mark that triggered the termination.
            trade = self._close(self._bar(self._i), "liquidation")
            equity = self._equity()

        if self.broker.position is not None and self.steps >= self.max_steps:
            # Truncation must leave a flat, fully booked account: close through
            # the same quote path as ``data_end`` so balance == equity and every
            # dollar of PnL belongs to a trade.
            trade = self._close(self._bar(self._i), "max_steps")
            equity = self._equity()

        # Keep the broker's ATR current for the bar the next step will act on.
        # ``opened`` is only set after ``allow_new`` and ``size`` accepted the
        # entry, so it marks the one step that earns the risk-gate credit. The
        # bonus never repeats on hold steps, which would farm +0.02 per bar.
        self.broker.atr = self._atr(self._i)
        risk_ok = 1.0 if opened is not None else 0.0

        self._peak = max(self._peak, equity)
        drawdown = (self._peak - equity) / self._peak if self._peak > 0 else 0.0
        excess = max(0.0, drawdown - _DD_LIMIT)
        # Charge only the new depth. ``_prev_excess`` is monotone within the
        # episode, so recovering and falling back to the same level is free.
        penalty = _DD_PENALTY * max(0.0, excess - self._prev_excess)
        self._prev_excess = max(self._prev_excess, excess)
        reward = (equity - self._equity_before) / self._initial * 100.0
        reward -= penalty
        reward -= _OPEN_PENALTY * (1.0 if opened is not None else 0.0)
        reward += _RISK_BONUS * risk_ok
        reward = float(np.clip(reward, -_REWARD_CLIP, _REWARD_CLIP))

        self._terminated = equity <= _TERMINATE_FRACTION * self._initial
        truncated = False
        if self._terminated:
            reason = "equity_depleted"
        elif self.steps >= self.max_steps:
            truncated = True
            reason = "max_steps"
        elif reason == "data_end":
            truncated = True
        self._done = self._terminated or truncated
        self._reason = reason
        self._equity_value = equity

        # The returned observation describes the bar just advanced to, so pick
        # that bar's arm now. A step that could not advance (data end) keeps the
        # arm already chosen for it: one select per decision bar, never two.
        if self._i != decision_bar:
            self._choose_strategy()
        self._last_obs = self._observe()
        return StepResult(
            self._last_obs.copy(),
            reward,
            self._terminated,
            truncated,
            self._info(equity, trade, blocked, opened),
        )

    # --- test hooks --------------------------------------------------------

    def _force_equity(self, value: float) -> None:
        """Test-only hook: set the balance so ``equity == value``.

        Requires a flat account: overwriting the balance with a position open
        would drop a real ``Trade`` and desynchronise the broker ledger.
        Never call this from a strategy, backtest or training run.
        """
        if not isfinite(value) or value <= 0:
            raise ValueError(f"value must be finite and > 0, got {value!r}")
        if self.broker.position is not None:
            raise RuntimeError("_force_equity requires a flat account")
        self.balance = float(value)
        self._equity_value = float(value)

    # --- actions -----------------------------------------------------------

    def _apply(self, action: int, bar: Bar) -> tuple[Trade | None, Position | None, str | None]:
        """Execute ``action`` at ``bar``; return ``(trade, opened, blocked)``.

        ``opened`` is the :class:`Position` this call created, or ``None``. It
        is independent of ``broker.position`` afterwards, so a position that is
        stopped out on the very next mark is still reported as opened here.
        """
        trade: Trade | None = None
        blocked: str | None = None
        opened: Position | None = None

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
                        opened = self.broker.position
        elif action == _CLOSE:
            if self.broker.position is None:
                blocked = "no_position"
            else:
                trade = self._close(bar, "manual")

        return trade, opened, blocked

    def _close(self, bar: Bar, reason: str) -> Trade:
        trade = self.broker.close(bar, reason)
        self.balance += trade.pnl
        return trade

    # --- strategy ----------------------------------------------------------

    def _choose_strategy(self) -> None:
        """Select the arm for the current bar and stash its context.

        The current bar is the one the observation describes, so the policy
        sees the arm for the decision it is about to take. The selector gets
        that bar's context row only, so nothing past ``bar_index`` reaches it;
        its return value is validated here rather than failing later inside an
        index expression.
        """
        selector = self._strategy_selector
        if selector is None:
            return
        context = self._strategy_context
        strengths = self._strategy_strengths
        if context is None or strengths is None:  # pragma: no cover - built together
            raise RuntimeError("strategy context missing for a selector-equipped simulator")
        # A copy: a selector may keep or mutate the vector it is handed, and the
        # matrix is shared with every later bar.
        choice = selector(context[self._i].copy())
        if isinstance(choice, (bool, np.bool_)) or not isinstance(choice, (int, np.integer)):
            raise TypeError(f"strategy_selector must return an arm index, got {choice!r}")
        arm = int(choice)
        if not 0 <= arm < len(ALL_STRATEGIES):
            raise ValueError(f"strategy arm {arm} outside 0..{len(ALL_STRATEGIES) - 1}")
        self._arm = arm
        self._arm_strength = float(strengths[self._i, arm])
        self.strategy = ALL_STRATEGIES[arm].value

    # --- observation and accounting ---------------------------------------

    def _observe(self) -> np.ndarray:
        """Flattened window of rows ``i-59 .. i`` plus the five position features.

        With a selector, the chosen arm's one-hot and strength on this bar
        follow the position features.
        """
        start = self._i - WINDOW + 1
        window = self.features.X[start : self._i + 1].reshape(-1)
        flat = window.astype(np.float32, copy=True)
        if not np.isfinite(flat).all():
            raise ValueError(
                f"non-finite feature in the observation window ending at bar {self._i}"
            )
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
        position_features = np.asarray(extras, dtype=np.float32)
        if self._strategy_selector is None:
            return np.concatenate((flat, position_features))
        context = np.zeros(_STRATEGY_CONTEXT, dtype=np.float32)
        if self._arm >= 0:
            context[self._arm] = 1.0
            context[-1] = self._arm_strength
        return np.concatenate((flat, position_features, context))

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

    def _info(
        self,
        equity: float,
        trade: Trade | None,
        blocked: str | None,
        opened: Position | None = None,
    ) -> dict[str, Any]:
        return {
            "equity": equity,
            "position": self.broker.position,
            "opened": opened,
            "trade": trade,
            "blocked": blocked,
            "strategy": self.strategy,
            "reason": self._reason,
        }

    # --- bars --------------------------------------------------------------

    def _has_window(self) -> bool:
        return self._i >= WINDOW - 1 and self._i <= self._last_index

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
