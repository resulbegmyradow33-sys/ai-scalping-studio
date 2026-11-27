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

Kill switches
-------------
The simulator owns the state the Phase 11 limits need: ``week_start_equity``
(rolled on the same 17:00 UTC session boundary, at the ISO-week change),
``_peak`` (episode peak equity), ``consecutive_losses`` (reset by a win, by a
timed pause expiry, or at a session roll when ``pause_bars == 0``),
``paused_until`` and ``spread_points`` (bar spread / contract point). Every
entry passes ``RiskEngine.allow_new_extended``; a policy has no path around it.
A tripped weekly, drawdown or emergency switch force-closes an open position
with the switch name as ``Trade.reason``. With every Phase 11 field at ``0``
the loop and its results are the pre-Phase-11 behaviour.

The entry snapshot also carries ``slippage_points`` (the same
``ATR * slippage_atr_frac / point`` the fill will pay) and ``data_quality_ok``
(the decision bar's ATR, spread and gap from the previous bar). The engine
compares them only when ``max_slippage_points`` is non-zero and
``data_quality_block`` is on, so both gates are per-bar entry blocks that never
touch HOLD or CLOSE.

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

Action masks
------------
:meth:`Simulator.action_masks` reports which of the four actions the loop can
execute on the current bar: entries while flat, HOLD and CLOSE while in a
position, and HOLD only once the episode is over. It is a pure read of the
state, so a caller can ask before every decision without changing the episode.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime
from math import isfinite
from typing import Any, Final, NamedTuple, Protocol, runtime_checkable

import numpy as np
import polars as pl

from scalping.config import Settings
from scalping.data.registry import SessionSpec
from scalping.env.broker import PaperBroker
from scalping.env.rejections import RejectionConfig
from scalping.env.risk import RiskEngine
from scalping.features.build import FeatureFrame
from scalping.strategy.signals import ALL_STRATEGIES
from scalping.types import AccountState, Bar, Position, Side, Trade

__all__ = ["MAX_STEPS", "N_ACTIONS", "WINDOW", "ArmSelector", "Simulator", "StepResult"]


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

#: Flow observation: signed imbalance, its excess over the gate threshold, and
#: the 5-bar change.
_FLOW_CONTEXT: Final[int] = 3

#: Floor for the stale-bar gap tolerance, in minutes. Matches the data
#: validator's default (``scalping.data.validate``); the effective threshold is
#: the larger of this floor and the frame's own median bar spacing, so an H1
#: frame is not flagged for its normal 60-minute cadence while a missing H1 bar
#: (a 120-minute gap) still is.
_MIN_STALE_GAP_MINUTES: Final[float] = 10.0


def _gap_minutes(ts: np.ndarray) -> np.ndarray:
    """Open-to-open gap to the previous bar in minutes; ``NaN`` on row 0."""
    stamps = np.asarray(ts, dtype="datetime64[us]")
    gaps = np.full(stamps.shape, np.nan, dtype=np.float64)
    if stamps.size > 1:
        delta = np.diff(stamps).astype(np.timedelta64(1, "us")).astype(np.int64)
        gaps[1:] = delta / 60_000_000.0
    return gaps


def _stale_gap_threshold(gaps: np.ndarray) -> float:
    """Largest tolerated bar gap: the frame's median spacing, floored.

    Timestamps are schedule metadata, not market observations, and the cadence
    is fixed before the run, so deriving the threshold from the frame's own
    spacing does not look ahead at any price, feature or label.
    """
    positive = gaps[np.isfinite(gaps) & (gaps > 0)]
    nominal = float(np.median(positive)) if positive.size else 0.0
    return max(_MIN_STALE_GAP_MINUTES, nominal)


def _flow_features(flow: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """The three causal flow values appended to the observation."""
    flow64 = np.asarray(flow, dtype=np.float64)
    thr = np.asarray(threshold, dtype=np.float64)
    change = flow64 - np.concatenate(([flow64[0]] * 5, flow64[:-5]))
    with np.errstate(divide="ignore", invalid="ignore"):
        excess = np.where(thr > 0, (np.abs(flow64) - thr) / thr, 0.0)
    out = np.column_stack((flow64, excess, change)).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


_RISK_BONUS: Final[float] = 0.02
_REWARD_CLIP: Final[float] = 5.0

_HOLD: Final[int] = 0
_BUY: Final[int] = 1
_SELL: Final[int] = 2
_CLOSE: Final[int] = 3
_ACTIONS: Final[frozenset[int]] = frozenset({_HOLD, _BUY, _SELL, _CLOSE})

#: Size of the discrete action space: HOLD, BUY, SELL and CLOSE.
N_ACTIONS: Final[int] = len(_ACTIONS)


class StepResult(NamedTuple):
    """One environment transition; iterable as ``(obs, reward, term, trunc, info)``."""

    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class _BarPrep(NamedTuple):
    """State ``step`` will have after its pre-entry rolls, without mutating."""

    equity_now: float
    spread_points: float | None
    session: date
    day_start_equity: float
    week: tuple[int, int]
    week_start_equity: float
    consecutive_losses: int
    paused_until: datetime | None


class _PendingEntry(NamedTuple):
    """A BUY/SELL/CLOSE decided on one bar and applied on the next."""

    action: int
    decision_bar: int
    strategy: str
    arm: int
    decision_close: float
    decision_atr: float


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
        entry_penalty: float | None = None,
        gate_signals: bool = False,
        window: int = WINDOW,
        flow: np.ndarray | None = None,
        flow_thr: np.ndarray | None = None,
        flow_obs: bool = False,
        session: SessionSpec | None = None,
        execution_delay: int = 0,
        rejections: RejectionConfig | None = None,
    ) -> None:
        if isinstance(execution_delay, bool) or not isinstance(execution_delay, int):
            raise TypeError(f"execution_delay must be an int, got {type(execution_delay).__name__}")
        if execution_delay not in (0, 1):
            raise ValueError(f"execution_delay must be 0 or 1, got {execution_delay!r}")
        self.features = features
        self.execution_delay = execution_delay
        self.rejections = rejections
        self._pending_entry: _PendingEntry | None = None
        self.session_spec = session
        self.settings = settings
        self.max_steps = MAX_STEPS if max_steps is None else int(max_steps)
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")
        if entry_penalty is None:
            self.entry_penalty = _OPEN_PENALTY
        else:
            if not isfinite(entry_penalty) or entry_penalty < 0.0:
                raise ValueError(f"entry_penalty must be finite and >= 0, got {entry_penalty!r}")
            self.entry_penalty = float(entry_penalty)
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
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive int, got {window!r}")
        self.window = int(window)
        height = int(features.X.shape[0])
        if flow is not None:
            flow_arr = np.asarray(flow, dtype=np.float64)
            if flow_arr.shape != (height,):
                raise ValueError(f"flow must have shape ({height},), got {flow_arr.shape}")
        else:
            flow_arr = None
        if flow_thr is not None:
            thr_arr = np.asarray(flow_thr, dtype=np.float64)
            if thr_arr.shape != (height,):
                raise ValueError(f"flow_thr must have shape ({height},), got {thr_arr.shape}")
        else:
            thr_arr = None
        if (flow_arr is None) != (thr_arr is None):
            raise ValueError("flow and flow_thr must be provided together")
        self._flow = flow_arr
        self._flow_thr = thr_arr
        self._flow_obs: np.ndarray | None = None
        if flow_arr is not None and thr_arr is not None and flow_obs:
            self._flow_obs = _flow_features(flow_arr, thr_arr)
        self.obs_dim = (
            self.window * self._n_features
            + POSITION_FEATURES
            + (0 if self._flow_obs is None else _FLOW_CONTEXT)
        )
        if self._strategy_selector is not None:
            self.obs_dim += _STRATEGY_CONTEXT
        self._last_index = (len(features.X) if end is None else min(int(end), len(features.X))) - 1
        self._first_index = max(int(start), features.valid_from + self.window - 1, self.window - 1)
        # Bar-gap metadata for the data-quality gate: the gap before each bar
        # and the cadence this frame is expected to keep.
        self._bar_gaps = _gap_minutes(features.ts)
        self._stale_gap_minutes = _stale_gap_threshold(self._bar_gaps)

        # Both matrices are frame-wide but causal for the rows they expose, so
        # building them once is equivalent to rebuilding a row per step without
        # the per-step labelling cost.
        self._strategy_context: np.ndarray | None = None
        self._strategy_strengths: np.ndarray | None = None
        self._strategy_signals: np.ndarray | None = None
        self.gate_signals = bool(gate_signals)
        if strategy_selector is not None:
            # Local import: labeling imports the env package (broker, risk), so a
            # module-level import here would break `import scalping.strategy.labeling`
            # while ``scalping.env`` is still initialising this module.
            from scalping.strategy.labeling import (
                context_matrix,
                signal_matrix,
                strength_matrix,
            )

            self._strategy_context = context_matrix(features, labels=labels)
            self._strategy_strengths = strength_matrix(features)
            self._strategy_signals = signal_matrix(features)

        # Episode state, fully established by reset().
        self.balance = self._initial
        self.day_start_equity = self._initial
        self.week_start_equity = self._initial
        self.session: date | None = None
        self.week: tuple[int, int] | None = None
        self.consecutive_losses = 0
        self.paused_until: datetime | None = None
        self.spread_points: float | None = None
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

    @property
    def slippage_points(self) -> float | None:
        """Estimated entry slippage on the current decision bar, in points."""
        return self._slippage_points_at(self._i)

    @property
    def data_quality_ok(self) -> bool:
        """Whether the current decision bar's data is usable for an entry."""
        return self._data_quality_at(self._i)

    @property
    def pending_action(self) -> int | None:
        """The queued BUY/SELL that fills on the next step, or ``None``.

        Only used when ``execution_delay=1``; the pending intent is never a
        position and is cleared by ``reset`` and by terminal steps.
        """
        return None if self._pending_entry is None else self._pending_entry.action


    def _prepare_bar_state(self, bar: Bar) -> _BarPrep:
        """Project the rolls ``step`` performs before an entry, without mutating.

        The mask must see the same state the loop will act on: the current
        bar's spread, an expired pause latch and the 17:00/ISO-week anchor
        resets. ``step`` applies this projection; ``action_masks`` only reads it.
        """
        equity_now = self._equity()
        spread_points = self._spread_points_at(self._i)
        paused_until = self.paused_until
        consecutive_losses = self.consecutive_losses
        if paused_until is not None and bar.ts >= paused_until:
            paused_until = None
            consecutive_losses = 0
        session = self.risk.session_anchor(bar.ts)
        day_start_equity = self.day_start_equity
        if session != self.session:
            day_start_equity = equity_now
            if self.risk.risk.pause_bars == 0:
                # A zero-pause consecutive-loss block lasts until the anchor
                # resets; the roll is that reset.
                consecutive_losses = 0
        week = self.risk.week_anchor(bar.ts)
        week_start_equity = self.week_start_equity
        if week != self.week:
            week_start_equity = equity_now
        return _BarPrep(
            equity_now=equity_now,
            spread_points=spread_points,
            session=session,
            day_start_equity=day_start_equity,
            week=week,
            week_start_equity=week_start_equity,
            consecutive_losses=consecutive_losses,
            paused_until=paused_until,
        )

    def _projected_account(self, bar: Bar) -> AccountState:
        """The account snapshot the next entry will be judged against."""
        prep = self._prepare_bar_state(bar)
        state = self.broker.state(self.balance, prep.day_start_equity)
        return replace(
            state,
            week_start_equity=prep.week_start_equity,
            peak_equity=self._peak,
            consecutive_losses=prep.consecutive_losses,
            paused_until=prep.paused_until,
            spread_points=prep.spread_points,
            slippage_points=self._slippage_points_at(self._i),
            data_quality_ok=self._data_quality_at(self._i),
        )

    def action_masks(self) -> np.ndarray:
        """Boolean availability of HOLD, BUY, SELL and CLOSE at this bar.

        The mask mirrors the loop's own rejections and is a pure read: it never
        steps, marks or mutates the account, and twice in a row returns the same
        vector.

        * done, or a frame with no usable bars: only HOLD;
        * position open: HOLD and CLOSE (a second entry would be rejected);
        * flat: HOLD plus BUY/SELL only when :meth:`_entry_reason` finds no block
          — session hours, daily/weekly loss, drawdown, kill switches,
          consecutive-loss pause, spread limit, slippage cap, unusable bar
          data, flow/signal gate, missing ATR or a lot below the contract
          minimum;
        * flat while every entry is blocked: HOLD only.

        It is evaluated against the same projected rolls ``step`` applies
        (current-bar spread, pause expiry, 17:00 and ISO-week anchors), so it
        never hides an action the loop will accept and never allows an entry the
        loop will reject.
        """
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[_HOLD] = True
        if self._done or self._no_bars:
            return mask
        if self.broker.position is not None:
            mask[_CLOSE] = True
            return mask
        mask[_BUY] = self._entry_reason(_BUY) is None
        mask[_SELL] = self._entry_reason(_SELL) is None
        return mask

    def _entry_reason(
        self,
        action: int,
        *,
        arm: int | None = None,
        signal_index: int | None = None,
    ) -> str | None:
        """The reason an entry is blocked right now, or ``None`` when it may open.

        Shared by :meth:`action_masks` and :meth:`_apply` so the mask and the
        accounting loop cannot disagree. Read-only: it sizes the order to check
        the minimum lot but opens nothing.
        """
        if action not in (_BUY, _SELL):
            raise ValueError(f"entry action must be BUY or SELL, got {action!r}")
        if self._done or self._no_bars:
            return "no_bars"
        bar = self._bar(self._i)
        if self.session_spec is not None and not self.session_spec.is_open(bar.ts):
            return "session_closed"
        if self.rejections is not None and self.rejections.reject_zero_volume and bar.volume <= 0:
            return "rejected_no_liquidity"
        account = self._projected_account(bar)
        allowed, reason = self.risk.allow_new_extended(account, bar.ts)
        if not allowed:
            return reason or "risk_blocked"
        if self.broker.position is not None:  # only reachable with max_positions > 1
            return "position_open"
        gate_reason = self._gate_reason(action, arm=arm, signal_index=signal_index)
        if gate_reason is not None:
            return gate_reason
        atr = self._atr(self._i)
        if atr <= 0:
            return "no_atr"
        side = Side.LONG if action == _BUY else Side.SHORT
        slippage = self.settings.costs.slippage_atr_frac * atr
        entry = bar.ask_close + slippage if side is Side.LONG else bar.bid_close - slippage
        sl, _tp = self.risk.sl_tp(entry, side, atr)
        sizing = self.risk.size(account.equity, entry, sl)
        if sizing.lot <= 0:
            return sizing.reason or "lot_too_small"
        return None

    # --- episode -----------------------------------------------------------

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Restart the episode and return the first observation."""
        self._pending_entry = None
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
        self.week_start_equity = self._initial
        self._peak = self._initial
        self._prev_excess = 0.0
        self._equity_before = self._initial
        self._equity_value = self._initial
        self._done = False
        self._terminated = False
        self._reason = None
        self.session = None
        self.week = None
        self.consecutive_losses = 0
        self.paused_until = None
        self.spread_points = None
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
        self.week = self.risk.week_anchor(self._ts(self._i))
        self.spread_points = self._spread_points_at(self._i)
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
        decision_strategy = self.strategy
        decision_arm = self._arm
        decision_close = float(self.features.ohlc[self._i, 3])
        decision_atr = self._atr(self._i)

        bar = self._bar(self._i)
        self.broker.atr = self._atr(self._i)
        prep = self._prepare_bar_state(bar)
        self.spread_points = prep.spread_points
        self.paused_until = prep.paused_until
        self.consecutive_losses = prep.consecutive_losses
        self.day_start_equity = prep.day_start_equity
        self.session = prep.session
        self.week = prep.week
        self.week_start_equity = prep.week_start_equity
        self._equity_before = prep.equity_now

        if self.execution_delay:
            trade, opened, blocked = self._fill_pending(bar)
            if action == _CLOSE and self.broker.position is None:
                # A close with nothing open is rejected now; queueing it would
                # only turn into the same rejection one bar later.
                _, _, blocked = self._apply(_CLOSE, bar)
        else:
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
                self._book(closed)
            if self.broker.position is not None:
                account = self.broker.state(self.balance, self.day_start_equity)
                if self.risk.must_force_close(account.bars_in_trade):
                    trade = self._close(next_bar, "timeout")
            equity = self._equity()
            reason = None

        # Peak-to-current drawdown is measured against the episode peak,
        # including the mark that just happened. The reward's own peak update
        # stays below, unchanged.
        peak = max(self._peak, equity)
        kill = self.risk.kill_switch_reason(self._account(peak))
        if kill is not None:
            self._latch_pause()
            if self.broker.position is not None:
                # Weekly/drawdown/emergency switches own the exit: the policy
                # cannot hold a position through a tripped equity kill switch.
                trade = self._close(self._bar(self._i), kill)
                equity = self._equity()
        self._maybe_latch_consecutive()

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
        reward -= self.entry_penalty * (1.0 if opened is not None else 0.0)
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
        if self.execution_delay:
            if not self._done and (
                action in (_BUY, _SELL)
                or (action == _CLOSE and self.broker.position is not None)
            ):
                self._pending_entry = _PendingEntry(
                    action=action,
                    decision_bar=decision_bar,
                    strategy=decision_strategy,
                    arm=decision_arm,
                    decision_close=decision_close,
                    decision_atr=decision_atr,
                )
            elif self._pending_entry is not None and not self._done:
                pass  # a pending entry queued earlier stays; it was not filled
            else:
                self._pending_entry = None

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

    def _signal_for_arm(
        self, arm: int | None = None, signal_index: int | None = None
    ) -> int:
        """The selected arm's signal, 0 when unknown.

        A delayed fill passes the arm and bar frozen at decision time; the flow
        gate still reads the fill bar.
        """
        chosen = self._arm if arm is None else arm
        index = self._i if signal_index is None else signal_index
        if self._strategy_signals is None or chosen < 0:
            return 0
        return int(self._strategy_signals[index, int(chosen)])

    def _flow_gate_blocked(self) -> bool:
        """True when the flow gate is active and this bar's imbalance is too weak.

        Shared by :meth:`_gate_reason` and :meth:`action_masks`, so the mask and
        the entry path can never disagree about what the flow gate allows. A
        non-finite imbalance or threshold counts as blocked, matching the entry
        path.
        """
        if self._flow is None or self._flow_thr is None:
            return False
        value = float(self._flow[self._i])
        threshold = float(self._flow_thr[self._i])
        return not (isfinite(value) and isfinite(threshold)) or abs(value) <= threshold

    def _gate_reason(
        self,
        action: int,
        *,
        arm: int | None = None,
        signal_index: int | None = None,
    ) -> str | None:
        """The reason this entry is blocked, or ``None`` when it may proceed."""
        if self.gate_signals and not self._signals_unknown(arm=arm):
            signal = self._signal_for_arm(arm=arm, signal_index=signal_index)
            if signal == 0:
                return "no_signal"
            if (action == _BUY and signal < 0) or (action == _SELL and signal > 0):
                return "signal_conflict"
        if self._flow_gate_blocked():
            return "no_flow"
        return None

    def _signals_unknown(self, arm: int | None = None) -> bool:
        chosen = self._arm if arm is None else arm
        return self._strategy_signals is None or chosen < 0

    def _fill_pending(self, bar: Bar) -> tuple[Trade | None, Position | None, str | None]:
        """Fill the queued BUY/SELL at this bar, if any."""
        pending = self._pending_entry
        self._pending_entry = None
        if pending is None:
            return None, None, None
        if self.rejections is not None and pending.action in (_BUY, _SELL):
            if self.rejections.reject_zero_volume and bar.volume <= 0:
                return None, None, "rejected_no_liquidity"
            if (
                self.rejections.max_gap_atr > 0.0
                and pending.decision_atr > 0.0
                and abs(bar.close - pending.decision_close)
                > self.rejections.max_gap_atr * pending.decision_atr
            ):
                return None, None, "rejected_price_gap"
        return self._apply(
            pending.action,
            bar,
            strategy=pending.strategy,
            arm=pending.arm,
            signal_index=pending.decision_bar,
        )

    def _apply(
        self,
        action: int,
        bar: Bar,
        *,
        strategy: str | None = None,
        arm: int | None = None,
        signal_index: int | None = None,
    ) -> tuple[Trade | None, Position | None, str | None]:
        """Execute ``action`` at ``bar``; return ``(trade, opened, blocked)``.

        ``opened`` is the :class:`Position` this call created, or ``None``. It
        is independent of ``broker.position`` afterwards, so a position that is
        stopped out on the very next mark is still reported as opened here.
        """
        trade: Trade | None = None
        blocked: str | None = None
        opened: Position | None = None

        if action in (_BUY, _SELL):
            blocked = self._entry_reason(action, arm=arm, signal_index=signal_index)
            if blocked is None:
                account = self._account()
                atr = self._atr(self._i)
                side = Side.LONG if action == _BUY else Side.SHORT
                slippage = self.settings.costs.slippage_atr_frac * atr
                entry = bar.ask_close + slippage if side is Side.LONG else bar.bid_close - slippage
                sl, tp = self.risk.sl_tp(entry, side, atr)
                sizing = self.risk.size(account.equity, entry, sl)
                self.broker.open(
                    side,
                    sizing.lot,
                    bar.ts,
                    bar.bid_close,
                    bar.ask_close,
                    sl,
                    tp,
                    self.strategy if strategy is None else strategy,
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
        self._book(trade)
        return trade

    # --- kill-switch state --------------------------------------------------

    def _account(self, peak_equity: float | None = None) -> AccountState:
        """The broker snapshot plus the simulator-owned kill-switch state."""
        state = self.broker.state(self.balance, self.day_start_equity)
        return replace(
            state,
            week_start_equity=self.week_start_equity,
            peak_equity=self._peak if peak_equity is None else peak_equity,
            consecutive_losses=self.consecutive_losses,
            paused_until=self.paused_until,
            spread_points=self.spread_points,
            slippage_points=self._slippage_points_at(self._i),
            data_quality_ok=self._data_quality_at(self._i),
        )

    def _book(self, trade: Trade) -> None:
        """Apply a closed trade to the balance and the loss streak."""
        self.balance += trade.pnl
        self.consecutive_losses = self.consecutive_losses + 1 if trade.pnl < 0 else 0
        self._maybe_latch_consecutive()

    def _expire_pause(self, ts: datetime) -> None:
        """Clear a timed latch once its bar is reached and restart the streak."""
        if self.paused_until is not None and ts >= self.paused_until:
            self.paused_until = None
            self.consecutive_losses = 0

    def _latch_pause(self) -> None:
        """Hold entries for ``pause_bars`` decision bars after the trip bar.

        ``paused_until`` is the timestamp of the first bar at which entries are
        allowed again, so the next ``pause_bars`` decision bars stay blocked.
        The target is clamped to the last available bar. ``pause_bars == 0``
        leaves the switch condition-driven and does not latch.
        """
        if self.risk.risk.pause_bars <= 0 or self.paused_until is not None:
            return
        target = min(self._i + self.risk.risk.pause_bars + 1, self._last_index)
        if target > self._i:
            self.paused_until = self._ts(target)

    def _maybe_latch_consecutive(self) -> None:
        limit = self.risk.risk.consecutive_loss_pause
        if limit > 0 and self.consecutive_losses >= limit:
            self._latch_pause()

    def _spread_points_at(self, index: int) -> float | None:
        point = self.settings.contract.point
        if not isfinite(point) or point <= 0:
            return None
        spread = float(self.features.spread[index])
        if not isfinite(spread) or spread < 0:
            return None
        return spread / point

    def _slippage_points_at(self, index: int) -> float | None:
        """Estimated slippage of an entry at ``index``, in contract points.

        Built from the same two inputs the fill uses — ``_atr`` and
        ``CostConfig.slippage_atr_frac`` — so ``max_slippage_points`` caps the
        order that would actually be sent instead of a separate estimate.
        """
        point = self.settings.contract.point
        if not isfinite(point) or point <= 0:
            return None
        atr = self._atr(index)
        if atr <= 0:
            return None
        slippage = self.settings.costs.slippage_atr_frac * atr
        if not isfinite(slippage) or slippage < 0:
            return None
        return slippage / point

    def _data_quality_at(self, index: int) -> bool:
        """True when the bar at ``index`` is usable for an entry.

        Checks the decision bar only, and only its own inputs: ATR present and
        positive, spread present and non-negative (the raw feature, before
        :meth:`_bar` sanitises a bad spread to zero), and the gap from the
        previous bar within :attr:`_stale_gap_minutes`. An out-of-range index
        has no bar to judge and reports ``True``; :meth:`_entry_reason` refuses
        to act there for other reasons.
        """
        atr_rows = len(self.features.atr)
        if not 0 <= index < atr_rows:
            return True
        atr = float(self.features.atr[index])
        if not isfinite(atr) or atr <= 0:
            return False
        if index >= len(self.features.spread):
            return False
        spread = float(self.features.spread[index])
        if not isfinite(spread) or spread < 0:
            return False
        gap = float(self._bar_gaps[index]) if index < len(self._bar_gaps) else float("nan")
        return not (isfinite(gap) and gap > self._stale_gap_minutes)

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
        start = self._i - self.window + 1
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
        parts = [flat, position_features]
        if self._strategy_selector is not None:
            context = np.zeros(_STRATEGY_CONTEXT, dtype=np.float32)
            if self._arm >= 0:
                context[self._arm] = 1.0
                context[-1] = self._arm_strength
            parts.append(context)
        if self._flow_obs is not None:
            parts.append(self._flow_obs[self._i].astype(np.float32, copy=False))
        return np.concatenate(parts)

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
        return self._i >= self.window - 1 and self._i <= self._last_index

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
