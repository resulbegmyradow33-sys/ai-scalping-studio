"""Wire a DSL candidate to the simulator: one strategy, one validation frame.

:class:`DslEvaluator` is the production implementation of
:class:`scalping.discovery.protocols.Evaluator`. It interprets a
:class:`~scalping.strategy.dsl.nodes.StrategySpec` bar by bar, drives the same
:``Simulator`` the backtest engine drives, and turns the resulting trades and
equity curve into a :class:`~scalping.discovery.protocols.TrialResult` with
net-of-cost metrics.

What the evaluator guarantees
-----------------------------
* **One frame, reused.** The caller supplies an already built
  :class:`~scalping.features.build.FeatureFrame`; every candidate reads the
  same arrays. Nothing is rebuilt per candidate and the frame is never mutated
  (a cost-stress run gets a shallow copy with only ``spread`` scaled).
* **Bar order, no look-ahead.** The loop is the same sequence as
  :func:`scalping.backtest.engine.run_backtest`: ``reset()``, then one
  ``step(action)`` per bar. A decision at bar ``i`` reads row ``i`` plus the
  policy's history of rows before ``i``; no later row is reachable from the
  DSL and none is read here.
* **Validation only.** The evaluator is constructed with one dataset id and
  refuses any other. It also refuses, at construction, an id that names the
  test/holdout split: the final holdout is not a search input, and a search
  that touches it turns it into another training set.
* **Acceptance from the registry gate.** When ``gate`` enables thresholds,
  the verdict and rejection strings come from
  :func:`scalping.registry.gate.evaluate_candidate`. Missing calibration or
  regime evidence therefore rejects with the gate's own codes instead of
  passing silently. A default :class:`GateCriteria` accepts everything.
* **Optional cost stress.** Each requested multiplier reruns the *same*
  candidate with a wider spread and scaled slippage/commission, and its
  metrics are flattened into ``TrialResult.metrics`` as
  ``stress_<label>_<metric>``. A multiplier of ``1`` reuses the base run.

Mapping
-------
``Policy.decide`` returns ``BUY``/``SELL``/``HOLD``/``CLOSE``/``NO_TRADE``;
the simulator speaks ``0``/``1``/``2``/``3``. ``BUY``→``1``, ``SELL``→``2``,
``CLOSE``→``3``, and both ``HOLD`` and ``NO_TRADE``→``0``. ``NO_TRADE`` does
not flatten a position: the DSL evaluates it before exits, so flattening here
would contradict the interpreter's own precedence. The ``no_trade`` rule
suppresses entries; an open position is still closed by its exit rule or the
holding limit on a later bar.

Deliberate gaps
---------------
Sizing nodes (``FixedFraction``, ``AtrMultiple``) are part of the candidate
identity but do not reach the broker: the simulator's ``RiskEngine`` owns
position size from its frozen config, so a generated strategy cannot request
size. Regime conditioning, walk-forward folds, purge/embargo geometry and the
final test slice are the caller's responsibility; this evaluator runs one
candidate on one validation frame. Passing this evaluator proves the wiring,
not an edge.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Final

import numpy as np

from scalping.backtest.metrics import Metrics, compute
from scalping.config import Settings
from scalping.data.registry import SessionSpec, session_for
from scalping.discovery.protocols import Candidate, TrialResult
from scalping.env.simulator import Simulator
from scalping.features.build import FeatureFrame
from scalping.registry.gate import GateCriteria, evaluate_candidate
from scalping.strategy.dsl import (
    DslValidationError,
    Policy,
    StrategySpec,
    compile_policy,
)
from scalping.types import Side, Trade

__all__ = ["DslEvaluator", "SimulationResult"]

#: Dataset id tokens that name the untouched test/holdout split. Any one of
#: them makes the id unusable for search, whatever the prefix says.
_HOLDOUT_TOKENS: Final[frozenset[str]] = frozenset({"test", "tests", "holdout", "final", "locked"})

_TOKEN_SPLIT: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")

#: The disabled gate. Frozen and empty, so one shared instance is safe; ruff's
#: B008 forbids constructing it in the signature itself.
_EMPTY_GATE: Final[GateCriteria] = GateCriteria()

#: DSL action -> simulator action. ``NO_TRADE`` maps to HOLD by contract.
_ACTIONS: Final[dict[str, int]] = {
    "HOLD": 0,
    "NO_TRADE": 0,
    "BUY": 1,
    "SELL": 2,
    "CLOSE": 3,
}


@dataclass(frozen=True, slots=True)
class _Run:
    """One simulator pass: metrics plus the raw evidence behind them."""

    metrics: Metrics
    trades: list[Trade]
    equity: np.ndarray


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """One simulator pass with the raw rows a caller needs for deeper evidence.

    ``evaluate`` only needs the metrics, but fold-level evidence also needs the
    closed trades (R multiples, entry timestamps for regime attribution) and the
    equity path. ``trades`` and ``equity`` are the exact objects the summary was
    computed from; nothing is re-priced or re-simulated.
    """

    metrics: Metrics
    trades: tuple[Trade, ...]
    equity: np.ndarray


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_SPLIT.split(text) if token}


def _holdout_id(dataset_id: str) -> bool:
    """True when the id's tokens name a test/holdout split.

    Two spellings are inspected: the plain tokens and the tokens after hyphens
    are removed, so ``final-test``, ``final-holdout`` and ``hold-out`` all
    match. The check is token-based, not a substring search, so ``latest`` does
    not match ``test``.
    """
    lowered = dataset_id.lower()
    return not (_tokens(lowered) | _tokens(lowered.replace("-", ""))).isdisjoint(_HOLDOUT_TOKENS)


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _stress_label(multiplier: float) -> str:
    """Stable label for one multiplier, matching the gate's ``x<multiplier>``."""
    return f"x{multiplier:g}"


def _stressed_settings(settings: Settings, multiplier: float) -> Settings:
    """A new settings object with spread, slippage and commission scaled.

    The caller's settings are frozen and never mutated. ``spread_floor`` alone
    cannot move a fill because the simulator reads the quote from the frame, so
    :func:`_stressed_features` scales the array as well.
    """
    costs = settings.costs
    return replace(
        settings,
        costs=replace(
            costs,
            spread_floor=costs.spread_floor * multiplier,
            slippage_atr_frac=costs.slippage_atr_frac * multiplier,
            commission_per_lot=costs.commission_per_lot * multiplier,
        ),
    )


def _stressed_features(features: FeatureFrame, multiplier: float) -> FeatureFrame:
    """Shallow copy of the frame with only ``spread`` scaled.

    ``X`` stays as observed: the stress is a wider broker quote, not a
    different market observation, so the candidate takes the same decisions.
    """
    if multiplier == 1.0:
        return features
    return replace(features, spread=np.asarray(features.spread, dtype=np.float64) * multiplier)


def _metric_dict(metrics: Metrics, *, bars: int) -> dict[str, float]:
    """Numeric view of ``metrics``; non-finite values are omitted, never faked.

    ``profit_factor`` is ``inf`` for wins without losses. ``TrialResult`` only
    accepts finite numbers, and clamping infinity would invent a value. The key
    is omitted instead, so a gate that requires ``profit_factor`` rejects with
    ``missing_metric`` rather than reading a fabricated pass.
    """
    payload: dict[str, float] = {}
    for name, value in asdict(metrics).items():
        number = float(value)
        if math.isfinite(number):
            payload[name] = number
    payload["bars"] = float(bars)
    return payload


class DslEvaluator:
    """Evaluate DSL candidates by driving one :class:`Simulator` per run.

    ``features`` must be a built frame the caller already owns; it is validated
    once here (shapes, column names, finiteness after ``valid_from``) and then
    reused for every candidate. ``settings`` supplies the contract, risk and
    cost blocks. ``symbol`` selects the registry's weekly session, so entries
    outside the symbol's window are blocked exactly as in the backtest engine.
    ``timeframe`` is provenance only; it never reaches the arithmetic.

    ``gate`` is the acceptance policy. Its default is empty, which accepts
    every candidate: that is the search-fitness mode, not a promotion. A run
    with real thresholds must also request the cost-stress multipliers the gate
    requires, or the gate rejects with ``missing_cost_stress``.

    ``stress`` reruns the candidate with each multiplier applied to spread,
    slippage and commission. ``max_steps`` overrides the simulator's cap (the
    default is :data:`scalping.env.simulator.MAX_STEPS`); pass the frame length
    when a long validation window must be evaluated end to end.
    """

    def __init__(
        self,
        features: FeatureFrame,
        settings: Settings,
        *,
        symbol: str,
        timeframe: str,
        dataset_id: str,
        gate: GateCriteria = _EMPTY_GATE,
        window: int = 60,
        stress: Sequence[float] = (),
        max_steps: int | None = None,
    ) -> None:
        if not isinstance(features, FeatureFrame):
            raise TypeError(f"features must be a FeatureFrame, got {type(features).__name__}")
        if not isinstance(settings, Settings):
            raise TypeError(f"settings must be a Settings, got {type(settings).__name__}")
        if not isinstance(gate, GateCriteria):
            raise TypeError(f"gate must be a GateCriteria, got {type(gate).__name__}")

        self._symbol = _require_text(symbol, "symbol")
        self._timeframe = _require_text(timeframe, "timeframe")
        self._dataset_id = _require_text(dataset_id, "dataset_id")
        if _holdout_id(self._dataset_id):
            raise ValueError(
                f"dataset id {self._dataset_id!r} names a test/holdout split; "
                "the discovery evaluator is validation-only"
            )
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive int, got {window!r}")
        if max_steps is not None and (
            isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1
        ):
            raise ValueError(f"max_steps must be a positive int or None, got {max_steps!r}")

        multipliers: list[float] = []
        for value in stress:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"stress multipliers must be numbers, got {type(value).__name__}")
            number = float(value)
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"stress multipliers must be finite and > 0, got {value!r}")
            multipliers.append(number)
        labels = [_stress_label(value) for value in multipliers]
        if len(set(labels)) != len(labels):
            raise ValueError(f"stress multipliers must be distinct, got {tuple(multipliers)!r}")

        self._features = features
        self._settings = settings
        self._gate = gate
        self._window = window
        self._stress: tuple[float, ...] = tuple(multipliers)
        self._max_steps = max_steps
        self._session: SessionSpec = session_for(self._symbol)
        self._columns = self._validate_frame(features)

    # --- public contract ---------------------------------------------------

    @property
    def dataset_id(self) -> str:
        """The one dataset id this evaluator will accept."""
        return self._dataset_id

    @property
    def symbol(self) -> str:
        """Symbol whose weekly session blocks out-of-hours entries."""
        return self._symbol

    @property
    def timeframe(self) -> str:
        """Provenance label; it does not reach the arithmetic."""
        return self._timeframe

    def evaluate(
        self,
        candidate: Candidate,
        *,
        dataset_id: str,
        code_version: str,
        budget: int,
    ) -> TrialResult:
        """Simulate ``candidate`` on the bound validation frame.

        ``dataset_id`` must equal the constructor's id; any other id, including
        a test/holdout spelling, raises before a bar is read. ``code_version``
        and ``budget`` are accepted for the protocol and do not change the
        arithmetic: the cost is one full pass per candidate plus one per
        additional stress multiplier.
        """
        _require_text(code_version, "code_version")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError(f"budget must be a positive int, got {budget!r}")
        self._check_dataset(dataset_id)

        spec = self._spec(candidate)
        policy = compile_policy(spec)
        base = self._simulate_spec(spec, policy, 1.0)
        metrics = _metric_dict(base.metrics, bars=int(base.equity.size))

        stress: dict[str, dict[str, float]] = {}
        for multiplier in self._stress:
            label = _stress_label(multiplier)
            if multiplier == 1.0:
                row = dict(metrics)
            else:
                stressed = self._simulate_spec(spec, policy, multiplier)
                row = _metric_dict(stressed.metrics, bars=int(stressed.equity.size))
            stress[label] = {"net_profit": row["net_profit"]}
            for name, value in row.items():
                metrics[f"stress_{label}_{name}"] = value

        decision = evaluate_candidate(self._gate, metrics, stress=stress or None)
        return TrialResult(
            metrics=metrics,
            accepted=decision.accepted,
            rejections=decision.rejections,
        )

    def simulate(
        self,
        candidate: Candidate,
        *,
        multiplier: float = 1.0,
    ) -> SimulationResult:
        """Run one pass and return its metrics, closed trades and equity path.

        ``multiplier`` scales spread, slippage and commission exactly like a
        cost-stress run in :meth:`evaluate`; ``1.0`` is the base run. The frame
        is never mutated: a stressed pass reads a shallow copy with only
        ``spread`` scaled. This is the same arithmetic :meth:`evaluate` uses,
        exposed so a fold-level evidence builder can attribute trades to a
        regime and bootstrap their R multiples.
        """
        spec = self._spec(candidate)
        return self._simulate_spec(spec, compile_policy(spec), multiplier)

    def _simulate_spec(
        self,
        spec: StrategySpec,
        policy: Policy,
        multiplier: float,
    ) -> SimulationResult:
        """Shared pass used by :meth:`evaluate` and :meth:`simulate`."""
        if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)):
            raise TypeError(f"multiplier must be a number, got {type(multiplier).__name__}")
        number = float(multiplier)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"multiplier must be finite and > 0, got {multiplier!r}")
        run = self._simulate(
            policy,
            spec,
            features=_stressed_features(self._features, number),
            settings=_stressed_settings(self._settings, number),
        )
        return SimulationResult(
            metrics=run.metrics,
            trades=tuple(run.trades),
            equity=run.equity,
        )

    # --- guards ------------------------------------------------------------

    @staticmethod
    def _spec(candidate: Candidate) -> StrategySpec:
        if not isinstance(candidate, StrategySpec):
            raise DslValidationError(
                f"DslEvaluator only evaluates StrategySpec, got {type(candidate).__name__}"
            )
        return candidate

    def _check_dataset(self, dataset_id: object) -> None:
        if not isinstance(dataset_id, str):
            raise TypeError(f"dataset_id must be a str, got {type(dataset_id).__name__}")
        if dataset_id != self._dataset_id:
            raise ValueError(
                f"evaluator is bound to dataset {self._dataset_id!r}; refused {dataset_id!r}"
            )

    @staticmethod
    def _validate_frame(features: FeatureFrame) -> tuple[str, ...]:
        """One-time shape/column check; returns the column names."""
        matrix = features.X
        if matrix.ndim != 2:
            raise ValueError(f"features.X must be 2-D, got shape {matrix.shape}")
        rows = int(matrix.shape[0])
        columns = tuple(features.columns)
        if not columns:
            raise ValueError(
                "features.columns must name every column; an unnamed frame is ambiguous"
            )
        if not all(isinstance(name, str) and name for name in columns):
            raise ValueError("features.columns must be non-empty strings")
        if len(columns) != int(matrix.shape[1]):
            raise ValueError(
                f"features.columns has {len(columns)} names for {matrix.shape[1]} columns"
            )
        for name in ("ts", "spread", "atr"):
            if getattr(features, name).shape != (rows,):
                raise ValueError(f"features.{name} must have shape ({rows},)")
        if features.ohlc.shape != (rows, 4):
            raise ValueError(
                f"features.ohlc must have shape ({rows}, 4), got {features.ohlc.shape}"
            )
        if not 0 <= int(features.valid_from) <= rows:
            raise ValueError(f"valid_from {features.valid_from} is outside [0, {rows}]")
        if rows and not bool(np.isfinite(matrix[int(features.valid_from) :]).all()):
            raise ValueError("features.X contains a non-finite value after valid_from")
        return columns

    # --- simulation --------------------------------------------------------

    def _row(self, index: int) -> dict[str, float]:
        """Feature row ``index`` keyed by name; only rows that already exist."""
        return dict(zip(self._columns, self._features.X[index].tolist(), strict=True))

    def _simulate(
        self,
        policy: Policy,
        spec: StrategySpec,
        *,
        features: FeatureFrame | None = None,
        settings: Settings | None = None,
    ) -> _Run:
        """One full pass in bar order, mirroring :func:`run_backtest`."""
        frame = self._features if features is None else features
        config = self._settings if settings is None else settings
        sim = Simulator(
            frame,
            config,
            max_steps=self._max_steps,
            window=self._window,
            session=self._session,
        )
        sim.reset()

        state = policy.new_state()
        # Seed history with rows that already happened before the first
        # decision bar, never with a future row. Without this, every lookback
        # would be unknown for the first `max_lookback` decisions.
        start = sim.episode_start
        for index in range(max(0, start - spec.max_lookback), start):
            state.push(self._row(index))

        equity: list[float] = []
        trades: list[Trade] = []
        while True:
            row = self._row(sim.bar_index)
            account = sim.broker.state(sim.balance, sim.day_start_equity)
            position = account.position
            state.position = (
                None if position is None else ("long" if position.side is Side.LONG else "short")
            )
            state.bars_held = int(account.bars_in_trade)

            decision = policy.decide(row, state)
            result = sim.step(_ACTIONS[decision.action])

            state.push(row)
            equity.append(float(result.info["equity"]))
            trade = result.info["trade"]
            if isinstance(trade, Trade):
                trades.append(trade)
            if result.terminated or result.truncated:
                break

        values = np.asarray(equity, dtype=np.float64)
        return _Run(
            metrics=compute(
                trades,
                values,
                float(config.contract.initial_balance),
                ts=frame.ts,
            ),
            trades=trades,
            equity=values,
        )
