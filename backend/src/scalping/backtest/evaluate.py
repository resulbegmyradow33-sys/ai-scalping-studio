"""Walk-forward and cost-stress evaluation for one fixed policy.

v1 does not retrain between folds. :func:`evaluate_walkforward` evaluates the
*same* callable out of sample on each fold's test window. That is deliberate:
the policy under test is the frozen bandit/PPO pair from the earlier tasks, and
retraining it per fold would change the thing being measured. The train window
is still used for one short run whose per-trade expectancy is the per-fold
train signal, so an in-sample/out-of-sample divergence can raise
``overfit_warning``.

No look-ahead
-------------
Every window is sliced out of the raw frame *before* features are built, so a
test window's feature matrix only contains bars from that window. A train run
never sees test bars, and the test run never sees train bars. The windows
themselves come from :func:`~scalping.backtest.walkforward.walkforward_split`,
which guarantees ``train.end <= val.start <= test.start`` and
``test.end <= last timestamp``.

Short frames
------------
The splitter needs ``train_months + 2`` months before the first test window
fits. When the frame is too short for the requested ``train_months`` and
``folds``, the train window is shortened until enough folds fit (down to one
month). A 200-day frame therefore evaluates with a two-month train window; a
frame long enough for the requested history keeps it unchanged.

Cost stress
-----------
:func:`cost_stress` reruns the policy with ``spread_floor``,
``slippage_atr_frac`` and ``commission_per_lot`` scaled by each multiplier. It
builds a new frozen :class:`~scalping.config.Settings` per run, so the caller's
settings object is never mutated and needs no restore step. Because fills read
the spread from the feature frame, not from settings, each run also gets a
shallow copy of the frame whose ``spread`` array is multiplied by the same
multiplier (``ask = bid + spread`` then follows automatically). The policy's
``X`` matrix is left untouched: the stress models a wider broker quote, not a
different market observation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from math import isfinite

import numpy as np
import polars as pl

from scalping.backtest.engine import run_backtest
from scalping.backtest.metrics import Metrics, compute
from scalping.backtest.walkforward import Split, walkforward_split
from scalping.config import Settings
from scalping.env.simulator import WINDOW, ArmSelector
from scalping.features.build import FeatureFrame, build_features
from scalping.strategy.labeling import label_events
from scalping.types import Trade

__all__ = [
    "FoldResult",
    "StressResult",
    "WalkForwardReport",
    "cost_stress",
    "evaluate_walkforward",
]


@dataclass(frozen=True, slots=True)
class FoldResult:
    """One fold: its windows, test metrics and the trades the test run closed.

    ``train_expectancy`` and ``test_expectancy`` (``metrics.expectancy_r``) are
    per-trade R, so the two sides of the overfit trend share one unit.
    ``evaluated_range`` is the span of bars the test run actually stepped
    through: the first decision bar to the last frame bar. It is narrower than
    ``test_range`` by the feature warm-up and any weekend/holiday boundary.
    ``equity`` is kept so the report can chain fold PnL into ``overall``; it is
    not part of the public contract.
    """

    train_range: tuple[datetime, datetime]
    test_range: tuple[datetime, datetime]
    metrics: Metrics
    trades: list[Trade]
    train_expectancy: float
    equity: np.ndarray
    evaluated_range: tuple[datetime, datetime]

    @property
    def train_reward(self) -> float:
        """Backward-compatible alias; the value is now per-trade expectancy in R."""
        return self.train_expectancy


@dataclass(frozen=True, slots=True)
class WalkForwardReport:
    """Per-fold results, their aggregate and the overfit verdict.

    ``train_expectancy_trend`` and ``test_expectancy_trend`` are least-squares
    slopes of per-trade expectancy (R) against fold index. Both sides use the
    same unit. With three folds the slope is endpoint-based:
    ``(y2 - y0) / 2``. The warning requires at least three folds and fires only
    when train expectancy is rising while test expectancy is falling, both
    strictly.
    """

    folds: list[FoldResult]
    overall: Metrics
    overfit_warning: bool
    folds_requested: int
    folds_produced: int
    train_expectancy_trend: float = 0.0
    test_expectancy_trend: float = 0.0
    train_months: int = 6

    @property
    def train_reward_trend(self) -> float:
        """Backward-compatible alias; the value is now an expectancy slope in R."""
        return self.train_expectancy_trend


@dataclass(frozen=True, slots=True)
class StressResult:
    """One cost-stress run: the multiplier and the metrics it produced."""

    multiplier: float
    metrics: Metrics
    trades: list[Trade]


def _select_splits(df: pl.DataFrame, train_months: int, folds: int) -> tuple[list[Split], int]:
    """Pick ``folds`` non-overlapping splits and return them with the history used.

    A frame long enough for ``train_months`` keeps it. A shorter frame tries
    one month less at a time until enough folds fit; the selected folds are the
    most recent ones. When even a one-month train window cannot produce a
    single test window, there is nothing to evaluate, so this raises instead of
    returning an empty report that would look like a flat policy.
    """
    if int(folds) < 1:
        raise ValueError(f"folds must be >= 1, got {folds!r}")
    if int(train_months) < 1:
        raise ValueError(f"train_months must be >= 1, got {train_months!r}")

    candidates: list[Split] = []
    used_months = int(train_months)
    for months in range(int(train_months), 0, -1):
        candidates = walkforward_split(df, train_months=months, val_months=1, test_months=1)
        used_months = months
        if len(candidates) >= int(folds):
            break
    if not candidates:
        raise ValueError(f"not enough data for {int(folds)} folds")
    return candidates[-int(folds) :], used_months


def _window_features(
    df: pl.DataFrame, start: datetime, end: datetime, settings: Settings
) -> FeatureFrame:
    """Slice raw bars to ``[start, end)`` and build features from that slice alone."""
    window = df.filter((pl.col("ts") >= start) & (pl.col("ts") < end))
    return build_features(window, settings.costs)


def _window_labels(
    df: pl.DataFrame, start: datetime, end: datetime, settings: Settings
) -> pl.DataFrame:
    """Label the same raw window; used only when a strategy selector is present."""
    window = df.filter((pl.col("ts") >= start) & (pl.col("ts") < end))
    return label_events(window, settings)


def _empty_metrics(settings: Settings) -> Metrics:
    return compute([], np.empty(0, dtype=np.float64), float(settings.contract.initial_balance))


def _evaluated_range(
    features: FeatureFrame, window: tuple[datetime, datetime]
) -> tuple[datetime, datetime]:
    """Span from the first decision bar to the last bar in ``features``."""
    if len(features.ts) == 0:
        return window
    first = max(int(features.valid_from) + WINDOW - 1, WINDOW - 1)
    first = min(first, len(features.ts) - 1)
    start = features.ts[first].astype("datetime64[us]").astype(datetime).replace(tzinfo=UTC)
    end = features.ts[-1].astype("datetime64[us]").astype(datetime).replace(tzinfo=UTC)
    return (start, end)


def _chain_equity(results: Sequence[FoldResult], initial: float) -> np.ndarray:
    """Concatenate fold equity curves, carrying each fold's PnL into the next."""
    offset = 0.0
    parts: list[np.ndarray] = []
    for result in results:
        equity = np.asarray(result.equity, dtype=np.float64)
        if equity.size == 0:
            continue
        parts.append(equity + offset)
        offset += float(equity[-1]) - initial
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)


def _slope(values: Sequence[float]) -> float:
    """Least-squares slope of ``values`` against their index; 0.0 if undefined."""
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size < 2 or not bool(np.isfinite(data).all()):
        return 0.0
    x = np.arange(data.size, dtype=np.float64)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(x_centered, data - data.mean()) / denominator)


def evaluate_walkforward(
    policy: Callable[[np.ndarray], int],
    df: pl.DataFrame,
    settings: Settings,
    train_months: int = 6,
    folds: int = 3,
    strategy_selector: ArmSelector | Callable[[np.ndarray], int] | None = None,
) -> WalkForwardReport:
    """Evaluate ``policy`` out of sample on ``folds`` walk-forward windows.

    Features are rebuilt inside each slice, so a fold's test run cannot see
    train bars and vice versa. Trades from every test window are aggregated
    into ``overall``; the per-fold train and test expectancies feed the overfit
    warning. No retraining happens between folds in v1.

    ``strategy_selector``, when given, is the frozen bandit a strategy-aware
    policy expects: every window builds its own labels and passes the same
    selector to the simulator, so the observation width matches training and
    the context stays causal inside the window.

    Each run raises the simulator's step cap to its window length, so a
    one-month M1 window (~29k bars) is evaluated end to end instead of stopping
    at the legacy 20k default. That applies to the train window too: its
    expectancy is a full-window signal, not the first 20k bars of it.
    """
    if not callable(policy):
        raise TypeError(f"policy must be callable, got {type(policy).__name__}")

    splits, used_months = _select_splits(df, train_months, folds)
    initial = float(settings.contract.initial_balance)
    fold_results: list[FoldResult] = []
    fold_ts: list[np.ndarray] = []

    for split in splits:
        train_features = _window_features(df, split.train.start, split.train.end, settings)
        test_features = _window_features(df, split.test.start, split.test.end, settings)
        train_labels = (
            _window_labels(df, split.train.start, split.train.end, settings)
            if strategy_selector
            else None
        )
        test_labels = (
            _window_labels(df, split.test.start, split.test.end, settings)
            if strategy_selector
            else None
        )
        train_result = run_backtest(
            policy,
            train_features,
            settings,
            max_steps=max(len(train_features.ts) + 1, 1),
            strategy_selector=strategy_selector,
            labels=train_labels,
        )
        test_result = run_backtest(
            policy,
            test_features,
            settings,
            max_steps=max(len(test_features.ts) + 1, 1),
            strategy_selector=strategy_selector,
            labels=test_labels,
        )
        fold_results.append(
            FoldResult(
                train_range=(split.train.start, split.train.end),
                test_range=(split.test.start, split.test.end),
                metrics=test_result.metrics,
                trades=test_result.trades,
                train_expectancy=float(train_result.metrics.expectancy_r),
                equity=test_result.equity,
                evaluated_range=_evaluated_range(test_features, (split.test.start, split.test.end)),
            )
        )
        fold_ts.append(np.asarray(test_features.ts))

    all_trades = [trade for fold in fold_results for trade in fold.trades]
    combined_ts = np.concatenate(fold_ts) if fold_ts else None
    overall = (
        compute(all_trades, _chain_equity(fold_results, initial), initial, ts=combined_ts)
        if fold_results
        else _empty_metrics(settings)
    )

    train_trend = _slope([fold.train_expectancy for fold in fold_results])
    test_trend = _slope([fold.metrics.expectancy_r for fold in fold_results])
    enough_folds = len(fold_results) >= 3
    return WalkForwardReport(
        folds=fold_results,
        overall=overall,
        overfit_warning=enough_folds and train_trend > 0.0 and test_trend < 0.0,
        folds_requested=int(folds),
        folds_produced=len(fold_results),
        train_expectancy_trend=train_trend,
        test_expectancy_trend=test_trend,
        train_months=used_months,
    )


def _scaled_features(features: FeatureFrame, multiplier: float) -> FeatureFrame:
    """Shallow copy of ``features`` with only ``spread`` scaled.

    Fills read the quote from the frame, so this is the lever that actually
    widens the broker's spread. ``X`` stays as observed: the stress is a cost
    change, not a different market.
    """
    if multiplier == 1.0:
        return features
    return replace(features, spread=np.asarray(features.spread, dtype=np.float64) * multiplier)


def _stressed_settings(settings: Settings, multiplier: float) -> Settings:
    """Return a new settings object with the three chargeable costs scaled."""
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


def cost_stress(
    policy: Callable[[np.ndarray], int],
    features: FeatureFrame,
    settings: Settings,
    multipliers: Sequence[float] = (1, 2, 3),
    strategy_selector: ArmSelector | Callable[[np.ndarray], int] | None = None,
    labels: pl.DataFrame | None = None,
) -> list[StressResult]:
    """Rerun ``policy`` with each cost multiplier and return the results.

    The caller's ``settings`` is never modified: each run gets a fresh frozen
    copy with ``spread_floor``, ``slippage_atr_frac`` and
    ``commission_per_lot`` multiplied. ``spread_floor`` alone cannot move a
    fill, because fills read ``FeatureFrame.spread``; each run therefore also
    receives a shallow frame copy with ``spread * multiplier``. Multipliers
    must be finite and positive. Each run's step cap is the frame length, so a
    multi-month frame is stressed end to end. ``strategy_selector``/``labels``
    mirror :func:`evaluate_walkforward` for a strategy-aware policy.
    """
    if not callable(policy):
        raise TypeError(f"policy must be callable, got {type(policy).__name__}")
    values = [float(m) for m in multipliers]
    if not values:
        raise ValueError("multipliers must not be empty")
    if any(not isfinite(m) or m <= 0.0 for m in values):
        raise ValueError(f"multipliers must be finite and positive, got {multipliers!r}")

    results: list[StressResult] = []
    for multiplier in values:
        stressed = _stressed_settings(settings, multiplier)
        result = run_backtest(
            policy,
            _scaled_features(features, multiplier),
            stressed,
            max_steps=max(len(features.ts) + 1, 1),
            strategy_selector=strategy_selector,
            labels=labels,
        )
        results.append(
            StressResult(multiplier=multiplier, metrics=result.metrics, trades=result.trades)
        )
    return results
