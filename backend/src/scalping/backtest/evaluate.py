"""Walk-forward and cost-stress evaluation for one fixed policy.

v1 does not retrain between folds. :func:`evaluate_walkforward` evaluates the
*same* callable out of sample on each fold's test window. That is deliberate:
the policy under test is the frozen bandit/PPO pair from the earlier tasks, and
retraining it per fold would change the thing being measured. The train window
is still used for one short run whose normalised net profit is the per-fold
train reward, so an in-sample/out-of-sample divergence can raise
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
settings object is never mutated and needs no restore step.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from math import isfinite

import numpy as np
import polars as pl

from scalping.backtest.engine import run_backtest
from scalping.backtest.metrics import Metrics, compute
from scalping.backtest.walkforward import Split, walkforward_split
from scalping.config import Settings
from scalping.features.build import FeatureFrame, build_features
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

    ``train_reward`` is the train run's net profit as a fraction of the initial
    balance (percent return / 100), the scalar used for the overfit trend.
    ``equity`` is kept so the report can chain fold PnL into ``overall``; it is
    not part of the public contract.
    """

    train_range: tuple[datetime, datetime]
    test_range: tuple[datetime, datetime]
    metrics: Metrics
    trades: list[Trade]
    train_reward: float
    equity: np.ndarray


@dataclass(frozen=True, slots=True)
class WalkForwardReport:
    """Per-fold results, their aggregate and the overfit verdict.

    ``train_reward_trend`` and ``test_expectancy_trend`` are least-squares
    slopes over fold index. The warning fires only when train reward is rising
    while test expectancy is falling, both strictly.
    """

    folds: list[FoldResult]
    overall: Metrics
    overfit_warning: bool
    train_reward_trend: float = 0.0
    test_expectancy_trend: float = 0.0
    train_months: int = 6


@dataclass(frozen=True, slots=True)
class StressResult:
    """One cost-stress run: the multiplier and the metrics it produced."""

    multiplier: float
    metrics: Metrics
    trades: list[Trade]


def _select_splits(df: pl.DataFrame, train_months: int, folds: int) -> list[Split]:
    """Pick ``folds`` non-overlapping splits, shrinking train history if needed.

    A frame long enough for ``train_months`` keeps it. A shorter frame tries
    one month less at a time until enough folds fit; the selected folds are the
    most recent ones.
    """
    if int(folds) < 1:
        raise ValueError(f"folds must be >= 1, got {folds!r}")
    if int(train_months) < 1:
        raise ValueError(f"train_months must be >= 1, got {train_months!r}")

    candidates: list[Split] = []
    for months in range(int(train_months), 0, -1):
        candidates = walkforward_split(df, train_months=months, val_months=1, test_months=1)
        if len(candidates) >= int(folds):
            break
    return candidates[-int(folds) :] if candidates else []


def _window_features(
    df: pl.DataFrame, start: datetime, end: datetime, settings: Settings
) -> FeatureFrame:
    """Slice raw bars to ``[start, end)`` and build features from that slice alone."""
    window = df.filter((pl.col("ts") >= start) & (pl.col("ts") < end))
    return build_features(window, settings.costs)


def _empty_metrics(settings: Settings) -> Metrics:
    return compute([], np.empty(0, dtype=np.float64), float(settings.contract.initial_balance))


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
) -> WalkForwardReport:
    """Evaluate ``policy`` out of sample on ``folds`` walk-forward windows.

    Features are rebuilt inside each slice, so a fold's test run cannot see
    train bars and vice versa. Trades from every test window are aggregated
    into ``overall``; the per-fold train and test series feed the overfit
    warning. No retraining happens between folds in v1.
    """
    if not callable(policy):
        raise TypeError(f"policy must be callable, got {type(policy).__name__}")

    splits = _select_splits(df, train_months, folds)
    initial = float(settings.contract.initial_balance)
    fold_results: list[FoldResult] = []

    for split in splits:
        train_features = _window_features(df, split.train.start, split.train.end, settings)
        test_features = _window_features(df, split.test.start, split.test.end, settings)
        train_result = run_backtest(policy, train_features, settings)
        test_result = run_backtest(policy, test_features, settings)
        fold_results.append(
            FoldResult(
                train_range=(split.train.start, split.train.end),
                test_range=(split.test.start, split.test.end),
                metrics=test_result.metrics,
                trades=test_result.trades,
                train_reward=float(train_result.metrics.net_profit) / initial,
                equity=test_result.equity,
            )
        )

    all_trades = [trade for fold in fold_results for trade in fold.trades]
    overall = (
        compute(all_trades, _chain_equity(fold_results, initial), initial)
        if fold_results
        else _empty_metrics(settings)
    )

    train_trend = _slope([fold.train_reward for fold in fold_results])
    test_trend = _slope([fold.metrics.expectancy_r for fold in fold_results])
    return WalkForwardReport(
        folds=fold_results,
        overall=overall,
        overfit_warning=train_trend > 0.0 and test_trend < 0.0,
        train_reward_trend=train_trend,
        test_expectancy_trend=test_trend,
        train_months=_selected_train_months(df, train_months, folds),
    )


def _selected_train_months(df: pl.DataFrame, train_months: int, folds: int) -> int:
    """Report the train history actually used by :func:`_select_splits`."""
    if int(folds) < 1 or int(train_months) < 1:
        return int(train_months)
    for months in range(int(train_months), 0, -1):
        if len(walkforward_split(df, months, 1, 1)) >= int(folds):
            return months
    return 1


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
) -> list[StressResult]:
    """Rerun ``policy`` with each cost multiplier and return the results.

    The caller's ``settings`` is never modified: each run gets a fresh frozen
    copy with ``spread_floor``, ``slippage_atr_frac`` and
    ``commission_per_lot`` multiplied. Multipliers must be finite and positive.
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
        result = run_backtest(policy, features, stressed)
        results.append(
            StressResult(multiplier=multiplier, metrics=result.metrics, trades=result.trades)
        )
    return results
