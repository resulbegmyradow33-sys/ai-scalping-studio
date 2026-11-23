"""One-axis-at-a-time parameter sensitivity for one fixed policy.

The question is not "which settings are best". It is "how much of this result
depends on the cost assumptions we chose?" A profitable run whose edge vanishes
when commission doubles is a different result from one that survives a 3x
spread, and the headline number alone cannot tell them apart.

Each axis reruns the *same* policy on the *same* features through the *same*
:func:`~scalping.backtest.engine.run_backtest`, changing exactly one execution
cost:

* ``spread_multiplier`` scales the frame's ``spread`` array (the lever fills
  actually read) and ``CostConfig.spread_floor`` by the value;
* ``slippage_atr_frac`` sets the absolute ATR slippage fraction;
* ``commission_per_lot`` sets the absolute per-lot commission.

Risk, contract, feature and label settings never move, so no axis can change
what the policy is allowed to see or what it is asked to predict. That rules
out look-ahead by construction: only the broker's prices change.

``breakeven_multiplier`` is the multiple of the base cost at which net profit
crosses zero between two adjacent grid rows, linearly interpolated. It is
``None`` when the sign never changes, when the base cost is zero (so no
multiplier exists), or when the axis is flat. It is one crossing of one grid,
not a precisely located economic threshold.

What this does not prove
------------------------
A grid over costs says nothing about unseen data, parameter overfitting or
whether the strategy has an edge. Moving one axis while holding everything else
fixed also assumes the axes do not interact; the true cost surface can bend.
Use the result to test fragility, never as a profitability proof.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any, Final

import numpy as np
import polars as pl

from scalping.backtest.engine import BacktestResult, run_backtest
from scalping.config import Settings
from scalping.env.simulator import WINDOW, ArmSelector
from scalping.features.build import FeatureFrame

__all__ = [
    "COMMISSION_AXIS",
    "SLIPPAGE_AXIS",
    "SPREAD_AXIS",
    "SUPPORTED_AXES",
    "SensitivityAxis",
    "SensitivityResult",
    "SensitivityRow",
    "default_axes",
    "parameter_sensitivity",
]

SPREAD_AXIS: Final[str] = "spread_multiplier"
SLIPPAGE_AXIS: Final[str] = "slippage_atr_frac"
COMMISSION_AXIS: Final[str] = "commission_per_lot"

#: Canonical axis order; the report follows it regardless of dict order.
SUPPORTED_AXES: Final[tuple[str, ...]] = (SPREAD_AXIS, SLIPPAGE_AXIS, COMMISSION_AXIS)

#: Spread multipliers used when the caller supplies no grid.
_DEFAULT_SPREADS: Final[tuple[float, ...]] = (0.5, 1.0, 2.0, 3.0)


@dataclass(frozen=True, slots=True)
class SensitivityRow:
    """One grid point: the value tried and the headline metrics it produced.

    ``multiplier`` is ``value`` relative to the base cost and is ``None`` for
    an absolute axis whose base is zero. ``profit_factor`` keeps the metrics
    convention and can be ``inf`` when a run has wins and no losses; the CLI
    maps it to JSON ``null``.
    """

    parameter: str
    value: float
    multiplier: float | None
    trades: int
    net_profit: float
    profit_factor: float
    expectancy_r: float
    max_drawdown: float
    total_costs: float
    run_id: str


@dataclass(frozen=True, slots=True)
class SensitivityAxis:
    """Every point of one axis, sorted by ``value``, plus its zero crossing."""

    parameter: str
    rows: tuple[SensitivityRow, ...]
    breakeven_multiplier: float | None


@dataclass(frozen=True, slots=True)
class SensitivityResult:
    """The unmodified base run plus one axis per cost parameter."""

    base: SensitivityRow
    axes: tuple[SensitivityAxis, ...]


def default_axes(settings: Settings) -> dict[str, tuple[float, ...]]:
    """Grids that scale with ``settings`` instead of hard-coding one config.

    Spread is expressed as multipliers. Slippage and commission are absolute
    values derived from the base settings, so the same defaults remain sensible
    when a caller passes a non-default cost block. Zero-valued base costs
    collapse to the single point ``0.0`` rather than duplicating it.
    """
    costs = settings.costs
    slippage = float(costs.slippage_atr_frac)
    commission = float(costs.commission_per_lot)
    return {
        SPREAD_AXIS: _DEFAULT_SPREADS,
        SLIPPAGE_AXIS: tuple(sorted({0.0, slippage / 2.0, slippage, slippage * 2.0})),
        COMMISSION_AXIS: tuple(sorted({0.0, commission / 2.0, commission, commission * 2.0})),
    }


def _axis_values(axis: str, values: Sequence[float]) -> tuple[float, ...]:
    """Validate one grid and return it sorted and duplicate-free."""
    numbers = tuple(float(value) for value in values)
    if not numbers:
        raise ValueError(f"sensitivity axis {axis!r} needs at least one value")
    if any(not np.isfinite(value) for value in numbers):
        raise ValueError(f"sensitivity axis {axis!r} values must be finite, got {values!r}")
    if axis == SPREAD_AXIS:
        if any(value <= 0.0 for value in numbers):
            raise ValueError(f"spread multipliers must be > 0, got {values!r}")
    elif any(value < 0.0 for value in numbers):
        raise ValueError(f"{axis} values must be >= 0, got {values!r}")
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"sensitivity axis {axis!r} has duplicate values: {values!r}")
    return tuple(sorted(numbers))


def _resolve_axes(
    axes: Mapping[str, Sequence[float]] | None, settings: Settings
) -> dict[str, tuple[float, ...]]:
    """Merge caller grids with the defaults and reject unknown axis names."""
    if axes is None:
        raw: Mapping[str, Sequence[float]] = default_axes(settings)
    else:
        if not axes:
            raise ValueError("at least one sensitivity axis is required")
        unknown = sorted(set(axes) - set(SUPPORTED_AXES))
        if unknown:
            raise ValueError(
                f"unknown sensitivity axes {unknown!r}; choose from {list(SUPPORTED_AXES)}"
            )
        raw = axes
    return {axis: _axis_values(axis, raw[axis]) for axis in SUPPORTED_AXES if axis in raw}


def _base_multiplier(axis: str, settings: Settings) -> float | None:
    """Base cost an *absolute* axis divides by; ``None`` for spread or a zero base."""
    if axis == SLIPPAGE_AXIS:
        base = float(settings.costs.slippage_atr_frac)
    elif axis == COMMISSION_AXIS:
        base = float(settings.costs.commission_per_lot)
    else:  # pragma: no cover - callers only pass supported axes
        return None
    return base if base > 0.0 else None


def _scaled_features(features: FeatureFrame, multiplier: float) -> FeatureFrame:
    """Shallow frame copy with only ``spread`` scaled, like ``cost_stress``."""
    if multiplier == 1.0:
        return features
    return replace(features, spread=np.asarray(features.spread, dtype=np.float64) * multiplier)


def _settings_for(axis: str, value: float, settings: Settings) -> Settings:
    """A fresh frozen settings object with exactly one cost field moved."""
    costs = settings.costs
    if axis == SPREAD_AXIS:
        return replace(settings, costs=replace(costs, spread_floor=costs.spread_floor * value))
    if axis == SLIPPAGE_AXIS:
        return replace(settings, costs=replace(costs, slippage_atr_frac=value))
    if axis == COMMISSION_AXIS:
        return replace(settings, costs=replace(costs, commission_per_lot=value))
    raise ValueError(f"unknown sensitivity axis {axis!r}")  # pragma: no cover


def _features_for(axis: str, value: float, features: FeatureFrame) -> FeatureFrame:
    return _scaled_features(features, value) if axis == SPREAD_AXIS else features


def _row(
    result: BacktestResult, parameter: str, value: float, multiplier: float | None
) -> SensitivityRow:
    metrics = result.metrics
    return SensitivityRow(
        parameter=parameter,
        value=value,
        multiplier=multiplier,
        trades=int(metrics.trades),
        net_profit=float(metrics.net_profit),
        profit_factor=float(metrics.profit_factor),
        expectancy_r=float(metrics.expectancy_r),
        max_drawdown=float(metrics.max_drawdown),
        total_costs=float(metrics.total_costs),
        run_id=result.run_id,
    )


def _breakeven(rows: Sequence[SensitivityRow]) -> float | None:
    """First multiplier where net profit crosses zero, interpolated linearly.

    Rows are sorted by value, so the first sign change is the lowest cost at
    which the run stops being profitable. A grid that only ever sits on one
    side of zero — including an all-zero, trade-free axis — has no crossing and
    returns ``None``. A row that lands exactly on zero counts only when the
    grid also contains both signs somewhere.
    """
    usable = [(row.multiplier, row.net_profit) for row in rows if row.multiplier is not None]
    signs = {np.sign(net) for _, net in usable}
    signs.discard(0.0)
    if len(signs) < 2:
        return None
    for (first_x, before), (second_x, after) in pairwise(usable):
        if before == 0.0:
            return first_x
        if after == 0.0:
            return second_x
        if (before > 0.0) != (after > 0.0):
            span = after - before
            if span == 0.0:  # pragma: no cover - equal signs already excluded
                continue
            return first_x + (0.0 - before) * (second_x - first_x) / span
    return None


def _run(
    policy: Callable[[np.ndarray], int],
    features: FeatureFrame,
    settings: Settings,
    *,
    strategy_selector: ArmSelector | Callable[[np.ndarray], int] | None,
    labels: pl.DataFrame | None,
    window: int,
    symbol: str | None,
) -> BacktestResult:
    """The single engine call every grid point shares."""
    return run_backtest(
        policy,
        features,
        settings,
        max_steps=max(len(features.ts) + 1, 1),
        strategy_selector=strategy_selector,
        labels=labels,
        window=window,
        symbol=symbol,
    )


def parameter_sensitivity(
    policy: Callable[[np.ndarray], int],
    features: FeatureFrame,
    settings: Settings,
    *,
    axes: Mapping[str, Sequence[float]] | None = None,
    strategy_selector: ArmSelector | Callable[[np.ndarray], int] | None = None,
    labels: pl.DataFrame | None = None,
    window: int = WINDOW,
    symbol: str | None = None,
) -> SensitivityResult:
    """Rerun ``policy`` across cost grids and report the headline metrics.

    ``axes`` maps axis name to its grid; omitted axes are not run and an
    entirely omitted mapping uses :func:`default_axes`. Each point is a fresh
    frozen :class:`~scalping.config.Settings`, so the caller's object is never
    mutated, and the spread axis gets a shallow frame copy because fills read
    the quote from ``FeatureFrame.spread`` rather than from settings. Rows are
    sorted by value, so the result does not depend on the order the caller
    listed the grid.

    ``strategy_selector``, ``labels``, ``window`` and ``symbol`` are forwarded
    unchanged, so a strategy-aware policy sees the same context on every point.
    Feature and label inputs are identical across the grid by construction.
    """
    if not callable(policy):
        raise TypeError(f"policy must be callable, got {type(policy).__name__}")
    if int(window) < 1:
        raise ValueError(f"window must be >= 1, got {window!r}")

    resolved = _resolve_axes(axes, settings)
    kwargs: dict[str, Any] = {
        "strategy_selector": strategy_selector,
        "labels": labels,
        "window": window,
        "symbol": symbol,
    }
    base_result = _run(policy, features, settings, **kwargs)
    base = _row(base_result, "base", 1.0, 1.0)

    reports: list[SensitivityAxis] = []
    for axis in SUPPORTED_AXES:
        values = resolved.get(axis)
        if values is None:
            continue
        divisor = _base_multiplier(axis, settings)
        rows: list[SensitivityRow] = []
        for value in values:
            if axis == SPREAD_AXIS:
                multiplier: float | None = value
            else:
                multiplier = None if divisor is None else value / divisor
            result = _run(
                policy,
                _features_for(axis, value, features),
                _settings_for(axis, value, settings),
                **kwargs,
            )
            rows.append(_row(result, axis, value, multiplier))
        reports.append(
            SensitivityAxis(
                parameter=axis,
                rows=tuple(rows),
                breakeven_multiplier=_breakeven(rows),
            )
        )

    return SensitivityResult(base=base, axes=tuple(reports))
