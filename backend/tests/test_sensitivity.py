"""Tests for one-axis execution-cost sensitivity.

The grid must reuse the simulator, degrade monotonically on a genuinely
profitable sequence, report a breakeven only when the sign actually changes,
and never touch risk, contract, feature or label inputs.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scalping.backtest.sensitivity import (
    COMMISSION_AXIS,
    SLIPPAGE_AXIS,
    SPREAD_AXIS,
    SensitivityAxis,
    _features_for,
    _settings_for,
    default_axes,
    parameter_sensitivity,
)
from scalping.config import Settings
from scalping.features.build import FeatureFrame


def trend_frame(n: int = 900, step: float = 0.02) -> FeatureFrame:
    """A steadily rising M1 frame: every long timeout trade books a gross gain.

    Stops and targets sit five and ten ATR(14) away from entry and the price
    moves ~2.4 over the 120-bar time stop, so neither level is reached. Only
    the time stop closes trades, which keeps the trade count identical across
    cost grids and leaves the cost as the single moving part.
    """
    price = 1_000.0 + step * np.arange(n)
    ts = np.datetime64("2026-01-05T00:00", "us") + np.arange(n).astype("timedelta64[m]")
    ohlc = np.column_stack([price, price + 0.05, price - 0.05, price])
    return FeatureFrame(
        X=np.zeros((n, 3), dtype=np.float32),
        ts=ts.astype("datetime64[us]"),
        ohlc=ohlc,
        spread=np.full(n, 0.20),
        atr=np.full(n, 5.0),
        valid_from=60,
        columns=["a", "b", "c"],
    )


def buy(obs: np.ndarray) -> int:
    return 1


def hold(obs: np.ndarray) -> int:
    return 0


@pytest.mark.parametrize(
    ("axis", "values"),
    [
        (COMMISSION_AXIS, (0.0, 7.0, 14.0, 28.0)),
        (SLIPPAGE_AXIS, (0.0, 0.05, 0.1, 0.2)),
        (SPREAD_AXIS, (0.5, 1.0, 2.0, 3.0)),
    ],
)
def test_each_cost_axis_degrades_a_profitable_sequence(axis, values):
    features = trend_frame()
    report = parameter_sensitivity(buy, features, Settings(), axes={axis: values})

    assert len(report.axes) == 1
    rows = report.axes[0].rows
    nets = [row.net_profit for row in rows]
    costs = [row.total_costs for row in rows]
    trades = {row.trades for row in rows}

    assert report.base.trades > 0
    assert report.base.net_profit > 0.0
    assert len(trades) == 1 and trades.pop() == report.base.trades
    assert nets == sorted(nets, reverse=True)
    assert len(set(nets)) == len(nets)
    assert costs == sorted(costs)
    assert costs[0] < costs[-1]
    if axis == SPREAD_AXIS:
        assert [row.multiplier for row in rows] == list(values)
    else:
        assert all(row.multiplier is not None for row in rows)


def test_breakeven_is_interpolated_between_the_crossing_rows():
    """Commission is the cleanest axis: it is charged per lot and per trade."""
    features = trend_frame()
    base = Settings()
    report = parameter_sensitivity(buy, features, base, axes={COMMISSION_AXIS: (0.0, 2_000.0)})

    axis = report.axes[0]
    assert isinstance(axis, SensitivityAxis)
    first, second = axis.rows
    assert first.net_profit > 0.0
    assert second.net_profit < 0.0
    assert axis.breakeven_multiplier is not None
    expected = first.multiplier + (
        (0.0 - first.net_profit)
        * (second.multiplier - first.multiplier)
        / (second.net_profit - first.net_profit)
    )
    assert axis.breakeven_multiplier == pytest.approx(expected)
    assert 0.0 < axis.breakeven_multiplier < 2_000.0 / base.costs.commission_per_lot


def test_a_flat_axis_has_no_breakeven():
    report = parameter_sensitivity(
        hold, trend_frame(), Settings(), axes={COMMISSION_AXIS: (0.0, 7.0, 14.0)}
    )

    axis = report.axes[0]
    assert {row.trades for row in axis.rows} == {0}
    assert {row.net_profit for row in axis.rows} == {0.0}
    assert axis.breakeven_multiplier is None
    assert report.base.trades == 0


def test_rows_are_sorted_and_the_result_is_deterministic():
    features = trend_frame(n=400)
    axes = {SPREAD_AXIS: (3.0, 0.5, 1.0)}
    first = parameter_sensitivity(buy, features, Settings(), axes=axes)
    second = parameter_sensitivity(buy, features, Settings(), axes=axes)

    assert first == second
    assert [row.value for row in first.axes[0].rows] == [0.5, 1.0, 3.0]


def test_axes_follow_the_canonical_order_not_the_mapping_order():
    report = parameter_sensitivity(
        buy,
        trend_frame(n=400),
        Settings(),
        axes={COMMISSION_AXIS: (7.0,), SPREAD_AXIS: (1.0,), SLIPPAGE_AXIS: (0.05,)},
    )
    assert [axis.parameter for axis in report.axes] == [SPREAD_AXIS, SLIPPAGE_AXIS, COMMISSION_AXIS]


def test_unknown_axis_is_rejected():
    with pytest.raises(ValueError, match="unknown sensitivity axes"):
        parameter_sensitivity(buy, trend_frame(), Settings(), axes={"leverage": (1.0, 2.0)})


@pytest.mark.parametrize(
    "axes",
    [
        {SPREAD_AXIS: (1.0, 1.0)},
        {SPREAD_AXIS: (0.0,)},
        {COMMISSION_AXIS: (-1.0,)},
        {SLIPPAGE_AXIS: (float("nan"),)},
        {COMMISSION_AXIS: ()},
    ],
)
def test_invalid_grids_are_rejected(axes):
    with pytest.raises(ValueError):
        parameter_sensitivity(buy, trend_frame(), Settings(), axes=axes)


def test_empty_axis_mapping_is_rejected():
    with pytest.raises(ValueError, match="at least one sensitivity axis"):
        parameter_sensitivity(buy, trend_frame(), Settings(), axes={})


def test_default_grids_scale_with_the_base_costs():
    settings = Settings()
    axes = default_axes(settings)

    assert axes[SPREAD_AXIS] == (0.5, 1.0, 2.0, 3.0)
    assert axes[SLIPPAGE_AXIS] == (0.0, 0.025, 0.05, 0.1)
    assert axes[COMMISSION_AXIS] == (0.0, 3.5, 7.0, 14.0)

    zeroed = replace(settings, costs=replace(settings.costs, commission_per_lot=0.0))
    assert default_axes(zeroed)[COMMISSION_AXIS] == (0.0,)


def test_only_the_selected_cost_field_moves():
    settings = Settings()
    for axis, value in (
        (SPREAD_AXIS, 2.0),
        (SLIPPAGE_AXIS, 0.2),
        (COMMISSION_AXIS, 14.0),
    ):
        moved = _settings_for(axis, value, settings)
        assert moved.risk == settings.risk
        assert moved.contract == settings.contract
        changed = [
            field
            for field in ("spread_floor", "slippage_atr_frac", "commission_per_lot")
            if getattr(moved.costs, field) != getattr(settings.costs, field)
        ]
        assert len(changed) == 1


def test_only_spread_multiplier_touches_the_feature_frame():
    features = trend_frame(n=200)
    scaled = _features_for(SPREAD_AXIS, 2.0, features)
    assert np.array_equal(scaled.spread, features.spread * 2.0)
    assert np.array_equal(scaled.X, features.X)

    for axis, value in ((SLIPPAGE_AXIS, 0.2), (COMMISSION_AXIS, 14.0)):
        same = _features_for(axis, value, features)
        assert same is features


def test_settings_and_features_are_not_mutated():
    features = trend_frame(n=400)
    spread_before = features.spread.copy()
    settings = Settings()

    parameter_sensitivity(
        buy,
        features,
        settings,
        axes={SPREAD_AXIS: (0.5, 3.0), COMMISSION_AXIS: (0.0, 28.0)},
    )

    assert np.array_equal(features.spread, spread_before)
    assert settings == Settings()
