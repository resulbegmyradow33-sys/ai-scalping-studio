"""Walk-forward and cost-stress evaluation tests.

The walk-forward report is the out-of-sample story of one fixed policy: the
same callable is run on every fold's test window and the per-fold trades are
aggregated. Nothing is retrained between folds in v1, so these tests also pin
the "no look-ahead" contract: features are rebuilt inside each window and a
train window never reaches into its test window.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from scalping.backtest.engine import BacktestResult
from scalping.backtest.evaluate import (
    _scaled_features,
    _select_splits,
    cost_stress,
    evaluate_walkforward,
)
from scalping.backtest.metrics import Metrics
from scalping.config import CostConfig, Settings
from scalping.data.synthetic import SyntheticProvider
from scalping.env.simulator import MAX_STEPS
from scalping.features.build import FeatureFrame, build_features

START = datetime(2026, 1, 5, tzinfo=UTC)


def make(days: int = 200):
    df = SyntheticProvider(seed=33).fetch("XAUUSD", START, START + timedelta(days=days))
    return df, build_features(df)


def test_walkforward_produces_non_overlapping_folds():
    df, _ = make()
    rep = evaluate_walkforward(lambda o: 0, df, Settings(), folds=3)
    assert len(rep.folds) == 3
    assert rep.folds_requested == 3
    assert rep.folds_produced == 3
    for a, b in itertools.pairwise(rep.folds):
        assert a.test_range[1] <= b.test_range[0]
        # A one-month M1 test window is ~29k bars: the legacy 20k cap would
        # truncate the tail, so full evaluation must go past MAX_STEPS.
        assert len(a.equity) > MAX_STEPS


def test_fold_train_window_never_reaches_into_its_test_window():
    df, _ = make()
    rep = evaluate_walkforward(lambda o: 0, df, Settings(), folds=3)
    for fold in rep.folds:
        assert fold.train_range[0] < fold.train_range[1] <= fold.test_range[0]
        assert fold.test_range[0] < fold.test_range[1]
        assert fold.test_range[1] <= df["ts"].max()
        assert fold.test_range[0] <= fold.evaluated_range[0] <= fold.evaluated_range[1]
        assert fold.evaluated_range[1] <= fold.test_range[1]


def test_zero_trade_policy_has_zero_expectancy_everywhere():
    df, _ = make()
    rep = evaluate_walkforward(lambda o: 0, df, Settings(), folds=3)
    assert rep.overall.trades == 0
    assert rep.overall.expectancy_r == 0.0
    assert all(fold.metrics.trades == 0 for fold in rep.folds)
    assert all(fold.metrics.expectancy_r == 0.0 for fold in rep.folds)
    assert all(fold.train_expectancy == 0.0 for fold in rep.folds)
    assert not rep.overfit_warning


def flat_frame(n: int = 1000):
    """Flat bars with stops far outside the range, so only timeouts close trades.

    A constant price and ATR 5.0 keep SL/TP unreachable, which makes the trade
    count identical across cost multipliers; only the charged costs move.
    """
    price = 1000.0
    ts = np.datetime64("2026-01-05T00:00", "us") + np.arange(n).astype("timedelta64[m]")
    ohlc = np.column_stack(
        [
            np.full(n, price),
            np.full(n, price + 0.001),
            np.full(n, price - 0.001),
            np.full(n, price),
        ]
    )
    return FeatureFrame(
        X=np.zeros((n, 3), dtype=np.float32),
        ts=ts.astype("datetime64[us]"),
        ohlc=ohlc,
        spread=np.full(n, 0.20),
        atr=np.full(n, 5.0),
        valid_from=60,
        columns=["a", "b", "c"],
    )


def test_cost_stress_monotonic_costs():
    out = cost_stress(lambda o: 1, flat_frame(), Settings())
    costs = [r.metrics.total_costs for r in out if r.metrics.trades > 0]
    assert len({r.metrics.trades for r in out}) == 1
    assert costs == sorted(costs)
    assert len(costs) == 3
    assert costs[0] < costs[1] < costs[2]


def test_scaling_only_spread_changes_total_costs():
    """Fills read ``features.spread``; settings alone must not be the lever."""
    _, f = make(30)
    policy = lambda o: 1 if o[0] > 0 else 0
    scaled = _scaled_features(f, 3.0)
    assert np.array_equal(scaled.spread, f.spread * 3.0)
    assert not np.array_equal(scaled.spread, f.spread)

    # Zero commission and slippage so the only chargeable cost left is spread.
    base_settings = Settings(costs=CostConfig(slippage_atr_frac=0.0, commission_per_lot=0.0))
    base = cost_stress(policy, f, base_settings, multipliers=(1.0, 3.0))
    assert base[0].metrics.trades > 0
    assert base[1].metrics.total_costs > base[0].metrics.total_costs


def test_cost_stress_does_not_mutate_settings_or_features():
    _, f = make(30)
    settings = Settings()
    snapshot = replace(settings)
    spread_before = f.spread.copy()
    cost_stress(lambda o: 1 if o[0] > 0 else 0, f, settings)
    assert settings == snapshot
    assert settings.costs.spread_floor == snapshot.costs.spread_floor
    assert settings.costs.slippage_atr_frac == snapshot.costs.slippage_atr_frac
    assert settings.costs.commission_per_lot == snapshot.costs.commission_per_lot
    assert np.array_equal(f.spread, spread_before)


def _flat_metrics(expectancy: float) -> Metrics:
    return Metrics(
        net_profit=0.0,
        profit_factor=0.0,
        sharpe=0.0,
        max_drawdown=0.0,
        win_rate=0.0,
        expectancy_r=expectancy,
        trades=1,
        total_costs=0.0,
        cost_ratio=0.0,
        avg_bars=1.0,
    )


def _patched_report(monkeypatch, train: tuple[float, ...], test: tuple[float, ...], folds: int):
    """Drive ``evaluate_walkforward`` with scripted per-fold expectancies."""
    import scalping.backtest.evaluate as evaluate_module

    calls = {"n": 0}

    def fake_run(policy, features, settings, **kwargs):
        index = calls["n"]
        calls["n"] += 1
        values = train if index % 2 == 0 else test
        value = values[index // 2]
        return BacktestResult(
            run_id="fake",
            metrics=_flat_metrics(value),
            equity=np.array([10_000.0]),
            trades=[],
            config_hash="fake",
        )

    def fake_features(df, start, end, settings):
        stamps = np.datetime64(start.replace(tzinfo=None), "us") + np.arange(130).astype(
            "timedelta64[m]"
        )
        return FeatureFrame(
            X=np.zeros((130, 1), dtype=np.float32),
            ts=stamps,
            ohlc=np.zeros((130, 4), dtype=np.float64),
            spread=np.zeros(130, dtype=np.float64),
            atr=np.zeros(130, dtype=np.float64),
            valid_from=0,
        )

    monkeypatch.setattr(evaluate_module, "run_backtest", fake_run)
    monkeypatch.setattr(evaluate_module, "_window_features", fake_features)
    stamps = pl.datetime_range(
        START, START + timedelta(days=420), interval="1d", time_zone="UTC", eager=True
    )
    return evaluate_walkforward(lambda o: 0, pl.DataFrame({"ts": stamps}), Settings(), folds=folds)


def test_overfit_warning_fires_when_train_rises_and_test_falls(monkeypatch):
    rep = _patched_report(monkeypatch, train=(-1.0, 0.0, 1.0), test=(1.0, 0.0, -1.0), folds=3)
    assert rep.overfit_warning
    assert rep.train_expectancy_trend == pytest.approx(1.0)
    assert rep.test_expectancy_trend == pytest.approx(-1.0)


def test_overfit_warning_requires_at_least_three_folds(monkeypatch):
    rep = _patched_report(monkeypatch, train=(0.0, 1.0), test=(1.0, -1.0), folds=2)
    assert not rep.overfit_warning
    assert rep.folds_produced == 2


def test_select_splits_gives_three_folds_for_420_days():
    stamps = pl.datetime_range(
        START, START + timedelta(days=420), interval="1d", time_zone="UTC", eager=True
    )
    splits, months = _select_splits(pl.DataFrame({"ts": stamps}), train_months=6, folds=3)
    assert months == 6
    assert len(splits) == 3
    for earlier, later in itertools.pairwise(splits):
        assert earlier.test.end <= later.test.start
        assert earlier.test.end <= stamps.max()


def test_select_splits_raises_when_no_fold_fits():
    stamps = pl.datetime_range(
        START, START + timedelta(days=2), interval="1d", time_zone="UTC", eager=True
    )
    with pytest.raises(ValueError, match="not enough data for 3 folds"):
        _select_splits(pl.DataFrame({"ts": stamps}), train_months=6, folds=3)
