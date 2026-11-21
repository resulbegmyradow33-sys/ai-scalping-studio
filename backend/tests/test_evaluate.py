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

import polars as pl

from scalping.backtest.evaluate import (
    _select_splits,
    cost_stress,
    evaluate_walkforward,
)
from scalping.config import Settings
from scalping.data.synthetic import SyntheticProvider
from scalping.features.build import build_features

START = datetime(2026, 1, 5, tzinfo=UTC)


def make(days: int = 200):
    df = SyntheticProvider(seed=33).fetch("XAUUSD", START, START + timedelta(days=days))
    return df, build_features(df)


def test_walkforward_produces_non_overlapping_folds():
    df, _ = make()
    rep = evaluate_walkforward(lambda o: 0, df, Settings(), folds=3)
    assert len(rep.folds) == 3
    for a, b in zip(rep.folds, rep.folds[1:]):
        assert a.test_range[1] <= b.test_range[0]


def test_fold_train_window_never_reaches_into_its_test_window():
    df, _ = make()
    rep = evaluate_walkforward(lambda o: 0, df, Settings(), folds=3)
    for fold in rep.folds:
        assert fold.train_range[0] < fold.train_range[1] <= fold.test_range[0]
        assert fold.test_range[0] < fold.test_range[1]
        assert fold.test_range[1] <= df["ts"].max()


def test_zero_trade_policy_has_zero_expectancy_everywhere():
    df, _ = make()
    rep = evaluate_walkforward(lambda o: 0, df, Settings(), folds=3)
    assert rep.overall.trades == 0
    assert rep.overall.expectancy_r == 0.0
    assert all(fold.metrics.trades == 0 for fold in rep.folds)
    assert all(fold.metrics.expectancy_r == 0.0 for fold in rep.folds)
    assert not rep.overfit_warning


def test_cost_stress_monotonic_costs():
    _, f = make(30)
    out = cost_stress(lambda o: 1 if o[0] > 0 else 0, f, Settings())
    costs = [r.metrics.total_costs for r in out if r.metrics.trades > 0]
    assert costs == sorted(costs)
    assert len(costs) == 3


def test_cost_stress_does_not_mutate_settings():
    _, f = make(30)
    settings = Settings()
    snapshot = replace(settings)
    cost_stress(lambda o: 1 if o[0] > 0 else 0, f, settings)
    assert settings == snapshot
    assert settings.costs.spread_floor == snapshot.costs.spread_floor
    assert settings.costs.slippage_atr_frac == snapshot.costs.slippage_atr_frac
    assert settings.costs.commission_per_lot == snapshot.costs.commission_per_lot


def test_select_splits_gives_three_folds_for_420_days():
    stamps = pl.datetime_range(
        START, START + timedelta(days=420), interval="1d", time_zone="UTC", eager=True
    )
    splits = _select_splits(pl.DataFrame({"ts": stamps}), train_months=6, folds=3)
    assert len(splits) == 3
    for earlier, later in itertools.pairwise(splits):
        assert earlier.test.end <= later.test.start
        assert earlier.test.end <= stamps.max()
