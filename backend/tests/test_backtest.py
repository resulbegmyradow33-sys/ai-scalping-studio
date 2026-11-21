"""Tests for the backtest engine, the trade metrics and walk-forward splits.

The engine must drive the existing :class:`~scalping.env.simulator.Simulator`
instead of re-implementing fills or equity, so several tests compare a
backtest against a hand-driven simulator loop.
"""

from __future__ import annotations

import itertools
import math
from datetime import UTC, datetime, timedelta
from functools import lru_cache

import numpy as np
import pytest

from scalping.backtest.engine import run_backtest
from scalping.backtest.metrics import compute
from scalping.backtest.walkforward import walkforward_split
from scalping.config import CostConfig, Settings
from scalping.data.synthetic import SyntheticProvider
from scalping.env.simulator import Simulator
from scalping.features.build import build_features
from scalping.types import Side, Trade

BASE = datetime(2026, 1, 5, tzinfo=UTC)


def make(days: int = 3):
    # ``start + timedelta`` (not day overflow) so the brief's ``days=420`` works.
    start = datetime(2026, 1, 5, tzinfo=UTC)
    df = SyntheticProvider(seed=9).fetch("XAUUSD", start, start + timedelta(days=days))
    return build_features(df), df


@lru_cache(maxsize=1)
def long_frame():
    """The 14-month frame shared by the walk-forward tests (build is ~5 s)."""
    return make(days=420)


def make_trade(pnl: float, costs: float, r_multiple: float, entry_min: int, exit_min: int) -> Trade:
    return Trade(
        entry_ts=BASE + timedelta(minutes=entry_min),
        exit_ts=BASE + timedelta(minutes=exit_min),
        side=Side.LONG,
        lot=0.1,
        entry=2000.0,
        exit=2000.0 + pnl,
        sl=1990.0,
        tp=2010.0,
        pnl=pnl,
        costs=costs,
        r_multiple=r_multiple,
        reason="tp" if pnl > 0 else "sl",
        strategy="test",
    )


def test_hold_policy_has_no_trades_and_flat_equity():
    f, _ = make()
    res = run_backtest(lambda obs: 0, f, Settings())
    assert res.trades == []
    assert np.allclose(res.equity, 10_000.0)


def test_same_seed_same_result_bit_exact():
    f, _ = make()
    pol = lambda obs: 1 if (obs[0] > 0) else 0
    a = run_backtest(pol, f, Settings())
    b = run_backtest(pol, f, Settings())
    assert np.array_equal(a.equity, b.equity)
    assert a.run_id == b.run_id


def test_metrics_known_answer():
    m = compute(trades=[], equity=np.array([10_000.0] * 10), initial_balance=10_000.0)
    assert m.net_profit == 0 and m.profit_factor == 0 and m.max_drawdown == 0


def test_metrics_two_trades_hand_computed():
    """Two trades, hand arithmetic.

    +200 win / -100 loss, costs 30, r multiples +2/-1, holding 10 and 6 bars.
    Equity 10 000 -> 10 200 -> 10 100.

    net_profit       = 10 100 - 10 000            = 100
    profit_factor    = 200 / 100                  = 2.0
    max_drawdown     = (10 200 - 10 100) / 10 200 = 1/102
    win_rate         = 1 / 2                      = 0.5
    expectancy_r     = (2 + -1) / 2               = 0.5
    total_costs      = 20 + 10                    = 30
    cost_ratio       = 30 / 200                   = 0.15
    avg_bars         = (10 + 6) / 2               = 8
    bar returns      = +2%, -100/10200%
    sharpe           = (13/38) * sqrt(525 600)    = 248.02...
    """
    trades = [
        make_trade(pnl=200.0, costs=20.0, r_multiple=2.0, entry_min=0, exit_min=10),
        make_trade(pnl=-100.0, costs=10.0, r_multiple=-1.0, entry_min=20, exit_min=26),
    ]
    m = compute(trades, equity=np.array([10_000.0, 10_200.0, 10_100.0]), initial_balance=10_000.0)

    assert m.net_profit == pytest.approx(100.0)
    assert m.profit_factor == pytest.approx(2.0)
    assert m.max_drawdown == pytest.approx(1.0 / 102.0)
    assert m.win_rate == pytest.approx(0.5)
    assert m.expectancy_r == pytest.approx(0.5)
    assert m.trades == 2
    assert m.total_costs == pytest.approx(30.0)
    assert m.cost_ratio == pytest.approx(0.15)
    assert m.avg_bars == pytest.approx(8.0)
    assert m.sharpe == pytest.approx(13.0 / 38.0 * math.sqrt(525_600))


def test_profit_factor_is_inf_with_wins_and_no_losses():
    trades = [make_trade(pnl=50.0, costs=5.0, r_multiple=1.0, entry_min=0, exit_min=5)]
    m = compute(trades, equity=np.array([10_000.0, 10_050.0]), initial_balance=10_000.0)
    assert m.profit_factor == math.inf
    assert m.cost_ratio == pytest.approx(0.1)


def test_cost_ratio_is_capped_at_one():
    trades = [make_trade(pnl=100.0, costs=500.0, r_multiple=1.0, entry_min=0, exit_min=5)]
    m = compute(trades, equity=np.array([10_000.0, 10_100.0]), initial_balance=10_000.0)
    assert m.cost_ratio == 1.0


def test_max_drawdown_is_a_positive_fraction():
    m = compute([], equity=np.array([10_000.0, 11_000.0, 9_900.0]), initial_balance=10_000.0)
    assert m.max_drawdown == pytest.approx(1_100.0 / 11_000.0)
    assert 0.0 < m.max_drawdown <= 1.0
    flat = compute([], equity=np.array([10_000.0, 10_000.0]), initial_balance=10_000.0)
    assert flat.max_drawdown == 0.0


def test_zero_trade_metrics_are_finite_not_nan():
    m = compute([], equity=np.array([10_000.0, 10_000.0]), initial_balance=10_000.0)
    for value in (
        m.net_profit,
        m.profit_factor,
        m.sharpe,
        m.max_drawdown,
        m.win_rate,
        m.expectancy_r,
        m.total_costs,
        m.cost_ratio,
        m.avg_bars,
    ):
        assert math.isfinite(value)
    assert m.trades == 0
    assert m.expectancy_r == 0.0 and m.win_rate == 0.0 and m.sharpe == 0.0


def test_walkforward_splits_do_not_overlap():
    _, df = long_frame()
    splits = walkforward_split(df)
    assert len(splits) >= 3
    for s in splits:
        assert s.train.end <= s.val.start and s.val.end <= s.test.start


def test_walkforward_test_end_never_exceeds_last_bar():
    _, df = long_frame()
    last = df["ts"].max()
    splits = walkforward_split(df)
    assert splits
    for s in splits:
        assert s.test.end <= last
        assert s.train.start < s.val.start < s.test.start
    # Fold 0 starts on the first bar; every later train window steps one month.
    for earlier, later in itertools.pairwise(splits):
        assert later.train.start == _next_month(earlier.train.start)


def test_walkforward_rejects_empty_frame():
    import polars as pl

    from scalping.data.base import BAR_SCHEMA

    assert walkforward_split(pl.DataFrame(schema=BAR_SCHEMA)) == []


def test_backtest_matches_a_direct_simulator_loop():
    """The engine may not re-implement accounting: same steps, same equity, same trades."""
    f, _ = make()
    pol = lambda obs: 1 if (obs[0] > 0) else 0
    res = run_backtest(pol, f, Settings())

    sim = Simulator(f, Settings())
    obs = sim.reset(0)
    equity: list[float] = []
    trades: list[Trade] = []
    while True:
        step = sim.step(int(pol(obs)))
        obs = step.obs
        equity.append(float(step.info["equity"]))
        if step.info["trade"] is not None:
            trades.append(step.info["trade"])
        if step.terminated or step.truncated:
            break

    assert len(res.equity) == len(equity)
    assert np.array_equal(res.equity, np.asarray(equity))
    assert [t.pnl for t in res.trades] == [t.pnl for t in trades]
    assert res.trades, "the chosen policy must actually trade"
    assert res.metrics.net_profit == pytest.approx(res.equity[-1] - 10_000.0)


def test_run_id_changes_with_the_feature_matrix():
    settings = Settings()
    short, _ = make(days=2)
    long, _ = make(days=3)
    assert run_backtest(lambda o: 0, short, settings).run_id != (
        run_backtest(lambda o: 0, long, settings).run_id
    )


def test_backtest_instantiates_one_simulator(monkeypatch):
    from scalping.backtest import engine

    created = []
    real = engine.Simulator

    def counting(*args, **kwargs):
        sim = real(*args, **kwargs)
        created.append(sim)
        return sim

    monkeypatch.setattr(engine, "Simulator", counting)
    f, _ = make()
    run_backtest(lambda obs: 0, f, Settings())
    assert len(created) == 1


def test_same_config_same_run_id_different_config_differs():
    f, _ = make()
    s = Settings()
    assert run_backtest(lambda o: 0, f, s).run_id == run_backtest(lambda o: 0, f, s).run_id
    s2 = Settings(costs=CostConfig(spread_floor=0.5))
    assert run_backtest(lambda o: 0, f, s).run_id != run_backtest(lambda o: 0, f, s2).run_id
    assert run_backtest(lambda o: 0, f, s).config_hash != run_backtest(lambda o: 0, f, s2).config_hash


def _next_month(value: datetime) -> datetime:
    year, month = (value.year + 1, 1) if value.month == 12 else (value.year, value.month + 1)
    return value.replace(year=year, month=month)
