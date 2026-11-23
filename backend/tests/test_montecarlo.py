"""Tests for Monte Carlo trade-sequence analysis.

The module is pure list arithmetic: it must be deterministic for a seed, keep
the observed terminal wealth under shuffling (multiplication is commutative),
vary it under bootstrap, and never crash on an empty or one-trade run. Ruin is
a touch of the floor, not only a final state.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import pairwise

import numpy as np
import pytest

from scalping.backtest.engine import BacktestResult
from scalping.backtest.metrics import Metrics
from scalping.backtest.montecarlo import (
    MonteCarloConfig,
    _equity_path,
    _path_indices,
    run_montecarlo,
)
from scalping.types import Side, Trade

BASE = datetime(2026, 1, 5, tzinfo=UTC)


def make_trade(pnl: float, minute: int = 0) -> Trade:
    return Trade(
        entry_ts=BASE + timedelta(minutes=minute),
        exit_ts=BASE + timedelta(minutes=minute + 5),
        side=Side.LONG,
        lot=0.1,
        entry=2_000.0,
        exit=2_000.0 + pnl / 10.0,
        sl=1_990.0,
        tp=2_010.0,
        pnl=pnl,
        costs=1.0,
        r_multiple=pnl / 100.0,
        reason="tp" if pnl > 0 else "sl",
        strategy="test",
    )


def traded(*pnls: float) -> list[Trade]:
    return [make_trade(pnl, minute=index * 10) for index, pnl in enumerate(pnls)]


MIXED = traded(300.0, -200.0, 150.0, -100.0, 50.0, -25.0)


def test_same_seed_gives_identical_result():
    config = MonteCarloConfig(n_paths=25, seed=11, mode="bootstrap", block_size=3)
    first = run_montecarlo(MIXED, config)
    second = run_montecarlo(MIXED, config)

    assert first == second
    assert first.outcomes == second.outcomes
    assert first.final_pnl_percentiles == second.final_pnl_percentiles


def test_different_seed_changes_the_paths():
    trades = MIXED
    a = run_montecarlo(trades, MonteCarloConfig(n_paths=40, seed=0))
    b = run_montecarlo(trades, MonteCarloConfig(n_paths=40, seed=1))

    assert a.outcomes != b.outcomes


def test_shuffle_keeps_terminal_wealth_but_moves_the_drawdown():
    """Multiplication is commutative: shuffling changes the path, not the end."""
    result = run_montecarlo(MIXED, MonteCarloConfig(n_paths=60, seed=4))

    endings = {round(outcome.ending_balance, 9) for outcome in result.outcomes}
    assert len(endings) == 1
    expected = 10_000.0
    for trade in MIXED:
        expected *= 1.0 + trade.pnl / 10_000.0
    assert endings.pop() == pytest.approx(expected)

    assert len({round(outcome.max_drawdown, 9) for outcome in result.outcomes}) > 1
    assert result.final_pnl_percentiles["p05"] == pytest.approx(result.final_pnl_percentiles["p95"])


def test_bootstrap_changes_the_multiset_and_so_the_terminal_wealth():
    shuffle = run_montecarlo(MIXED, MonteCarloConfig(n_paths=80, seed=5, mode="shuffle"))
    bootstrap = run_montecarlo(MIXED, MonteCarloConfig(n_paths=80, seed=5, mode="bootstrap"))

    assert len({round(o.ending_balance, 9) for o in shuffle.outcomes}) == 1
    assert len({round(o.ending_balance, 9) for o in bootstrap.outcomes}) > 1
    assert bootstrap.final_pnl_percentiles["p05"] < bootstrap.final_pnl_percentiles["p95"]


def test_empty_trade_list_is_a_documented_degenerate_result():
    result = run_montecarlo([], MonteCarloConfig(n_paths=50, seed=2, ruin_fraction=0.5))

    assert result.paths == 0
    assert result.trades == 0
    assert result.outcomes == ()
    assert result.worst_path is None
    assert result.degenerate is True
    assert result.probability_of_ruin == 0.0
    assert result.ruin_threshold == pytest.approx(5_000.0)
    assert result.final_pnl_percentiles == {"p05": 0.0, "p50": 0.0, "p95": 0.0}
    assert result.max_drawdown_percentiles == {"p05": 0.0, "p50": 0.0, "p95": 0.0}
    assert result.ending_balance_percentiles == {"p05": 0.0, "p50": 0.0, "p95": 0.0}


def test_single_trade_has_no_sequence_risk_but_is_priced():
    result = run_montecarlo(traded(250.0), MonteCarloConfig(n_paths=7, seed=0))

    assert result.paths == 7
    assert result.degenerate is True
    assert {outcome.ending_balance for outcome in result.outcomes} == {10_250.0}
    assert result.final_pnl_percentiles["p50"] == pytest.approx(250.0)
    assert result.max_drawdown_percentiles["p95"] == 0.0
    assert result.probability_of_ruin == 0.0


def test_ruin_is_a_touch_of_the_floor_not_only_the_ending_balance():
    losing = run_montecarlo(
        traded(-6_000.0, -6_000.0),
        MonteCarloConfig(n_paths=10, seed=0, ruin_fraction=0.5),
    )
    winning = run_montecarlo(
        traded(100.0, 100.0),
        MonteCarloConfig(n_paths=10, seed=0, ruin_fraction=0.5),
    )

    assert losing.ruin_threshold == pytest.approx(5_000.0)
    assert losing.probability_of_ruin == 1.0
    assert all(outcome.ruined for outcome in losing.outcomes)
    assert winning.probability_of_ruin == 0.0


def test_percentiles_are_ordered_and_follow_custom_levels():
    result = run_montecarlo(
        MIXED,
        MonteCarloConfig(n_paths=100, seed=8, mode="bootstrap", confidence_levels=(0.1, 0.9)),
    )

    assert set(result.final_pnl_percentiles) == {"p10", "p90"}
    for table in (
        result.final_pnl_percentiles,
        result.max_drawdown_percentiles,
        result.ending_balance_percentiles,
    ):
        assert table["p10"] <= table["p90"]


def test_default_percentiles_are_ordered():
    result = run_montecarlo(MIXED, MonteCarloConfig(n_paths=100, seed=9, mode="bootstrap"))
    for table in (
        result.final_pnl_percentiles,
        result.max_drawdown_percentiles,
        result.ending_balance_percentiles,
    ):
        assert table["p05"] <= table["p50"] <= table["p95"]


def test_block_bootstrap_preserves_contiguous_blocks():
    """With a block equal to the sample, every path is a circular rotation."""
    config = MonteCarloConfig(n_paths=6, seed=3, mode="bootstrap", block_size=8)
    paths = _path_indices(8, config)

    assert len(paths) == 6
    for path in paths:
        assert sorted(path.tolist()) == list(range(8))
        steps = [(int(b) - int(a)) % 8 for a, b in pairwise(path)]
        assert steps == [1] * 7


def test_block_size_is_clamped_to_the_sample():
    paths = _path_indices(3, MonteCarloConfig(n_paths=4, seed=1, mode="bootstrap", block_size=10))

    for path in paths:
        assert sorted(path.tolist()) == [0, 1, 2]
        steps = [(int(b) - int(a)) % 3 for a, b in pairwise(path)]
        assert steps == [1, 1]


def test_worst_path_matches_the_lowest_ending_balance():
    result = run_montecarlo(MIXED, MonteCarloConfig(n_paths=30, seed=7, mode="bootstrap"))

    assert result.worst_path is not None
    assert result.worst_path.ending_balance == min(o.ending_balance for o in result.outcomes)
    assert result.worst_path == min(
        result.outcomes, key=lambda outcome: (outcome.ending_balance, outcome.index)
    )


def test_input_trades_are_not_mutated_or_reordered():
    trades = traded(120.0, -80.0, 40.0)
    before = [trade.pnl for trade in trades]
    before_ids = [id(trade) for trade in trades]

    run_montecarlo(trades, MonteCarloConfig(n_paths=20, seed=0, mode="bootstrap"))

    assert [trade.pnl for trade in trades] == before
    assert [id(trade) for trade in trades] == before_ids


def test_a_backtest_result_is_accepted_as_input():
    metrics = Metrics(
        net_profit=100.0,
        profit_factor=2.0,
        sharpe=0.0,
        max_drawdown=0.01,
        win_rate=0.5,
        expectancy_r=0.5,
        trades=len(MIXED),
        total_costs=5.0,
        cost_ratio=0.05,
        avg_bars=5.0,
    )
    result = BacktestResult(
        run_id="abc",
        metrics=metrics,
        equity=np.asarray([10_100.0]),
        trades=MIXED,
        config_hash="def",
    )

    via_result = run_montecarlo(result, MonteCarloConfig(n_paths=15, seed=2))
    via_list = run_montecarlo(MIXED, MonteCarloConfig(n_paths=15, seed=2))

    assert via_result == via_list


def test_equity_path_compounds_returns_on_the_starting_balance():
    path = _equity_path(np.asarray([1_000.0, 1_000.0]), 10_000.0)
    assert np.allclose(path, [11_000.0, 12_100.0])

    recovered = _equity_path(np.asarray([-500.0, 1_000.0]), 10_000.0)
    assert np.allclose(recovered, [9_500.0, 10_450.0])


def test_equity_path_stops_at_zero():
    path = _equity_path(np.asarray([-20_000.0, 500.0]), 10_000.0)
    assert np.array_equal(path, [0.0, 0.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_paths": 0},
        {"mode": "iid"},
        {"block_size": 0},
        {"starting_balance": 0.0},
        {"ruin_fraction": 1.0},
        {"confidence_levels": ()},
        {"confidence_levels": (0.0, 0.5)},
        {"confidence_levels": (0.95, 0.95)},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        MonteCarloConfig(**kwargs)
