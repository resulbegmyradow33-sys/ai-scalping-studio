"""Backtesting layer: simulator-driven runs, metrics and walk-forward splits.

Public surface:

* :func:`~scalping.backtest.engine.run_backtest` runs a policy through the
  single :class:`~scalping.env.simulator.Simulator` accounting loop and returns
  a :class:`~scalping.backtest.engine.BacktestResult`.
* :func:`~scalping.backtest.metrics.compute` summarises trades and the equity
  curve into :class:`~scalping.backtest.metrics.Metrics`.
* :func:`~scalping.backtest.walkforward.walkforward_split` produces
  non-overlapping calendar-month folds.
* :func:`~scalping.backtest.montecarlo.run_montecarlo` reports the distribution
  of equity paths over resampled trade sequences.
* :func:`~scalping.backtest.sensitivity.parameter_sensitivity` reruns one
  policy across execution-cost grids.
"""

from __future__ import annotations

from scalping.backtest.engine import BacktestResult, data_hash, run_backtest, run_id
from scalping.backtest.metrics import Metrics, compute
from scalping.backtest.montecarlo import (
    MonteCarloConfig,
    MonteCarloResult,
    PathOutcome,
    run_montecarlo,
)
from scalping.backtest.sensitivity import (
    SensitivityAxis,
    SensitivityResult,
    SensitivityRow,
    default_axes,
    parameter_sensitivity,
)
from scalping.backtest.walkforward import Split, Window, walkforward_split

__all__ = [
    "BacktestResult",
    "Metrics",
    "MonteCarloConfig",
    "MonteCarloResult",
    "PathOutcome",
    "SensitivityAxis",
    "SensitivityResult",
    "SensitivityRow",
    "Split",
    "Window",
    "compute",
    "data_hash",
    "default_axes",
    "parameter_sensitivity",
    "run_backtest",
    "run_id",
    "run_montecarlo",
    "walkforward_split",
]
