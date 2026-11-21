"""Backtesting layer: simulator-driven runs, metrics and walk-forward splits.

Public surface:

* :func:`~scalping.backtest.engine.run_backtest` runs a policy through the
  single :class:`~scalping.env.simulator.Simulator` accounting loop and returns
  a :class:`~scalping.backtest.engine.BacktestResult`.
* :func:`~scalping.backtest.metrics.compute` summarises trades and the equity
  curve into :class:`~scalping.backtest.metrics.Metrics`.
* :func:`~scalping.backtest.walkforward.walkforward_split` produces
  non-overlapping calendar-month folds.
"""

from __future__ import annotations

from scalping.backtest.engine import BacktestResult, data_hash, run_backtest, run_id
from scalping.backtest.metrics import Metrics, compute
from scalping.backtest.walkforward import Split, Window, walkforward_split

__all__ = [
    "BacktestResult",
    "Metrics",
    "Split",
    "Window",
    "compute",
    "data_hash",
    "run_backtest",
    "run_id",
    "walkforward_split",
]
