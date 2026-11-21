"""Trade-level and equity-curve metrics for one backtest run.

``compute`` is the single place where a run's outcome is summarised. It reads
only the closed ``Trade`` objects and the equity curve the simulator already
produced, so it never re-derives fills or costs.

Conventions:

* ``net_profit`` is the change of the final marked equity from
  ``initial_balance``; it is not the sum of ``Trade.pnl`` because an open
  position is marked, and data-end billing already closed it.
* ``profit_factor`` is ``gross_win / gross_loss`` using absolute losses. It is
  ``0.0`` when there is neither, and ``inf`` when there are wins without losses
  (a finite denominator would invent a number).
* ``max_drawdown`` is the deepest drop from a running peak as a positive
  fraction of that peak, clamped to ``[0, 1]``.
* ``sharpe`` annualises the per-bar equity returns with
  ``sqrt(bars_per_year)`` and returns ``0.0`` when the return series is flat or
  shorter than two returns.
* ``cost_ratio`` reports costs against gross profit, capped at ``1.0``; with no
  gross profit it is ``0.0`` because a ratio against zero is undefined.
* ``avg_bars`` is the mean holding time in M1 bars, derived from the trade
  timestamps (one minute per bar).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from scalping.types import Trade

__all__ = ["Metrics", "compute"]

#: M1 bars in a year (365 * 24 * 60), the default annualisation base.
BARS_PER_YEAR: int = 525_600


@dataclass(frozen=True, slots=True)
class Metrics:
    """Summary of one backtest run. All values are plain floats, never NaN."""

    net_profit: float
    profit_factor: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    expectancy_r: float
    trades: int
    total_costs: float
    cost_ratio: float
    avg_bars: float


def _profit_factor(gross_profit: float, gross_loss: float) -> float:
    if gross_profit <= 0.0 and gross_loss <= 0.0:
        return 0.0
    if gross_loss <= 0.0:
        return math.inf
    return gross_profit / gross_loss


def _sharpe(equity: np.ndarray, bars_per_year: int) -> float:
    if equity.size < 2:
        return 0.0
    previous = equity[:-1]
    current = equity[1:]
    valid = np.isfinite(previous) & np.isfinite(current) & (previous > 0.0)
    if not valid.any():
        return 0.0
    returns = (current[valid] - previous[valid]) / previous[valid]
    if returns.size < 2:
        return 0.0
    std = float(np.std(returns))
    if not math.isfinite(std) or std <= 0.0:
        return 0.0
    return float(np.mean(returns) / std * math.sqrt(bars_per_year))


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    drawdowns = np.where(peaks > 0.0, (peaks - equity) / np.where(peaks > 0.0, peaks, 1.0), 0.0)
    return float(np.clip(np.max(drawdowns), 0.0, 1.0))


def _avg_bars(trades: Sequence[Trade]) -> float:
    if not trades:
        return 0.0
    holding = [(trade.exit_ts - trade.entry_ts).total_seconds() / 60.0 for trade in trades]
    return float(np.mean(holding))


def compute(
    trades: Sequence[Trade],
    equity: np.ndarray,
    initial_balance: float,
    bars_per_year: int = BARS_PER_YEAR,
) -> Metrics:
    """Summarise ``trades`` and the per-step ``equity`` curve."""
    values = np.asarray(equity, dtype=np.float64).reshape(-1)
    pnls = np.asarray([trade.pnl for trade in trades], dtype=np.float64)
    wins = pnls[pnls > 0.0]
    losses = pnls[pnls < 0.0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    total_costs = float(sum(trade.costs for trade in trades))

    net_profit = float(values[-1] - initial_balance) if values.size else 0.0
    return Metrics(
        net_profit=net_profit,
        profit_factor=_profit_factor(gross_profit, gross_loss),
        sharpe=_sharpe(values, bars_per_year),
        max_drawdown=_max_drawdown(values),
        win_rate=float(wins.size / pnls.size) if pnls.size else 0.0,
        expectancy_r=float(np.mean([trade.r_multiple for trade in trades])) if trades else 0.0,
        trades=len(trades),
        total_costs=total_costs,
        cost_ratio=min(total_costs / gross_profit, 1.0) if gross_profit > 0.0 else 0.0,
        avg_bars=_avg_bars(trades),
    )
