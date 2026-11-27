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
  fraction of that peak, clamped to ``[0, 1]``. The equity series is prefixed
  with ``initial_balance`` so a first-bar loss is included.
* ``sharpe`` annualises the per-bar equity returns with
  ``sqrt(bars_per_year)`` and returns ``0.0`` when the return series is flat or
  shorter than two returns. The same prefixed series is used.
* ``sortino`` uses the same prefixed returns but divides by the root-mean-square
  of the negative returns; a run with no losing bar reads ``0.0`` rather than
  an infinite ratio. ``calmar`` is the annualised **mean** bar return divided by
  ``max_drawdown``; a flat or non-positive curve reads ``0.0``.
* ``cost_ratio`` reports the broker's cost breakdown against gross profit,
  capped at ``1.0``; with no gross profit it is ``0.0``. The breakdown omits
  the exit half-spread on SL/TP fills, so this is not economic cost drag.
* ``avg_bars`` is the mean holding time. With the frame's ``ts`` array it is
  the index distance between entry and exit (weekend gaps excluded); without
  it the fallback is wall-clock minutes, which equals bars on a gapless M1
  series.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from scalping.types import Trade

__all__ = ["Metrics", "compute"]

#: M1 bars in a year (365 * 24 * 60), the default annualisation base.
BARS_PER_YEAR: int = 525_600


@dataclass(frozen=True, slots=True)
class Metrics:
    """Summary of one backtest run. All values are plain floats, never NaN.

    ``total_costs`` is the broker's reported cost breakdown, not economic cost
    drag: it contains the entry half-spread, entry slippage and the round-turn
    commission, plus the exit half-spread only on quote exits. SL/TP fills are
    level fills and report no exit half-spread, so ``total_costs`` and the
    ``cost_ratio`` derived from it understate the true friction. ``net_profit``
    stays the authoritative PnL; costs are already embedded in the fill prices
    and are never subtracted from it again.
    """

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
    #: Annualised mean return divided by downside deviation. Zero when there is
    #: no negative return or the curve is too short.
    sortino: float = 0.0
    #: Annualised return divided by max drawdown. Zero without a drawdown.
    calmar: float = 0.0


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


def _bar_returns(equity: np.ndarray) -> np.ndarray:
    if equity.size < 2:
        return np.zeros(0, dtype=np.float64)
    previous = equity[:-1]
    current = equity[1:]
    valid = np.isfinite(previous) & np.isfinite(current) & (previous > 0.0)
    if not valid.any():
        return np.zeros(0, dtype=np.float64)
    return (current[valid] - previous[valid]) / previous[valid]


def _sortino(equity: np.ndarray, bars_per_year: int) -> float:
    returns = _bar_returns(equity)
    if returns.size < 2:
        return 0.0
    downside = returns[returns < 0.0]
    if downside.size == 0:
        return 0.0
    deviation = float(np.sqrt(np.mean(np.square(downside))))
    if not math.isfinite(deviation) or deviation <= 0.0:
        return 0.0
    return float(np.mean(returns) / deviation * math.sqrt(bars_per_year))


def _calmar(equity: np.ndarray, bars_per_year: int) -> float:
    """Annualised mean bar return over max drawdown.

    The mean-return annualisation is used instead of CAGR so a one-bar curve
    cannot overflow ``exp``; the number is a scale-free fragility ratio, not a
    forecast.
    """
    returns = _bar_returns(equity)
    if returns.size < 1:
        return 0.0
    drawdown = _max_drawdown(equity)
    annualised = float(np.mean(returns)) * bars_per_year
    if not math.isfinite(annualised) or drawdown <= 0.0:
        return 0.0
    return annualised / drawdown


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    drawdowns = np.where(peaks > 0.0, (peaks - equity) / np.where(peaks > 0.0, peaks, 1.0), 0.0)
    return float(np.clip(np.max(drawdowns), 0.0, 1.0))


def _avg_bars(trades: Sequence[Trade]) -> float:
    """Fallback holding time in minutes (the frame is M1, so minutes are bars)."""
    if not trades:
        return 0.0
    holding = [(trade.exit_ts - trade.entry_ts).total_seconds() / 60.0 for trade in trades]
    return float(np.mean(holding))


def _to_us(value: datetime) -> np.datetime64:
    """Convert an aware or naive datetime to the naive UTC unit used by ``ts``."""
    if value.tzinfo is not None:
        value = value.astimezone(UTC).replace(tzinfo=None)
    return np.datetime64(value, "us")


def _avg_bars_indexed(trades: Sequence[Trade], ts: np.ndarray) -> float:
    """Mean index distance between entry and exit in the sorted ``ts`` array."""
    if not trades:
        return 0.0
    stamps = np.asarray(ts, dtype="datetime64[us]").reshape(-1)
    if stamps.size == 0:  # pragma: no cover - a trade cannot exist without bars
        return _avg_bars(trades)
    holding = []
    for trade in trades:
        entry = int(np.searchsorted(stamps, _to_us(trade.entry_ts), side="left"))
        exit_ = int(np.searchsorted(stamps, _to_us(trade.exit_ts), side="left"))
        holding.append(max(exit_ - entry, 0))
    return float(np.mean(holding))


def compute(
    trades: Sequence[Trade],
    equity: np.ndarray,
    initial_balance: float,
    bars_per_year: int = BARS_PER_YEAR,
    *,
    ts: np.ndarray | None = None,
) -> Metrics:
    """Summarise ``trades`` and the post-step ``equity`` curve.

    ``equity`` holds the marked equity after each simulator step and therefore
    starts *after* the first bar. Drawdown and Sharpe prepend ``initial_balance``
    so a loss on the first bar is visible; ``net_profit`` is still
    ``equity[-1] - initial_balance``.

    ``total_costs`` is the broker's reported cost breakdown. It excludes the
    exit half-spread on SL/TP fills, so ``cost_ratio`` is a reporting ratio, not
    economic cost drag; it is clamped to ``[0, 1]``. When ``ts`` (the frame's
    sorted timestamps) is given, ``avg_bars`` is the mean index distance between
    a trade's entry and exit in that array, so weekend and holiday gaps count as
    bars that actually traded. Without ``ts`` the fallback is wall-clock
    minutes, which equals bars only on a gapless M1 series.
    """
    values = np.asarray(equity, dtype=np.float64).reshape(-1)
    curve = (
        np.concatenate(([float(initial_balance)], values))
        if values.size
        else np.asarray([float(initial_balance)])
    )
    pnls = np.asarray([trade.pnl for trade in trades], dtype=np.float64)
    wins = pnls[pnls > 0.0]
    losses = pnls[pnls < 0.0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    total_costs = float(sum(trade.costs for trade in trades))

    net_profit = float(values[-1] - initial_balance) if values.size else 0.0
    cost_ratio = (
        min(max(total_costs / gross_profit, 0.0), 1.0) if gross_profit > 0.0 else 0.0
    )
    return Metrics(
        net_profit=net_profit,
        profit_factor=_profit_factor(gross_profit, gross_loss),
        sharpe=_sharpe(curve, bars_per_year),
        max_drawdown=_max_drawdown(curve),
        win_rate=float(wins.size / pnls.size) if pnls.size else 0.0,
        expectancy_r=float(np.mean([trade.r_multiple for trade in trades])) if trades else 0.0,
        trades=len(trades),
        total_costs=total_costs,
        cost_ratio=cost_ratio,
        avg_bars=_avg_bars_indexed(trades, ts) if ts is not None else _avg_bars(trades),
        sortino=_sortino(curve, bars_per_year),
        calmar=_calmar(curve, bars_per_year),
    )
