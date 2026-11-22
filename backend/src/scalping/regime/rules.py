"""Causal market-regime labels.

The regime engine describes the recent path of a symbol; it does not forecast
the next bar and must never be reported as a price predictor. Each label is
computed from strictly backward-looking windows, so a row can be classified in
live replay exactly as it is classified offline.

Axes (one column each, all aligned to the input rows):

``trend``
    ``strong_up`` / ``weak_up`` / ``sideways`` / ``weak_down`` / ``strong_down``
    from the efficiency ratio of the last ``trend_window`` closes. The ratio is
    the net move divided by the sum of absolute bar moves: a clean trend scores
    near 1, a choppy path near 0.
``volatility``
    ``low`` / ``normal`` / ``high`` from ATR relative to its own trailing
    median. It answers "is this symbol quiet or busy *for itself*", not "is this
    a cheap or expensive asset".
``expansion``
    ``contracting`` / ``stable`` / ``expanding`` from ATR now versus ATR
    ``expansion_window`` bars ago.
``breakout``
    ``up`` / ``down`` / ``none`` for the first close outside the prior
    ``breakout_window`` high/low (plus an optional ATR buffer) while the
    previous close was still inside that range. It is an event, not a state.
``unusual``
    ``True`` when the latest log return is a ``>= unusual_z`` standard-deviation
    outlier of the trailing return distribution.
``regime``
    one summary label for screens and tables: ``unusual`` and ``breakout_*``
    take precedence, otherwise ``<direction>_<volatility>`` (for example
    ``up_high`` or ``sideways_low``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from scalping.data.base import conform_bars
from scalping.features.indicators import atr

__all__ = [
    "REGIME_COLUMNS",
    "RegimeConfig",
    "classify",
]

#: Columns a classified frame carries, in order, after ``ts``.
REGIME_COLUMNS: Final[tuple[str, ...]] = (
    "trend",
    "trend_strength",
    "efficiency_ratio",
    "volatility",
    "vol_ratio",
    "expansion",
    "expansion_ratio",
    "breakout",
    "unusual",
    "return_z",
    "regime",
)

_MIN_RETURN_SAMPLES: Final[int] = 20


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    """Thresholds for :func:`classify`; defaults are research choices, not laws."""

    trend_window: int = 60
    trend_threshold: float = 0.30
    strong_trend_threshold: float = 0.50
    atr_window: int = 14
    vol_window: int = 480
    vol_low_ratio: float = 0.75
    vol_high_ratio: float = 1.50
    expansion_window: int = 10
    expansion_ratio: float = 1.30
    contraction_ratio: float = 0.77
    breakout_window: int = 20
    breakout_atr_buffer: float = 0.0
    return_window: int = 60
    unusual_z: float = 4.0

    def __post_init__(self) -> None:
        windows = {
            "trend_window": self.trend_window,
            "atr_window": self.atr_window,
            "vol_window": self.vol_window,
            "expansion_window": self.expansion_window,
            "breakout_window": self.breakout_window,
            "return_window": self.return_window,
        }
        for name, value in windows.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if not 0.0 < self.trend_threshold <= 1.0:
            raise ValueError("trend_threshold must be in (0, 1]")
        if not self.trend_threshold <= self.strong_trend_threshold <= 1.0:
            raise ValueError("strong_trend_threshold must be >= trend_threshold and <= 1")
        if not 0.0 < self.vol_low_ratio < self.vol_high_ratio:
            raise ValueError("vol_low_ratio must be positive and below vol_high_ratio")
        if not 0.0 < self.contraction_ratio < self.expansion_ratio:
            raise ValueError("contraction_ratio must be positive and below expansion_ratio")
        if self.breakout_atr_buffer < 0.0:
            raise ValueError("breakout_atr_buffer must be >= 0")
        if self.unusual_z <= 0.0:
            raise ValueError("unusual_z must be > 0")


def _empty_labels() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ts": pl.Datetime("us", "UTC"),
            "trend": pl.String,
            "trend_strength": pl.String,
            "efficiency_ratio": pl.Float64,
            "volatility": pl.String,
            "vol_ratio": pl.Float64,
            "expansion": pl.String,
            "expansion_ratio": pl.Float64,
            "breakout": pl.String,
            "unusual": pl.Boolean,
            "return_z": pl.Float64,
            "regime": pl.String,
        }
    )


def _ratio(numerator: pl.Expr, denominator: pl.Expr, default: float) -> pl.Expr:
    """Elementwise division that falls back to ``default`` on a zero denominator."""
    return (
        pl.when(denominator > 0)
        .then(numerator / denominator)
        .otherwise(default)
        .cast(pl.Float64)
    )


def classify(frame: pl.DataFrame, config: RegimeConfig | None = None) -> pl.DataFrame:
    """Return ``ts`` plus the regime axes, one row per input bar.

    The input is normalised with :func:`~scalping.data.base.conform_bars` and
    sorted by ``ts``; extra columns are dropped. The result has the same height
    and is aligned row for row with that sorted frame, so callers can join it
    back on ``ts``.
    """
    cfg = RegimeConfig() if config is None else config
    bars = conform_bars(frame).sort("ts")
    if bars.is_empty():
        return _empty_labels()

    work = bars.with_columns(
        atr(bars, cfg.atr_window).alias("_atr"),
        pl.col("close").log().diff().alias("_logret"),
    )

    net = pl.col("close") - pl.col("close").shift(cfg.trend_window)
    path = (
        pl.col("close")
        .diff()
        .abs()
        .rolling_sum(window_size=cfg.trend_window, min_samples=2)
    )
    efficiency = _ratio(net.abs(), path, 0.0).clip(0.0, 1.0)
    trending = efficiency >= cfg.trend_threshold
    strong = efficiency >= cfg.strong_trend_threshold

    trend = (
        pl.when(trending & (net > 0) & strong)
        .then(pl.lit("strong_up"))
        .when(trending & (net > 0))
        .then(pl.lit("weak_up"))
        .when(trending & (net < 0) & strong)
        .then(pl.lit("strong_down"))
        .when(trending & (net < 0))
        .then(pl.lit("weak_down"))
        .otherwise(pl.lit("sideways"))
    )
    trend_strength = (
        pl.when(trending & strong)
        .then(pl.lit("strong"))
        .when(trending)
        .then(pl.lit("weak"))
        .otherwise(pl.lit("none"))
    )

    atr_now = pl.col("_atr")
    vol_ratio = _ratio(
        atr_now,
        atr_now.rolling_median(window_size=cfg.vol_window, min_samples=1),
        1.0,
    )
    volatility = (
        pl.when(vol_ratio <= cfg.vol_low_ratio)
        .then(pl.lit("low"))
        .when(vol_ratio >= cfg.vol_high_ratio)
        .then(pl.lit("high"))
        .otherwise(pl.lit("normal"))
    )

    expansion_ratio = _ratio(atr_now, atr_now.shift(cfg.expansion_window), 1.0)
    expansion = (
        pl.when(expansion_ratio >= cfg.expansion_ratio)
        .then(pl.lit("expanding"))
        .when(expansion_ratio <= cfg.contraction_ratio)
        .then(pl.lit("contracting"))
        .otherwise(pl.lit("stable"))
    )

    prior_high = pl.col("high").rolling_max(
        window_size=cfg.breakout_window, min_samples=1
    ).shift(1)
    prior_low = pl.col("low").rolling_min(
        window_size=cfg.breakout_window, min_samples=1
    ).shift(1)
    buffer = cfg.breakout_atr_buffer * atr_now
    close = pl.col("close")
    # An event, not a state: the first close outside the range that was still
    # holding on the previous bar. Otherwise every bar of a trend would be a
    # "breakout" and the label would say nothing.
    was_inside = (close.shift(1) <= prior_high.shift(1)) & (
        close.shift(1) >= prior_low.shift(1)
    )
    breakout = (
        pl.when(((close > prior_high + buffer) & was_inside).fill_null(False))
        .then(pl.lit("up"))
        .when(((close < prior_low - buffer) & was_inside).fill_null(False))
        .then(pl.lit("down"))
        .otherwise(pl.lit("none"))
    )

    mean = pl.col("_logret").rolling_mean(
        window_size=cfg.return_window, min_samples=_MIN_RETURN_SAMPLES
    )
    std = pl.col("_logret").rolling_std(
        window_size=cfg.return_window, min_samples=_MIN_RETURN_SAMPLES
    )
    return_z = _ratio(pl.col("_logret") - mean, std, 0.0)
    unusual = (return_z.abs() >= cfg.unusual_z).fill_null(False)

    direction = (
        pl.when(trend.is_in(["strong_up", "weak_up"]))
        .then(pl.lit("up"))
        .when(trend.is_in(["strong_down", "weak_down"]))
        .then(pl.lit("down"))
        .otherwise(pl.lit("sideways"))
    )
    regime = (
        pl.when(unusual)
        .then(pl.lit("unusual"))
        .when(breakout == "up")
        .then(pl.lit("breakout_up"))
        .when(breakout == "down")
        .then(pl.lit("breakout_down"))
        .otherwise(direction + pl.lit("_") + volatility)
    )

    return work.select(
        pl.col("ts"),
        trend.alias("trend"),
        trend_strength.alias("trend_strength"),
        efficiency.alias("efficiency_ratio"),
        volatility.alias("volatility"),
        vol_ratio.alias("vol_ratio"),
        expansion.alias("expansion"),
        expansion_ratio.alias("expansion_ratio"),
        breakout.alias("breakout"),
        unusual.alias("unusual"),
        return_z.alias("return_z"),
        regime.alias("regime"),
    )
