"""Liquidity features: fair value gaps and sweep rejections.

Fair value gaps
---------------
An up gap exists at bar ``i`` when ``low[i] > high[i-2]``: the middle bar jumped
so hard that the range between ``high[i-2]`` and ``low[i]`` was never traded.
The down gap is the mirror, ``high[i] < low[i-2]``. The zone is
``[high[i-2], low[i]]`` (up) or ``[high[i], low[i-2]]`` (down).

``fvg_dist`` is the distance from the current bar to the nearest **unfilled**
gap in ATR units. A gap stays open until price trades through its far edge
(a later ``low`` at or below ``high[i-2]`` for an up gap, mirrored for a down
gap). The sign follows the side: positive above an up gap, negative below a
down gap. ``0.0`` means no open gap.

The scan is strictly backward looking: gaps are created and tested against bars
that have already printed.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from scalping.features.indicators import atr

__all__ = ["fvg", "liquidity_sweep"]


def fvg(df: pl.DataFrame, n_atr: int = 14) -> pl.DataFrame:
    """``fvg_up``/``fvg_down`` booleans and ``fvg_dist`` in ATR units."""
    highs = df["high"].to_numpy().astype(np.float64)
    lows = df["low"].to_numpy().astype(np.float64)
    atr_values = atr(df, n_atr).to_numpy().astype(np.float64)
    height = df.height

    up_flag = np.zeros(height, dtype=bool)
    down_flag = np.zeros(height, dtype=bool)
    distance = np.zeros(height, dtype=np.float64)
    # Open zones as (bottom, top). An up gap sits below price, a down gap above.
    up_gaps: list[tuple[float, float]] = []
    down_gaps: list[tuple[float, float]] = []

    for i in range(height):
        # 1. Drop gaps price has traded all the way through by this bar.
        low = lows[i]
        high = highs[i]
        up_gaps = [gap for gap in up_gaps if low > gap[0]]
        down_gaps = [gap for gap in down_gaps if high < gap[1]]

        # 2. Register a gap completed by this bar.
        if i >= 2:
            if low > highs[i - 2]:
                up_flag[i] = True
                up_gaps.append((highs[i - 2], low))
            if high < lows[i - 2]:
                down_flag[i] = True
                down_gaps.append((highs[i], lows[i - 2]))

        # 3. Nearest open gap, signed by side, in ATR units.
        scale = atr_values[i]
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        best = 0.0
        for bottom, top in up_gaps:
            candidate = (low - top) / scale
            if best == 0.0 or abs(candidate) < abs(best):
                best = candidate
        for bottom, top in down_gaps:
            candidate = (high - bottom) / scale
            if best == 0.0 or abs(candidate) < abs(best):
                best = candidate
        distance[i] = best

    return pl.DataFrame(
        {
            "fvg_up": pl.Series("fvg_up", up_flag, dtype=pl.Boolean),
            "fvg_down": pl.Series("fvg_down", down_flag, dtype=pl.Boolean),
            "fvg_dist": pl.Series("fvg_dist", distance, dtype=pl.Float64),
        }
    )


def liquidity_sweep(df: pl.DataFrame, lookback: int = 20) -> pl.Series:
    """Wick through the prior ``lookback`` bar extreme that closes back inside.

    A high sweep is ``high[i] > prior_high`` with ``close[i] < prior_high``: the
    bar traded above the range but was rejected. The low side is the mirror.
    ``prior_high``/``prior_low`` exclude the current bar.
    """
    if lookback < 1:
        raise ValueError(f"sweep lookback must be >= 1, got {lookback}")
    close = pl.col("close")
    prior_high = pl.col("high").rolling_max(lookback, min_samples=lookback).shift(1)
    prior_low = pl.col("low").rolling_min(lookback, min_samples=lookback).shift(1)
    swept_high = (pl.col("high") > prior_high) & (close < prior_high)
    swept_low = (pl.col("low") < prior_low) & (close > prior_low)
    return df.select(
        (swept_high.fill_null(False) | swept_low.fill_null(False)).alias("liquidity_sweep")
    ).to_series()
