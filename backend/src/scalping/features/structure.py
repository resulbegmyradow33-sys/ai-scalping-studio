"""Causal market structure: swings, breaks of structure, order blocks and character changes.

A swing is a local extreme of the ``2k + 1`` bar window centred on the pivot.
The label is written at the **confirmation** bar ``i + k``, the first bar at
which the pivot can be known, so a row never learns about a future extreme.
That is the one place a centred window is allowed; the label shift keeps the
whole feature causal.

A break of structure (BOS) is a close that crosses the most recent structure
level: the last confirmed swing high/low, or the recent ``2k + 1`` bar extreme
while no swing has confirmed yet (a fresh series must still produce signals).
A change of character (CHoCH) is a BOS that flips the running trend.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import polars as pl

from scalping.features.indicators import atr

__all__ = ["market_structure", "order_block_distance", "swings"]


def swings(df: pl.DataFrame, k: int = 2) -> tuple[pl.Series, pl.Series]:
    """Boolean ``(swing_high, swing_low)`` aligned to ``df`` rows.

    Both are ``True`` at the confirmation bar ``i + k`` when ``high[i]`` (or
    ``low[i]``) is the extreme of the ``2k + 1`` bars around ``i``. Ties count:
    a plateau of equal highs produces a label for every equal extreme.
    """
    if k < 1:
        raise ValueError(f"swing window k must be >= 1, got {k}")
    size = 2 * k + 1
    high = pl.col("high")
    low = pl.col("low")
    is_high = (high == high.rolling_max(size, center=True, min_samples=size)).fill_null(False)
    is_low = (low == low.rolling_min(size, center=True, min_samples=size)).fill_null(False)
    out = df.select(
        # +k moves the pivot label forward to the bar that confirms it.
        is_high.shift(k).fill_null(False).alias("swing_high"),
        is_low.shift(k).fill_null(False).alias("swing_low"),
    )
    return out.to_series(0), out.to_series(1)


def market_structure(df: pl.DataFrame, k: int = 2) -> pl.DataFrame:
    """Trend state and break events, one row per input bar.

    Returns ``trend``, ``bos`` and ``choch`` as ``Int8`` in ``{-1, 0, 1}`` plus
    the ``swing_high``/``swing_low`` labels and the ``structure_high``/
    ``structure_low`` levels (the last confirmed swing, forward filled, with the
    recent range extreme as fallback). A ``bos`` of ``1`` is a close crossing up
    through ``structure_high``; ``-1`` is the mirror. ``choch`` repeats the BOS
    sign only when it flips a non-zero trend.
    """
    if k < 1:
        raise ValueError(f"swing window k must be >= 1, got {k}")
    frame = df.sort("ts") if "ts" in df.columns else df
    size = 2 * k + 1
    swing_high, swing_low = swings(frame, k)

    work = frame.with_columns(swing_high, swing_low)
    work = work.with_columns(
        pl.when(pl.col("swing_high"))
        .then(pl.col("high").shift(k))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("_swing_high_raw"),
        pl.when(pl.col("swing_low"))
        .then(pl.col("low").shift(k))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("_swing_low_raw"),
    )
    work = work.with_columns(
        pl.col("_swing_high_raw").forward_fill().alias("_swing_high_level"),
        pl.col("_swing_low_raw").forward_fill().alias("_swing_low_level"),
        # Fallback while no pivot has confirmed: the recent range, current bar excluded.
        pl.col("high").rolling_max(size, min_samples=1).shift(1).alias("_range_high"),
        pl.col("low").rolling_min(size, min_samples=1).shift(1).alias("_range_low"),
    )
    work = work.with_columns(
        pl.coalesce("_swing_high_level", "_range_high").alias("structure_high"),
        pl.coalesce("_swing_low_level", "_range_low").alias("structure_low"),
    )

    close = pl.col("close")
    prev_close = close.shift(1)
    bos_up = (close > pl.col("structure_high")) & (prev_close <= pl.col("structure_high"))
    bos_down = (close < pl.col("structure_low")) & (prev_close >= pl.col("structure_low"))
    bos = (
        pl.when(bos_up.fill_null(False))
        .then(1)
        .when(bos_down.fill_null(False))
        .then(-1)
        .otherwise(0)
        .cast(pl.Int8)
    )
    trend = (
        pl.when(bos != 0).then(bos).otherwise(None).forward_fill().fill_null(0).cast(pl.Int8)
    )
    prev_trend = trend.shift(1).fill_null(0)
    choch = (
        pl.when((bos != 0) & (prev_trend != 0) & (bos != prev_trend))
        .then(bos)
        .otherwise(0)
        .cast(pl.Int8)
    )
    return work.select(
        trend.alias("trend"),
        bos.alias("bos"),
        choch.alias("choch"),
        pl.col("swing_high"),
        pl.col("swing_low"),
        pl.col("structure_high"),
        pl.col("structure_low"),
    )


def order_block_distance(df: pl.DataFrame, n_atr: int = 14) -> pl.Series:
    """Signed distance to the nearest recent opposing candle body, in ATR units.

    Two bodies are tracked per side: the two most recent **down** candles (the
    opposing block for a long) and the two most recent **up** candles (the block
    for a short). The body is ``[min(open, close), max(open, close)]``.

    Distance from the current close to a body is ``(close - body_top) / ATR``
    above it, ``(close - body_bottom) / ATR`` below it, and ``0`` inside it. The
    nearest candidate by absolute distance wins, so the result is ``0`` whenever
    the close sits in any tracked body. The current bar's body joins the book
    only after its own row, so a row never measures against itself. ``0.0``
    before any body exists.
    """
    if n_atr < 1:
        raise ValueError(f"atr window must be >= 1, got {n_atr}")
    opens = df["open"].to_numpy().astype(np.float64)
    closes = df["close"].to_numpy().astype(np.float64)
    scale = atr(df, n_atr).to_numpy().astype(np.float64)

    out = np.zeros(df.height, dtype=np.float64)
    up_bodies: deque[tuple[float, float]] = deque(maxlen=2)
    down_bodies: deque[tuple[float, float]] = deque(maxlen=2)

    for i in range(df.height):
        candidates = list(up_bodies) + list(down_bodies)
        if candidates:
            distances: list[float] = []
            unit = scale[i] if np.isfinite(scale[i]) and scale[i] > 0.0 else 1.0
            for bottom, top in candidates:
                if closes[i] > top:
                    distances.append((closes[i] - top) / unit)
                elif closes[i] < bottom:
                    distances.append((closes[i] - bottom) / unit)
                else:
                    distances.append(0.0)
            out[i] = min(distances, key=abs)

        if closes[i] < opens[i]:
            down_bodies.append((closes[i], opens[i]))
        elif closes[i] > opens[i]:
            up_bodies.append((opens[i], closes[i]))

    return pl.Series("order_block_dist_atr", out)
