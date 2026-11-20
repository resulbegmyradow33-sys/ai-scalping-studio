"""Downsample M1 bars to a coarser timeframe.

OHLC aggregates the bucket; ``bid_close``/``ask_close`` come from the last bar
so the quote matches ``close``; ``spread_mean`` is the bucket mean.
"""

from __future__ import annotations

from typing import Final

import polars as pl
from polars.datatypes import DataType, DataTypeClass

from scalping.data.base import conform_bars

__all__ = ["RESAMPLE_COLUMNS", "RESAMPLE_SCHEMA", "resample"]

RESAMPLE_COLUMNS: Final[tuple[str, ...]] = (
    "ts",
    "open",
    "high",
    "low",
    "close",
    "bid_close",
    "ask_close",
    "spread_mean",
    "volume",
    "provider",
)

_RESAMPLE_DTYPES: dict[str, DataType | DataTypeClass] = {
    "ts": pl.Datetime("us", "UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "bid_close": pl.Float64,
    "ask_close": pl.Float64,
    "spread_mean": pl.Float64,
    "volume": pl.Int64,
    "provider": pl.String,
}

RESAMPLE_SCHEMA: Final[pl.Schema] = pl.Schema(_RESAMPLE_DTYPES)


def resample(df: pl.DataFrame, rule: str) -> pl.DataFrame:
    """Aggregate M1 ``df`` into ``rule``-wide buckets (``"5m"``, ``"15m"``).

    Buckets are left-closed and left-labelled, so a bar at ``10:00`` belongs to
    the ``10:00`` bucket and the ``10:05`` bar to the next one. Input need not be
    sorted; the frame is sorted first so ``first``/``last`` mean what they say.
    """
    frame = conform_bars(df)
    if frame.is_empty():
        return pl.DataFrame(schema=RESAMPLE_SCHEMA).select(RESAMPLE_COLUMNS)
    out = (
        frame.sort("ts")
        .group_by_dynamic("ts", every=rule, closed="left", label="left")
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("bid_close").last(),
            pl.col("ask_close").last(),
            pl.col("spread").mean().alias("spread_mean"),
            pl.col("volume").sum(),
            pl.col("provider").last(),
        )
    )
    return out.select(RESAMPLE_COLUMNS).sort("ts")
