"""Session flags and higher timeframe context.

Sessions are UTC buckets: Asia ``00:00-07:59``, London ``08:00-12:59``, New York
``13:00-20:59``. They are mutually exclusive; ``ny_open_30`` is the first half
hour of the New York session and sits inside ``ny``. Hours after 21:00 UTC are
labelled by no session.

Higher timeframe context
------------------------
``htf_context`` only joins a bucket once it **closed**. Each aggregated bar is
moved forward by one bucket width (its availability time) and then joined with
``join_asof(strategy="backward")``, so a row at ``10:07`` sees the 5m bucket
that ended at ``10:00``, never the one still forming. Aggregating a truncated
frame therefore cannot change any joined value.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from scalping.features.indicators import atr, rsi

__all__ = ["HTF_COLUMNS", "htf_context", "session_flags"]

#: Feature suffixes computed on each higher timeframe.
_HTF_SUFFIXES: tuple[str, ...] = ("ret_1", "rsi_14", "atr_norm", "ema_gap_21", "pos_range_20")

_ASIA_HOURS: tuple[int, int] = (0, 8)
_LONDON_HOURS: tuple[int, int] = (8, 13)
_NY_HOURS: tuple[int, int] = (13, 21)

#: Every column ``htf_context`` produces, in order.
HTF_COLUMNS: tuple[str, ...] = tuple(
    f"{timeframe}_{suffix}"
    for timeframe in ("m5", "m15")
    for suffix in _HTF_SUFFIXES
)


def session_flags(ts: pl.Series) -> pl.DataFrame:
    """Boolean ``asia``/``london``/``ny``/``ny_open_30`` flags for UTC ``ts``."""
    hour = ts.dt.hour()
    minute = ts.dt.minute()
    asia = (hour >= _ASIA_HOURS[0]) & (hour < _ASIA_HOURS[1])
    london = (hour >= _LONDON_HOURS[0]) & (hour < _LONDON_HOURS[1])
    ny = (hour >= _NY_HOURS[0]) & (hour < _NY_HOURS[1])
    return pl.DataFrame(
        {
            "asia": asia.fill_null(False),
            "london": london.fill_null(False),
            "ny": ny.fill_null(False),
            "ny_open_30": (ny & (hour == _NY_HOURS[0]) & (minute >= 30)).fill_null(False),
        }
    )


def _htf_features(frame: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Five causal features for one aggregated timeframe."""
    close = pl.col("close")
    width = pl.col("high").rolling_max(20, min_samples=1) - pl.col(
        "low"
    ).rolling_min(20, min_samples=1)
    rsi_14 = rsi(frame, 14, min_samples=2)
    return frame.select(
        pl.col("ts"),
        (close / close.shift(1) - 1.0).alias(f"{prefix}_ret_1"),
        (rsi_14 / 100.0).alias(f"{prefix}_rsi_14"),
        (atr(frame, 14) / close).alias(f"{prefix}_atr_norm"),
        (close / close.ewm_mean(span=21, adjust=False) - 1.0).alias(f"{prefix}_ema_gap_21"),
        pl.when(width > 0.0)
        .then(
            (close - pl.col("low").rolling_min(20, min_samples=1)) / width
        )
        .otherwise(0.5)
        .alias(f"{prefix}_pos_range_20"),
    )


def _bucket_width(frame: pl.DataFrame) -> pl.Expr:
    """Availability offset: one bucket width, taken from the bucket spacing."""
    step = frame["ts"].diff().drop_nulls().min()
    if not isinstance(step, timedelta) or step <= timedelta(0):
        raise ValueError("a higher timeframe frame needs at least two distinct buckets")
    return pl.duration(microseconds=round(step.total_seconds() * 1_000_000))


def htf_context(m1: pl.DataFrame, m5: pl.DataFrame, m15: pl.DataFrame) -> pl.DataFrame:
    """Join the last closed 5m and 15m bar onto every M1 ``ts``.

    Returns one row per M1 row, in input order, with ``ts`` plus the columns in
    :data:`HTF_COLUMNS`. Rows before the first closed bucket carry nulls; the
    builder forward fills them.
    """
    left = m1.select("ts").with_row_index("_row")
    joined = left
    for frame, prefix in ((m5, "m5"), (m15, "m15")):
        if frame.height < 2:
            joined = joined.with_columns(
                [pl.lit(None, dtype=pl.Float64).alias(f"{prefix}_{s}") for s in _HTF_SUFFIXES]
            )
            continue
        ready = (
            _htf_features(frame, prefix)
            .sort("ts")
            .with_columns((pl.col("ts") + _bucket_width(frame)).alias("ts"))
        )
        joined = joined.sort("ts").join_asof(ready, on="ts", strategy="backward")
    return joined.sort("_row").drop("_row")
