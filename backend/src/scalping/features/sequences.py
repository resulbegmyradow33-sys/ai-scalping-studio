"""Optional causal sequence features: log returns and candle streaks.

The default 59-column matrix keeps simple returns. Research that wants
explicit log returns or run lengths can build them here without changing the
stable feature contract that datasets, models and the DSL already pin.

Every column is causal: row ``i`` uses closes at or before ``i``. The first row
has no previous close, so its values are zero rather than null.
"""

from __future__ import annotations

from typing import Final

import polars as pl
from polars.datatypes import DataType, DataTypeClass

from scalping.data.base import conform_bars

__all__ = ["SEQUENCE_COLUMNS", "sequence_features"]

SEQUENCE_COLUMNS: Final[tuple[str, ...]] = (
    "log_ret_1",
    "log_ret_3",
    "log_ret_5",
    "up_streak",
    "down_streak",
)


def sequence_features(df: pl.DataFrame) -> pl.DataFrame:
    """Return ``ts`` plus :data:`SEQUENCE_COLUMNS`, aligned to sorted bars."""
    bars = conform_bars(df).sort("ts")
    if bars.is_empty():
        schema: dict[str, DataType | DataTypeClass] = {"ts": pl.Datetime("us", "UTC")}
        schema.update({name: pl.Float64 for name in SEQUENCE_COLUMNS})
        return pl.DataFrame(schema=schema)

    close = pl.col("close")
    log_close = close.log()
    direction = (
        pl.when(close > close.shift(1))
        .then(1)
        .when(close < close.shift(1))
        .then(-1)
        .otherwise(0)
        .cast(pl.Int64)
        .alias("_direction")
    )
    work = bars.with_columns(direction).with_columns(
        (pl.col("_direction") != pl.col("_direction").shift(1))
        .fill_null(True)
        .cum_sum()
        .alias("_run")
    )
    run = pl.int_range(1, pl.len() + 1).over("_run")
    out = work.select(
        pl.col("ts"),
        (log_close - log_close.shift(1)).fill_null(0.0).alias("log_ret_1"),
        (log_close - log_close.shift(3)).fill_null(0.0).alias("log_ret_3"),
        (log_close - log_close.shift(5)).fill_null(0.0).alias("log_ret_5"),
        pl.when(pl.col("_direction") == 1)
        .then(run.cast(pl.Float64))
        .otherwise(0.0)
        .alias("up_streak"),
        pl.when(pl.col("_direction") == -1)
        .then(run.cast(pl.Float64))
        .otherwise(0.0)
        .alias("down_streak"),
    )
    return out
