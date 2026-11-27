"""Aggregate signed tick volume into flow and cumulative volume delta.

Flow is the microstructure side-channel used by the research rule:

```
flow = (buy_volume - sell_volume) / (buy_volume + sell_volume)
cvd  = cumulative sum of (buy_volume - sell_volume)
```

The result is a separate table, not a bar column: adding it to the stable bar
schema would make every dataset depend on tick data that most sources lack.
Input is any tick frame with ``ts``, ``bid_volume`` and ``ask_volume``.
"""

from __future__ import annotations

from typing import Final

import polars as pl
from polars.datatypes import DataType, DataTypeClass

__all__ = ["FLOW_COLUMNS", "aggregate_flow"]

FLOW_COLUMNS: Final[tuple[str, ...]] = (
    "ts",
    "buy_vol",
    "sell_vol",
    "flow",
    "cvd",
    "ticks",
)

_REQUIRED: Final[tuple[str, ...]] = ("ts", "bid_volume", "ask_volume")

_FLOW_DTYPES: dict[str, DataType | DataTypeClass] = {
    "ts": pl.Datetime("us", "UTC"),
    "buy_vol": pl.Float64,
    "sell_vol": pl.Float64,
    "flow": pl.Float64,
    "cvd": pl.Float64,
    "ticks": pl.Int64,
}

_FLOW_SCHEMA: Final[pl.Schema] = pl.Schema(_FLOW_DTYPES)


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=_FLOW_DTYPES)


def aggregate_flow(ticks: pl.DataFrame, *, rule: str = "1m") -> pl.DataFrame:
    """Aggregate ticks into ``FLOW_COLUMNS`` buckets (left-closed, left-labelled).

    ``bid_volume`` is sell-side aggression and ``ask_volume`` buy-side. A bucket
    with zero total volume gets ``flow = 0`` instead of a division artifact.
    ``cvd`` accumulates across the supplied frame from zero.
    """
    missing = [name for name in _REQUIRED if name not in ticks.columns]
    if missing:
        raise ValueError(f"missing tick columns: {missing}")
    if ticks.is_empty():
        return _empty()

    frame = ticks
    dtype = frame.schema["ts"]
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
        frame = frame.with_columns(pl.col("ts").dt.replace_time_zone("UTC"))
    frame = frame.select(
        pl.col("ts"),
        pl.col("bid_volume").cast(pl.Float64),
        pl.col("ask_volume").cast(pl.Float64),
    ).sort("ts")

    out = (
        frame.group_by_dynamic("ts", every=rule, closed="left", label="left")
        .agg(
            pl.col("ask_volume").sum().alias("buy_vol"),
            pl.col("bid_volume").sum().alias("sell_vol"),
            pl.len().alias("ticks"),
        )
        .with_columns(
            pl.when((pl.col("buy_vol") + pl.col("sell_vol")) > 0)
            .then(
                (pl.col("buy_vol") - pl.col("sell_vol"))
                / (pl.col("buy_vol") + pl.col("sell_vol"))
            )
            .otherwise(0.0)
            .alias("flow"),
            (pl.col("buy_vol") - pl.col("sell_vol")).cum_sum().alias("cvd"),
        )
    )
    return out.select(FLOW_COLUMNS).cast(_FLOW_SCHEMA).sort("ts")
