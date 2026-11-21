"""Spread model for bar sources that quote only one side.

A broker CSV without bid/ask columns carries OHLC only: ``spread``,
``ask_close`` and ``bid_close`` are null, and a feature matrix built from a null
spread is NaN, which the simulator rejects at reset. :func:`prepare_bars` fills
those bars from :class:`~scalping.types.CostConfig`, so an OHLC-only file still
produces a finite, strictly positive spread.

The model is::

    spread = spread_floor * (1 + spread_vol_mult * atr14 / mean(atr14))

clamped to ``[spread_floor, 5 * spread_floor]``. The ATR mean is taken over the
whole frame, so this is a calibration, not a causal feature; it only runs on the
rows that have no spread of their own.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from scalping.types import CostConfig

__all__ = ["prepare_bars"]

#: Upper clamp of the model, as a multiple of ``CostConfig.spread_floor``.
_MAX_FLOOR_MULTIPLE: float = 5.0


def prepare_bars(df: pl.DataFrame, costs: CostConfig, atr: np.ndarray) -> pl.DataFrame:
    """Fill a null or non-positive spread with the cost-model spread.

    ``atr`` is the raw ATR(14) of ``df``, aligned row by row. Rows that already
    carry a positive spread are left untouched, including their quotes. A
    filled row gets ``ask_close = close + spread`` and ``bid_close = close``.
    """
    if df.is_empty():
        return df
    values = np.asarray(atr, dtype=np.float64).reshape(-1)
    if values.size != df.height:
        raise ValueError(f"atr has {values.size} values for {df.height} bars")
    floor = float(costs.spread_floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError(f"spread_floor must be finite and > 0, got {costs.spread_floor!r}")

    missing = pl.col("spread").is_null() | (pl.col("spread") <= 0.0)
    if not df.filter(missing.fill_null(True)).height:
        return df

    finite = values[np.isfinite(values)]
    reference = float(finite.mean()) if finite.size else 0.0
    if reference > 0.0:
        ratio = np.where(np.isfinite(values), values, reference) / reference
    else:
        ratio = np.ones(values.size, dtype=np.float64)
    model = floor * (1.0 + float(costs.spread_vol_mult) * ratio)
    model = np.clip(model, floor, _MAX_FLOOR_MULTIPLE * floor)

    work = df.with_columns(
        pl.Series("_model_spread", model, dtype=pl.Float64),
        missing.fill_null(True).alias("_fill_spread"),
    )
    work = work.with_columns(
        pl.when(pl.col("_fill_spread"))
        .then(pl.col("_model_spread"))
        .otherwise(pl.col("spread"))
        .alias("spread")
    )
    return work.with_columns(
        pl.when(pl.col("_fill_spread"))
        .then(pl.col("close") + pl.col("spread"))
        .otherwise(pl.col("ask_close"))
        .alias("ask_close"),
        pl.when(pl.col("_fill_spread"))
        .then(pl.col("close"))
        .otherwise(pl.col("bid_close"))
        .alias("bid_close"),
    ).drop("_model_spread", "_fill_spread")
