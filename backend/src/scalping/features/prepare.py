"""Spread model for bar sources that quote only one side.

A broker CSV without bid/ask columns carries OHLC only: ``spread``,
``ask_close`` and ``bid_close`` are null, and a feature matrix built from a null
spread is NaN, which the simulator rejects at reset. :func:`prepare_bars` fills
those bars from :class:`~scalping.types.CostConfig`, so an OHLC-only file still
produces a finite, strictly positive spread.

The model is::

    spread = spread_floor * (1 + spread_vol_mult * atr14 / atr_ref)

where ``atr_ref`` is the causal rolling mean of ``atr14`` over the current and
the previous :data:`ATR_REF_WINDOW - 1` bars. The result is clamped to
``[spread_floor, 5 * spread_floor]``, and a zero or non-finite reference falls
back to ``spread_floor``. Bar ``t`` therefore reads only bars at or before ``t``,
matching the feature layer's causality rule.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from scalping.types import CostConfig

__all__ = ["ATR_REF_WINDOW", "prepare_bars"]

#: Upper clamp of the model, as a multiple of ``CostConfig.spread_floor``.
_MAX_FLOOR_MULTIPLE: float = 5.0

#: Bars in the causal reference mean (the current bar plus the prior 239).
ATR_REF_WINDOW: int = 240


def prepare_bars(df: pl.DataFrame, costs: CostConfig, atr: np.ndarray) -> pl.DataFrame:
    """Fill a null or non-positive spread with the cost-model spread.

    ``atr`` is the raw ATR(14) of ``df``, aligned row by row. Rows that already
    carry a positive spread are left untouched, including their quotes. A
    filled row gets ``ask_close = close + spread`` and ``bid_close = close``.

    The volatility reference is a backward-looking rolling mean, so the spread
    at bar ``t`` never depends on a bar after ``t``.
    """
    if df.is_empty():
        return df
    values = np.asarray(atr, dtype=np.float64).reshape(-1)
    values = np.where(np.isfinite(values), values, np.nan)
    if values.size != df.height:
        raise ValueError(f"atr has {values.size} values for {df.height} bars")
    floor = float(costs.spread_floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError(f"spread_floor must be finite and > 0, got {costs.spread_floor!r}")

    missing = pl.col("spread").is_null() | (pl.col("spread") <= 0.0)
    if not df.filter(missing.fill_null(True)).height:
        return df

    reference = (
        pl.Series("_atr_ref", values, dtype=pl.Float64)
        .rolling_mean(window_size=ATR_REF_WINDOW, min_samples=1)
        .to_numpy()
    )
    model = np.full(values.size, floor, dtype=np.float64)
    usable = np.isfinite(values) & np.isfinite(reference) & (reference > 0.0)
    model[usable] = floor * (
        1.0 + float(costs.spread_vol_mult) * values[usable] / reference[usable]
    )
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
