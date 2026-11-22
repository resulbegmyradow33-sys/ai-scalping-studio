"""Model inputs for the regime classifier.

The feature set is deliberately small and causal. Clustering 59 trading columns
would mix signal definitions that were designed for a supervised edge into an
unsupervised description; a regime model needs a compact summary of return,
volatility, range and participation instead.

Every column is finite from the first row (warm-up gaps fall back to zero), so
``feature_matrix`` can hand a dense array to scikit-learn without imputation.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from polars.datatypes import DataType, DataTypeClass

from scalping.data.base import conform_bars
from scalping.features.indicators import atr

__all__ = ["REGIME_FEATURES", "build_regime_features", "feature_matrix", "validate_features"]

#: Model input columns, in a fixed order.
REGIME_FEATURES: Final[tuple[str, ...]] = (
    "ret_1",
    "ret_5",
    "ret_15",
    "realized_vol_20",
    "atr_norm",
    "range_atr",
    "efficiency_20",
    "volume_z20",
)

_TREND_WINDOW: Final[int] = 20
_VOL_WINDOW: Final[int] = 20


def _empty() -> pl.DataFrame:
    schema: dict[str, DataType | DataTypeClass] = {"ts": pl.Datetime("us", "UTC")}
    schema.update({name: pl.Float64 for name in REGIME_FEATURES})
    return pl.DataFrame(schema=schema)


def _safe_div(numerator: pl.Expr, denominator: pl.Expr, default: float = 0.0) -> pl.Expr:
    return (
        pl.when(denominator > 0)
        .then(numerator / denominator)
        .otherwise(default)
        .cast(pl.Float64)
    )


def build_regime_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Return ``ts`` plus :data:`REGIME_FEATURES`, one row per sorted bar."""
    bars = conform_bars(frame).sort("ts")
    if bars.is_empty():
        return _empty()

    work = bars.with_columns(atr(bars, 14).alias("_atr"))
    close = pl.col("close")
    log_close = close.log()
    ret_1 = log_close.diff()
    atr_now = pl.col("_atr")
    volume = pl.col("volume").cast(pl.Float64)

    net = (close - close.shift(_TREND_WINDOW)).abs()
    path = (
        close.diff().abs().rolling_sum(window_size=_TREND_WINDOW, min_samples=2)
    )
    volume_mean = volume.rolling_mean(window_size=_VOL_WINDOW, min_samples=2)
    volume_std = volume.rolling_std(window_size=_VOL_WINDOW, min_samples=2)

    out = work.select(
        pl.col("ts"),
        ret_1.alias("ret_1"),
        (log_close - log_close.shift(5)).alias("ret_5"),
        (log_close - log_close.shift(15)).alias("ret_15"),
        ret_1.rolling_std(window_size=_VOL_WINDOW, min_samples=2).alias("realized_vol_20"),
        _safe_div(atr_now, close).alias("atr_norm"),
        _safe_div(pl.col("high") - pl.col("low"), atr_now).alias("range_atr"),
        _safe_div(net, path).alias("efficiency_20"),
        _safe_div(volume - volume_mean, volume_std).alias("volume_z20"),
    )
    return out.with_columns(
        pl.col(name).fill_null(0.0).fill_nan(0.0) for name in REGIME_FEATURES
    )


def validate_features(features: pl.DataFrame) -> pl.DataFrame:
    """Validate the feature frame before it reaches a model.

    Kept separate from the builder so a caller that cached or filtered features
    gets the same guarantee: :data:`REGIME_FEATURES` are present and finite.
    """
    missing = [name for name in REGIME_FEATURES if name not in features.columns]
    if missing:
        raise ValueError(f"missing regime feature columns: {missing}")
    nulls = features.select(
        [pl.col(name).is_null().sum().alias(name) for name in REGIME_FEATURES]
    ).row(0)
    if any(int(count) for count in nulls):
        raise ValueError(f"regime features contain nulls: {dict(zip(REGIME_FEATURES, nulls))}")
    if features.select(
        [pl.col(name).is_nan().sum().alias(name) for name in REGIME_FEATURES]
    ).sum_horizontal().item():
        raise ValueError("regime features contain NaN values")
    return features


def feature_matrix(features: pl.DataFrame) -> np.ndarray:
    """Dense ``float64`` matrix in :data:`REGIME_FEATURES` order, validated first."""
    validate_features(features)
    matrix = features.select(REGIME_FEATURES).to_numpy().astype(np.float64, copy=False)
    if not np.isfinite(matrix).all():  # pragma: no cover - validate_features catches this
        raise ValueError("regime feature matrix is not finite")
    return matrix
