"""Causal price indicators shared by the feature builder.

Every function returns one value per input row and is computed from strictly
backward looking windows, so row ``i`` never depends on row ``i + 1``. That is
the property :func:`scalping.features.build.build_features` relies on.

``atr`` is the simple mean of the true range rather than a recursive Wilder
average: it has no warm-up hole, which keeps the "no NaN after ``valid_from``"
guarantee independent of how a series starts. For a constant range series both
definitions agree exactly.
"""

from __future__ import annotations

import polars as pl

__all__ = ["atr", "macd_hist", "rsi", "true_range"]

#: EMA spans used by :func:`macd_hist`.
_MACD_FAST: int = 12
_MACD_SLOW: int = 26
_MACD_SIGNAL: int = 9


def _true_range_expr() -> pl.Expr:
    """``max(high - low, |high - prev close|, |low - prev close|)``.

    ``max_horizontal`` ignores nulls, so the first bar falls back to its own
    range instead of poisoning the series.
    """
    prev_close = pl.col("close").shift(1)
    return pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )


def true_range(df: pl.DataFrame) -> pl.Series:
    """Per-bar true range, defined from the first row."""
    return df.select(_true_range_expr().alias("true_range")).to_series()


def atr(df: pl.DataFrame, n: int = 14) -> pl.Series:
    """Average true range: rolling mean of the true range over ``n`` bars.

    ``min_samples=1`` means the first rows use the bars available so far, so the
    series is finite everywhere while the window fills up.
    """
    if n < 1:
        raise ValueError(f"atr window must be >= 1, got {n}")
    return df.select(
        _true_range_expr().rolling_mean(window_size=n, min_samples=1).alias("atr")
    ).to_series()


def rsi(df: pl.DataFrame, n: int = 14, *, min_samples: int | None = None) -> pl.Series:
    """Wilder's relative strength index in ``[0, 100]``.

    Gains and losses are smoothed with ``alpha = 1 / n`` and ``adjust=False``, so
    the value only depends on the bars already seen. ``min_samples`` defaults to
    ``n`` (a full window before the first value); higher timeframes pass a
    smaller value so the first few aggregated buckets are still usable.

    Flat prices (no gain and no loss) return 50. An unbroken run of gains
    returns 100 and an unbroken run of losses returns 0.
    """
    if n < 1:
        raise ValueError(f"rsi window must be >= 1, got {n}")
    period = n if min_samples is None else min_samples
    delta = pl.col("close").diff()
    gain = delta.clip(lower_bound=0.0)
    loss = (-delta).clip(lower_bound=0.0)
    avg_gain = gain.ewm_mean(alpha=1.0 / n, adjust=False, min_samples=period)
    avg_loss = loss.ewm_mean(alpha=1.0 / n, adjust=False, min_samples=period)
    relative = avg_gain / avg_loss
    out = (
        pl.when((avg_gain == 0.0) & (avg_loss == 0.0))
        .then(50.0)
        .when(avg_loss == 0.0)
        .then(100.0)
        .when(avg_gain == 0.0)
        .then(0.0)
        .otherwise(100.0 - 100.0 / (1.0 + relative))
    )
    return df.select(out.alias("rsi")).to_series()


def macd_hist(df: pl.DataFrame) -> pl.Series:
    """MACD histogram: ``(ema12 - ema26) - ema9(ema12 - ema26)``.

    Exponentials are seeded on the first bar (``adjust=False``), so the value is
    finite from row 0 and depends only on past closes.
    """
    close = pl.col("close")
    fast = close.ewm_mean(span=_MACD_FAST, adjust=False)
    slow = close.ewm_mean(span=_MACD_SLOW, adjust=False)
    macd = fast - slow
    signal = macd.ewm_mean(span=_MACD_SIGNAL, adjust=False)
    return df.select((macd - signal).alias("macd_hist")).to_series()
