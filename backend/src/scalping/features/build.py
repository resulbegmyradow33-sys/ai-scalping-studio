"""Assemble the causal feature matrix used by the learning agents.

``build_features`` takes M1 bars only. It derives the 5m and 15m context
internally, shifts each aggregated series by its own bucket width so only the
last **closed** bucket joins, and forward fills the resulting nulls. Nothing in
the pipeline looks past the current bar:

* every rolling window is backward looking (``center=True`` appears only in the
  swing detector, whose label is shifted to the confirmation bar),
* no ``shift(-n)`` is used,
* no statistic is taken over the whole frame.

``valid_from`` is the first row at which every window is warm (60 bars), and
``X[valid_from:]`` contains no NaN. The raw ATR(14) is exposed next to ``X`` as
``FeatureFrame.atr`` so the simulator can size stops in price units; it is not
part of the normalised matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import tau

import numpy as np
import polars as pl

from scalping.data.base import conform_bars
from scalping.data.resample import resample
from scalping.features.context import htf_context, session_flags
from scalping.features.indicators import atr, macd_hist, rsi
from scalping.features.liquidity import fvg, liquidity_sweep
from scalping.features.prepare import prepare_bars
from scalping.features.structure import market_structure, order_block_distance
from scalping.types import CostConfig

__all__ = ["FEATURE_COLUMNS", "WARMUP", "FeatureFrame", "build_features"]

#: Bars of warm-up before the matrix is considered usable.
WARMUP: int = 60

#: Short windows used by the normalised features.
_Z_WINDOW: int = 20
_RANGE_WINDOW: int = 20

#: Window of the realised volatility estimate and of its tercile history.
_VOL_WINDOW: int = 30
_VOL_TIER_WINDOW: int = 120

#: Price tolerance for an equal high/low, in price units.
_EQUAL_EXTREME_TOLERANCE: float = 0.05

#: The FX day rolls at 17:00 UTC, matching ``RiskEngine.session_anchor``.
_SESSION_ROLL_HOUR: int = 17

# Units and signs
# ---------------
# ``*_atr``: value divided by ATR(14) of the same timeframe, dimensionless.
# ``*_z20``: (x - 20-bar mean) / 20-bar std; 0 when the window has no variance.
# ``ret_1/5/15/60``: simple close-to-close returns. ``ret_3``: log return.
# ``dist_high_20``/``dist_low_20``: unsigned room to the 20-bar extreme, ATR
#   units, so both are non-negative and large when price sits mid-range.
# ``dist_structure_high``/``dist_structure_low``: signed close-to-level
#   distance, positive above the level (the swing high/low has been broken).
# ``fvg_dist``: ATR distance to the nearest unfilled gap, positive above an up
#   gap, negative below a down gap, 0 when no gap is open.
# ``order_block_dist_atr``: signed ATR distance to the nearest tracked candle
#   body, positive above it, negative below it, 0 when price is inside one.
# ``eq_high_20``/``eq_low_20``: 1.0 when the high/low is within 0.05 price
#   units of the prior 20-bar extreme (current bar excluded).
# ``vwap_dist_atr``: (close - session VWAP) / ATR(14), 17:00 UTC session roll.
# ``day_position``: close position inside the current 17:00-to-17:00 session
#   range, in [0, 1], 0.5 on a zero range.
# ``realized_vol_30``: std of 1-bar log returns over 30 bars.
# ``vol_regime``: -1/0/1 from ``realized_vol_30`` against its own prior
#   120-bar terciles (thresholds never include the current bar).
# ``trend``/``bos``/``choch``: -1/0/1 integer states. Booleans are 0.0/1.0.
# ``rsi_14`` and the M5/M15 RSI columns are divided by 100.
# HTF columns come from the last closed 5m/15m bucket; ``htf_bos`` is the M5
#   BOS state and ``htf_fvg_dist`` the M15 FVG distance.

#: Column order of :attr:`FeatureFrame.X`. One entry per feature.
FEATURE_COLUMNS: tuple[str, ...] = (
    # --- price action and momentum (29) ---
    "ret_1",
    "ret_5",
    "ret_15",
    "ret_60",
    "ret_3",
    "body_atr",
    "upper_wick_atr",
    "lower_wick_atr",
    "range_atr",
    "realized_vol_30",
    "vol_regime",
    "ret_z20",
    "rsi_14",
    "macd_hist_atr",
    "ema_gap_9_21",
    "close_ema_60",
    "atr_norm",
    "atr_ratio_14_60",
    "volume_z20",
    "volume_ratio_20",
    "spread_norm",
    "spread_z20",
    "dist_high_20",
    "dist_low_20",
    "pos_range_20",
    "eq_high_20",
    "eq_low_20",
    "tod_sin",
    "tod_cos",
    # --- structure and liquidity (12) ---
    "trend",
    "bos",
    "choch",
    "swing_high",
    "swing_low",
    "dist_structure_high",
    "dist_structure_low",
    "order_block_dist_atr",
    "liquidity_sweep",
    "fvg_up",
    "fvg_down",
    "fvg_dist",
    # --- session context (6) ---
    "asia",
    "london",
    "ny",
    "ny_open_30",
    "vwap_dist_atr",
    "day_position",
    # --- higher timeframe context (12) ---
    "m5_ret_1",
    "m5_rsi_14",
    "m5_atr_norm",
    "m5_ema_gap_21",
    "m5_pos_range_20",
    "m15_ret_1",
    "m15_rsi_14",
    "m15_atr_norm",
    "m15_ema_gap_21",
    "m15_pos_range_20",
    "htf_bos",
    "htf_fvg_dist",
)


@dataclass
class FeatureFrame:
    """Feature matrix plus the raw market data each row was built from.

    ``X`` is ``(T, F)`` ``float32``; ``ohlc`` is ``(T, 4)`` open/high/low/close
    and ``spread`` is the bid/ask spread, both in price units. ``atr`` is the
    raw causal ATR(14) of the M1 bars, also in price units, kept outside ``X``
    because the normalised matrix cannot express the stop distance the
    simulator needs for slippage, stops and targets. ``valid_from`` indexes the
    first fully warm row.
    """

    X: np.ndarray
    ts: np.ndarray
    ohlc: np.ndarray
    spread: np.ndarray
    atr: np.ndarray
    valid_from: int
    columns: list[str] = field(default_factory=list)


def _guarded(condition: pl.Expr, value: pl.Expr, fallback: float | pl.Expr) -> pl.Expr:
    """``value`` where ``condition`` holds, else ``fallback`` (null-safe)."""
    return pl.when(condition.fill_null(False)).then(value).otherwise(fallback)


def _scaled(expr: pl.Expr, scale: pl.Expr) -> pl.Expr:
    """Divide by ``scale`` where it is positive, otherwise return ``0.0``."""
    return _guarded(scale > 0.0, expr / scale, 0.0)


def _with_helpers(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach the intermediate series every feature expression reads."""
    close = pl.col("close")
    volume = pl.col("volume").cast(pl.Float64)
    spread = pl.col("spread")
    high20 = pl.col("high").rolling_max(_RANGE_WINDOW, min_samples=1)
    low20 = pl.col("low").rolling_min(_RANGE_WINDOW, min_samples=1)
    ret1 = close / close.shift(1) - 1.0
    realized_vol = (
        (close / close.shift(1)).log().rolling_std(_VOL_WINDOW, min_samples=1).fill_null(0.0)
    )
    work = frame.with_columns(
        atr(frame, 14).alias("_atr14"),
        atr(frame, 60).alias("_atr60"),
        rsi(frame, 14).alias("_rsi14"),
        macd_hist(frame).alias("_macd_hist"),
        close.ewm_mean(span=9, adjust=False).alias("_ema9"),
        close.ewm_mean(span=21, adjust=False).alias("_ema21"),
        close.ewm_mean(span=60, adjust=False).alias("_ema60"),
        high20.alias("_high20"),
        low20.alias("_low20"),
        (high20 - low20).alias("_width20"),
        ret1.alias("_ret1"),
        ret1.rolling_mean(_Z_WINDOW, min_samples=1).alias("_ret1_mean"),
        ret1.rolling_std(_Z_WINDOW, min_samples=1).alias("_ret1_std"),
        volume.rolling_mean(_Z_WINDOW, min_samples=1).alias("_vol_mean"),
        volume.rolling_std(_Z_WINDOW, min_samples=1).alias("_vol_std"),
        spread.rolling_mean(_Z_WINDOW, min_samples=1).alias("_spread_mean"),
        spread.rolling_std(_Z_WINDOW, min_samples=1).alias("_spread_std"),
        (pl.col("ts").dt.hour() * 60 + pl.col("ts").dt.minute())
        .cast(pl.Float64)
        .alias("_minute_of_day"),
        liquidity_sweep(frame, 20).alias("_liquidity_sweep"),
        order_block_distance(frame, 14).alias("_order_block"),
        realized_vol.alias("_realized_vol"),
    )
    work = work.with_columns(_vol_regime_expr().alias("_vol_regime"))
    work = work.with_columns(_session_anchor_expr())
    return work.with_columns(
        _vwap_distance_expr().alias("_vwap_dist"),
        _day_position_expr().alias("_day_position"),
    )


def _vol_regime_expr() -> pl.Expr:
    """-1/0/1 tercile bucket of realised vol against its prior 120 bars.

    Both thresholds are shifted by one bar, so the current bar's realised
    volatility never contributes to the terciles it is compared against.
    """
    realized = pl.col("_realized_vol")
    lower = realized.rolling_quantile(
        0.33, interpolation="linear", window_size=_VOL_TIER_WINDOW, min_samples=1
    ).shift(1)
    upper = realized.rolling_quantile(
        0.67, interpolation="linear", window_size=_VOL_TIER_WINDOW, min_samples=1
    ).shift(1)
    return pl.when(realized < lower).then(-1.0).when(realized > upper).then(1.0).otherwise(0.0)


def _session_anchor_expr() -> pl.Expr:
    """The 17:00 UTC FX day a row belongs to.

    Mirrors ``RiskEngine.session_anchor``: at or after 17:00 the row anchors to
    its own date, earlier to the previous date. The engine itself is not
    imported, so the feature layer stays free of execution dependencies.
    """
    day = pl.col("ts").dt.date()
    return (
        pl.when(pl.col("ts").dt.hour() >= _SESSION_ROLL_HOUR)
        .then(day)
        .otherwise(day - pl.duration(days=1))
        .alias("_session")
    )


def _vwap_distance_expr() -> pl.Expr:
    """(close - session VWAP) / ATR(14), with the VWAP reset per 17:00 session.

    The typical price (``(high + low + close) / 3``) is volume weighted and
    accumulated over rows already printed in the current session.
    """
    volume = pl.col("volume").cast(pl.Float64)
    typical = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    notional = (typical * volume).cum_sum().over("_session")
    weight = volume.cum_sum().over("_session")
    vwap = _guarded(weight > 0.0, notional / weight, pl.col("close"))
    return _scaled(pl.col("close") - vwap, pl.col("_atr14"))


def _day_position_expr() -> pl.Expr:
    """Close position inside the running 17:00-to-17:00 session range."""
    day_high = pl.col("high").cum_max().over("_session")
    day_low = pl.col("low").cum_min().over("_session")
    width = day_high - day_low
    return _guarded(width > 0.0, (pl.col("close") - day_low) / width, 0.5)


def _feature_exprs() -> list[pl.Expr]:
    """The 59 feature expressions, in :data:`FEATURE_COLUMNS` order."""
    close = pl.col("close")
    high = pl.col("high")
    low = pl.col("low")
    open_ = pl.col("open")
    atr14 = pl.col("_atr14")

    ret1 = pl.col("_ret1")
    body = close - open_
    minutes = pl.col("_minute_of_day")
    angle = minutes * (tau / 1440.0)

    return [
        ret1,
        close / close.shift(5) - 1.0,
        close / close.shift(15) - 1.0,
        close / close.shift(60) - 1.0,
        (close / close.shift(3)).log(),  # log return, unlike the simple ret_* above
        _scaled(body, atr14),
        _scaled(high - pl.max_horizontal(open_, close), atr14),
        _scaled(pl.min_horizontal(open_, close) - low, atr14),
        _scaled(high - low, atr14),
        pl.col("_realized_vol"),
        pl.col("_vol_regime"),
        _scaled(ret1 - pl.col("_ret1_mean"), pl.col("_ret1_std")),
        pl.col("_rsi14") / 100.0,
        _scaled(pl.col("_macd_hist"), atr14),
        pl.col("_ema9") / pl.col("_ema21") - 1.0,
        close / pl.col("_ema60") - 1.0,
        _scaled(atr14, close),
        _guarded(pl.col("_atr60") > 0.0, atr14 / pl.col("_atr60"), 1.0),
        _scaled(
            pl.col("volume").cast(pl.Float64) - pl.col("_vol_mean"), pl.col("_vol_std")
        ),
        _scaled(pl.col("volume").cast(pl.Float64), pl.col("_vol_mean")),
        _scaled(pl.col("spread"), atr14),
        _scaled(pl.col("spread") - pl.col("_spread_mean"), pl.col("_spread_std")),
        _scaled(pl.col("_high20") - close, atr14),
        _scaled(close - pl.col("_low20"), atr14),
        _guarded(
            pl.col("_width20") > 0.0,
            (close - pl.col("_low20")) / pl.col("_width20"),
            0.5,
        ),
        _equal_extreme(high, pl.col("high").rolling_max(_RANGE_WINDOW, min_samples=1)),
        _equal_extreme(low, pl.col("low").rolling_min(_RANGE_WINDOW, min_samples=1)),
        angle.sin(),
        angle.cos(),
        pl.col("_trend").cast(pl.Float64),
        pl.col("_bos").cast(pl.Float64),
        pl.col("_choch").cast(pl.Float64),
        pl.col("_swing_high").cast(pl.Float64),
        pl.col("_swing_low").cast(pl.Float64),
        _scaled(close - pl.col("_structure_high"), atr14),
        _scaled(close - pl.col("_structure_low"), atr14),
        pl.col("_order_block"),
        pl.col("_liquidity_sweep").cast(pl.Float64),
        pl.col("_fvg_up").cast(pl.Float64),
        pl.col("_fvg_down").cast(pl.Float64),
        pl.col("_fvg_dist"),
        pl.col("_asia").cast(pl.Float64),
        pl.col("_london").cast(pl.Float64),
        pl.col("_ny").cast(pl.Float64),
        pl.col("_ny_open_30").cast(pl.Float64),
        pl.col("_vwap_dist"),
        pl.col("_day_position"),
        pl.col("m5_ret_1"),
        pl.col("m5_rsi_14"),
        pl.col("m5_atr_norm"),
        pl.col("m5_ema_gap_21"),
        pl.col("m5_pos_range_20"),
        pl.col("m15_ret_1"),
        pl.col("m15_rsi_14"),
        pl.col("m15_atr_norm"),
        pl.col("m15_ema_gap_21"),
        pl.col("m15_pos_range_20"),
        pl.col("m5_bos").cast(pl.Float64),
        pl.col("m15_fvg_dist"),
    ]


def _equal_extreme(price: pl.Expr, rolling: pl.Expr) -> pl.Expr:
    """1.0 when ``price`` is within tolerance of the prior-window extreme."""
    prior = rolling.shift(1)
    return pl.when((price - prior).abs() <= _EQUAL_EXTREME_TOLERANCE).then(1.0).otherwise(0.0)


def _empty_frame() -> FeatureFrame:
    return FeatureFrame(
        X=np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float32),
        ts=np.zeros(0, dtype="datetime64[us]"),
        ohlc=np.zeros((0, 4), dtype=np.float64),
        spread=np.zeros(0, dtype=np.float64),
        atr=np.zeros(0, dtype=np.float64),
        valid_from=0,
        columns=list(FEATURE_COLUMNS),
    )


def build_features(df: pl.DataFrame, costs: CostConfig | None = None) -> FeatureFrame:
    """Build the causal feature frame for M1 ``df``.

    ``df`` is normalised with :func:`~scalping.data.base.conform_bars` and sorted
    by ``ts``. Rows are returned in that sorted order. A source without a quoted
    spread (the OHLC-only broker CSV) is filled first by
    :func:`~scalping.features.prepare.prepare_bars` with ``costs`` (default
    :class:`~scalping.types.CostConfig`), so the matrix stays finite.

    ``X[valid_from:]`` is guaranteed finite. A non-finite value raises
    :class:`ValueError` naming the offending columns, so a data problem fails at
    ingestion instead of deep inside an episode reset.
    """
    frame = conform_bars(df).sort("ts")
    if frame.is_empty():
        return _empty_frame()
    frame = prepare_bars(
        frame, CostConfig() if costs is None else costs, atr(frame, 14).to_numpy()
    )

    structure = market_structure(frame, 2).rename(
        {
            "trend": "_trend",
            "bos": "_bos",
            "choch": "_choch",
            "swing_high": "_swing_high",
            "swing_low": "_swing_low",
            "structure_high": "_structure_high",
            "structure_low": "_structure_low",
        }
    )
    gaps = fvg(frame).rename(
        {"fvg_up": "_fvg_up", "fvg_down": "_fvg_down", "fvg_dist": "_fvg_dist"}
    )
    sessions = session_flags(frame["ts"]).rename(
        {"asia": "_asia", "london": "_london", "ny": "_ny", "ny_open_30": "_ny_open_30"}
    )
    context = htf_context(frame, resample(frame, "5m"), resample(frame, "15m"))
    context = context.drop("ts").select(pl.all().forward_fill().fill_null(0.0))

    work = pl.concat(
        [_with_helpers(frame), structure, gaps, sessions, context], how="horizontal"
    )
    features = work.select(
        [expr.alias(name) for expr, name in zip(_feature_exprs(), FEATURE_COLUMNS)]
    )
    if features.columns != list(FEATURE_COLUMNS):  # pragma: no cover - guard only
        raise RuntimeError("feature column order drifted from FEATURE_COLUMNS")

    X = features.cast(pl.Float32).to_numpy().copy(order="C")
    valid_from = min(WARMUP, frame.height)
    tail = X[valid_from:]
    if tail.size and not np.isfinite(tail).all():
        names = [
            name
            for index, name in enumerate(FEATURE_COLUMNS)
            if not np.isfinite(tail[:, index]).all()
        ]
        raise ValueError(f"non-finite feature columns: {names}")

    return FeatureFrame(
        X=X,
        ts=frame["ts"].to_numpy(),
        ohlc=frame.select("open", "high", "low", "close").to_numpy(),
        spread=frame["spread"].to_numpy(),
        atr=work["_atr14"].to_numpy().astype(np.float64, copy=True),
        valid_from=valid_from,
        columns=list(FEATURE_COLUMNS),
    )
