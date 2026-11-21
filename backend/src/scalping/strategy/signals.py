"""Five causal strategy signal families.

Each family turns the feature layer's causal indicators into a discrete
direction (``-1``/``0``/``+1``) plus a confidence in ``[0, 1]``. The order of
:data:`ALL_STRATEGIES` is frozen because the bandit's arm index is a position in
that tuple; :func:`active_strategy` is the bridge between the two.

Rules
-----
``momentum``
    ``close > ema20`` and ``close > close[-5]``, with expansion evidence:
    ATR(14)/ATR(60) above ``1.1`` **or** the 5-bar displacement above ``1.1``
    ATR. The displacement branch exists because a trend with a constant bar
    range has an ATR ratio of exactly ``1.0``; without it a clean, steady trend
    could never fire. Mirror for ``-1``.
``reversal``
    RSI(14) above ``75`` on a bearish candle -> ``-1``; below ``25`` on a
    bullish candle -> ``+1``.
``sweep``
    A sweep is a wick through the prior 20-bar extreme that closes back inside
    (:func:`~scalping.features.liquidity.liquidity_sweep`). The signal fires on
    the **next** bar, once that bar holds above the swept low (or below the
    swept high) without extending the wick: a deep wick alone is not a signal,
    the recovery is. Mirror for ``-1``.
``fvg``
    Price trading back into the nearest unfilled fair value gap
    (:func:`~scalping.features.liquidity.fvg`), at most ``1.5`` ATR past the
    near edge. Inside an up gap -> ``+1``, inside a down gap -> ``-1``.
``trend``
    A break of structure aligned with the running trend
    (:func:`~scalping.features.structure.market_structure`) whose close sits
    within ``0.5`` ATR of the bar extreme (a shallow pullback).

Causality: every expression is a backward window, a lag or an exponential
seeded on the left edge. There is no ``shift(-n)``, no ``center=True`` and no
statistic over the whole frame, so row ``i`` never depends on a bar after it.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

import polars as pl

from scalping.features.indicators import atr, rsi
from scalping.features.liquidity import fvg, liquidity_sweep
from scalping.features.structure import market_structure

__all__ = [
    "ALL_STRATEGIES",
    "SIGNAL_COLUMNS",
    "STRENGTH_COLUMNS",
    "Strategy",
    "active_strategy",
    "strategy_signals",
]


class Strategy(StrEnum):
    """The five signal families. The value is the column-name suffix."""

    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    SWEEP = "sweep"
    FVG = "fvg"
    TREND = "trend"


#: Frozen order. The bandit's arm index is a position in this tuple, so a
#: reordering would silently re-label every stored arm.
ALL_STRATEGIES: tuple[Strategy, ...] = (
    Strategy.MOMENTUM,
    Strategy.REVERSAL,
    Strategy.SWEEP,
    Strategy.FVG,
    Strategy.TREND,
)

SIGNAL_COLUMNS: tuple[str, ...] = tuple(f"sig_{s.value}" for s in ALL_STRATEGIES)
STRENGTH_COLUMNS: tuple[str, ...] = tuple(f"strength_{s.value}" for s in ALL_STRATEGIES)

#: Columns :func:`strategy_signals` needs from the bar frame.
_REQUIRED_COLUMNS: tuple[str, ...] = ("ts", "open", "high", "low", "close")

_MOMENTUM_EMA: int = 20
_MOMENTUM_LOOKBACK: int = 5
#: Expansion threshold shared by the ATR ratio and the ATR displacement.
_MOMENTUM_EXPANSION: float = 1.1
#: Expansion that maps to full strength, in multiples of the threshold unit.
_MOMENTUM_STRENGTH_SCALE: float = 1.5

_REVERSAL_RSI_N: int = 14
_REVERSAL_OVERSOLD: float = 25.0
_REVERSAL_OVERBOUGHT: float = 75.0

_SWEEP_LOOKBACK: int = 20

_FVG_ATR_LIMIT: float = 1.5

_TREND_PULLBACK_ATR: float = 0.5


def _over_atr(expr: pl.Expr) -> pl.Expr:
    """Divide by ATR(14), returning ``0.0`` on a zero range instead of inf."""
    scale = pl.col("_atr14")
    return pl.when(scale > 0.0).then(expr / scale).otherwise(0.0)


def _pair(long: pl.Expr, short: pl.Expr, up: pl.Expr, down: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """Signal from the two sides; strength follows the side that fired."""
    long = long.fill_null(False)
    short = short.fill_null(False)
    sig = pl.when(long).then(1).when(short).then(-1).otherwise(0)
    strength = (
        pl.when(long)
        .then(up)
        .when(short)
        .then(down)
        .otherwise(0.0)
        .clip(0.0, 1.0)
    )
    return sig.cast(pl.Int8).alias("_sig"), strength.cast(pl.Float32).alias("_strength")


def _momentum() -> tuple[pl.Expr, pl.Expr]:
    """Trend continuation with volatility or displacement expansion."""
    close = pl.col("close")
    ema = close.ewm_mean(span=_MOMENTUM_EMA, adjust=False)
    atr_ratio = (
        pl.when(pl.col("_atr60") > 0.0)
        .then(pl.col("_atr14") / pl.col("_atr60"))
        .otherwise(1.0)
    )
    displacement = _over_atr((close - close.shift(_MOMENTUM_LOOKBACK)).abs())
    expansion = pl.max_horizontal(atr_ratio, displacement).fill_null(0.0)
    gate = expansion > _MOMENTUM_EXPANSION
    up = (close > ema) & (close > close.shift(_MOMENTUM_LOOKBACK)) & gate
    down = (close < ema) & (close < close.shift(_MOMENTUM_LOOKBACK)) & gate
    strength = ((expansion - 1.0) / _MOMENTUM_STRENGTH_SCALE).clip(0.0, 1.0)
    return _pair(up, down, strength, strength)


def _reversal() -> tuple[pl.Expr, pl.Expr]:
    """RSI exhaustion confirmed by a candle in the opposite direction."""
    close = pl.col("close")
    open_ = pl.col("open")
    rsi14 = pl.col("_rsi14")
    up = (rsi14 < _REVERSAL_OVERSOLD) & (close > open_)
    down = (rsi14 > _REVERSAL_OVERBOUGHT) & (close < open_)
    up_strength = (_REVERSAL_OVERSOLD - rsi14) / _REVERSAL_OVERSOLD
    down_strength = (rsi14 - _REVERSAL_OVERBOUGHT) / (100.0 - _REVERSAL_OVERBOUGHT)
    return _pair(up, down, up_strength, down_strength)


def _sweep() -> tuple[pl.Expr, pl.Expr]:
    """Wick through the prior 20-bar extreme, confirmed by the next bar."""
    low = pl.col("low")
    high = pl.col("high")
    close = pl.col("close")
    # The sweep event at t-1 referenced these levels (rolling min/max up to t-2).
    level_low = low.rolling_min(_SWEEP_LOOKBACK, min_samples=_SWEEP_LOOKBACK).shift(2)
    level_high = high.rolling_max(_SWEEP_LOOKBACK, min_samples=_SWEEP_LOOKBACK).shift(2)
    swept_at_prev = pl.col("_sweep_event").shift(1).fill_null(False)
    swept_low = swept_at_prev & (low.shift(1) < level_low)
    swept_high = swept_at_prev & (high.shift(1) > level_high)
    up = swept_low & (close > level_low) & (low > low.shift(1))
    down = swept_high & (close < level_high) & (high < high.shift(1))
    up_strength = _over_atr(close - level_low)
    down_strength = _over_atr(level_high - close)
    return _pair(up, down, up_strength, down_strength)


def _fvg() -> tuple[pl.Expr, pl.Expr]:
    """Re-entry into the nearest unfilled fair value gap."""
    dist = pl.col("_fvg_dist")
    prev = dist.shift(1)
    # Negative distance: price is inside an up gap. Positive: inside a down gap.
    # Zero is the module's "no open gap" sentinel and never fires. Requiring the
    # previous bar to sit outside the gap makes this a re-entry event instead of
    # a level that stays lit for as long as price trades inside the zone.
    up = (dist < 0.0) & (dist >= -_FVG_ATR_LIMIT) & (prev >= 0.0)
    down = (dist > 0.0) & (dist <= _FVG_ATR_LIMIT) & (prev <= 0.0)
    strength = (1.0 - dist.abs() / _FVG_ATR_LIMIT).clip(0.0, 1.0)
    return _pair(up, down, strength, strength)


def _trend() -> tuple[pl.Expr, pl.Expr]:
    """BOS in the trend direction with a shallow retracement on the bar."""
    close = pl.col("close")
    high = pl.col("high")
    low = pl.col("low")
    trend = pl.col("_trend")
    bos = pl.col("_bos")
    has_atr = pl.col("_atr14") > 0.0
    pullback_up = _over_atr(high - close)
    pullback_down = _over_atr(close - low)
    up = (bos == 1) & (trend == 1) & has_atr & (pullback_up < _TREND_PULLBACK_ATR)
    down = (bos == -1) & (trend == -1) & has_atr & (pullback_down < _TREND_PULLBACK_ATR)
    up_strength = 1.0 - pullback_up / _TREND_PULLBACK_ATR
    down_strength = 1.0 - pullback_down / _TREND_PULLBACK_ATR
    return _pair(up, down, up_strength, down_strength)


_RULES = {
    Strategy.MOMENTUM: _momentum,
    Strategy.REVERSAL: _reversal,
    Strategy.SWEEP: _sweep,
    Strategy.FVG: _fvg,
    Strategy.TREND: _trend,
}

_EMPTY_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "ts": pl.Datetime("us", "UTC"),
    **{name: pl.Int8 for name in SIGNAL_COLUMNS},
    **{name: pl.Float32 for name in STRENGTH_COLUMNS},
}


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(schema=_EMPTY_SCHEMA)


def _work_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Attach every helper series the rules read, one row per input bar."""
    missing = [name for name in _REQUIRED_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f"missing bar columns: {missing}")
    frame = df.sort("ts")
    structure = market_structure(frame, 2).select(
        pl.col("trend").alias("_trend"), pl.col("bos").alias("_bos")
    )
    gaps = fvg(frame).select(
        pl.col("fvg_dist").alias("_fvg_dist"),
        pl.col("fvg_up").alias("_fvg_up"),
        pl.col("fvg_down").alias("_fvg_down"),
    )
    return pl.concat(
        [
            frame.with_columns(
                atr(frame, 14).alias("_atr14"),
                atr(frame, 60).alias("_atr60"),
                rsi(frame, _REVERSAL_RSI_N).alias("_rsi14"),
                liquidity_sweep(frame, _SWEEP_LOOKBACK).alias("_sweep_event"),
            ),
            structure,
            gaps,
        ],
        how="horizontal",
    )


def strategy_signals(df: pl.DataFrame) -> pl.DataFrame:
    """Per-bar direction and strength for every strategy family.

    ``df`` is a frame of M1 bars (at least ``ts``/``open``/``high``/``low``/
    ``close``, as in :data:`~scalping.data.base.BAR_SCHEMA`). Rows are returned
    sorted by ``ts``. The result has one ``sig_<name>`` (``Int8`` in
    ``{-1, 0, 1}``) and one ``strength_<name>`` (``Float32`` in ``[0, 1]``,
    ``0.0`` wherever the signal is ``0``) per entry of :data:`ALL_STRATEGIES`.
    """
    if df.is_empty():
        return _empty_output()
    work = _work_frame(df)
    exprs: list[pl.Expr] = [pl.col("ts")]
    for strategy in ALL_STRATEGIES:
        sig, strength = _RULES[strategy]()
        exprs.append(sig.alias(f"sig_{strategy.value}"))
        exprs.append(strength.alias(f"strength_{strategy.value}"))
    return work.select(exprs)


def active_strategy(row: Mapping[str, object], bandit_choice: int) -> tuple[int, float]:
    """The ``(signal, strength)`` pair the bandit selected for this row.

    ``bandit_choice`` is a position in :data:`ALL_STRATEGIES`; an out-of-range
    index raises :class:`ValueError` instead of silently wrapping. ``row`` is a
    named row of :func:`strategy_signals` output, e.g.
    ``signals.row(i, named=True)``.
    """
    if not 0 <= bandit_choice < len(ALL_STRATEGIES):
        raise ValueError(
            f"bandit choice {bandit_choice} outside 0..{len(ALL_STRATEGIES) - 1}"
        )
    strategy = ALL_STRATEGIES[bandit_choice]
    return (
        int(row[f"sig_{strategy.value}"]),  # type: ignore[call-overload]
        float(row[f"strength_{strategy.value}"]),  # type: ignore[arg-type]
    )
