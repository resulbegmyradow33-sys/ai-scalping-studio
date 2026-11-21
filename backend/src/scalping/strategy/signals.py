"""Five causal strategy signal families.

Each family turns the feature layer's causal indicators into a discrete
direction (``-1``/``0``/``+1``) plus a confidence in ``[0, 1]``. The order of
:data:`ALL_STRATEGIES` is frozen because the bandit's arm index is a position in
that tuple; :func:`active_strategy` is the bridge between the two.

Rules
-----
``momentum``
    ``close > ema20`` and ``close > close[-5]``, with expansion evidence:
    ``max(ATR(14)/ATR(60), 5-bar displacement in ATR)`` above ``1.1``.
    The displacement branch exists because a trend with a constant bar range
    has an ATR ratio of exactly ``1.0``; without it a clean, steady trend
    could never fire. Mirror for ``-1``. (Amended rule, see the plan.)
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
    (:func:`~scalping.features.liquidity.fvg`), less than ``1.5`` ATR past the
    near edge. Inside an up gap -> ``+1``, inside a down gap -> ``-1``.
``trend``
    A break of structure aligned with the running trend
    (:func:`~scalping.features.structure.market_structure`), **excluding**
    change-of-character breaks (continuation only), whose close sits within
    ``0.5`` ATR of the bar extreme (a shallow pullback). ``trend`` is ``bos``
    forward filled, so the alignment conjunct alone is a tautology; the CHoCH
    exclusion is what makes this a continuation filter.

Causality: every expression is a backward window, a lag or an exponential
seeded on the left edge. There is no ``shift(-n)``, no ``center=True`` and no
statistic over the whole frame, so row ``i`` never depends on a bar after it.

Non-finite prices are treated as unusable bars, not as evidence. Helpers are
computed on a copy whose non-finite prices are nulled (so one bad bar cannot
poison the ATR or the structure pass), the recursive momentum indicators read a
causally forward-filled close instead of the raw one (an EMA would otherwise
stay NaN forever), the signal predicate requires finite prices on the bar
itself, and strength is dropped to ``0.0`` if it is ever NaN or null.
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

#: Columns an unusable bar may not leave non-finite.
_PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")

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
    """Signal from the two sides; strength follows the side that fired.

    A bar with a non-finite price abstains: Polars compares NaN above every
    number (``close < NaN`` is true), so a raw predicate would fire on missing
    inputs. Strength is NaN- and null-free before the cast because a NaN would
    otherwise survive ``clip`` and compare above every float.
    """
    usable = pl.col("_finite")
    long = long & usable
    short = short & usable
    sig = pl.when(long).then(1).when(short).then(-1).otherwise(0)
    strength = (
        pl.when(long)
        .then(up)
        .when(short)
        .then(down)
        .otherwise(0.0)
        .clip(0.0, 1.0)
        .fill_nan(None)
        .fill_null(0.0)
    )
    return sig.cast(pl.Int8).alias("_sig"), strength.cast(pl.Float32).alias("_strength")


def _momentum() -> tuple[pl.Expr, pl.Expr]:
    """Trend continuation with volatility or displacement expansion.

    Reads ``_close_clean`` rather than the raw close: an EMA is recursive, so a
    single non-finite price (NaN or +/-inf) would otherwise poison every later
    bar and flip the whole tail to the short side. The clean series forward
    fills the gap (causal) and the row itself still abstains through the
    ``_finite`` gate.
    """
    close = pl.col("_close_clean")
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
    # Zero is the module's "no open gap" sentinel and never fires. The bounds are
    # strict so a signal always carries positive evidence (strength > 0), and the
    # previous bar must sit outside the gap, making this a re-entry event instead
    # of a level that stays lit for as long as price trades inside the zone.
    up = (dist < 0.0) & (dist > -_FVG_ATR_LIMIT) & (prev >= 0.0)
    down = (dist > 0.0) & (dist < _FVG_ATR_LIMIT) & (prev <= 0.0)
    strength = (1.0 - dist.abs() / _FVG_ATR_LIMIT).clip(0.0, 1.0)
    return _pair(up, down, strength, strength)


def _trend() -> tuple[pl.Expr, pl.Expr]:
    """BOS continuation in the trend direction with a shallow retracement."""
    close = pl.col("close")
    high = pl.col("high")
    low = pl.col("low")
    trend = pl.col("_trend")
    bos = pl.col("_bos")
    # ``trend`` is ``bos`` forward filled, so ``trend == bos`` is a tautology on a
    # break bar. Continuation means "not a change of character".
    choch = pl.col("_choch")
    has_atr = pl.col("_atr14") > 0.0
    pullback_up = _over_atr(high - close)
    pullback_down = _over_atr(close - low)
    up = (
        (bos == 1)
        & (trend == 1)
        & (choch == 0)
        & has_atr
        & (pullback_up < _TREND_PULLBACK_ATR)
    )
    down = (
        (bos == -1)
        & (trend == -1)
        & (choch == 0)
        & has_atr
        & (pullback_down < _TREND_PULLBACK_ATR)
    )
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
    **{
        name: dtype
        for strategy in ALL_STRATEGIES
        for name, dtype in (
            (f"sig_{strategy.value}", pl.Int8),
            (f"strength_{strategy.value}", pl.Float32),
        )
    },
}


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(schema=_EMPTY_SCHEMA)


def _validate_columns(df: pl.DataFrame) -> None:
    missing = [name for name in _REQUIRED_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f"missing bar columns: {missing}")


def _work_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Attach every helper series the rules read, one row per input bar.

    Helpers are computed on a sanitized copy: a non-finite price becomes null,
    which the EWM and rolling windows skip instead of carrying forward. The
    rules themselves read the original prices and are gated by ``_finite``, so
    an unusable bar abstains without contributing half-truths.
    """
    frame = df.sort("ts")
    usable = (
        pl.col("open").is_finite()
        & pl.col("high").is_finite()
        & pl.col("low").is_finite()
        & pl.col("close").is_finite()
    ).fill_null(False)
    clean = frame.with_columns(
        pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None).alias(name)
        for name in _PRICE_COLUMNS
    )
    # Causal close for the recursive momentum indicators: null anything
    # non-finite (NaN and +/-inf), then forward fill from the last printed
    # value. No raw fallback: re-inserting a leading NaN would poison the EWM
    # seed, while a null start simply delays the EMA.
    close_clean = (
        pl.when(pl.col("close").is_finite())
        .then(pl.col("close"))
        .otherwise(None)
        .forward_fill()
        .alias("_close_clean")
    )
    structure = market_structure(clean, 2).select(
        pl.col("trend").alias("_trend"),
        pl.col("bos").alias("_bos"),
        pl.col("choch").alias("_choch"),
    )
    gaps = fvg(clean).select(
        pl.col("fvg_dist").alias("_fvg_dist"),
        pl.col("fvg_up").alias("_fvg_up"),
        pl.col("fvg_down").alias("_fvg_down"),
    )
    return pl.concat(
        [
            frame.with_columns(
                close_clean,
                atr(clean, 14).alias("_atr14"),
                atr(clean, 60).alias("_atr60"),
                rsi(clean, _REVERSAL_RSI_N).alias("_rsi14"),
                liquidity_sweep(clean, _SWEEP_LOOKBACK).alias("_sweep_event"),
                usable.alias("_finite"),
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

    A bar whose ``open``/``high``/``low``/``close`` is not finite produces no
    signal (and therefore no strength) in any family.
    """
    _validate_columns(df)
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
