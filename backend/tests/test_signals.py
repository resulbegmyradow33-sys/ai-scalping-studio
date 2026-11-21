"""Strategy signal tests.

The five families are the learning layer's action source, so the contract is
narrow: a stable strategy order (the bandit arm index depends on it), one
``int8`` signal plus one ``float32`` strength per family, strengths in ``[0, 1]``
and exactly zero where a family abstains. Causality is the load bearing
property: rebuilding from a truncated prefix must reproduce every row bit for
bit.
"""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from scalping.features.structure import market_structure
from scalping.strategy.signals import (
    ALL_STRATEGIES,
    SIGNAL_COLUMNS,
    STRENGTH_COLUMNS,
    Strategy,
    strategy_signals,
)


def bars(closes, highs=None, lows=None):
    n = len(closes)
    # The brief wrote ``minute=i``; that overflows for n > 60, so the fixture
    # rolls over the hour instead (same one-minute spacing).
    start = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    ts = [start + timedelta(minutes=i) for i in range(n)]
    return pl.DataFrame({
        "ts": ts, "open": closes, "close": closes,
        "high": highs or [c + 0.5 for c in closes],
        "low": lows or [c - 0.5 for c in closes],
        "bid_close": closes, "ask_close": [c + 0.2 for c in closes],
        "spread": [0.2] * n, "volume": [10] * n,
    })


def candles(closes, opens):
    """M1 bars with real bodies: each wick clears its own body by 0.3."""
    n = len(closes)
    start = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    return pl.DataFrame({
        "ts": [start + timedelta(minutes=i) for i in range(n)],
        "open": opens,
        "close": closes,
        "high": [max(o, c) + 0.3 for o, c in zip(opens, closes)],
        "low": [min(o, c) - 0.3 for o, c in zip(opens, closes)],
        "bid_close": closes, "ask_close": closes,
        "spread": [0.2] * n, "volume": [10] * n,
    })


def staircase_bars():
    """Up leg with close near the high, then a down leg with close near the low."""
    closes = [10.0]
    for i in range(1, 40):
        closes.append(closes[-1] + (0.4 if i % 4 else -0.2))
    for _ in range(10):
        closes.append(closes[-1] - 0.8)
    highs, lows = [], []
    for i, close in enumerate(closes):
        if i < 40:
            highs.append(close + 0.1)
            lows.append(close - 1.0)
        else:
            highs.append(close + 1.0)
            lows.append(close - 0.1)
    return bars(closes, highs=highs, lows=lows)


def test_all_strategies_are_stable_order():
    assert [s.value for s in ALL_STRATEGIES] == ["momentum", "reversal", "sweep", "fvg", "trend"]


def test_strategy_enum_values_are_the_column_suffixes():
    assert [s.value for s in Strategy] == [s.value for s in ALL_STRATEGIES]


def test_output_has_one_column_pair_per_strategy():
    out = strategy_signals(bars([10 + (i % 7) * 0.1 for i in range(300)]))
    for s in ALL_STRATEGIES:
        assert f"sig_{s.value}" in out.columns
        assert f"strength_{s.value}" in out.columns
    assert [c for c in out.columns if c.startswith("sig_")] == list(SIGNAL_COLUMNS)
    assert [c for c in out.columns if c.startswith("strength_")] == list(STRENGTH_COLUMNS)
    assert out.height == 300


def test_signals_only_take_valid_values():
    out = strategy_signals(bars([10 + (i % 11) * 0.2 for i in range(500)]))
    for s in ALL_STRATEGIES:
        assert set(out[f"sig_{s.value}"].unique()).issubset({-1, 0, 1})
        assert out[f"strength_{s.value}"].min() >= 0.0
        assert out[f"strength_{s.value}"].max() <= 1.0


def test_momentum_fires_on_sustained_uptrend():
    closes = [10 + i * 0.5 for i in range(60)]
    out = strategy_signals(bars(closes))
    assert (out["sig_momentum"] == 1).sum() > 0


def test_sweep_requires_wick_recovery():
    closes = [10.0] * 50
    lows = [9.8] * 50
    lows[45], lows[46] = 9.0, 9.9   # deep wick, recovered
    highs = [10.2] * 50
    out = strategy_signals(bars(closes, highs=highs, lows=lows))
    assert out["sig_sweep"][46] == 1
    # The wick bar itself is not a signal: the recovery bar is.
    assert out["sig_sweep"][45] == 0


def test_signals_are_causal():
    closes = [10 + (i % 9) * 0.3 for i in range(200)]
    df = bars(closes)
    full = strategy_signals(df)
    part = strategy_signals(df.head(150))
    for s in ALL_STRATEGIES:
        assert full[f"sig_{s.value}"][:150].to_list() == part[f"sig_{s.value}"].to_list()


def test_strengths_are_causal_too():
    closes = [10 + (i % 13) * 0.4 for i in range(240)]
    df = bars(closes)
    full = strategy_signals(df)
    part = strategy_signals(df.head(170))
    for s in ALL_STRATEGIES:
        assert full[f"strength_{s.value}"][:170].to_list() == part[f"strength_{s.value}"].to_list()


# --- semantic coverage of the individual rules -----------------------------


@pytest.mark.parametrize("direction", [1, -1])
def test_reversal_fires_on_an_exhausted_candle(direction):
    # direction=+1: straight up, last candle opens above its close (bearish).
    # direction=-1: straight down, last candle opens below its close (bullish).
    closes = [10 + direction * i * 0.5 for i in range(30)]
    opens = list(closes)
    opens[-1] = closes[-1] + 0.2 * direction
    out = strategy_signals(candles(closes, opens))
    assert out["sig_reversal"][-1] == -direction
    assert out["strength_reversal"][-1] > 0


def test_fvg_fires_when_price_reenters_a_gap():
    # Bars 0-3 sit in a 9.5-10.5 range; bar 4's wick tops the range, bar 5
    # jumps to leave a 10.5-11.0 up gap, bar 6 dips back into it.
    closes = [10.0, 10.05, 10.0, 10.05, 10.4, 11.6, 10.75]
    highs = [10.5, 10.55, 10.5, 10.55, 10.75, 11.8, 10.9]
    lows = [9.5, 9.55, 9.5, 9.55, 9.6, 11.0, 10.7]
    out = strategy_signals(bars(closes, highs=highs, lows=lows))
    assert out["sig_fvg"][5] == 0        # the bar that creates the gap is not an entry
    assert out["sig_fvg"][6] == 1        # re-entry from above -> bull
    assert 0.0 < out["strength_fvg"][6] <= 1.0


def test_trend_fires_on_continuation_but_not_on_choch():
    df = staircase_bars()
    structure = market_structure(df, 2)
    out = strategy_signals(df)
    # Index 1 is the first up break, with no reversal break before it.
    assert structure["bos"][1] == 1
    assert structure["choch"][1] == 0
    assert out["sig_trend"][1] == 1
    # Index 42 flips the running trend down: a CHoCH, not a continuation.
    assert structure["choch"][42] == -1
    assert out["sig_trend"][42] == 0


def test_abstaining_signal_has_zero_strength_everywhere():
    out = strategy_signals(bars([10 + (i % 17) * 0.35 for i in range(400)]))
    for s in ALL_STRATEGIES:
        sig = out[f"sig_{s.value}"]
        strength = out[f"strength_{s.value}"]
        assert ((sig == 0) & (strength != 0)).sum() == 0
        assert ((sig != 0) & (strength <= 0)).sum() == 0


def test_empty_frame_keeps_the_schema():
    out = strategy_signals(bars([]))
    reference = strategy_signals(bars([10 + i * 0.1 for i in range(10)]))
    assert out.height == 0
    assert out.columns == reference.columns
    assert out.schema["ts"] == pl.Datetime("us", "UTC")
    for s in ALL_STRATEGIES:
        assert out.schema[f"sig_{s.value}"] == pl.Int8
        assert out.schema[f"strength_{s.value}"] == pl.Float32


def test_missing_bar_columns_raise():
    with pytest.raises(ValueError, match="missing bar columns"):
        strategy_signals(bars([10.0] * 10).drop("high"))


def test_nan_price_never_invents_a_signal():
    # A NaN breaks every comparison (`x < NaN` is true in Polars) and, fed to
    # the EMA, poisons it for the rest of the frame. A rising series can never
    # justify a short, so any -1 here is an artifact.
    closes = [10 + i * 0.1 for i in range(120)]
    closes[50] = float("nan")
    out = strategy_signals(bars(closes))
    for s in ALL_STRATEGIES:
        sig = out[f"sig_{s.value}"]
        strength = out[f"strength_{s.value}"]
        assert strength.is_finite().all()
        assert strength.min() >= 0.0
        assert strength.max() <= 1.0
        assert sig[50] == 0
        assert ((sig != 0) & (strength <= 0.0)).sum() == 0
    assert (out["sig_momentum"][51:] == -1).sum() == 0
