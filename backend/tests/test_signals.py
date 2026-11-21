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

from scalping.strategy.signals import ALL_STRATEGIES, Strategy, strategy_signals


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


def test_all_strategies_are_stable_order():
    assert [s.value for s in ALL_STRATEGIES] == ["momentum", "reversal", "sweep", "fvg", "trend"]


def test_strategy_enum_values_are_the_column_suffixes():
    assert [s.value for s in Strategy] == [s.value for s in ALL_STRATEGIES]


def test_output_has_one_column_pair_per_strategy():
    out = strategy_signals(bars([10 + (i % 7) * 0.1 for i in range(300)]))
    for s in ALL_STRATEGIES:
        assert f"sig_{s.value}" in out.columns
        assert f"strength_{s.value}" in out.columns
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
