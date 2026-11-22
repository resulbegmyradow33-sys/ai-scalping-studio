"""Rule-based regime labels: determinism, causality and sanity.

The fixtures are hand-built paths with a known shape (steady trend, quiet
range, volatility jump, clean breakout) so the assertions test the rules rather
than a fitted model. The leakage test mutates *future* bars and requires the
already-classified prefix to stay byte-identical.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from scalping.regime import REGIME_COLUMNS, RegimeConfig, classify

START = datetime(2026, 1, 5, tzinfo=UTC)


def bars(closes: object, *, wick: float = 0.05) -> pl.DataFrame:
    """Build a schema-correct bar frame from a close path."""
    close = np.asarray(closes, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pl.DataFrame(
        {
            "ts": [START + timedelta(minutes=i) for i in range(close.size)],
            "open": open_,
            "high": np.maximum(open_, close) + wick,
            "low": np.minimum(open_, close) - wick,
            "close": close,
            "bid_close": close,
            "ask_close": close + 0.2,
            "spread": np.full(close.size, 0.2),
            "volume": np.full(close.size, 10, dtype=np.int64),
        }
    )


def uptrend(n: int = 300) -> np.ndarray:
    return 100.0 * np.exp(np.linspace(0.0, 0.15, n))


def quiet_range(n: int = 300) -> np.ndarray:
    return 100.0 + 0.3 * np.sin(np.arange(n) / 7.0)


def test_clean_uptrend_is_labelled_up() -> None:
    labels = classify(bars(uptrend()))

    tail = labels.tail(50)
    assert set(tail["trend"]).issubset({"strong_up", "weak_up"})
    assert all(value.startswith(("up_", "breakout_up")) for value in tail["regime"])


def test_quiet_range_is_mostly_sideways() -> None:
    labels = classify(bars(quiet_range()))

    sideways = labels["trend"].eq("sideways").sum()
    assert sideways / labels.height >= 0.80


def test_sustained_volatility_jump_is_high_volatility() -> None:
    rng = np.random.default_rng(3)
    quiet = 100.0 + 0.02 * np.sin(np.arange(200) / 4.0)
    noisy = quiet[-1] + np.cumsum(rng.normal(0.0, 0.6, 200))

    labels = classify(bars(np.concatenate([quiet, noisy])))

    # A 480-bar baseline means a persistent jump stays "high" instead of
    # normalising itself after one window.
    assert labels.tail(150)["volatility"].eq("high").all()


def test_breakout_is_an_event_not_a_state() -> None:
    base = quiet_range(120)
    # A one-bar step out of the range, then a slow drift: the step is the event.
    jump = np.concatenate([base, base[-1] + 1.0 + np.linspace(0.0, 0.5, 40)])

    labels = classify(bars(jump))

    events = labels["breakout"].to_list()
    # The quiet prefix has at most a couple of genuine range extremes; the jump
    # itself must be flagged on entry and must not repeat on every bar.
    assert events[:120].count("up") == 0
    assert events[120:].count("up") >= 1
    assert events[120:].count("up") <= 2


def test_labels_do_not_use_future_bars() -> None:
    path = np.concatenate([uptrend(300), quiet_range(300)])
    frame = bars(path)

    full = classify(frame)
    truncated = classify(frame.head(300))
    assert_frame_equal(full.head(300), truncated)

    # Rebuild with a wild future; rows before the cut must not move.
    future = path.copy()
    future[300:] = future[300:] * 3.0
    mutated = classify(bars(future))
    assert_frame_equal(full.head(300), mutated.head(300))


def test_classification_is_deterministic_and_order_independent() -> None:
    frame = bars(np.concatenate([uptrend(200), quiet_range(200)]))

    first = classify(frame)
    second = classify(frame)
    shuffled = classify(frame.reverse())

    assert_frame_equal(first, second)
    assert_frame_equal(first, shuffled)


def test_empty_and_short_frames_keep_the_contract() -> None:
    empty = classify(bars([100.0]).head(0))
    assert empty.height == 0
    assert empty.columns == ["ts", *REGIME_COLUMNS]

    short = classify(bars([100.0, 100.1, 100.05, 100.2]))
    assert short.height == 4
    assert short.columns == ["ts", *REGIME_COLUMNS]
    assert short["unusual"].to_list() == [False, False, False, False]


def test_config_rejects_impossible_thresholds() -> None:
    with pytest.raises(ValueError, match="trend_window"):
        RegimeConfig(trend_window=0)
    with pytest.raises(ValueError, match="strong_trend_threshold"):
        RegimeConfig(trend_threshold=0.6, strong_trend_threshold=0.5)
    with pytest.raises(ValueError, match="vol_low_ratio"):
        RegimeConfig(vol_low_ratio=2.0, vol_high_ratio=1.5)
    with pytest.raises(ValueError, match="breakout_atr_buffer"):
        RegimeConfig(breakout_atr_buffer=-0.1)
