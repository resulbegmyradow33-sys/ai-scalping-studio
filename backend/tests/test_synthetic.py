from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from scalping.data.base import BAR_SCHEMA
from scalping.data.synthetic import SyntheticProvider

START = datetime(2026, 1, 5, tzinfo=UTC)
END = datetime(2026, 1, 7, tzinfo=UTC)


def test_deterministic_for_same_seed():
    a = SyntheticProvider(seed=11).fetch("XAUUSD", START, END)
    b = SyntheticProvider(seed=11).fetch("XAUUSD", START, END)
    assert a.equals(b)


def test_different_seed_differs():
    a = SyntheticProvider(seed=11).fetch("XAUUSD", START, END)
    b = SyntheticProvider(seed=12).fetch("XAUUSD", START, END)
    assert not a["close"].equals(b["close"])


def test_ohlc_valid_and_no_weekend_bars():
    df = SyntheticProvider().fetch("XAUUSD", START, END)
    assert df.height > 2000
    assert (df["high"] >= df[["open", "close"]].max_horizontal()).all()
    assert (df["low"] <= df[["open", "close"]].min_horizontal()).all()
    assert df.filter(pl.col("ts").dt.weekday() >= 6).height == 0


def test_spread_positive_and_ask_above_bid():
    df = SyntheticProvider().fetch("XAUUSD", START, END)
    assert (df["spread"] > 0).all()
    assert (df["ask_close"] > df["bid_close"]).all()


def test_schema_and_provider_literal():
    df = SyntheticProvider().fetch("XAUUSD", START, END)
    assert df.schema == BAR_SCHEMA
    assert df["provider"].unique().to_list() == ["synthetic"]
    assert df["ts"].is_sorted()
    assert df["ts"].n_unique() == df.height
    # Every bar sits on the whole-minute UTC grid.
    assert (df["ts"].dt.second() == 0).all()
    assert (df["ts"].dt.nanosecond() == 0).all()


def test_prices_positive_continuous_and_mid_quoted():
    df = SyntheticProvider().fetch("XAUUSD", START, END)
    for col in ("open", "high", "low", "close", "bid_close", "ask_close"):
        assert (df[col] > 0).all()
        assert np.isfinite(df[col].to_numpy()).all()
    # Candles chain into one continuous path and quotes straddle the close.
    assert np.allclose(df["open"][1:].to_numpy(), df["close"][:-1].to_numpy(), rtol=0, atol=1e-9)
    assert np.allclose(
        df["close"].to_numpy(),
        (df["bid_close"] + df["ask_close"]).to_numpy() / 2,
        rtol=0,
        atol=1e-9,
    )


def test_short_window_under_one_hour():
    start = datetime(2026, 1, 5, 9, 30, tzinfo=UTC)
    end = datetime(2026, 1, 5, 9, 45, tzinfo=UTC)
    df = SyntheticProvider().fetch("XAUUSD", start, end)
    assert df.height == 16
    assert df.schema == BAR_SCHEMA
    assert df["ts"].min() == start
    assert df["ts"].max() == end


def test_weekend_only_window_is_empty_with_schema():
    # 2026-01-10 is a Saturday, 2026-01-11 a Sunday.
    start = datetime(2026, 1, 10, tzinfo=UTC)
    end = datetime(2026, 1, 11, 23, 59, tzinfo=UTC)
    df = SyntheticProvider().fetch("XAUUSD", start, end)
    assert df.height == 0
    assert df.schema == BAR_SCHEMA


def test_values_do_not_depend_on_requested_window():
    full = SyntheticProvider(seed=11).fetch("XAUUSD", START, END)
    cutoff = datetime(2026, 1, 6, tzinfo=UTC)
    tail = SyntheticProvider(seed=11).fetch("XAUUSD", cutoff, END)
    assert tail.equals(full.filter(pl.col("ts") >= cutoff))


def test_naive_timestamps_are_read_as_utc():
    aware = SyntheticProvider(seed=11).fetch("XAUUSD", START, END)
    naive = SyntheticProvider(seed=11).fetch(
        "XAUUSD", START.replace(tzinfo=None), END.replace(tzinfo=None)
    )
    assert naive.equals(aware)


def test_empty_range_is_empty_with_schema():
    df = SyntheticProvider().fetch("XAUUSD", END, START)
    assert df.height == 0
    assert df.schema == BAR_SCHEMA


def _median_spread(start: datetime, end: datetime) -> float:
    df = SyntheticProvider().fetch("XAUUSD", start, end)
    assert df.height > 0
    return float(df["spread"].median())


def test_session_spread_tiers():
    # Monday 2026-01-05: Asia 00:00-07:59, London 08:00-12:59, NY 13:00-20:59.
    asia = _median_spread(
        datetime(2026, 1, 5, 0, tzinfo=UTC), datetime(2026, 1, 5, 2, 59, tzinfo=UTC)
    )
    london = _median_spread(
        datetime(2026, 1, 5, 12, tzinfo=UTC), datetime(2026, 1, 5, 12, 59, tzinfo=UTC)
    )
    ny = _median_spread(
        datetime(2026, 1, 5, 18, tzinfo=UTC), datetime(2026, 1, 5, 20, 59, tzinfo=UTC)
    )
    late_asia = _median_spread(
        datetime(2026, 1, 5, 21, tzinfo=UTC), datetime(2026, 1, 5, 23, 59, tzinfo=UTC)
    )

    # Tier order and bases (0.18/0.22/0.28) within the +/-10% noise band.
    assert london < ny < asia
    assert np.isclose(asia, 0.28, rtol=0.05)
    assert np.isclose(london, 0.18, rtol=0.05)
    assert np.isclose(ny, 0.22, rtol=0.05)
    assert np.isclose(london / asia, 0.18 / 0.28, rtol=0.10)
    assert np.isclose(ny / asia, 0.22 / 0.28, rtol=0.10)
    # 21:00-23:59 rolls over to the Asian book.
    assert np.isclose(late_asia, asia, rtol=0.03)


def test_spread_is_ask_minus_bid():
    df = SyntheticProvider().fetch("XAUUSD", START, END)
    assert np.allclose(
        (df["ask_close"] - df["bid_close"]).to_numpy(),
        df["spread"].to_numpy(),
        rtol=0,
        atol=1e-9,
    )


@pytest.mark.parametrize("bad_price", [float("nan"), float("inf"), float("-inf")])
def test_start_price_must_be_finite(bad_price):
    with pytest.raises(ValueError, match="start_price"):
        SyntheticProvider(start_price=bad_price)
