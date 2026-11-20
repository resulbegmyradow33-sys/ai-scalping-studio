from datetime import UTC, datetime

import numpy as np
import polars as pl

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
