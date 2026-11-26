"""Tick store: partitioning, dedupe, range queries and coverage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from scalping.data.ticks import TICK_SCHEMA, TickStore, conform_ticks


def ticks(start: datetime, n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [start + timedelta(seconds=i) for i in range(n)],
            "bid": [2000.0 + i for i in range(n)],
            "ask": [2000.1 + i for i in range(n)],
            "bid_volume": [1.0] * n,
            "ask_volume": [2.0] * n,
        }
    )


def test_round_trip_across_month_partitions(tmp_path) -> None:
    store = TickStore(tmp_path)
    frame = ticks(datetime(2026, 1, 31, 23, 58, tzinfo=UTC), 300)

    assert store.write("XAUUSD", frame) == 300

    out = store.read("XAUUSD")
    assert out.height == 300
    assert out["ts"].is_sorted()
    assert out.columns == ["ts", "bid", "ask", "bid_volume", "ask_volume", "provider"]
    assert out.schema == TICK_SCHEMA
    assert (tmp_path / "XAUUSD" / "2026" / "01.parquet").is_file()
    assert (tmp_path / "XAUUSD" / "2026" / "02.parquet").is_file()


def test_reimport_adds_nothing_and_keeps_distinct_ticks(tmp_path) -> None:
    store = TickStore(tmp_path)
    frame = ticks(datetime(2026, 1, 5, 10, 0, tzinfo=UTC), 3)

    assert store.write("XAUUSD", frame) == 3
    assert store.write("XAUUSD", frame) == 0

    same_ts = frame.head(1).with_columns(pl.lit(999.0).alias("bid"))
    assert store.write("XAUUSD", same_ts) == 1
    assert store.read("XAUUSD").height == 4


def test_range_and_limit_are_inclusive_and_oldest_first(tmp_path) -> None:
    store = TickStore(tmp_path)
    store.write("XAUUSD", ticks(datetime(2026, 1, 5, 10, 0, tzinfo=UTC), 10))
    start = datetime(2026, 1, 5, 10, 0, 2, tzinfo=UTC)
    end = datetime(2026, 1, 5, 10, 0, 4, tzinfo=UTC)

    window = store.read("XAUUSD", start, end)
    limited = store.read("XAUUSD", limit=2)

    assert window.height == 3
    assert [row.second for row in window["ts"].to_list()] == [2, 3, 4]
    assert limited.height == 2
    assert limited["ts"][0] < limited["ts"][1]
    with pytest.raises(ValueError, match="limit"):
        store.read("XAUUSD", limit=0)
    assert store.read("XAUUSD", end, start).is_empty()


def test_coverage_and_missing_symbol(tmp_path) -> None:
    store = TickStore(tmp_path)
    assert store.read("MISSING").is_empty()
    assert store.coverage("MISSING").rows == 0

    frame = ticks(datetime(2026, 1, 5, 10, 0, 0, 500000, tzinfo=UTC), 5)
    store.write("XAUUSD", frame)
    coverage = store.coverage("XAUUSD")

    assert coverage.rows == 5
    assert coverage.first_ts == frame["ts"][0]
    assert coverage.last_ts == frame["ts"][-1]


def test_conform_ticks_fills_provider_and_utc() -> None:
    raw = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 5, 10, 0)],
            "bid": [1.0],
            "ask": [2.0],
            "bid_volume": [1],
            "ask_volume": [1],
            "ignored": ["x"],
        }
    )
    out = conform_ticks(raw)

    assert out["provider"].to_list() == ["unknown"]
    assert out["ts"].dtype.time_zone == "UTC"
    assert "ignored" not in out.columns

    with pytest.raises(ValueError, match="missing tick columns"):
        conform_ticks(raw.drop("ask"))


def test_symbol_must_be_a_single_safe_component(tmp_path) -> None:
    store = TickStore(tmp_path)
    frame = ticks(datetime(2026, 1, 5, 10, 0, tzinfo=UTC), 1)

    for bad in ("../escape", "a/b", "..", ".", "", "x" * 65):
        with pytest.raises(ValueError, match="invalid symbol"):
            store.write(bad, frame)
        with pytest.raises(ValueError, match="invalid symbol"):
            store.read(bad)
        with pytest.raises(ValueError, match="invalid symbol"):
            store.coverage(bad)
