from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from scalping.data.base import BAR_SCHEMA
from scalping.data.resample import resample
from scalping.data.store import ParquetStore


def bars(n=10):
    ts = [datetime(2026, 1, 5, 10, i, tzinfo=UTC) for i in range(n)]
    return pl.DataFrame({
        "ts": ts, "open": [1.0+i for i in range(n)], "high": [2.0+i for i in range(n)],
        "low": [0.5+i for i in range(n)], "close": [1.5+i for i in range(n)],
        "bid_close": [1.4+i for i in range(n)], "ask_close": [1.6+i for i in range(n)],
        "spread": [0.2]*n, "volume": [1]*n,
    })

def test_write_read_roundtrip_and_dedup(tmp_path):
    s = ParquetStore(tmp_path)
    assert s.write("XAUUSD", "M1", bars(10)) == 10
    assert s.write("XAUUSD", "M1", bars(10)) == 0  # duplicate ts dropped
    out = s.read("XAUUSD", "M1")
    assert out.height == 10
    assert out["ts"].is_sorted()

def test_read_missing_range_returns_empty_with_schema(tmp_path):
    s = ParquetStore(tmp_path)
    s.write("XAUUSD", "M1", bars(3))
    out = s.read("XAUUSD", "M1", datetime(2030, 1, 1, tzinfo=UTC))
    assert out.height == 0
    assert "spread" in out.columns

def test_coverage(tmp_path):
    s = ParquetStore(tmp_path)
    s.write("XAUUSD", "M1", bars(5))
    cov = s.coverage("XAUUSD")
    assert (cov.rows, cov.first_ts.hour, cov.last_ts.minute) == (5, 10, 4)

def test_resample_5m_ohlc_and_spread():
    out = resample(bars(10), "5m")
    assert out.height == 2
    first = out.row(0, named=True)
    assert (first["open"], first["high"], first["low"], first["close"]) == (1.0, 6.0, 0.5, 5.5)
    assert first["volume"] == 5
    assert first["spread_mean"] == 0.2


def test_dedup_keeps_last_row(tmp_path):
    s = ParquetStore(tmp_path)
    s.write("XAUUSD", "M1", bars(3))
    newer = bars(3).with_columns(pl.lit(9.0).alias("close"))
    assert s.write("XAUUSD", "M1", newer) == 0
    assert s.read("XAUUSD", "M1")["close"].to_list() == [9.0, 9.0, 9.0]


def test_write_counts_only_new_timestamps(tmp_path):
    s = ParquetStore(tmp_path)
    s.write("XAUUSD", "M1", bars(5))
    assert s.write("XAUUSD", "M1", bars(7)) == 2
    assert s.read("XAUUSD", "M1").height == 7


def test_empty_store_read_has_canonical_schema(tmp_path):
    out = ParquetStore(tmp_path).read("XAUUSD", "M1")
    assert out.height == 0
    assert out.columns == list(BAR_SCHEMA)
    assert out.schema == BAR_SCHEMA


def test_read_filters_to_requested_range(tmp_path):
    s = ParquetStore(tmp_path)
    s.write("XAUUSD", "M1", bars(10))
    out = s.read(
        "XAUUSD",
        "M1",
        datetime(2026, 1, 5, 10, 3, tzinfo=UTC),
        datetime(2026, 1, 5, 10, 5, tzinfo=UTC),
    )
    assert out.height == 3
    assert out["ts"].min() == datetime(2026, 1, 5, 10, 3, tzinfo=UTC)


def test_write_partitions_by_year_and_month(tmp_path):
    s = ParquetStore(tmp_path)
    jan = bars(1)
    feb = bars(1).with_columns(pl.lit(datetime(2026, 2, 1, tzinfo=UTC)).alias("ts"))
    assert s.write("XAUUSD", "M1", pl.concat([jan, feb])) == 2
    assert (tmp_path / "XAUUSD" / "M1" / "2026" / "01.parquet").exists()
    assert (tmp_path / "XAUUSD" / "M1" / "2026" / "02.parquet").exists()


def test_coverage_empty_store(tmp_path):
    cov = ParquetStore(tmp_path).coverage("XAUUSD")
    assert (cov.first_ts, cov.last_ts, cov.rows) == (None, None, 0)


def test_resample_uses_last_bar_quotes_and_provider(tmp_path):
    df = bars(10).with_columns(pl.lit("csv").alias("provider"))
    first = resample(df, "5m").row(0, named=True)
    assert first["ts"] == datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    assert (first["bid_close"], first["ask_close"]) == (5.4, 5.6)
    assert first["provider"] == "csv"


def test_resample_15m_single_bucket():
    out = resample(bars(10), "15m")
    assert out.height == 1
    row = out.row(0, named=True)
    assert (row["high"], row["low"], row["close"], row["volume"]) == (11.0, 0.5, 10.5, 10)


def test_provider_survives_store_roundtrip(tmp_path):
    s = ParquetStore(tmp_path)
    s.write("XAUUSD", "M1", bars(3).with_columns(pl.lit("dukascopy").alias("provider")))
    assert s.read("XAUUSD", "M1")["provider"].unique().to_list() == ["dukascopy"]


def test_read_normalizes_partitions_with_divergent_schemas(tmp_path):
    s = ParquetStore(tmp_path)
    feb = bars(1).with_columns(
        pl.lit(datetime(2026, 2, 1, tzinfo=UTC)).alias("ts"),
        pl.lit("synthetic").alias("provider"),
    )
    s.write("XAUUSD", "M1", feb)
    # Hand-written partition: no provider column, plus a stray extra column.
    raw = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 5, 10, 0, tzinfo=UTC)],
            "open": [1.0],
            "high": [1.5],
            "low": [0.5],
            "close": [1.0],
            "bid_close": [0.9],
            "ask_close": [1.1],
            "spread": [0.2],
            "volume": [1],
            "legacy_source": ["old-export"],
        }
    )
    raw_path = tmp_path / "XAUUSD" / "M1" / "2026" / "01.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw.write_parquet(raw_path)

    out = s.read("XAUUSD", "M1")

    assert out.height == 2
    assert out.columns == list(BAR_SCHEMA)
    assert out["provider"].to_list() == ["unknown", "synthetic"]


def test_failed_write_leaves_previous_partition_intact(tmp_path, monkeypatch):
    s = ParquetStore(tmp_path)
    s.write("XAUUSD", "M1", bars(3))
    path = tmp_path / "XAUUSD" / "M1" / "2026" / "01.parquet"
    before = path.read_bytes()

    def failing_write(self, file, **kwargs):
        Path(file).write_bytes(b"truncated")
        raise OSError("disk full")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", failing_write)
    with pytest.raises(OSError, match="disk full"):
        s.write("XAUUSD", "M1", bars(5))

    assert path.read_bytes() == before
    assert not list(path.parent.glob("*.tmp"))
    out = s.read("XAUUSD", "M1")
    assert out.height == 3
    assert out["close"].to_list() == [1.5, 2.5, 3.5]


def minute_bars(start: datetime, n: int) -> pl.DataFrame:
    ts = [start + timedelta(minutes=i) for i in range(n)]
    return pl.DataFrame(
        {
            "ts": ts,
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "bid_close": [100.4 + i for i in range(n)],
            "ask_close": [100.6 + i for i in range(n)],
            "spread": [0.2] * n,
            "volume": [1] * n,
        }
    )


def test_resample_m30_and_h4_bucket_alignment():
    frame = minute_bars(datetime(2026, 1, 5, 0, 0, tzinfo=UTC), 8 * 60)

    m30 = resample(frame, "30m")
    h4 = resample(frame, "4h")

    assert m30.height == 16
    assert h4.height == 2
    assert m30["ts"].to_list()[:3] == [
        datetime(2026, 1, 5, 0, 0, tzinfo=UTC),
        datetime(2026, 1, 5, 0, 30, tzinfo=UTC),
        datetime(2026, 1, 5, 1, 0, tzinfo=UTC),
    ]
    assert h4["ts"].to_list() == [
        datetime(2026, 1, 5, 0, 0, tzinfo=UTC),
        datetime(2026, 1, 5, 4, 0, tzinfo=UTC),
    ]

    first = m30.row(0, named=True)
    assert (first["open"], first["high"], first["low"], first["close"]) == (
        100.0,
        130.0,
        99.0,
        129.5,
    )
    assert first["volume"] == 30
    assert first["bid_close"] == 129.4
    assert first["ask_close"] == 129.6
    assert first["spread_mean"] == pytest.approx(0.2)

    first_h4 = h4.row(0, named=True)
    assert (first_h4["open"], first_h4["close"], first_h4["volume"]) == (100.0, 339.5, 240)


def test_resample_boundaries_are_left_closed():
    frame = minute_bars(datetime(2026, 1, 5, 3, 55, tzinfo=UTC), 10)  # 03:55..04:04

    h4 = resample(frame, "4h")

    assert h4["volume"].to_list() == [5, 5]
    assert h4["ts"].to_list() == [
        datetime(2026, 1, 5, 0, 0, tzinfo=UTC),
        datetime(2026, 1, 5, 4, 0, tzinfo=UTC),
    ]
    assert h4["close"].to_list() == [100.5 + 4, 100.5 + 9]


def test_derived_m30_store_matches_resample(tmp_path):
    from scalping.data.registry import timeframe_rule

    store = ParquetStore(tmp_path)
    source = minute_bars(datetime(2026, 1, 5, 10, 0, tzinfo=UTC), 120)
    store.write("XAUUSD", "M1", source)

    expected = resample(store.read("XAUUSD", "M1"), timeframe_rule("M30")).rename(
        {"spread_mean": "spread"}
    )
    store.write("XAUUSD", "M30", expected)

    stored = store.read("XAUUSD", "M30")
    assert stored.height == 4
    assert stored["ts"].to_list() == expected["ts"].to_list()
    assert stored["volume"].to_list() == [30, 30, 30, 30]
    assert stored["close"].to_list() == expected["close"].to_list()


def _month_bars(year: int, month: int, n: int = 3) -> pl.DataFrame:
    ts = [datetime(year, month, 1, 0, minute, tzinfo=UTC) for minute in range(n)]
    return pl.DataFrame(
        {
            "ts": ts,
            "open": [1.0] * n,
            "high": [2.0] * n,
            "low": [0.5] * n,
            "close": [1.5] * n,
            "bid_close": [1.4] * n,
            "ask_close": [1.6] * n,
            "spread": [0.2] * n,
            "volume": [1] * n,
        }
    )


def test_coverage_matches_full_read_across_partitions(tmp_path):
    s = ParquetStore(tmp_path)
    for year, month in ((2025, 12), (2026, 1), (2026, 2)):
        s.write("XAUUSD", "M1", _month_bars(year, month))

    frame = s.read("XAUUSD", "M1")
    cov = s.coverage("XAUUSD", "M1")

    assert cov.rows == frame.height == 9
    assert cov.first_ts == frame["ts"].min()
    assert cov.last_ts == frame["ts"].max()


def test_coverage_never_calls_read(tmp_path, monkeypatch):
    s = ParquetStore(tmp_path)
    s.write("XAUUSD", "M1", _month_bars(2026, 1))

    def fail(*args, **kwargs):
        raise AssertionError("coverage must not materialise the frame")

    monkeypatch.setattr(ParquetStore, "read", fail)
    cov = s.coverage("XAUUSD", "M1")

    assert (cov.rows, cov.first_ts.minute, cov.last_ts.minute) == (3, 0, 2)


def test_coverage_falls_back_when_statistics_are_missing(tmp_path, monkeypatch):
    import scalping.data.store as store_module

    s = ParquetStore(tmp_path)
    s.write("XAUUSD", "M1", _month_bars(2026, 1))

    def unavailable(path):
        raise store_module._MetadataUnavailable("no statistics")

    monkeypatch.setattr(store_module, "_metadata_partition_coverage", unavailable)
    cov = s.coverage("XAUUSD", "M1")

    assert (cov.rows, cov.first_ts.minute, cov.last_ts.minute) == (3, 0, 2)


def test_providers_reads_only_the_provider_column(tmp_path):
    s = ParquetStore(tmp_path)
    s.write("XAUUSD", "M1", _month_bars(2026, 1).with_columns(pl.lit("csv").alias("provider")))
    s.write("XAUUSD", "M1", _month_bars(2026, 2).with_columns(pl.lit("synthetic").alias("provider")))

    assert s.providers("XAUUSD", "M1") == ["csv", "synthetic"]
    assert s.providers("XAUUSD", "M5") == []
