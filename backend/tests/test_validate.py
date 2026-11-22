"""Bar/tick validation reports without repair."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from scalping.data.validate import validate_bars, validate_ticks


def _bars(n: int = 5) -> pl.DataFrame:
    start = datetime(2026, 1, 5, tzinfo=UTC)
    return pl.DataFrame(
        {
            "ts": [start + timedelta(minutes=i) for i in range(n)],
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.5] * n,
            "bid_close": [100.4] * n,
            "ask_close": [100.6] * n,
            "spread": [0.2] * n,
            "volume": [10] * n,
            "provider": ["test"] * n,
        }
    )


def test_clean_frame_passes() -> None:
    report = validate_bars(_bars(), symbol="XAUUSD")
    assert report.ok
    assert report.rows == 5
    assert report.issues == ()


def test_duplicate_and_broken_ohlc_are_errors() -> None:
    frame = _bars()
    frame = pl.concat([frame, frame.head(1)])
    frame = frame.with_columns(pl.when(pl.int_range(pl.len()) == 2).then(102.0).otherwise(pl.col("low")).alias("low"))
    report = validate_bars(frame)
    codes = {issue.code for issue in report.issues}
    assert not report.ok
    assert "duplicate_timestamps" in codes
    assert "ohlc_inconsistent" in codes


def test_negative_spread_and_crossed_quotes_are_errors() -> None:
    frame = _bars().with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(-0.1).otherwise(pl.col("spread")).alias("spread"),
        pl.when(pl.int_range(pl.len()) == 1).then(100.0).otherwise(pl.col("ask_close")).alias("ask_close"),
    )
    report = validate_bars(frame)
    codes = {issue.code for issue in report.issues}
    assert "negative_spread" in codes
    assert "crossed_quotes" in codes


def test_gap_is_a_warning_not_an_error() -> None:
    frame = _bars(3)
    gap = frame.tail(1).with_columns((pl.col("ts") + timedelta(minutes=30)).alias("ts"))
    report = validate_bars(pl.concat([frame, gap]))
    assert report.ok
    assert any(issue.code == "time_gaps" and issue.severity == "warning" for issue in report.issues)


def test_tick_validation_flags_crossed_book() -> None:
    frame = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 5, tzinfo=UTC)],
            "bid": [100.0],
            "ask": [99.5],
            "bid_volume": [1.0],
            "ask_volume": [2.0],
        }
    )
    report = validate_ticks(frame)
    assert not report.ok
    assert any(issue.code == "crossed_quotes" for issue in report.issues)
