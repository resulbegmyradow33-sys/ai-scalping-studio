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


def test_bid_only_nulls_are_a_warning_with_honest_capabilities() -> None:
    frame = _bars().with_columns(
        pl.lit(None, dtype=pl.Float64).alias("bid_close"),
        pl.lit(None, dtype=pl.Float64).alias("ask_close"),
        pl.lit(None, dtype=pl.Float64).alias("spread"),
    )

    report = validate_bars(frame, symbol="XAUUSD")

    codes = {issue.code for issue in report.issues}
    assert report.ok is True
    assert "missing_quotes" in codes
    assert "null_values" not in codes
    assert "non_finite_prices" not in codes
    assert report.capabilities is not None
    assert report.capabilities["real_quotes"] is False
    assert report.capabilities["modeled_spread"] is True
    assert report.to_dict()["capabilities"] == report.capabilities


def test_gap_breakdown_separates_in_session_and_session_closed() -> None:
    start = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)  # Monday
    frame = pl.DataFrame(
        {
            "ts": [start, start + timedelta(minutes=30)],
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.5, 100.5],
            "bid_close": [100.4, 100.4],
            "ask_close": [100.6, 100.6],
            "spread": [0.2, 0.2],
            "volume": [10, 10],
            "provider": ["test", "test"],
        }
    )

    report = validate_bars(frame, symbol="XAUUSD", timeframe="M1")

    assert report.ok is True
    breakdown = report.gap_breakdown
    assert breakdown is not None
    assert breakdown["gaps"] == 1
    assert breakdown["in_session"] == 1
    assert breakdown["session_closed"] == 0
    assert breakdown["missing_minutes"] == 29


def test_weekend_gap_is_session_closed() -> None:
    friday = datetime(2026, 1, 9, 20, 0, tzinfo=UTC)
    sunday = datetime(2026, 1, 11, 22, 0, tzinfo=UTC)
    frame = _bars(1)
    frame = pl.concat(
        [
            frame.with_columns(pl.lit(friday).alias("ts")),
            frame.with_columns(pl.lit(sunday).alias("ts")),
        ]
    )

    report = validate_bars(frame, symbol="XAUUSD", timeframe="M1")

    assert report.gap_breakdown is not None
    assert report.gap_breakdown["in_session"] == 0
    assert report.gap_breakdown["session_closed"] == 1
    assert report.gap_breakdown["missing_minutes"] == 0


def test_price_outlier_skips_the_first_bar_after_a_gap() -> None:
    from scalping.data.quality import OutlierConfig, detect_bar_outliers

    frame = _bars(60)
    jumped = frame.tail(1).with_columns(
        (pl.col("ts") + timedelta(days=2)).alias("ts"),
        pl.lit(200.0).alias("open"),
        pl.lit(201.0).alias("high"),
        pl.lit(199.0).alias("low"),
        pl.lit(200.0).alias("close"),
    )
    frame = pl.concat([frame.head(59), jumped])
    config = OutlierConfig(return_z=1.0, min_rows=30)

    without = {issue.code for issue in detect_bar_outliers(frame, config)}
    with_gap = {
        issue.code
        for issue in detect_bar_outliers(frame, config, gap_threshold_minutes=5)
    }

    assert "price_outliers" in without
    assert "price_outliers" not in with_gap
