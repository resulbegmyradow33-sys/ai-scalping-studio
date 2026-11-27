"""Tick flow aggregation: signed volume, CVD and guard rails."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from scalping.data.flow import FLOW_COLUMNS, aggregate_flow


def ticks(rows: list[tuple[int, float, float]]) -> pl.DataFrame:
    start = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    return pl.DataFrame(
        {
            "ts": [start + timedelta(seconds=offset) for offset, _, _ in rows],
            "bid_volume": [sell for _, sell, _ in rows],
            "ask_volume": [buy for _, _, buy in rows],
        }
    )


def test_flow_and_cvd_across_buckets() -> None:
    frame = ticks([(0, 3.0, 1.0), (30, 1.0, 1.0), (60, 1.0, 4.0)])

    out = aggregate_flow(frame, rule="1m")

    assert out.columns == list(FLOW_COLUMNS)
    assert out.height == 2
    first = out.row(0, named=True)
    assert first["buy_vol"] == 2.0 and first["sell_vol"] == 4.0
    assert first["flow"] == pytest.approx(-1 / 3)
    assert first["cvd"] == pytest.approx(-2.0)
    assert first["ticks"] == 2

    second = out.row(1, named=True)
    assert second["flow"] == pytest.approx(0.6)
    assert second["cvd"] == pytest.approx(1.0)


def test_zero_volume_bucket_has_zero_flow() -> None:
    out = aggregate_flow(ticks([(0, 0.0, 0.0)]), rule="1m")

    assert out["flow"].to_list() == [0.0]
    assert out["cvd"].to_list() == [0.0]


def test_unsorted_and_naive_timestamps_are_normalised() -> None:
    frame = ticks([(60, 1.0, 3.0), (0, 1.0, 1.0)]).with_columns(
        pl.col("ts").dt.replace_time_zone(None, ambiguous="earliest")
    )

    out = aggregate_flow(frame, rule="1m")

    assert out["ts"].dtype.time_zone == "UTC"
    assert out["ts"].is_sorted()
    assert out["cvd"].to_list() == [0.0, 2.0]


def test_empty_and_missing_columns_are_rejected() -> None:
    empty = pl.DataFrame(
        schema={"ts": pl.Datetime("us", "UTC"), "bid_volume": pl.Float64, "ask_volume": pl.Float64}
    )
    assert aggregate_flow(empty).height == 0

    with pytest.raises(ValueError, match="missing tick columns"):
        aggregate_flow(ticks([(0, 1.0, 1.0)]).drop("ask_volume"))
