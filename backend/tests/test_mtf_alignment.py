"""M30/H4 alignment and chained-resampling semantics on synthetic data.

The load-bearing property is information availability: a higher-timeframe value
may only reach a decision bar once its bucket has closed. Every frame here is
synthetic, so the tests describe the pipeline contract, not market evidence.

The feature layer currently consumes M5 and M15; M30/H4 are data-layer
timeframes. ``htf_context`` is exercised with M30/H4 frames because its
availability rule is timeframe-agnostic (one bucket width taken from the frame
spacing), and that rule is what any future H4 feature will inherit.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scalping.data.registry import session_for, timeframe_rule
from scalping.data.resample import resample
from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider
from scalping.data.validate import validate_bars
from scalping.features.build import FEATURE_COLUMNS, build_features
from scalping.features.context import htf_context

M30 = timedelta(minutes=30)
H4 = timedelta(hours=4)


def minute_bars(start: datetime, count: int, *, base: float = 2350.0) -> pl.DataFrame:
    """``count`` M1 bars from ``start``, one per minute, with a rising trend.

    ``high``/``low`` are a fixed distance from the close and the spread is
    constant, so a bucket's extrema and mean can be checked by hand.
    """
    ts = [start + timedelta(minutes=i) for i in range(count)]
    close = [base + 0.01 * i for i in range(count)]
    return pl.DataFrame(
        {
            "ts": ts,
            "open": [value - 0.01 for value in close],
            "high": [value + 0.10 for value in close],
            "low": [value - 0.10 for value in close],
            "close": close,
            "bid_close": [value - 0.02 for value in close],
            "ask_close": [value + 0.18 for value in close],
            "spread": [0.2] * count,
            "volume": [10] * count,
            "provider": ["synthetic"] * count,
        }
    )


def synthetic(start: datetime, end: datetime, *, seed: int = 3) -> pl.DataFrame:
    return SyntheticProvider(seed=seed).fetch("XAUUSD", start, end)


# --- bucket geometry -------------------------------------------------------


def test_m30_and_h4_labels_come_from_the_utc_epoch_grid() -> None:
    frame = minute_bars(datetime(2026, 1, 5, 10, 7, tzinfo=UTC), 180)
    m30 = resample(frame, timeframe_rule("M30"))
    h4 = resample(frame, timeframe_rule("H4"))

    # The first bucket starts at 10:00 even though the frame starts at 10:07:
    # labels are grid windows, never the first source timestamp.
    assert m30["ts"].to_list()[:2] == [
        datetime(2026, 1, 5, 10, 0, tzinfo=UTC),
        datetime(2026, 1, 5, 10, 30, tzinfo=UTC),
    ]
    assert all(ts.minute % 30 == 0 and ts.second == 0 for ts in m30["ts"])
    assert m30["ts"].is_sorted() and m30["ts"].n_unique() == m30.height

    # 4h buckets are 00/04/08/12/16/20 UTC, independent of the first bar.
    assert h4["ts"].to_list()[0] == datetime(2026, 1, 5, 8, 0, tzinfo=UTC)
    assert all(ts.hour % 4 == 0 and ts.minute == 0 for ts in h4["ts"])
    assert h4["open"][0] == frame["open"][0]


def test_bucket_ohlc_is_first_high_low_last_with_last_bar_quote() -> None:
    start = datetime(2026, 1, 5, 0, 0, tzinfo=UTC)
    frame = minute_bars(start, 240)
    row = resample(frame, "4h").row(0, named=True)

    assert row["ts"] == start
    assert row["open"] == frame["open"][0]
    assert row["high"] == frame["high"].max()
    assert row["low"] == frame["low"].min()
    assert row["close"] == frame["close"][-1]
    # bid/ask come from the last bar so the quote matches ``close``.
    assert row["bid_close"] == frame["bid_close"][-1]
    assert row["ask_close"] == frame["ask_close"][-1]
    assert row["volume"] == 2400
    assert row["spread_mean"] == pytest.approx(0.2)
    assert row["provider"] == "synthetic"


# --- no look-ahead ---------------------------------------------------------


def test_truncating_the_source_cannot_change_a_closed_bucket() -> None:
    frame = minute_bars(datetime(2026, 1, 5, 10, 0, tzinfo=UTC), 240)
    full = resample(frame, "30m")
    columns = ("open", "high", "low", "close", "bid_close", "ask_close", "volume")

    for cut in (1, 17, 60, 119, 200):
        part = resample(frame.head(cut), "30m")
        closed = part.head(part.height - 1)  # the final bucket is still forming
        np.testing.assert_array_equal(
            closed["ts"].to_numpy(), full["ts"].to_numpy()[: closed.height]
        )
        for column in columns:
            np.testing.assert_array_equal(
                closed[column].to_numpy(), full[column].to_numpy()[: closed.height]
            )
        # A partial bucket may use only the bars that have printed.
        assert part["close"][-1] == frame["close"][cut - 1]


def test_htf_join_never_uses_an_unclosed_h4_bucket_at_an_m30_bar() -> None:
    m1 = synthetic(datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 6, 23, 59, tzinfo=UTC))
    m30 = resample(m1, "30m")
    m5 = resample(m1, "5m")
    h4 = resample(m1, "4h")

    # Prefixes are positional: the H4 payload lands in the ``m15_*`` slots.
    joined = htf_context(m30, m5, h4)
    labels = h4["ts"].to_list()
    h4_ret = (h4["close"] / h4["close"].shift(1) - 1.0).to_numpy()

    def expected(decision: datetime) -> float:
        closed = [index for index, label in enumerate(labels) if label + H4 <= decision]
        assert closed, "the test window has at least one closed bucket"
        return float(h4_ret[closed[-1]])

    def value(day: int, hour: int, minute: int) -> float:
        row = joined.filter(pl.col("ts") == datetime(2026, 1, day, hour, minute, tzinfo=UTC))
        assert row.height == 1
        return float(row["m15_ret_1"][0])

    # 11:30 is inside the 08:00-11:59 bucket, which has not closed yet.
    assert value(5, 11, 30) == pytest.approx(expected(datetime(2026, 1, 5, 11, 30, tzinfo=UTC)))
    assert value(5, 11, 30) == pytest.approx(
        h4_ret[labels.index(datetime(2026, 1, 5, 4, 0, tzinfo=UTC))]
    )
    # At 12:00 the 08:00 bucket has closed and becomes the reference.
    assert value(5, 12, 0) == pytest.approx(
        h4_ret[labels.index(datetime(2026, 1, 5, 8, 0, tzinfo=UTC))]
    )
    # Still the same closed bucket at 15:30; the 12:00 bucket closes at 16:00.
    assert value(5, 15, 30) == value(5, 12, 0)
    assert value(5, 16, 0) == pytest.approx(
        h4_ret[labels.index(datetime(2026, 1, 5, 12, 0, tzinfo=UTC))]
    )
    assert value(5, 11, 30) != value(5, 12, 0)


def test_mtf_feature_columns_are_frozen_for_past_rows() -> None:
    m1 = synthetic(datetime(2026, 1, 5, 9, tzinfo=UTC), datetime(2026, 1, 5, 14, tzinfo=UTC))
    full = build_features(m1)
    indices = [
        index
        for index, name in enumerate(FEATURE_COLUMNS)
        if name.startswith(("m5_", "m15_", "htf_"))
    ]
    assert indices
    for t in (90, 120, 137, 180, 240, 290):
        part = build_features(m1.head(t + 1))
        np.testing.assert_allclose(
            full.X[t, indices],
            part.X[-1, indices],
            rtol=0,
            atol=1e-6,
            err_msg=f"higher-timeframe lookahead at {t}",
        )


def test_mtf_feature_columns_are_the_documented_m5_m15_ladder() -> None:
    prefixes = ("m5_", "m15_", "m30_", "h4_", "htf_")
    ladder = tuple(name for name in FEATURE_COLUMNS if name.startswith(prefixes))
    assert ladder == (
        "m5_ret_1",
        "m5_rsi_14",
        "m5_atr_norm",
        "m5_ema_gap_21",
        "m5_pos_range_20",
        "m15_ret_1",
        "m15_rsi_14",
        "m15_atr_norm",
        "m15_ema_gap_21",
        "m15_pos_range_20",
        "htf_bos",
        "htf_fvg_dist",
    )


# --- chained derivation ----------------------------------------------------


def test_chained_derivation_matches_the_m1_buckets_when_definitionally_equal() -> None:
    m1 = synthetic(datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 7, 23, 59, tzinfo=UTC))
    direct = resample(m1, "4h")
    assert direct.height == 18

    for rule in ("30m", "5m"):
        child = resample(m1, rule)
        # The derive CLI renames the intermediate mean back to ``spread``
        # before writing, which is what makes chained resampling possible.
        assert "spread_mean" in child.columns and "spread" not in child.columns
        with pytest.raises(ValueError, match="missing bar columns"):
            resample(child, "4h")
        chained = resample(child.rename({"spread_mean": "spread"}), "4h")
        joined = direct.join(chained, on="ts", suffix="_chained")
        assert joined.height == direct.height
        for column in (
            "open",
            "high",
            "low",
            "close",
            "bid_close",
            "ask_close",
            "volume",
            "provider",
        ):
            np.testing.assert_array_equal(
                joined[column].to_numpy(),
                joined[f"{column}_chained"].to_numpy(),
                err_msg=f"{rule} -> 4h differs in {column}",
            )
        # A mean of means equals the overall mean only on complete buckets.
        counts = m1.group_by_dynamic("ts", every="4h", closed="left", label="left").agg(
            pl.len().alias("bars")
        )
        complete = joined.join(
            counts.filter(pl.col("bars") == 240).select("ts"), on="ts", how="semi"
        ).select("spread_mean", "spread_mean_chained")
        assert complete.height == direct.height  # this window is fully complete
        np.testing.assert_allclose(
            complete["spread_mean"].to_numpy(),
            complete["spread_mean_chained"].to_numpy(),
            rtol=1e-12,
        )


# --- session and week boundaries -------------------------------------------


def test_xauusd_week_boundary_is_closed_friday_2100_to_sunday_2200() -> None:
    session = session_for("XAUUSD")
    assert datetime(2026, 1, 9).weekday() == 4  # Friday
    assert datetime(2026, 1, 11).weekday() == 6  # Sunday
    assert session.is_open(datetime(2026, 1, 9, 20, 59, tzinfo=UTC))
    assert not session.is_open(datetime(2026, 1, 9, 21, 0, tzinfo=UTC))
    assert not session.is_open(datetime(2026, 1, 11, 21, 59, tzinfo=UTC))
    assert session.is_open(datetime(2026, 1, 11, 22, 0, tzinfo=UTC))


def test_weekend_gap_keeps_friday_and_sunday_in_separate_h4_buckets() -> None:
    friday = minute_bars(datetime(2026, 1, 9, 20, 0, tzinfo=UTC), 60)
    sunday = minute_bars(datetime(2026, 1, 11, 22, 0, tzinfo=UTC), 120)
    h4 = resample(pl.concat([friday, sunday]), "4h")

    # No Saturday bucket, no merge across the closed window. Sunday 22:00 falls
    # into the grid bucket labelled 20:00, which closes at 24:00.
    assert h4["ts"].to_list() == [
        datetime(2026, 1, 9, 20, 0, tzinfo=UTC),
        datetime(2026, 1, 11, 20, 0, tzinfo=UTC),
    ]
    assert h4["volume"].to_list() == [600, 1200]
    assert h4["close"][0] == friday["close"][-1]
    assert h4["open"][1] == sunday["open"][0]
    assert h4["low"][0] == friday["low"].min()


# --- missing and duplicate source bars -------------------------------------


def test_duplicate_source_bars_are_reported_and_buckets_do_not_shift() -> None:
    start = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    frame = minute_bars(start, 60)
    duplicated = pl.concat([frame, frame.slice(5, 1)])

    m30 = resample(duplicated, "30m")
    assert m30["ts"].to_list() == resample(frame, "30m")["ts"].to_list()
    assert m30["volume"].to_list() == [310, 300]

    report = validate_bars(duplicated, symbol="XAUUSD", timeframe="M1", include_outliers=False)
    assert not report.ok
    assert "duplicate_timestamps" in {issue.code for issue in report.issues}


def test_store_dedupes_before_derivation(tmp_path: Path) -> None:
    frame = minute_bars(datetime(2026, 1, 5, 10, 0, tzinfo=UTC), 60)
    store = ParquetStore(tmp_path)
    store.write("XAUUSD", "M1", pl.concat([frame, frame.slice(5, 1)]))

    stored = store.read("XAUUSD", "M1")
    assert stored.height == frame.height
    assert resample(stored, "30m")["volume"].to_list() == [300, 300]


def test_missing_source_bars_are_reported_and_labels_stay_on_the_grid() -> None:
    start = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    frame = minute_bars(start, 60)
    # 10:03-10:07 are missing; 10:00 is short, 10:30 is complete.
    gapped = pl.concat([frame.head(3), frame.slice(8)])

    m30 = resample(gapped, "30m")
    assert m30["ts"].to_list() == [start, start + M30]
    assert m30["volume"].to_list() == [250, 300]
    assert m30["close"][0] == frame["close"][29]

    report = validate_bars(
        gapped, symbol="XAUUSD", timeframe="M1", gap_threshold_minutes=1, include_outliers=False
    )
    assert report.ok  # a gap is a warning, not a structural error
    assert "time_gaps" in {issue.code for issue in report.issues}

    # A whole missing 30m bucket leaves a hole, never a shifted label.
    assert resample(frame.slice(30), "30m")["ts"].to_list() == [start + M30]
