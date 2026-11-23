"""Data quality: robust outliers and cross-source reconciliation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from scalping.data.quality import (
    OutlierConfig,
    ReconcileConfig,
    detect_bar_outliers,
    reconcile_bars,
    robust_z,
)

START = datetime(2026, 1, 5, tzinfo=UTC)


def bars(n: int = 240, seed: int = 4) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.08, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.full(n, 0.2)
    return pl.DataFrame(
        {
            "ts": [START + timedelta(minutes=i) for i in range(n)],
            "open": open_,
            "high": np.maximum(open_, close) + 0.05,
            "low": np.minimum(open_, close) - 0.05,
            "close": close,
            "bid_close": close,
            "ask_close": close + spread,
            "spread": spread,
            "volume": rng.integers(10, 100, n).astype(np.int64),
        }
    )


def test_clean_frame_has_no_outlier_warnings() -> None:
    assert detect_bar_outliers(bars()) == ()


def test_return_and_volume_spikes_are_reported() -> None:
    frame = bars()
    close = frame["close"].to_numpy()
    close = close.copy()
    close[150] = close[149] * 1.5
    volume = frame["volume"].to_numpy().copy()
    volume[200] = 1_000_000
    mutated = frame.with_columns(
        pl.Series("close", close),
        pl.Series("volume", volume.astype(np.int64)),
    )

    issues = {issue.code: issue for issue in detect_bar_outliers(mutated)}

    assert issues["price_outliers"].severity == "warning"
    assert issues["price_outliers"].count >= 1
    assert issues["price_outliers"].sample
    assert issues["volume_outliers"].count >= 1
    assert "01-05 02:30" in issues["price_outliers"].sample[0]


def test_short_frame_is_skipped() -> None:
    assert detect_bar_outliers(bars(10)) == ()


def test_robust_z_handles_degenerate_input() -> None:
    assert np.allclose(robust_z(np.ones(10)), 0.0)
    assert robust_z(np.array([1.0, 2.0, 3.0])).shape == (3,)
    with pytest.raises(ValueError, match="min_rows"):
        OutlierConfig(min_rows=1)
    with pytest.raises(ValueError, match="z thresholds"):
        OutlierConfig(return_z=0.0)


def test_identical_sources_reconcile_cleanly() -> None:
    frame = bars(60)
    report = reconcile_bars(frame, frame, symbol="XAUUSD", timeframe="M1")

    assert report.ok is True
    assert report.matched == 60
    assert report.issues == ()


def test_close_and_spread_mismatches_are_errors() -> None:
    left = bars(60)
    close = left["close"].to_numpy().copy()
    close[10] += 1.0
    spread = left["spread"].to_numpy().copy()
    spread[20] = 0.5
    right = left.with_columns(
        pl.Series("close", close),
        pl.Series("spread", spread),
    )

    report = reconcile_bars(
        left, right, config=ReconcileConfig(price_tolerance=0.01, spread_tolerance=0.01)
    )

    codes = {issue.code: issue for issue in report.issues}
    assert report.ok is False
    assert codes["close_mismatch"].count == 1
    assert codes["spread_mismatch"].count == 1
    assert codes["close_mismatch"].sample

    loose = reconcile_bars(
        left, right, config=ReconcileConfig(price_tolerance=2.0, spread_tolerance=1.0)
    )
    assert loose.ok is True


def test_coverage_gaps_are_warnings() -> None:
    left = bars(60)
    right = left.head(55)

    report = reconcile_bars(left, right, config=ReconcileConfig(price_tolerance=0.01))

    assert report.ok is True
    assert report.matched == 55
    assert [(issue.code, issue.count) for issue in report.issues] == [("only_in_primary", 5)]


def test_duplicate_timestamps_are_errors() -> None:
    left = bars(60)
    duplicated = pl.concat([left, left.head(1)])

    report = reconcile_bars(left, duplicated)

    assert report.ok is False
    assert report.issues[0].code == "duplicate_timestamps"
    assert report.issues[0].severity == "error"


def test_reconcile_config_is_validated() -> None:
    with pytest.raises(ValueError, match="tolerances"):
        ReconcileConfig(price_tolerance=-1.0)
    with pytest.raises(ValueError, match="sample"):
        ReconcileConfig(sample=0)
