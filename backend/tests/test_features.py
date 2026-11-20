"""Feature engineering tests. The load bearing property is causality.

Every feature row may use only bars at or before its own timestamp. The
``test_no_lookahead_*`` tests are the contract: rebuilding the frame from a
truncated prefix must reproduce the same row, bit for bit, at every ``t``.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from scalping.data.resample import resample
from scalping.data.synthetic import SyntheticProvider
from scalping.features.build import FeatureFrame, build_features
from scalping.features.context import htf_context, session_flags
from scalping.features.indicators import atr, macd_hist, rsi
from scalping.features.liquidity import fvg, liquidity_sweep
from scalping.features.structure import market_structure, swings


def bars(closes: list[float]) -> pl.DataFrame:
    """Minimal M1 frame; range is 1.0 wide so ATR-based checks are exact."""
    n = len(closes)
    ts = [datetime(2026, 1, 5, 10, i, tzinfo=UTC) for i in range(n)]
    return pl.DataFrame(
        {
            "ts": ts,
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "bid_close": closes,
            "ask_close": [c + 0.2 for c in closes],
            "spread": [0.2] * n,
            "volume": [10] * n,
        }
    )


def ohlc_bars(highs: list[float], lows: list[float], closes: list[float]) -> pl.DataFrame:
    n = len(highs)
    ts = [datetime(2026, 1, 5, 10, i, tzinfo=UTC) for i in range(n)]
    return pl.DataFrame(
        {
            "ts": ts,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "bid_close": closes,
            "ask_close": closes,
            "spread": [0.02] * n,
            "volume": [10] * n,
        }
    )


@pytest.fixture(scope="module")
def day() -> pl.DataFrame:
    provider = SyntheticProvider(seed=3)
    return provider.fetch(
        "XAUUSD", datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 6, tzinfo=UTC)
    )


# --- brief's tests ---------------------------------------------------------


def test_atr_matches_manual_mean_range() -> None:
    df = bars([10.0] * 20)
    assert atr(df, 14)[-1] == pytest.approx(1.0)


def test_rsi_all_up_is_100() -> None:
    df = bars([10.0 + i for i in range(30)])
    assert rsi(df, 14)[-1] == pytest.approx(100.0, abs=1e-6)


def test_bos_after_higher_high() -> None:
    closes = [10, 11, 10.5, 12, 11.5, 13, 12.5, 14]
    out = market_structure(bars([float(c) for c in closes]))
    assert out["bos"].to_list()[-1] in (1, 0)
    assert out.filter(pl.col("bos") == 1).height >= 1


def test_fvg_detects_gap_up() -> None:
    df = bars([10.0, 10.2, 10.2])
    df = df.with_columns(
        pl.Series("high", [10.1, 10.9, 11.3]), pl.Series("low", [9.9, 10.3, 10.6])
    )
    out = fvg(df)
    # gap: low[i] > high[i-2] -> bar index 2
    assert out["fvg_up"].to_list()[2] is True


def test_no_lookahead_last_row_identical(day: pl.DataFrame) -> None:
    full = build_features(day)
    for t in (60, 61, 90, 200, 300, 500, 700, 999, 1100, 1300, day.height - 1):
        part = build_features(day.head(t + 1))
        assert np.allclose(full.X[t], part.X[-1], equal_nan=True), f"lookahead at {t}"


def test_feature_frame_shapes_and_dtype(day: pl.DataFrame) -> None:
    f = build_features(day)
    assert f.X.dtype == np.float32
    assert f.X.shape == (day.height, len(f.columns))
    assert f.valid_from >= 60


# --- frame contract --------------------------------------------------------


def test_feature_frame_fields(day: pl.DataFrame) -> None:
    f = build_features(day)
    assert isinstance(f, FeatureFrame)
    assert isinstance(f.columns, list)
    assert f.ohlc.shape == (day.height, 4)
    assert f.spread.shape == (day.height,)
    assert f.ts.shape == (day.height,)
    assert f.ts[0] == np.datetime64("2026-01-05T00:00:00.000000")
    assert np.issubdtype(f.ts.dtype, np.datetime64)


def test_column_count_is_between_40_and_60(day: pl.DataFrame) -> None:
    f = build_features(day)
    assert 40 <= len(f.columns) <= 60, len(f.columns)
    assert len(set(f.columns)) == len(f.columns), "duplicate column names"


def test_no_nan_from_valid_from(day: pl.DataFrame) -> None:
    f = build_features(day)
    assert f.valid_from == 60
    tail = f.X[f.valid_from :]
    assert np.isfinite(tail).all(), "non-finite value after valid_from"


def test_ohlc_and_spread_match_input(day: pl.DataFrame) -> None:
    f = build_features(day)
    flat = day.sort("ts")
    np.testing.assert_allclose(f.ohlc[:, 0], flat["open"].to_numpy(), rtol=0, atol=1e-9)
    np.testing.assert_allclose(f.ohlc[:, 3], flat["close"].to_numpy(), rtol=0, atol=1e-9)
    np.testing.assert_allclose(f.spread, flat["spread"].to_numpy(), rtol=0, atol=1e-9)
    assert (f.ohlc[:, 1] >= f.ohlc[:, 2]).all()


# --- causality helpers -----------------------------------------------------


def test_prefix_equality_across_every_100th_bar(day: pl.DataFrame) -> None:
    full = build_features(day)
    for t in range(60, day.height, 100):
        part = build_features(day.head(t + 1))
        np.testing.assert_allclose(
            full.X[t], part.X[-1], rtol=0, atol=1e-6, err_msg=f"lookahead at {t}"
        )
        np.testing.assert_allclose(full.ohlc[t], part.ohlc[-1], rtol=0, atol=0)
        assert full.ts[t] == part.ts[-1]


def test_every_bar_is_causal() -> None:
    # Dense check on a window that straddles the 15m and hour boundaries.
    window = SyntheticProvider(seed=17).fetch(
        "XAUUSD", datetime(2026, 1, 5, 9, 40, tzinfo=UTC), datetime(2026, 1, 5, 12, tzinfo=UTC)
    )
    full = build_features(window)
    for t in range(full.valid_from, window.height):
        part = build_features(window.head(t + 1))
        np.testing.assert_allclose(
            full.X[t], part.X[-1], rtol=0, atol=1e-6, err_msg=f"lookahead at {t}"
        )


def test_unsorted_input_is_sorted_before_building(day: pl.DataFrame) -> None:
    ordered = build_features(day.head(200))
    shuffled = build_features(day.head(200).sample(fraction=1.0, shuffle=True, seed=5))
    np.testing.assert_allclose(ordered.X, shuffled.X, rtol=0, atol=0)
    assert ordered.ts.tolist() == shuffled.ts.tolist()


def test_htf_join_uses_last_closed_bucket(day: pl.DataFrame) -> None:
    f = build_features(day)
    col = f.columns.index("m5_ret_1")
    m5 = resample(day, "5m")
    current = m5.filter(pl.col("ts") == datetime(2026, 1, 5, 12, 0, tzinfo=UTC))["close"][0]
    previous = m5.filter(pl.col("ts") == datetime(2026, 1, 5, 11, 55, tzinfo=UTC))["close"][0]
    row = day.with_row_index().filter(
        pl.col("ts") == datetime(2026, 1, 5, 12, 7, tzinfo=UTC)
    )["index"][0]
    # 12:07 sits inside the 12:05..12:09 bucket; only the 12:00 bucket is closed.
    assert f.X[row, col] == pytest.approx(current / previous - 1.0, rel=1e-5)


def test_htf_context_switches_only_when_a_bucket_closes(day: pl.DataFrame) -> None:
    m5, m15 = resample(day, "5m"), resample(day, "15m")
    out = htf_context(day, m5, m15)
    assert out.height == day.height

    def m15_close(hour: int, minute: int) -> float:
        return m15.filter(pl.col("ts") == datetime(2026, 1, 5, hour, minute, tzinfo=UTC))[
            "close"
        ][0]

    def value(hour: int, minute: int) -> float:
        return out.filter(pl.col("ts") == datetime(2026, 1, 5, hour, minute, tzinfo=UTC))[
            "m15_ret_1"
        ][0]

    # Inside 12:00..12:14 the only closed 15m bucket is the one starting 11:45.
    assert value(12, 7) == pytest.approx(m15_close(11, 45) / m15_close(11, 30) - 1.0, rel=1e-5)
    # At 12:15 that bucket closes and becomes the reference.
    assert value(12, 15) == pytest.approx(m15_close(12, 0) / m15_close(11, 45) - 1.0, rel=1e-5)


# --- structure -------------------------------------------------------------


def test_swing_high_is_labelled_at_the_confirmation_bar() -> None:
    highs = [1.0, 2, 3, 4, 5, 9, 4, 3, 2, 1, 1, 1]
    df = ohlc_bars(highs, [h - 0.5 for h in highs], highs)
    sh, _ = swings(df, k=2)
    flags = sh.to_list()
    assert flags[5] is False, "peak must not be labelled before confirmation"
    assert flags[7] is True, "peak at 5 confirmed at 5 + 2"
    assert flags.count(True) == 1


def test_swing_low_is_labelled_at_the_confirmation_bar() -> None:
    lows = [9.0, 8, 7, 6, 5, 1, 6, 7, 8, 9, 9, 9]
    df = ohlc_bars([low + 0.5 for low in lows], lows, lows)
    _, sl = swings(df, k=2)
    flags = sl.to_list()
    assert flags[5] is False
    assert flags[7] is True


def test_market_structure_dtypes_and_domains() -> None:
    closes = [10, 11, 10.5, 12, 11.5, 13, 12.5, 14, 13, 12, 11, 10, 9, 10, 11, 12]
    out = market_structure(bars([float(c) for c in closes]))
    for name in ("trend", "bos", "choch"):
        assert out[name].dtype == pl.Int8
        assert set(out[name].to_list()) <= {-1, 0, 1}
    assert out.height == len(closes)


def test_structure_finds_a_down_break() -> None:
    closes = [10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 9, 8, 7, 6, 5, 4]
    out = market_structure(bars([float(c) for c in closes]))
    assert (out["bos"] == -1).sum() >= 1
    assert (out["trend"] == -1).sum() >= 1


# --- liquidity -------------------------------------------------------------


def test_fvg_detects_gap_down() -> None:
    df = bars([11.0, 10.8, 10.8])
    df = df.with_columns(
        pl.Series("high", [11.1, 10.4, 10.1]), pl.Series("low", [10.9, 10.0, 9.7])
    )
    out = fvg(df)
    assert out["fvg_down"].to_list()[2] is True
    assert out["fvg_up"].to_list()[2] is False


def test_fvg_distance_is_in_atr_units_and_filled_gaps_drop_out() -> None:
    # One up gap at index 2: zone [high[0], low[2]] = [10.0, 10.4].
    highs = [10.0, 10.55, 10.7, 10.75, 10.7, 10.7, 10.7, 10.7]
    lows = [9.5, 10.35, 10.4, 10.55, 10.55, 10.55, 10.55, 9.2]
    closes = [9.7, 10.4, 10.6, 10.6, 10.6, 10.6, 10.6, 9.6]
    df = ohlc_bars(highs, lows, closes)
    out = fvg(df)
    assert out["fvg_up"].to_list()[2] is True
    dist = out["fvg_dist"].to_list()
    assert dist[3] > 0.0, "price sits above the unfilled gap"
    assert dist[6] > 0.0
    assert dist[7] == 0.0, "gap filled when price trades back through it"


def test_fvg_distance_measured_from_gap_edge() -> None:
    highs = [10.0, 10.55, 10.7, 10.75, 10.7, 10.7, 10.7, 10.7]
    lows = [9.5, 10.35, 10.4, 10.55, 10.55, 10.55, 10.55, 9.2]
    closes = [9.7, 10.4, 10.6, 10.6, 10.6, 10.6, 10.6, 9.6]
    df = ohlc_bars(highs, lows, closes)
    out = fvg(df)
    atr_at_6 = atr(df, 14)[6]
    expected = (lows[6] - lows[2]) / atr_at_6  # gap top is low[gap bar]
    assert out["fvg_dist"].to_list()[6] == pytest.approx(expected, rel=1e-6)


def test_liquidity_sweep_flags_wick_rejection() -> None:
    closes = [10.0] * 21 + [10.05]
    highs = [10.1] * 21 + [10.9]
    lows = [9.9] * 21 + [9.95]
    df = ohlc_bars(highs, lows, closes)
    sweep = liquidity_sweep(df, lookback=20)
    assert sweep.to_list()[:-1] == [False] * (len(df) - 1)
    assert sweep[-1] is True


def test_liquidity_sweep_ignores_plain_breakout() -> None:
    closes = [10.0] * 21 + [10.85]
    highs = [10.1] * 21 + [10.9]
    lows = [9.9] * 21 + [10.8]
    sweep = liquidity_sweep(ohlc_bars(highs, lows, closes), lookback=20)
    assert sweep[-1] is False


# --- sessions --------------------------------------------------------------


def test_session_flags_are_utc_and_mutually_exclusive() -> None:
    ts = pl.Series(
        "ts",
        [
            datetime(2026, 1, 5, 2, 0, tzinfo=UTC),
            datetime(2026, 1, 5, 9, 0, tzinfo=UTC),
            datetime(2026, 1, 5, 13, 45, tzinfo=UTC),
            datetime(2026, 1, 5, 18, 0, tzinfo=UTC),
        ],
        dtype=pl.Datetime("us", "UTC"),
    )
    flags = session_flags(ts)
    assert flags.columns == ["asia", "london", "ny", "ny_open_30"]
    assert flags.to_dicts() == [
        {"asia": True, "london": False, "ny": False, "ny_open_30": False},
        {"asia": False, "london": True, "ny": False, "ny_open_30": False},
        {"asia": False, "london": False, "ny": True, "ny_open_30": True},
        {"asia": False, "london": False, "ny": True, "ny_open_30": False},
    ]
    assert (flags.select("asia", "london", "ny").sum_horizontal().to_numpy() <= 1).all()


# --- indicators ------------------------------------------------------------


def test_macd_hist_is_causal_and_finite(day: pl.DataFrame) -> None:
    h = macd_hist(day)
    assert h.len() == day.height
    assert np.isfinite(h.to_numpy()).all()
    assert h[399] == pytest.approx(macd_hist(day.head(400))[-1])


def test_atr_uses_only_past_true_range() -> None:
    df = bars([10.0] * 40)
    a = atr(df, 14)
    assert a[13] == pytest.approx(1.0)
    # A late spike must not change earlier ATR values.
    spiked = df.with_columns(
        pl.when(pl.int_range(0, df.height) == 30)
        .then(pl.lit(20.0))
        .otherwise(pl.col("high"))
        .alias("high")
    )
    assert atr(spiked, 14)[20] == atr(df, 14)[20]


# --- performance -----------------------------------------------------------


def test_three_days_build_under_a_second() -> None:
    df = SyntheticProvider(seed=11).fetch(
        "XAUUSD", datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 8, tzinfo=UTC)
    )
    assert df.height > 4000
    started = time.perf_counter()
    f = build_features(df)
    elapsed = time.perf_counter() - started
    assert f.X.shape[0] == df.height
    assert elapsed < 1.0, f"build took {elapsed:.3f}s"
