"""Bit-exactness tests for the fair-value-gap scan.

``_fvg_reference`` is the original per-row implementation, kept verbatim as the
oracle. The shipped :func:`scalping.features.liquidity.fvg` must reproduce it
exactly, including its tie-breaking and the ``best == 0.0`` restart on a gap
edge that coincides with the current price.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scalping.data.store import ParquetStore
from scalping.features.indicators import atr
from scalping.features.liquidity import _fvg_scan, fvg

#: Row cap for the real-data M1 slice; the full 1.77M-row frame is benchmarked
#: separately (``scripts/bench_fvg.py``), not in the test suite.
_M1_SLICE = 60_000


def _reference(
    highs: np.ndarray, lows: np.ndarray, atr_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Original row loop, unchanged."""
    height = len(highs)
    up_flag = np.zeros(height, dtype=bool)
    down_flag = np.zeros(height, dtype=bool)
    distance = np.zeros(height, dtype=np.float64)
    up_gaps: list[tuple[float, float]] = []
    down_gaps: list[tuple[float, float]] = []

    for i in range(height):
        low = lows[i]
        high = highs[i]
        up_gaps = [gap for gap in up_gaps if low > gap[0]]
        down_gaps = [gap for gap in down_gaps if high < gap[1]]

        if i >= 2:
            if low > highs[i - 2]:
                up_flag[i] = True
                up_gaps.append((highs[i - 2], low))
            if high < lows[i - 2]:
                down_flag[i] = True
                down_gaps.append((highs[i], lows[i - 2]))

        scale = atr_values[i]
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        best = 0.0
        for bottom, top in up_gaps:
            candidate = (low - top) / scale
            if best == 0.0 or abs(candidate) < abs(best):
                best = candidate
        for bottom, top in down_gaps:
            candidate = (high - bottom) / scale
            if best == 0.0 or abs(candidate) < abs(best):
                best = candidate
        distance[i] = best

    return up_flag, down_flag, distance


def _frame(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray | None = None) -> pl.DataFrame:
    n = len(highs)
    closes = (highs + lows) / 2.0 if closes is None else closes
    start = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    return pl.DataFrame(
        {
            "ts": [start + timedelta(minutes=i) for i in range(n)],
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "bid_close": closes,
            "ask_close": closes,
            "spread": np.full(n, 0.02),
            "volume": np.full(n, 10.0),
        }
    )


def _random_ohlc(
    rng: np.random.Generator, n: int, grid: float | None
) -> tuple[np.ndarray, np.ndarray]:
    base = 100.0 + rng.normal(0.0, 1.0, n).cumsum() * 0.1
    span = rng.uniform(0.02, 1.0, n)
    highs = base + span * rng.uniform(0.2, 1.0, n)
    lows = base - span * rng.uniform(0.2, 1.0, n)
    if grid is not None:
        highs = np.round(highs / grid) * grid
        lows = np.round(lows / grid) * grid
    # Force exact repeats often enough to exercise the zero-restart branch.
    if n > 8 and rng.random() < 0.5:
        for step, shift in ((7, 1), (5, 2)):
            target = np.arange(0, n, step)
            source = np.arange(shift, n, step)
            count = min(len(target), len(source))
            highs[target[:count]] = highs[source[:count]]
            lows[target[:count]] = lows[source[:count]]
    return highs, lows


def _assert_equal(
    highs: np.ndarray, lows: np.ndarray, atr_values: np.ndarray, context: str
) -> None:
    want = _reference(highs, lows, atr_values)
    got = _fvg_scan(highs, lows, atr_values)
    for name, expected, actual in zip(("fvg_up", "fvg_down", "fvg_dist"), want, got):
        if not np.array_equal(expected, actual):
            bad = np.flatnonzero(np.asarray(expected) != np.asarray(actual))
            raise AssertionError(
                f"{context}: {name} differs at {bad[:10].tolist()} "
                f"(expected {np.asarray(expected)[bad[:10]]}, "
                f"got {np.asarray(actual)[bad[:10]]})"
            )


def test_random_frames_match_reference() -> None:
    for seed in range(500):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(0, 400))
        grid = None if seed % 4 else (0.05 if seed % 8 else 0.01)
        if n < 3:
            highs = np.zeros(n)
            lows = np.zeros(n)
        else:
            highs, lows = _random_ohlc(rng, n, grid)
        atr_values = rng.uniform(0.0, 2.0, n)
        atr_values[rng.random(n) < 0.05] = 0.0
        atr_values[rng.random(n) < 0.02] = np.nan
        atr_values[rng.random(n) < 0.02] = np.inf
        _assert_equal(highs, lows, atr_values, f"seed {seed} n {n} grid {grid}")


def test_flat_and_monotonic_series_match_reference() -> None:
    cases = {
        "flat": (np.full(50, 10.0), np.full(50, 10.0)),
        "rising": (np.arange(50.0) + 10.5, np.arange(50.0) + 9.5),
        "falling": (50.0 - np.arange(50.0), 49.0 - np.arange(50.0)),
        "zero_atr": (np.full(20, 3.0), np.full(20, 3.0)),
    }
    for name, (highs, lows) in cases.items():
        _assert_equal(highs, lows, np.zeros(len(highs)), name)


def test_existing_fixtures_match_reference() -> None:
    # The three fixtures already asserted in ``test_features.py``.
    highs = [10.1, 10.9, 11.3]
    lows = [9.9, 10.3, 10.6]
    _assert_equal(np.array(highs), np.array(lows), np.ones(3), "gap up")

    highs = [11.1, 10.4, 10.1]
    lows = [10.9, 10.0, 9.7]
    _assert_equal(np.array(highs), np.array(lows), np.ones(3), "gap down")

    highs = np.array([10.0, 10.55, 10.7, 10.75, 10.7, 10.7, 10.7, 10.7])
    lows = np.array([9.5, 10.35, 10.4, 10.55, 10.55, 10.55, 10.55, 9.2])
    closes = np.array([9.7, 10.4, 10.6, 10.6, 10.6, 10.6, 10.6, 9.6])
    frame = _frame(highs, lows, closes)
    _assert_equal(highs, lows, atr(frame, 14).to_numpy(), "fill fixture")


def test_public_fvg_matches_reference_on_frame() -> None:
    rng = np.random.default_rng(7)
    highs, lows = _random_ohlc(rng, 300, 0.05)
    frame = _frame(highs, lows)
    out = fvg(frame)
    want = _reference(highs, lows, atr(frame, 14).to_numpy())
    assert np.array_equal(out["fvg_up"].to_numpy(), want[0])
    assert np.array_equal(out["fvg_down"].to_numpy(), want[1])
    assert np.array_equal(out["fvg_dist"].to_numpy(), want[2])
    assert out["fvg_up"].dtype == pl.Boolean
    assert out["fvg_dist"].dtype == pl.Float64


def test_multi_block_random_frames_match_reference() -> None:
    # Every size crosses at least one 512-bar block boundary; 4097 and 5000
    # cross eight and nine. The coarse grids and forced repeats put gaps and
    # zero restarts on both sides of the boundaries.
    for n in (511, 512, 513, 1023, 1024, 1025, 4097, 5000):
        rng = np.random.default_rng(1000 + n)
        grid = 0.05 if n % 2 else 0.01
        highs, lows = _random_ohlc(rng, n, grid)
        atr_values = rng.uniform(0.1, 2.0, n)
        _assert_equal(highs, lows, atr_values, f"multi-block n {n}")


def test_gap_spans_multiple_blocks() -> None:
    # The up gap created at bar 2 stays open for the whole 2,000-bar frame, so
    # it is a static entry in blocks 1, 2 and 3.
    n = 2000
    highs = np.full(n, 101.4)
    lows = np.full(n, 100.6)
    highs[0], lows[0] = 100.0, 99.5
    highs[1], lows[1] = 100.9, 99.8
    highs[2], lows[2] = 101.5, 101.0
    atr_values = np.ones(n)
    _assert_equal(highs, lows, atr_values, "spanning gap")
    up, _, distance = _fvg_scan(highs, lows, atr_values)
    assert up[2]
    assert np.any(distance[3:] != 0.0)


def test_zero_restart_just_past_a_block_edge() -> None:
    # Up gap at bar 500 (inside block 0). Bar 513, the first bar after the
    # 512-boundary, revisits its top while a down gap is still open, so the
    # restart must return the down candidate, not 0.0.
    n = 1100
    highs = np.full(n, 102.8)
    lows = np.full(n, 102.6)
    highs[:400] = 100.0
    lows[:400] = 99.0
    highs[400], lows[400] = 104.0, 103.0
    highs[401], lows[401] = 102.8, 102.2
    highs[402:500] = 102.5
    lows[402:500] = 102.0
    atr_values = np.ones(n)
    _assert_equal(highs, lows, atr_values, "zero at block edge")
    up, _, distance = _fvg_scan(highs, lows, atr_values)
    assert up[500] and not up[513]
    assert distance[513] == pytest.approx(0.3)
    assert distance[513] != 0.0


def test_old_up_edge_zero_restarts_scan() -> None:
    # Bar 11 revisits the top of an up gap created at bar 9, so the running
    # best restarts and the result comes from the gaps after that zero.
    highs = np.array(
        [100.4, 100.9, 100.4, 100.7, 100.7, 100.6, 100.9, 100.9, 100.9, 101.2, 101.3, 101.2]
    )
    lows = np.array(
        [100.0, 99.5, 100.2, 99.6, 100.1, 100.4, 100.2, 100.7, 100.8, 101.1, 101.0, 101.1]
    )
    _assert_equal(highs, lows, np.ones(len(highs)), "old up zero")
    distance = _fvg_scan(highs, lows, np.ones(len(highs)))[2]
    assert distance[11] == pytest.approx(0.1)
    assert distance[11] != 0.0


def test_down_gap_creation_bar_has_zero_distance() -> None:
    rng = np.random.default_rng(11)
    for _ in range(50):
        highs, lows = _random_ohlc(rng, 200, 0.05)
        atr_values = rng.uniform(0.1, 2.0, 200)
        _, down, distance = _fvg_scan(highs, lows, atr_values)
        assert np.all(distance[down] == 0.0), "a fresh down gap is the last candidate"


def _store_root() -> Path | None:
    here = Path(__file__).resolve()
    candidates = [Path.cwd() / "data-ticks5-store", Path.cwd().parent / "data-ticks5-store"]
    for depth in (2, 4):
        if len(here.parents) > depth:
            candidates.append(here.parents[depth] / "data-ticks5-store")
    for candidate in candidates:
        if (candidate / "XAUUSD" / "M1").is_dir():
            return candidate
    return None


_STORE = _store_root()


@pytest.mark.skipif(_STORE is None, reason="data-ticks5-store is not available")
def test_real_h1_matches_reference() -> None:
    frame = ParquetStore(_STORE).read("XAUUSD", "H1")
    highs = frame["high"].to_numpy().astype(np.float64)
    lows = frame["low"].to_numpy().astype(np.float64)
    _assert_equal(highs, lows, atr(frame, 14).to_numpy().astype(np.float64), "H1")


@pytest.mark.skipif(_STORE is None, reason="data-ticks5-store is not available")
def test_real_m1_slice_matches_reference() -> None:
    frame = ParquetStore(_STORE).read("XAUUSD", "M1").head(_M1_SLICE)
    highs = frame["high"].to_numpy().astype(np.float64)
    lows = frame["low"].to_numpy().astype(np.float64)
    _assert_equal(highs, lows, atr(frame, 14).to_numpy().astype(np.float64), "M1 slice")
