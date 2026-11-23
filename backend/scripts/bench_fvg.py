"""Benchmark the fair-value-gap scan, old row loop vs the block-indexed scan.

The old implementation is the original per-row loop, kept here as the oracle.
The new one is ``scalping.features.liquidity.fvg``. Both are run on the same
frame and their outputs are compared bit for bit.

The script is manual: it needs ``data-ticks5-store`` and is not collected by
pytest, so CI never depends on the data store.

Examples
--------
    uv run python scripts/bench_fvg.py
    uv run python scripts/bench_fvg.py --timeframe H1
    uv run python scripts/bench_fvg.py --skip-old          # new timing only
    uv run python scripts/bench_fvg.py --build             # full build_features
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

from scalping.data.store import ParquetStore
from scalping.features.build import build_features
from scalping.features.indicators import atr
from scalping.features.liquidity import _fvg_scan, fvg

#: Like-for-like C port of the original loop, measured by the benchmark report.
C_PORT_SECONDS = 1.694


def _reference(
    highs: np.ndarray, lows: np.ndarray, atr_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Original O(n * open gaps) row loop."""
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
        for _bottom, top in up_gaps:
            candidate = (low - top) / scale
            if best == 0.0 or abs(candidate) < abs(best):
                best = candidate
        for bottom, _top in down_gaps:
            candidate = (high - bottom) / scale
            if best == 0.0 or abs(candidate) < abs(best):
                best = candidate
        distance[i] = best

    return up_flag, down_flag, distance


def _find_store(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    here = Path(__file__).resolve()
    candidates = [Path.cwd() / "data-ticks5-store", Path.cwd().parent / "data-ticks5-store"]
    for depth in (2, 4):
        if len(here.parents) > depth:
            candidates.append(here.parents[depth] / "data-ticks5-store")
    for candidate in candidates:
        if (candidate / "XAUUSD" / "M1").is_dir():
            return candidate
    raise SystemExit("data-ticks5-store not found; pass --store")


def _cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _peak_rss_mb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", help="data-ticks5-store root")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M1")
    parser.add_argument("--rows", type=int, default=0, help="limit rows (0 = all)")
    parser.add_argument("--skip-old", action="store_true", help="skip the row loop")
    parser.add_argument("--build", action="store_true", help="also time build_features")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    store = ParquetStore(_find_store(args.store))
    frame = store.read(args.symbol, args.timeframe)
    if args.rows:
        frame = frame.head(args.rows)
    highs = frame["high"].to_numpy().astype(np.float64)
    lows = frame["low"].to_numpy().astype(np.float64)
    atr_values = atr(frame, 14).to_numpy().astype(np.float64)
    rows = len(highs)
    print(f"{args.symbol} {args.timeframe}: {rows:,} rows")

    old: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    old_wall = 0.0
    if not args.skip_old:
        cpu = _cpu_seconds()
        start = time.perf_counter()
        old = _reference(highs, lows, atr_values)
        old_wall = time.perf_counter() - start
        print(f"  old row loop        {old_wall:9.3f} s wall  {_cpu_seconds() - cpu:9.3f} s cpu")

    best_wall = float("inf")
    best_cpu = float("inf")
    new: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for _ in range(args.repeat):
        cpu = _cpu_seconds()
        start = time.perf_counter()
        new = _fvg_scan(highs, lows, atr_values)
        wall = time.perf_counter() - start
        best_wall = min(best_wall, wall)
        best_cpu = min(best_cpu, _cpu_seconds() - cpu)
    print(
        f"  new block index     {best_wall:9.3f} s wall  {best_cpu:9.3f} s cpu"
        f"  ({rows / best_wall / 1e6:.2f} M rows/s)"
    )

    # Public entry point, including the polars frame construction.
    start = time.perf_counter()
    public = fvg(frame)
    public_wall = time.perf_counter() - start
    print(f"  public fvg(frame)   {public_wall:9.3f} s wall")
    assert new is not None
    for name, series, array in (
        ("fvg_up", public["fvg_up"], new[0]),
        ("fvg_down", public["fvg_down"], new[1]),
        ("fvg_dist", public["fvg_dist"], new[2]),
    ):
        if not np.array_equal(series.to_numpy(), array):
            raise SystemExit(f"public {name} drifted from the scan")

    if old is not None:
        equal = all(np.array_equal(a, b) for a, b in zip(old, new))
        print(f"  bit-identical       {'yes' if equal else 'NO'}")
        if not equal:
            for name, a, b in zip(("fvg_up", "fvg_down", "fvg_dist"), old, new):
                bad = int((np.asarray(a) != np.asarray(b)).sum())
                print(f"    {name}: {bad} differing rows")
        print(f"  speedup             {old_wall / best_wall:8.1f}x")
    print(f"  C port (reference)  {C_PORT_SECONDS:9.3f} s  (benchmark artifact)")

    if args.build:
        cpu = _cpu_seconds()
        start = time.perf_counter()
        built = build_features(frame)
        wall = time.perf_counter() - start
        print(
            f"  build_features      {wall:9.3f} s wall  {_cpu_seconds() - cpu:9.3f} s cpu"
            f"  {_peak_rss_mb():8.1f} MB peak RSS  X={built.X.shape}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
