"""Liquidity features: fair value gaps and sweep rejections.

Fair value gaps
---------------
An up gap exists at bar ``i`` when ``low[i] > high[i-2]``: the middle bar jumped
so hard that the range between ``high[i-2]`` and ``low[i]`` was never traded.
The down gap is the mirror, ``high[i] < low[i-2]``. The zone is
``[high[i-2], low[i]]`` (up) or ``[high[i], low[i-2]]`` (down).

``fvg_dist`` is the distance from the current bar to the nearest **unfilled**
gap in ATR units. A gap stays open until price trades through its far edge
(a later ``low`` at or below ``high[i-2]`` for an up gap, mirrored for a down
gap). The sign follows the side: positive above an up gap, negative below a
down gap. ``0.0`` means no open gap.

The scan is strictly backward looking: gaps are created and tested against bars
that have already printed.
"""

from __future__ import annotations

import heapq

import numpy as np
import polars as pl

from scalping.features.indicators import atr

__all__ = ["fvg", "liquidity_sweep"]

#: Bars per block in the nearest-open-gap index. The scan is O(n log n) in the
#: number of gaps, independent of how many gaps are open at once.
_BLOCK: int = 512


def _gap_deaths(
    creations: np.ndarray,
    values: np.ndarray,
    thresholds: np.ndarray,
    height: int,
    *,
    below: bool,
) -> np.ndarray:
    """Index of the first bar that closes each gap.

    A gap created at ``c`` with threshold ``t`` dies at the first ``j > c``
    where ``values[j] <= t`` (up gap, ``below=True``) or ``values[j] >= t``
    (down gap). A max/min-heap keyed by the threshold pops every gap the bar
    kills, so each gap is pushed and popped once.
    """
    deaths = np.full(len(creations), height, dtype=np.int64)
    if len(creations) == 0:
        return deaths

    heap: list[tuple[float, int]] = []
    ptr = 0
    bars = creations.tolist()
    limits = thresholds.tolist()
    series = values.tolist()
    if below:
        for j in range(height):
            value = series[j]
            while heap and -heap[0][0] >= value:
                _, gap = heapq.heappop(heap)
                deaths[gap] = j
            while ptr < len(bars) and bars[ptr] == j:
                heapq.heappush(heap, (-limits[ptr], ptr))
                ptr += 1
    else:
        for j in range(height):
            value = series[j]
            while heap and heap[0][0] <= value:
                _, gap = heapq.heappop(heap)
                deaths[gap] = j
            while ptr < len(bars) and bars[ptr] == j:
                heapq.heappush(heap, (limits[ptr], ptr))
                ptr += 1
    return deaths


def _nearest_active(
    height: int,
    creations: np.ndarray,
    deaths: np.ndarray,
    edges: np.ndarray,
    queries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest open gap edge for every bar.

    ``edges[c]`` is the far edge of the gap created at ``c`` (the top for an up
    gap, the bottom for a down gap) and ``queries[i]`` is the price the distance
    is measured from. A gap is open on bar ``i`` when ``creations <= i <
    deaths``. Returns ``(abs distance, edge value, creation index)``; ties go to
    the oldest gap, matching the insertion order of the row loop.

    The bar axis is split into blocks. A gap that covers a whole block is put
    in that block's sorted edge array, so all bars of the block are answered by
    one ``searchsorted``. Gaps only partly inside a block are expanded over
    their (short) slice and merged into the block result.
    """
    gaps = len(creations)
    abs_best = np.full(height, np.inf)
    edge_best = np.zeros(height)
    creation_best = np.full(height, -1, dtype=np.int64)
    if gaps == 0:
        return abs_best, edge_best, creation_best

    blocks = (height + _BLOCK - 1) // _BLOCK
    start_block = creations // _BLOCK
    end_block = (deaths - 1) // _BLOCK
    gap_index = np.arange(gaps)

    # --- gaps that cover whole blocks -------------------------------------
    full_counts = np.maximum(0, end_block - start_block - 1)
    total_full = int(full_counts.sum())
    if total_full:
        rep = np.repeat(gap_index, full_counts)
        offsets = np.repeat(np.cumsum(full_counts) - full_counts, full_counts)
        block_ids = (
            np.arange(total_full, dtype=np.int64)
            - offsets
            + np.repeat(start_block + 1, full_counts)
        )
        values = edges[rep]
        origins = creations[rep]
        order = np.lexsort((values, block_ids))
        block_ids = block_ids[order]
        values = values[order]
        origins = origins[order]
        counts = np.bincount(block_ids, minlength=blocks)
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        for block in np.flatnonzero(counts):
            lo = starts[block]
            hi = lo + counts[block]
            block_values = values[lo:hi]
            block_origins = origins[lo:hi]
            unique = np.r_[True, block_values[1:] != block_values[:-1]]
            uniq_values = block_values[unique]
            uniq_starts = np.flatnonzero(unique)
            uniq_origins = np.minimum.reduceat(block_origins, uniq_starts)

            first = int(block) * _BLOCK
            last = min(first + _BLOCK, height)
            query = queries[first:last]
            size = last - first
            pos = np.searchsorted(uniq_values, query, side="left")

            left_dist = np.full(size, np.inf)
            left_value = np.zeros(size)
            left_origin = np.full(size, -1, dtype=np.int64)
            mask = pos > 0
            left_dist[mask] = query[mask] - uniq_values[pos[mask] - 1]
            left_value[mask] = uniq_values[pos[mask] - 1]
            left_origin[mask] = uniq_origins[pos[mask] - 1]

            right_dist = np.full(size, np.inf)
            right_value = np.zeros(size)
            right_origin = np.full(size, -1, dtype=np.int64)
            mask = pos < len(uniq_values)
            right_dist[mask] = uniq_values[pos[mask]] - query[mask]
            right_value[mask] = uniq_values[pos[mask]]
            right_origin[mask] = uniq_origins[pos[mask]]

            right = (right_dist < left_dist) | (
                (right_dist == left_dist) & (right_origin < left_origin)
            )
            abs_best[first:last] = np.where(right, right_dist, left_dist)
            edge_best[first:last] = np.where(right, right_value, left_value)
            creation_best[first:last] = np.where(right, right_origin, left_origin)

    # --- gaps that only partly cover a block ------------------------------
    single = start_block == end_block
    spanning = ~single
    record_lo = np.concatenate(
        [creations[single], creations[spanning], end_block[spanning] * _BLOCK]
    )
    record_hi = np.concatenate(
        [deaths[single], (start_block[spanning] + 1) * _BLOCK, deaths[spanning]]
    )
    record_block = np.concatenate([start_block[single], start_block[spanning], end_block[spanning]])
    record_gap = np.concatenate([gap_index[single], gap_index[spanning], gap_index[spanning]])
    order = np.lexsort((record_gap, record_block))
    record_lo = record_lo[order]
    record_hi = record_hi[order]
    record_block = record_block[order]
    record_gap = record_gap[order]
    counts = np.bincount(record_block, minlength=blocks)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    for block in np.flatnonzero(counts):
        lo = starts[block]
        hi = lo + counts[block]
        slice_lo = record_lo[lo:hi]
        slice_hi = record_hi[lo:hi]
        slice_gap = record_gap[lo:hi]
        lengths = slice_hi - slice_lo
        total = int(lengths.sum())
        if total == 0:  # pragma: no cover - gaps always cover their own bar
            continue
        offsets = np.cumsum(lengths) - lengths
        rows = np.repeat(np.arange(len(slice_lo)), lengths)
        positions = (
            np.arange(total, dtype=np.int64)
            - np.repeat(offsets, lengths)
            + np.repeat(slice_lo, lengths)
        )
        block_lo = block * _BLOCK
        query = queries[block_lo : min(block_lo + _BLOCK, height)]
        dist = np.abs(edges[slice_gap][rows] - query[positions - block_lo])

        order = np.argsort(positions, kind="stable")
        positions = positions[order]
        dist = dist[order]
        gap = slice_gap[rows][order]
        group_start = np.flatnonzero(np.r_[True, positions[1:] != positions[:-1]])
        group_min = np.minimum.reduceat(dist, group_start)
        group_id = np.repeat(
            np.arange(len(group_start)), np.diff(np.r_[group_start, len(positions)])
        )
        first_min = np.flatnonzero(dist == group_min[group_id])
        keep = np.r_[True, group_id[first_min][1:] != group_id[first_min][:-1]]
        selected = first_min[keep]
        bar = positions[selected]
        new_dist = dist[selected]
        new_origin = creations[gap[selected]]
        old_dist = abs_best[bar]
        old_origin = creation_best[bar]
        take = (new_dist < old_dist) | ((new_dist == old_dist) & (new_origin < old_origin))
        abs_best[bar[take]] = new_dist[take]
        edge_best[bar[take]] = edges[gap[selected]][take]
        creation_best[bar[take]] = new_origin[take]

    return abs_best, edge_best, creation_best


def _resolve_zero_restart(
    bars: np.ndarray,
    creations: np.ndarray,
    deaths: np.ndarray,
    edges: np.ndarray,
    queries: np.ndarray,
    distances: np.ndarray,
    scale: np.ndarray,
    other_abs: np.ndarray | None,
    other_query: np.ndarray | None,
    other_value: np.ndarray | None,
) -> None:
    """Exact result for bars whose rightmost zero candidate is an old gap.

    The row loop restarts its minimum whenever the running best is exactly
    ``0.0``, so the answer is the nearest candidate *after* the last gap edge
    that coincides with the current price. Creation bars are handled by the
    caller; here the last zero is found with a per-price heap and only the gaps
    created after it are scanned.
    """
    if len(bars) == 0:
        return
    bars = np.sort(bars)
    wanted = set(queries[bars].tolist())
    heaps: dict[float, list[tuple[int, int, int]]] = {}
    ptr = 0
    count = len(creations)
    for i in bars.tolist():
        while ptr < count and creations[ptr] <= i:
            value = edges[ptr]
            if value in wanted:
                heapq.heappush(
                    heaps.setdefault(value, []),
                    (-int(creations[ptr]), int(deaths[ptr]), ptr),
                )
            ptr += 1
        heap = heaps.get(queries[i])
        if heap:
            while heap and heap[0][1] <= i:
                heapq.heappop(heap)
        zero = heap[0][2] if heap else -1

        best_abs = np.inf
        best_value = 0.0
        for gap in range(zero + 1, ptr):
            if deaths[gap] > i:
                candidate = queries[i] - edges[gap]
                magnitude = abs(candidate)
                if magnitude < best_abs:
                    best_abs = magnitude
                    best_value = candidate
        if other_abs is not None and other_abs[i] < best_abs:
            assert other_query is not None and other_value is not None
            distances[i] = (other_query[i] - other_value[i]) / scale[i]
        else:
            distances[i] = best_value / scale[i]


def _fvg_scan(
    highs: np.ndarray, lows: np.ndarray, atr_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised equivalent of the original per-row gap scan."""
    height = len(highs)
    up_flag = np.zeros(height, dtype=bool)
    down_flag = np.zeros(height, dtype=bool)
    if height < 3:
        return up_flag, down_flag, np.zeros(height, dtype=np.float64)

    up_created = np.nonzero(lows[2:] > highs[:-2])[0] + 2
    down_created = np.nonzero(highs[2:] < lows[:-2])[0] + 2
    up_flag[up_created] = True
    down_flag[down_created] = True

    up_edges = lows[up_created]  # gap top
    down_edges = highs[down_created]  # gap bottom
    up_deaths = _gap_deaths(up_created, lows, highs[up_created - 2], height, below=True)
    down_deaths = _gap_deaths(down_created, highs, lows[down_created - 2], height, below=False)

    up_abs, up_edge, _ = _nearest_active(height, up_created, up_deaths, up_edges, lows)
    down_abs, down_edge, _ = _nearest_active(height, down_created, down_deaths, down_edges, highs)

    scale = np.asarray(atr_values, dtype=np.float64).copy()
    bad = ~np.isfinite(scale) | (scale <= 0.0)
    scale[bad] = 1.0

    # Up gaps are tested first and a down gap must be strictly closer, so an
    # exact tie keeps the up side.
    use_down = down_abs < up_abs
    best_abs = np.where(use_down, down_abs, up_abs)
    best_value = np.where(use_down, highs - down_edge, lows - up_edge)
    valid = np.isfinite(best_abs)
    distance = np.divide(best_value, scale, out=np.zeros(height), where=valid)

    # A newly created gap is itself a zero-distance candidate. It is the last
    # up candidate, so an up gap created on this bar leaves the down side to
    # decide; a down gap created on this bar is the final candidate, so the
    # distance is exactly 0.0.
    up_created_flag = np.zeros(height, dtype=bool)
    up_created_flag[up_created] = True
    down_created_flag = np.zeros(height, dtype=bool)
    down_created_flag[down_created] = True
    up_zero = up_abs == 0.0
    down_zero = down_abs == 0.0

    old_down_zero = np.flatnonzero(down_zero & ~down_created_flag)
    old_up_zero = np.flatnonzero(up_zero & ~down_zero & ~up_created_flag)
    _resolve_zero_restart(
        old_up_zero,
        up_created,
        up_deaths,
        up_edges,
        lows,
        distance,
        scale,
        down_abs,
        highs,
        down_edge,
    )
    _resolve_zero_restart(
        old_down_zero,
        down_created,
        down_deaths,
        down_edges,
        highs,
        distance,
        scale,
        None,
        None,
        None,
    )

    fresh_up = up_created_flag & up_zero & ~down_zero
    distance[fresh_up] = np.divide(
        highs[fresh_up] - down_edge[fresh_up],
        scale[fresh_up],
        out=np.zeros(int(fresh_up.sum())),
        where=np.isfinite(down_abs[fresh_up]),
    )
    distance[down_created_flag] = 0.0
    return up_flag, down_flag, distance


def fvg(df: pl.DataFrame, n_atr: int = 14) -> pl.DataFrame:
    """``fvg_up``/``fvg_down`` booleans and ``fvg_dist`` in ATR units."""
    highs = df["high"].to_numpy().astype(np.float64)
    lows = df["low"].to_numpy().astype(np.float64)
    atr_values = atr(df, n_atr).to_numpy().astype(np.float64)
    up_flag, down_flag, distance = _fvg_scan(highs, lows, atr_values)

    return pl.DataFrame(
        {
            "fvg_up": pl.Series("fvg_up", up_flag, dtype=pl.Boolean),
            "fvg_down": pl.Series("fvg_down", down_flag, dtype=pl.Boolean),
            "fvg_dist": pl.Series("fvg_dist", distance, dtype=pl.Float64),
        }
    )


def liquidity_sweep(df: pl.DataFrame, lookback: int = 20) -> pl.Series:
    """Wick through the prior ``lookback`` bar extreme that closes back inside.

    A high sweep is ``high[i] > prior_high`` with ``close[i] < prior_high``: the
    bar traded above the range but was rejected. The low side is the mirror.
    ``prior_high``/``prior_low`` exclude the current bar.
    """
    if lookback < 1:
        raise ValueError(f"sweep lookback must be >= 1, got {lookback}")
    close = pl.col("close")
    prior_high = pl.col("high").rolling_max(lookback, min_samples=lookback).shift(1)
    prior_low = pl.col("low").rolling_min(lookback, min_samples=lookback).shift(1)
    swept_high = (pl.col("high") > prior_high) & (close < prior_high)
    swept_low = (pl.col("low") < prior_low) & (close > prior_low)
    return df.select(
        (swept_high.fill_null(False) | swept_low.fill_null(False)).alias("liquidity_sweep")
    ).to_series()
