"""Tick-volume flow: load the side table and compute the causal gate threshold."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

__all__ = ["flow_series", "flow_thresholds"]

_RULE = {"M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m", "H1": "1h", "H4": "4h", "D1": "1d"}


def flow_series(flow_root: Path, symbol: str, timeframe: str, frame: pl.DataFrame) -> np.ndarray:
    """Mean imbalance per bar of ``frame``, aligned on ``ts``; NaN when missing."""
    rule = _RULE.get(timeframe)
    if rule is None:
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    if (
        not symbol
        or symbol.strip() != symbol
        or "/" in symbol
        or "\\" in symbol
        or ".." in symbol
    ):
        raise ValueError(f"invalid flow symbol {symbol!r}")
    parts = sorted((Path(flow_root) / symbol / "M1").glob("*.parquet"))
    if not parts:
        raise ValueError(f"no flow partitions under {Path(flow_root) / symbol / 'M1'}")
    pieces = [pl.read_parquet(part) for part in parts]
    missing = sorted({"ts", "flow"} - set(pieces[0].columns))
    if missing:
        raise ValueError(f"flow partitions missing columns: {missing}")
    # The bar store keeps the last duplicate on write; the flow side table must
    # use the same rule instead of silently keeping the first row.
    flow = pl.concat(pieces).unique(subset=["ts"], keep="last", maintain_order=True).sort("ts")
    if rule != "1m":
        flow = flow.group_by_dynamic("ts", every=rule, closed="left", label="left").agg(
            pl.col("flow").mean()
        )
    flow = flow.with_columns(pl.col("ts").dt.cast_time_unit("us"))
    joined = (
        frame.select("ts")
        .with_columns(pl.col("ts").dt.cast_time_unit("us"))
        .join(flow, on="ts", how="left")
    )
    return joined["flow"].to_numpy()


def flow_thresholds(flow: np.ndarray, window: int = 240, quantile: float = 0.8) -> np.ndarray:
    """Causal ``|flow|`` quantile over the trailing window, shifted one bar."""
    series = pl.Series("f", np.abs(flow))
    thr = (
        series.rolling_quantile(quantile, window_size=window, min_samples=min(60, window))
        .shift(1)
        .to_numpy()
    )
    return np.asarray(thr, dtype=np.float64)
