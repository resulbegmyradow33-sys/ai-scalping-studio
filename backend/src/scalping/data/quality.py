"""Retrospective data-quality checks: statistical outliers and source reconciliation.

Structural validation (:mod:`scalping.data.validate`) answers "is this bar
well-formed?". This module adds two questions a merged store can still fail:

* **outliers** — a return or volume print that is many robust standard
  deviations away from the rest of the slice. They are warnings, not errors: a
  news spike is real data, but it deserves a look before training on it.
* **reconciliation** — two sources for the same symbol/timeframe disagree about
  prices, spread or coverage. Every disagreement is reported with a timestamp
  sample; nothing is merged or repaired here.

Both checks are retrospective. They are not features and must never feed the
model: the outlier threshold would leak the distribution of the whole slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import polars as pl

from scalping.data.base import conform_bars
from scalping.data.validate import Issue

__all__ = [
    "OutlierConfig",
    "ReconcileConfig",
    "ReconciliationReport",
    "detect_bar_outliers",
    "reconcile_bars",
    "robust_z",
]

_MAD_SCALE: Final[float] = 1.4826  # normal-consistent MAD multiplier


@dataclass(frozen=True, slots=True)
class OutlierConfig:
    """Thresholds for the robust z-scores; defaults are deliberately wide."""

    return_z: float = 8.0
    volume_z: float = 8.0
    min_rows: int = 30
    sample: int = 5

    def __post_init__(self) -> None:
        if self.return_z <= 0 or self.volume_z <= 0:
            raise ValueError("z thresholds must be > 0")
        if self.min_rows < 2:
            raise ValueError("min_rows must be >= 2")
        if self.sample < 1:
            raise ValueError("sample must be >= 1")


@dataclass(frozen=True, slots=True)
class ReconcileConfig:
    """Absolute tolerances in price units and spread units.

    Both defaults are ``0.0`` on purpose: a comparison is exact unless the
    caller states a tolerance. Two independent feeds usually need a small
    non-zero price/spread tolerance, because otherwise every last-digit
    difference is an error. Set them explicitly for a cross-provider check;
    leaving them at zero keeps the default strict for tests and same-source
    replays.
    """

    price_tolerance: float = 0.0
    spread_tolerance: float = 0.0
    sample: int = 5

    def __post_init__(self) -> None:
        if self.price_tolerance < 0 or self.spread_tolerance < 0:
            raise ValueError("tolerances must be non-negative")
        if self.sample < 1:
            raise ValueError("sample must be >= 1")


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """Coverage and mismatch summary for two frames of the same slice."""

    symbol: str
    timeframe: str
    primary_rows: int
    secondary_rows: int
    matched: int
    ok: bool
    issues: tuple[Issue, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "primary_rows": self.primary_rows,
            "secondary_rows": self.secondary_rows,
            "matched": self.matched,
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def robust_z(values: np.ndarray) -> np.ndarray:
    """``(x - median) / (1.4826 * MAD)`` with a std fallback for a zero MAD."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return array
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = _MAD_SCALE * mad
    if scale <= 0.0:
        scale = float(array.std())
    if scale <= 0.0 or not np.isfinite(scale):
        return np.zeros_like(array)
    return (array - median) / scale


def _sample_ts(frame: pl.DataFrame, mask: np.ndarray, limit: int) -> tuple[str, ...]:
    stamps = frame.filter(pl.Series(mask)).get_column("ts").head(limit).to_list()
    return tuple(str(stamp) for stamp in stamps)


def detect_bar_outliers(
    df: pl.DataFrame,
    config: OutlierConfig | None = None,
) -> tuple[Issue, ...]:
    """Return warning issues for extreme returns and volumes.

    The checks use the whole slice's median/MAD, so they are retrospective. A
    frame shorter than ``min_rows`` is skipped: a robust scale from a handful of
    bars is not a scale.
    """
    cfg = OutlierConfig() if config is None else config
    frame = conform_bars(df).sort("ts")
    if frame.height < cfg.min_rows:
        return ()

    issues: list[Issue] = []
    returns = np.asarray(
        frame.select(pl.col("close").log().diff()).to_series().to_numpy(),
        dtype=np.float64,
    )
    valid = np.isfinite(returns)
    z = np.zeros(frame.height, dtype=np.float64)
    z[valid] = np.abs(robust_z(returns[valid]))
    mask = z >= cfg.return_z
    if mask.any():
        count = int(mask.sum())
        issues.append(
            Issue(
                code="price_outliers",
                severity="warning",
                count=count,
                detail=f"{count} returns beyond {cfg.return_z:g} robust sigma",
                sample=_sample_ts(frame, mask, cfg.sample),
            )
        )

    volume = np.log1p(np.asarray(frame["volume"], dtype=np.float64))
    vz = np.abs(robust_z(volume))
    vmask = vz >= cfg.volume_z
    if vmask.any():
        count = int(vmask.sum())
        issues.append(
            Issue(
                code="volume_outliers",
                severity="warning",
                count=count,
                detail=f"{count} volume prints beyond {cfg.volume_z:g} robust sigma",
                sample=_sample_ts(frame, vmask, cfg.sample),
            )
        )
    return tuple(issues)


def reconcile_bars(
    primary: pl.DataFrame,
    secondary: pl.DataFrame,
    *,
    symbol: str = "",
    timeframe: str = "M1",
    config: ReconcileConfig | None = None,
) -> ReconciliationReport:
    """Compare two frames by timestamp; report coverage and price disagreements.

    ``primary`` is the store you trust; ``secondary`` is the new/other source.
    Only rows present in both are price-checked. Rows present in one side are
    warnings (coverage), price/spread mismatches are errors.
    """
    cfg = ReconcileConfig() if config is None else config
    left = conform_bars(primary).sort("ts")
    right = conform_bars(secondary).sort("ts")
    issues: list[Issue] = []

    for frame, label in ((left, "primary"), (right, "secondary")):
        duplicates = frame.height - frame["ts"].n_unique()
        if duplicates:
            issues.append(
                Issue(
                    code="duplicate_timestamps",
                    severity="error",
                    count=duplicates,
                    detail=f"{duplicates} duplicate timestamps in {label}",
                )
            )

    left_ts = set(left["ts"].to_list())
    right_ts = set(right["ts"].to_list())
    only_left = sorted(left_ts - right_ts)
    only_right = sorted(right_ts - left_ts)
    if only_left:
        issues.append(
            Issue(
                code="only_in_primary",
                severity="warning",
                count=len(only_left),
                detail=f"{len(only_left)} timestamps missing from the secondary source",
                sample=tuple(str(stamp) for stamp in only_left[: cfg.sample]),
            )
        )
    if only_right:
        issues.append(
            Issue(
                code="only_in_secondary",
                severity="warning",
                count=len(only_right),
                detail=f"{len(only_right)} timestamps missing from the primary source",
                sample=tuple(str(stamp) for stamp in only_right[: cfg.sample]),
            )
        )

    joined = left.select("ts", "close", "spread").join(
        right.select("ts", "close", "spread"),
        on="ts",
        how="inner",
        suffix="_other",
    )
    matched = joined.height
    if matched:
        close_gap = (joined["close"] - joined["close_other"]).abs()
        close_mask = (close_gap > cfg.price_tolerance).to_numpy()
        if close_mask.any():
            count = int(close_mask.sum())
            largest = float(cast(float, close_gap.max()))
            issues.append(
                Issue(
                    code="close_mismatch",
                    severity="error",
                    count=count,
                    detail=f"{count} closes differ by more than {cfg.price_tolerance:g} (max {largest:g})",
                    sample=_sample_ts(joined, close_mask, cfg.sample),
                )
            )
        spread_gap = (joined["spread"] - joined["spread_other"]).abs()
        spread_mask = (spread_gap > cfg.spread_tolerance).to_numpy()
        if spread_mask.any():
            count = int(spread_mask.sum())
            largest = float(cast(float, spread_gap.max()))
            issues.append(
                Issue(
                    code="spread_mismatch",
                    severity="error",
                    count=count,
                    detail=f"{count} spreads differ by more than {cfg.spread_tolerance:g} (max {largest:g})",
                    sample=_sample_ts(joined, spread_mask, cfg.sample),
                )
            )

    ok = not any(issue.severity == "error" for issue in issues)
    return ReconciliationReport(
        symbol=symbol,
        timeframe=timeframe,
        primary_rows=left.height,
        secondary_rows=right.height,
        matched=matched,
        ok=ok,
        issues=tuple(issues),
    )
