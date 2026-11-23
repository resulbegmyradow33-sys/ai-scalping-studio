"""Chronological, purged dataset splits with a reproducible manifest.

A random train/test split leaks future information into financial models. This
builder cuts the frame in time order and inserts a purge/embargo gap at each
boundary. The purge is at least the label horizon, so a training row's target
cannot overlap a validation row's features.

Labels use future closes **only as targets**; the feature matrix comes from
`build_features` and never sees them. Rows without a complete future horizon
are dropped from the tail, and purged rows are kept with ``split == -1`` so a
report can show how much data the boundary cost.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

from scalping.features.availability import Capabilities, ensure_columns_available
from scalping.features.build import FeatureFrame
from scalping.features.registry import DEFAULT_GROUPS, columns_for, indices_for

__all__ = [
    "DatasetBundle",
    "DatasetManifest",
    "SplitRange",
    "build_dataset",
    "save_manifest",
]

SPLIT_NAMES: Final[tuple[str, ...]] = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class SplitRange:
    """Absolute row indices ``[start, end)`` of one split in the source frame."""

    start: int
    end: int

    @property
    def rows(self) -> int:
        return max(0, self.end - self.start)

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end, "rows": self.rows}


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Everything needed to reproduce and compare a dataset build."""

    dataset_id: str
    symbol: str
    timeframe: str
    horizon: int
    groups: tuple[str, ...]
    columns: tuple[str, ...]
    rows: int
    train: SplitRange
    validation: SplitRange
    test: SplitRange
    purge_bars: int
    embargo_bars: int
    checksum: str
    created_at: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizon": self.horizon,
            "groups": list(self.groups),
            "columns": list(self.columns),
            "rows": self.rows,
            "splits": {
                "train": self.train.to_dict(),
                "validation": self.validation.to_dict(),
                "test": self.test.to_dict(),
            },
            "purge_bars": self.purge_bars,
            "embargo_bars": self.embargo_bars,
            "checksum": self.checksum,
            "created_at": self.created_at,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """Arrays + split labels for one build; ``split == -1`` is purged."""

    X: np.ndarray
    y_direction: np.ndarray
    y_return: np.ndarray
    ts: np.ndarray
    split: np.ndarray
    manifest: DatasetManifest

    def mask(self, name: str) -> np.ndarray:
        if name not in SPLIT_NAMES:
            raise ValueError(f"unknown split {name!r}")
        return self.split == SPLIT_NAMES.index(name)

    def rows(self, name: str) -> int:
        return int(self.mask(name).sum())


def _checksum(bundle: tuple[np.ndarray, ...]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for array in bundle:
        values = np.ascontiguousarray(array)
        digest.update(f"{values.dtype}:{values.shape}".encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def build_dataset(
    features: FeatureFrame,
    *,
    symbol: str = "XAUUSD",
    timeframe: str = "M1",
    groups: tuple[str, ...] | None = None,
    horizon: int = 1,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    purge_bars: int = 0,
    embargo_bars: int = 0,
    notes: str = "",
    capabilities: Capabilities | None = None,
) -> DatasetBundle:
    """Cut a causal feature frame into train/validation/test with purge/embargo.

    ``purge_bars`` is raised to at least ``horizon`` because two labels whose
    horizons overlap share future bars even if their rows do not.
    ``capabilities``, when given, fails the build if a selected column needs an
    input the source cannot provide (for example signed flow without tick
    volumes). ``None`` keeps the historical behaviour for callers that only have
    the feature matrix.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1")
    if purge_bars < 0 or embargo_bars < 0:
        raise ValueError("purge/embargo must be non-negative")

    n = int(features.X.shape[0])
    start = int(features.valid_from)
    end = n - horizon
    usable = end - start
    if usable <= 2:
        raise ValueError("not enough usable rows after warm-up and horizon")

    gap = max(horizon, purge_bars) + embargo_bars
    train_rows = int(usable * train_fraction)
    validation_rows = int(usable * validation_fraction)
    train = SplitRange(start, start + train_rows)
    validation = SplitRange(train.end + gap, train.end + gap + validation_rows)
    test = SplitRange(validation.end + gap, end)
    if validation.end > end or test.start >= end:
        raise ValueError("splits leave no test rows; reduce purge/embargo or fractions")

    indices = indices_for(groups)
    columns = columns_for(groups)
    if capabilities is not None:
        ensure_columns_available(columns, capabilities)
    X = np.ascontiguousarray(features.X[start:end][:, indices], dtype=np.float32)
    close = np.asarray(features.ohlc[start:end, 3], dtype=np.float64)
    future = np.asarray(features.ohlc[start + horizon : n, 3], dtype=np.float64)
    if close.shape != future.shape:
        raise ValueError("internal horizon alignment error")
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.where(close > 0, future / close - 1.0, np.nan)
    direction = np.sign(returns)
    direction = np.nan_to_num(direction, nan=0.0).astype(np.int8)
    returns = returns.astype(np.float32)

    split = np.full(X.shape[0], -1, dtype=np.int8)
    for name, span in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        lo = span.start - start
        hi = span.end - start
        if hi > lo:
            split[lo:hi] = SPLIT_NAMES.index(name)

    digest = _checksum((X, direction, returns, np.asarray(features.ts[start:end]), split))
    selected_groups = DEFAULT_GROUPS if groups is None else tuple(groups)
    manifest = DatasetManifest(
        dataset_id=f"{symbol}:{timeframe}:h{horizon}:{digest[:12]}",
        symbol=symbol,
        timeframe=timeframe,
        horizon=horizon,
        groups=selected_groups,
        columns=columns,
        rows=int(X.shape[0]),
        train=train,
        validation=validation,
        test=test,
        purge_bars=max(horizon, purge_bars),
        embargo_bars=embargo_bars,
        checksum=digest,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        notes=notes,
    )
    return DatasetBundle(
        X=X,
        y_direction=direction,
        y_return=returns,
        ts=np.asarray(features.ts[start:end]),
        split=split,
        manifest=manifest,
    )


def save_manifest(path: str | Path, bundle: DatasetBundle) -> Path:
    """Write the manifest atomically; arrays remain in memory/caller storage."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    payload = bundle.manifest.to_dict()
    payload["split_counts"] = {
        name: bundle.rows(name) for name in SPLIT_NAMES
    }
    payload["purged_rows"] = int((bundle.split == -1).sum())
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, target)
    return target
