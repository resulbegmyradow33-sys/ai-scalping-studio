"""Windowed neural-network datasets with train-only normalisation.

The classifier/regressor trainers need ``(batch, window, features)`` tensors.
Two rules matter more than the tensor shape:

* the mean/std come from the **training split only**, so a wild test period
  cannot move the scale the model learned on;
* a window never crosses a split boundary, so a validation sample cannot borrow
  test bars as context (or the reverse).

The label is already aligned by :func:`scalping.datasets.builder.build_dataset`
(``y`` at row ``i`` refers to the horizon after ``i``); this module only stacks
the trailing window that ends at ``i`` and never shifts the target again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from scalping.datasets.builder import DatasetBundle

__all__ = ["WindowConfig", "WindowedDataset", "build_windowed_dataset"]

_SPLIT_NAMES: Final[tuple[str, ...]] = ("train", "validation", "test")


class WindowError(ValueError):
    """The bundle cannot support the requested windows."""


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Window length and normalisation guard rails."""

    window: int = 60
    clip: float = 5.0
    min_train_windows: int = 1
    #: Refuse to materialise a tensor larger than this. 2 GiB leaves room for
    #: the model inside the 8 GB Mac mini budget; M1 history needs a longer
    #: timeframe or the lazy loader planned for Phase 14.
    max_bytes: int = 2 * 1024**3

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if not np.isfinite(self.clip) or self.clip <= 0.0:
            raise ValueError("clip must be a positive finite number")
        if self.min_train_windows < 1:
            raise ValueError("min_train_windows must be >= 1")
        if self.max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")


@dataclass(frozen=True, slots=True)
class WindowedDataset:
    """Chronological ``(n, window, features)`` arrays plus split metadata."""

    X_train: np.ndarray
    y_train: np.ndarray
    ts_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    ts_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    ts_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    columns: tuple[str, ...]
    dataset_id: str
    window: int

    @property
    def feature_count(self) -> int:
        return int(self.X_train.shape[2]) if self.X_train.size else int(self.mean.size)

    def counts(self) -> dict[str, int]:
        return {
            "train": int(self.X_train.shape[0]),
            "validation": int(self.X_validation.shape[0]),
            "test": int(self.X_test.shape[0]),
        }


def _normaliser(X: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = X[train_mask]
    if train.shape[0] == 0:
        raise WindowError("the training split is empty; cannot fit a normaliser")
    mean = train.mean(axis=0, dtype=np.float64)
    std = train.std(axis=0, dtype=np.float64)
    # A constant feature carries no information; scaling by 1 keeps it at 0.
    std = np.where(std > 0.0, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def _segment_ends(split: np.ndarray, index: int, window: int) -> np.ndarray:
    """End rows whose whole window sits inside split ``index``.

    Keep the window inside its own split: no context borrowed from a
    neighbouring split, purged rows included. The moving sum is O(n).
    """
    n = split.shape[0]
    if n < window:
        return np.empty(0, dtype=np.int64)
    in_split = (split == index).astype(np.int64)
    cumulative = np.concatenate(([0], np.cumsum(in_split)))
    ends = np.arange(window - 1, n)
    full = (cumulative[window:] - cumulative[:-window]) == window
    return ends[full]


def _stack(
    Z: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    ends: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ends.size == 0:
        empty_x = np.empty((0, window, Z.shape[1]), dtype=np.float32)
        return empty_x, np.empty(0, dtype=np.int64), np.empty(0, dtype=ts.dtype)
    stacked = np.stack([Z[end - window + 1 : end + 1] for end in ends])
    return stacked.astype(np.float32), y[ends].astype(np.int64), ts[ends]


def build_windowed_dataset(
    bundle: DatasetBundle,
    config: WindowConfig | None = None,
) -> WindowedDataset:
    """Stack trailing windows per split and normalise them from train rows only."""
    cfg = WindowConfig() if config is None else config
    X = np.asarray(bundle.X, dtype=np.float64)
    if X.ndim != 2:
        raise WindowError(f"X must be 2-D (rows, features), got shape {X.shape}")
    if X.shape[0] != bundle.y_direction.shape[0] or X.shape[0] != bundle.split.shape[0]:
        raise WindowError("X, y and split must have the same row count")
    if not np.isfinite(X).all():
        raise WindowError("X contains non-finite values")

    train_mask = bundle.split == 0
    mean, std = _normaliser(X, train_mask)
    Z = np.clip((X - mean.astype(np.float64)) / std.astype(np.float64), -cfg.clip, cfg.clip)

    y = np.asarray(bundle.y_direction)
    if np.asarray(bundle.ts).shape[0] != X.shape[0]:
        raise WindowError("ts and X must have the same row count")
    ts = np.asarray(bundle.ts)

    ends = {
        name: _segment_ends(bundle.split, index, cfg.window)
        for index, name in enumerate(_SPLIT_NAMES)
    }
    total_windows = int(sum(ends[name].size for name in _SPLIT_NAMES))
    estimate = total_windows * cfg.window * X.shape[1] * np.dtype(np.float32).itemsize
    if estimate > cfg.max_bytes:
        raise WindowError(
            f"windowed dataset needs about {estimate / 1024**3:.1f} GiB, above "
            f"max_bytes={cfg.max_bytes}; use a longer timeframe, a smaller window "
            "or the lazy loader planned for Phase 14"
        )
    splits = {
        name: _stack(Z, y, ts, ends[name], cfg.window) for name in _SPLIT_NAMES
    }
    if splits["train"][0].shape[0] < cfg.min_train_windows:
        raise WindowError(
            f"need at least {cfg.min_train_windows} training windows, "
            f"got {splits['train'][0].shape[0]}"
        )

    return WindowedDataset(
        X_train=splits["train"][0],
        y_train=splits["train"][1],
        ts_train=splits["train"][2],
        X_validation=splits["validation"][0],
        y_validation=splits["validation"][1],
        ts_validation=splits["validation"][2],
        X_test=splits["test"][0],
        y_test=splits["test"][1],
        ts_test=splits["test"][2],
        mean=mean,
        std=std,
        columns=tuple(bundle.manifest.columns),
        dataset_id=f"{bundle.manifest.dataset_id}:w{cfg.window}:c{cfg.clip:g}",
        window=cfg.window,
    )
