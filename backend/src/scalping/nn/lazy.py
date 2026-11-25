"""Lazy, batched access to windowed tensors.

:class:`~scalping.nn.data.WindowedDataset` materialises every window, which is
the right trade for a few hundred thousand rows and the wrong one for a
multi-GB M1 history. This module keeps the raw feature matrix and yields windows
batch by batch, so memory is the raw rows plus one batch instead of
``rows × window × features``.

The contract matches the materialised dataset: train-only normalisation, whole
window inside its split, purged rows excluded, same counts and labels.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import numpy as np

from scalping.datasets.builder import DatasetBundle
from scalping.nn.data import (
    WindowConfig,
    WindowError,
    _normaliser,
    _segment_ends,
)

__all__ = ["LazyWindowedDataset"]

_SPLITS: Final[tuple[str, ...]] = ("train", "validation", "test")


class LazyWindowedDataset:
    """Yield ``(X_batch, y_batch, ts_batch)`` without stacking all windows."""

    def __init__(
        self,
        bundle: DatasetBundle,
        config: WindowConfig | None = None,
    ) -> None:
        self.config = WindowConfig() if config is None else config
        X = np.asarray(bundle.X, dtype=np.float64)
        if X.ndim != 2:
            raise WindowError(f"X must be 2-D (rows, features), got shape {X.shape}")
        if X.shape[0] != bundle.y_direction.shape[0] or X.shape[0] != bundle.split.shape[0]:
            raise WindowError("X, y and split must have the same row count")
        if not np.isfinite(X).all():
            raise WindowError("X contains non-finite values")
        self._X = X
        self._y = np.asarray(bundle.y_direction)
        self._ts = np.asarray(bundle.ts)
        self._mean, self._std = _normaliser(X, bundle.split == 0)
        self._ends = {
            name: _segment_ends(bundle.split, index, self.config.window)
            for index, name in enumerate(_SPLITS)
        }
        if self._ends["train"].size < self.config.min_train_windows:
            raise WindowError(
                f"need at least {self.config.min_train_windows} training windows, "
                f"got {self._ends['train'].size}"
            )
        self.dataset_id = f"{bundle.manifest.dataset_id}:w{self.config.window}:lazy"
        self.window = self.config.window
        self.feature_count = int(X.shape[1])
        self.columns = tuple(bundle.manifest.columns)

    def counts(self) -> dict[str, int]:
        return {name: int(self._ends[name].size) for name in _SPLITS}

    def _normalised(self, first: int, last: int) -> np.ndarray:
        raw = self._X[first:last]
        scaled = (raw - self._mean.astype(np.float64)) / self._std.astype(np.float64)
        return np.clip(scaled, -self.config.clip, self.config.clip).astype(np.float32)

    def iter_batches(
        self, split: str = "train", *, batch_size: int = 1024
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield windows for one split in chronological order."""
        if split not in _SPLITS:
            raise ValueError(f"unknown split {split!r}; choose from {_SPLITS}")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        ends = self._ends[split]
        for start in range(0, ends.size, batch_size):
            chunk = ends[start : start + batch_size]
            windows = np.stack(
                [
                    self._normalised(end - self.window + 1, end + 1)
                    for end in chunk
                ]
            )
            yield windows, self._y[chunk].astype(np.int64), self._ts[chunk]

    def materialise(self, split: str = "test") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stack one split (for small splits and tests)."""
        parts = list(self.iter_batches(split, batch_size=max(1, self.counts()[split])))
        if not parts:
            return (
                np.empty((0, self.window, self.feature_count), dtype=np.float32),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=self._ts.dtype),
            )
        windows, labels, stamps = parts[0]
        return windows, labels, stamps
