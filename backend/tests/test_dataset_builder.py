"""Chronological purge/embargo dataset builder."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scalping.datasets.builder import build_dataset, save_manifest
from scalping.features.build import FEATURE_COLUMNS, FeatureFrame


def _frame(n: int = 200, valid_from: int = 10) -> FeatureFrame:
    ts = np.array(
        [np.datetime64("2026-01-05T00:00", "us") + np.timedelta64(i, "m") for i in range(n)]
    )
    close = 100.0 + np.arange(n, dtype=np.float64)
    ohlc = np.column_stack((close, close + 0.5, close - 0.5, close))
    return FeatureFrame(
        X=np.arange(n * len(FEATURE_COLUMNS), dtype=np.float32).reshape(n, len(FEATURE_COLUMNS)),
        ts=ts,
        ohlc=ohlc,
        spread=np.full(n, 0.2),
        atr=np.full(n, 0.5),
        valid_from=valid_from,
        columns=list(FEATURE_COLUMNS),
    )


def test_splits_are_chronological_with_purge_and_embargo() -> None:
    bundle = build_dataset(
        _frame(),
        symbol="XAUUSD",
        timeframe="H1",
        groups=("price",),
        horizon=3,
        train_fraction=0.6,
        validation_fraction=0.2,
        purge_bars=2,
        embargo_bars=1,
    )
    train = bundle.manifest.train
    validation = bundle.manifest.validation
    test = bundle.manifest.test
    assert train.end + 3 <= validation.start
    assert validation.end + 3 <= test.start
    assert bundle.rows("train") == train.rows
    assert bundle.rows("validation") == validation.rows
    assert bundle.rows("test") == test.rows
    assert (bundle.split == -1).sum() >= 6


def test_labels_use_the_future_close_only_as_target() -> None:
    frame = _frame()
    bundle = build_dataset(frame, horizon=4, groups=("price",))
    close = frame.ohlc[:, 3]
    expected = close[4:] / close[:-4] - 1.0
    # The bundle starts at valid_from, so compare its slice.
    start = frame.valid_from
    assert np.allclose(bundle.y_return, expected[start:], atol=1e-5)


def test_tail_without_the_full_horizon_is_dropped() -> None:
    frame = _frame(n=80, valid_from=5)
    bundle = build_dataset(frame, horizon=5, groups=("price",))
    assert bundle.y_return.shape[0] == frame.X.shape[0] - 5 - 5


def test_manifest_is_deterministic_and_persisted(tmp_path) -> None:
    frame = _frame()
    first = build_dataset(frame, horizon=2)
    second = build_dataset(frame, horizon=2)
    assert first.manifest.checksum == second.manifest.checksum
    assert first.manifest.dataset_id == second.manifest.dataset_id
    path = save_manifest(tmp_path / "dataset.json", first)
    payload = json.loads(path.read_text())
    assert payload["dataset_id"] == first.manifest.dataset_id
    assert payload["split_counts"]["train"] == first.rows("train")
    assert payload["purged_rows"] == int((first.split == -1).sum())


def test_invalid_fractions_raise() -> None:
    with pytest.raises(ValueError, match="must be < 1"):
        build_dataset(_frame(), train_fraction=0.8, validation_fraction=0.2)
