"""Lazy windowed batches must match the materialised dataset contract."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from scalping.data.synthetic import SyntheticProvider
from scalping.datasets.builder import build_dataset
from scalping.features.build import build_features
from scalping.nn import WindowConfig, build_windowed_dataset
from scalping.nn.data import WindowError
from scalping.nn.lazy import LazyWindowedDataset

START = datetime(2026, 1, 5, tzinfo=UTC)
CONFIG = WindowConfig(window=20, clip=5.0)


def make_bundle():
    frame = SyntheticProvider(seed=7).fetch("XAUUSD", START, START + timedelta(days=2))
    return build_dataset(build_features(frame), horizon=1, purge_bars=1)


def test_counts_and_batches_match_the_materialised_dataset() -> None:
    bundle = make_bundle()
    materialised = build_windowed_dataset(bundle, CONFIG)
    lazy = LazyWindowedDataset(bundle, CONFIG)

    assert lazy.counts() == materialised.counts()

    windows, labels, stamps = lazy.materialise("test")
    np.testing.assert_array_equal(windows, materialised.X_test)
    np.testing.assert_array_equal(labels, materialised.y_test)
    np.testing.assert_array_equal(stamps, materialised.ts_test)

    chunks = list(lazy.iter_batches("train", batch_size=37))
    assert sum(chunk[0].shape[0] for chunk in chunks) == lazy.counts()["train"]
    np.testing.assert_array_equal(np.concatenate([c[0] for c in chunks]), materialised.X_train)


def test_batch_size_larger_than_split_yields_one_batch() -> None:
    lazy = LazyWindowedDataset(make_bundle(), CONFIG)
    batches = list(lazy.iter_batches("test", batch_size=1_000_000))

    assert len(batches) == 1
    assert batches[0][0].shape[1:] == (CONFIG.window, lazy.feature_count)


def test_train_normalisation_ignores_future_rows() -> None:
    bundle = make_bundle()
    base = LazyWindowedDataset(bundle, CONFIG).materialise("train")[0]

    wild = bundle.X.copy()
    wild[bundle.split == 2] *= 100.0
    changed = LazyWindowedDataset(replace(bundle, X=wild), CONFIG).materialise("train")[0]

    np.testing.assert_array_equal(base, changed)


def test_guard_rails() -> None:
    bundle = make_bundle()

    with pytest.raises(WindowError, match="training windows"):
        LazyWindowedDataset(bundle, WindowConfig(window=bundle.rows("train") + 10))
    with pytest.raises(WindowError, match="non-finite"):
        LazyWindowedDataset(replace(bundle, X=np.full_like(bundle.X, np.nan)))

    lazy = LazyWindowedDataset(bundle, CONFIG)
    with pytest.raises(ValueError, match="unknown split"):
        next(lazy.iter_batches("holdout"))
    with pytest.raises(ValueError, match="batch_size"):
        next(lazy.iter_batches("train", batch_size=0))
