"""Windowed datasets: alignment, train-only scaling and split isolation."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from scalping.data.synthetic import SyntheticProvider
from scalping.datasets.builder import build_dataset
from scalping.features.build import build_features
from scalping.nn import WindowConfig, WindowError, build_windowed_dataset

START = datetime(2026, 1, 5, tzinfo=UTC)
CONFIG = WindowConfig(window=20, clip=5.0)


def make_bundle(*, days: int = 2, horizon: int = 1):
    frame = SyntheticProvider(seed=7).fetch("XAUUSD", START, START + timedelta(days=days))
    features = build_features(frame)
    return build_dataset(
        features,
        symbol="XAUUSD",
        timeframe="M1",
        horizon=horizon,
        purge_bars=horizon,
    )


def row_of(bundle, ts) -> int:
    index = np.flatnonzero(bundle.ts == ts)
    assert index.size == 1
    return int(index[0])


def test_shapes_and_alignment() -> None:
    bundle = make_bundle()
    data = build_windowed_dataset(bundle, CONFIG)

    assert data.counts()["train"] > 0
    assert data.counts()["validation"] > 0
    assert data.counts()["test"] > 0
    assert data.X_train.shape[1:] == (CONFIG.window, len(bundle.manifest.columns))
    assert data.X_train.dtype == np.float32
    assert data.y_train.dtype == np.int64
    assert data.X_train.shape[0] == data.y_train.shape[0] == data.ts_train.shape[0]

    # Window k ends at its timestamp and contains exactly the preceding bars.
    k = 7
    end = row_of(bundle, data.ts_train[k])
    expected = np.clip(
        (bundle.X[end - CONFIG.window + 1 : end + 1] - data.mean) / data.std,
        -CONFIG.clip,
        CONFIG.clip,
    )
    np.testing.assert_allclose(data.X_train[k], expected, rtol=0, atol=1e-6)
    assert data.y_train[k] == int(bundle.y_direction[end])


def test_normaliser_uses_training_rows_only() -> None:
    bundle = make_bundle()
    base = build_windowed_dataset(bundle, CONFIG)

    train_mask = bundle.split == 0
    np.testing.assert_allclose(
        base.mean,
        bundle.X[train_mask].astype(np.float64).mean(axis=0),
        rtol=0,
        atol=1e-6,
    )

    wild = bundle.X.copy()
    wild[bundle.split == 2] *= 100.0
    changed = build_windowed_dataset(replace(bundle, X=wild), CONFIG)

    np.testing.assert_array_equal(base.mean, changed.mean)
    np.testing.assert_array_equal(base.std, changed.std)
    np.testing.assert_array_equal(base.X_train, changed.X_train)
    assert np.abs(changed.X_test).max() <= CONFIG.clip


def test_constant_feature_is_not_divided_by_zero() -> None:
    bundle = make_bundle()
    X = bundle.X.copy()
    X[:, 0] = 42.0

    data = build_windowed_dataset(replace(bundle, X=X), CONFIG)

    assert data.std[0] == pytest.approx(1.0)
    assert np.allclose(data.X_train[..., 0], 0.0)


def test_windows_stay_inside_their_split() -> None:
    bundle = make_bundle()
    data = build_windowed_dataset(bundle, CONFIG)

    for name, ts in (
        ("train", data.ts_train),
        ("validation", data.ts_validation),
        ("test", data.ts_test),
    ):
        split_id = {"train": 0, "validation": 1, "test": 2}[name]
        for stamp in ts:
            row = row_of(bundle, stamp)
            assert bundle.split[row] == split_id
            assert row - CONFIG.window + 1 >= 0
            # Every input row belongs to the same split as the target.
            assert np.all(bundle.split[row - CONFIG.window + 1 : row + 1] == split_id)


def test_purged_rows_are_never_targets() -> None:
    bundle = make_bundle()
    assert np.any(bundle.split == -1)  # the purge really dropped rows

    data = build_windowed_dataset(bundle, CONFIG)

    purged = set(bundle.ts[bundle.split == -1].tolist())
    assert not (set(data.ts_train.tolist()) & purged)
    assert not (set(data.ts_validation.tolist()) & purged)
    assert not (set(data.ts_test.tolist()) & purged)


def test_build_is_deterministic() -> None:
    bundle = make_bundle()

    first = build_windowed_dataset(bundle, CONFIG)
    second = build_windowed_dataset(bundle, CONFIG)

    np.testing.assert_array_equal(first.X_train, second.X_train)
    np.testing.assert_array_equal(first.y_test, second.y_test)
    assert first.dataset_id == second.dataset_id


def test_window_too_long_for_training_split_is_rejected() -> None:
    bundle = make_bundle(days=1)
    train_rows = int(bundle.rows("train"))
    long = WindowConfig(window=train_rows + 10)

    with pytest.raises(WindowError, match="training windows"):
        build_windowed_dataset(bundle, long)


def test_bad_inputs_are_rejected() -> None:
    bundle = make_bundle()

    bad = bundle.X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(WindowError, match="non-finite"):
        build_windowed_dataset(replace(bundle, X=bad), CONFIG)

    with pytest.raises(WindowError, match="same row count"):
        build_windowed_dataset(replace(bundle, y_direction=bundle.y_direction[:-1]), CONFIG)

    with pytest.raises(ValueError, match="window"):
        WindowConfig(window=0)
    with pytest.raises(ValueError, match="clip"):
        WindowConfig(clip=0.0)
    with pytest.raises(ValueError, match="min_train_windows"):
        WindowConfig(min_train_windows=0)


def test_memory_guard_rejects_an_oversized_tensor() -> None:
    bundle = make_bundle()

    with pytest.raises(WindowError, match="max_bytes"):
        build_windowed_dataset(bundle, WindowConfig(window=20, max_bytes=1024))
    with pytest.raises(ValueError, match="max_bytes"):
        WindowConfig(max_bytes=0)
