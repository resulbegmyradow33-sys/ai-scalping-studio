"""Purged time-series cross-validation over the training split only."""

from __future__ import annotations

import numpy as np
import pytest

from scalping.datasets.builder import build_dataset
from scalping.features.build import FEATURE_COLUMNS, FeatureFrame
from scalping.ml import cv as cv_module
from scalping.ml.cv import CVConfig, cross_validate
from scalping.ml.train import TrainSpec


def _learnable_frame(n: int = 1200, valid_from: int = 10) -> FeatureFrame:
    """Feature 0 alternates +/-1 and drives the next bar's return."""
    signal = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    close = np.empty(n, dtype=np.float64)
    close[0] = 100.0
    for index in range(1, n):
        close[index] = close[index - 1] * (1.0 + 0.002 * signal[index - 1])
    X = np.zeros((n, len(FEATURE_COLUMNS)), dtype=np.float32)
    X[:, 0] = signal
    ts = np.array(
        [np.datetime64("2026-01-05T00:00", "us") + np.timedelta64(i, "m") for i in range(n)]
    )
    ohlc = np.column_stack((close, close + 0.01, close - 0.01, close))
    return FeatureFrame(
        X=X,
        ts=ts,
        ohlc=ohlc,
        spread=np.full(n, 0.2),
        atr=np.full(n, 0.5),
        valid_from=valid_from,
        columns=list(FEATURE_COLUMNS),
    )


def _spec(**overrides: object) -> TrainSpec:
    values: dict[str, object] = {
        "model": "logistic",
        "horizon": 1,
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "purge_bars": 1,
        "embargo_bars": 1,
    }
    values.update(overrides)
    return TrainSpec(**values)  # type: ignore[arg-type]


def _bundle(frame: FeatureFrame, spec: TrainSpec):
    return build_dataset(
        frame,
        symbol="XAUUSD",
        timeframe="M1",
        groups=spec.groups,
        horizon=spec.horizon,
        train_fraction=spec.train_fraction,
        validation_fraction=spec.validation_fraction,
        purge_bars=max(spec.purge_bars, spec.horizon),
        embargo_bars=spec.embargo_bars,
    )


def test_folds_are_chronological_contiguous_and_purged() -> None:
    frame = _learnable_frame()
    spec = _spec()
    config = CVConfig(n_splits=4, purge_bars=3, embargo_bars=2, min_train_rows=20)
    result = cross_validate(frame, symbol="XAUUSD", timeframe="M1", spec=spec, config=config)

    bundle = _bundle(frame, spec)
    positions = np.flatnonzero(bundle.split == 0)
    train_lo, train_hi = int(positions[0]), int(positions[-1]) + 1
    gap = max(config.purge_bars, spec.horizon) + config.embargo_bars

    assert len(result.folds) == config.n_splits
    previous_end: int | None = None
    for fold in result.folds:
        assert train_lo <= fold.train_start < fold.train_end <= fold.test_start < fold.test_end
        assert fold.test_end <= train_hi
        assert fold.train_start == train_lo
        assert fold.train_rows == fold.train_end - fold.train_start
        assert fold.test_rows == fold.test_end - fold.test_start
        assert fold.test_start - fold.train_end >= gap
        if previous_end is not None:
            assert fold.test_start == previous_end
        previous_end = fold.test_end

    assert result.folds[-1].test_end == train_hi
    assert [fold.fold for fold in result.folds] == list(range(config.n_splits))


def test_validation_and_test_rows_are_never_used() -> None:
    frame = _learnable_frame()
    spec = _spec()
    config = CVConfig(n_splits=4, purge_bars=3, embargo_bars=2, min_train_rows=20)
    bundle = _bundle(frame, spec)

    # Bundle row ``i`` is source row ``valid_from + i``. Poison everything the
    # CV must not touch: if a validation/test/purged row reaches the model the
    # fit or the probabilities would fail on NaN.
    outside = bundle.split != 0
    poisoned = frame.X.copy()
    poisoned[frame.valid_from + np.flatnonzero(outside)] = np.nan
    poisoned_frame = FeatureFrame(
        X=poisoned,
        ts=frame.ts,
        ohlc=frame.ohlc,
        spread=frame.spread,
        atr=frame.atr,
        valid_from=frame.valid_from,
        columns=frame.columns,
    )

    result = cross_validate(
        poisoned_frame, symbol="XAUUSD", timeframe="M1", spec=spec, config=config
    )

    assert len(result.folds) == config.n_splits
    assert result.split_counts["validation"] == bundle.rows("validation")
    assert result.split_counts["test"] == bundle.rows("test")
    assert result.split_counts["purged"] == int((bundle.split == -1).sum())
    for metrics in result.fold_metrics:
        assert np.isfinite(metrics["log_loss"])
        assert np.isfinite(metrics["brier"])


def test_metrics_are_finite_and_the_signal_is_learned() -> None:
    result = cross_validate(
        _learnable_frame(),
        symbol="XAUUSD",
        timeframe="M1",
        spec=_spec(),
        config=CVConfig(n_splits=4, purge_bars=3, embargo_bars=2, min_train_rows=20),
    )

    assert set(result.aggregate) == {
        "accuracy",
        "balanced_accuracy",
        "log_loss",
        "brier",
        "ece",
        "mce",
    }
    for metrics in result.fold_metrics:
        for key in result.aggregate:
            assert np.isfinite(metrics[key])
        assert metrics["rows"] > 0
        assert metrics["train_rows"] >= 20
        assert metrics["accuracy"] > 0.8
        assert metrics["ece"] >= 0.0
        assert metrics["mce"] >= 0.0
    for stats in result.aggregate.values():
        assert np.isfinite(stats["mean"])
        assert np.isfinite(stats["std"])
        assert stats["std"] >= 0.0
    assert result.aggregate["accuracy"]["mean"] > 0.8


def test_cross_validation_is_deterministic() -> None:
    frame = _learnable_frame()
    spec = _spec()
    config = CVConfig(n_splits=3, purge_bars=2, embargo_bars=1, min_train_rows=20)

    first = cross_validate(frame, symbol="XAUUSD", timeframe="M1", spec=spec, config=config)
    second = cross_validate(frame, symbol="XAUUSD", timeframe="M1", spec=spec, config=config)

    assert first.folds == second.folds
    assert first.dataset_id == second.dataset_id
    assert first.aggregate == second.aggregate
    for left, right in zip(first.fold_metrics, second.fold_metrics, strict=True):
        for key, value in left.items():
            assert right[key] == value


def test_too_small_train_slice_raises() -> None:
    with pytest.raises(ValueError, match="min_train_rows"):
        cross_validate(
            _learnable_frame(),
            symbol="XAUUSD",
            timeframe="M1",
            spec=_spec(),
            config=CVConfig(n_splits=4, purge_bars=1, embargo_bars=0, min_train_rows=10_000),
        )


def test_non_finite_probabilities_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NaNFoldModel:
        classes_ = np.array([-1, 1])

        def fit(self, X: np.ndarray, y: np.ndarray) -> _NaNFoldModel:
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.full((X.shape[0], 2), np.nan)

    monkeypatch.setattr(cv_module, "build_model", lambda name, seed=0: _NaNFoldModel())
    with pytest.raises(ValueError, match="non-finite"):
        cross_validate(
            _learnable_frame(),
            symbol="XAUUSD",
            timeframe="M1",
            spec=_spec(),
            config=CVConfig(n_splits=2, purge_bars=1, embargo_bars=0, min_train_rows=20),
        )


def test_config_is_validated() -> None:
    with pytest.raises(ValueError, match="n_splits"):
        CVConfig(n_splits=0)
    with pytest.raises(ValueError, match="purge_bars"):
        CVConfig(purge_bars=-1)
    with pytest.raises(ValueError, match="embargo_bars"):
        CVConfig(embargo_bars=-1)
    with pytest.raises(ValueError, match="min_train_rows"):
        CVConfig(min_train_rows=0)
