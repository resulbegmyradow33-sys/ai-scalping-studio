"""Purged time-series cross-validation over the training split only."""

from __future__ import annotations

import numpy as np
import pytest

from scalping.datasets.builder import build_dataset
from scalping.features.build import FEATURE_COLUMNS, FeatureFrame
from scalping.ml import cv as cv_module
from scalping.ml.cv import CVConfig, cross_validate
from scalping.ml.train import TrainSpec


def _learnable_frame(
    n: int = 1200,
    valid_from: int = 10,
    flat_every: int | None = None,
) -> FeatureFrame:
    """Feature 0 alternates +/-1 and drives the next bar's return."""
    signal = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    close = np.empty(n, dtype=np.float64)
    close[0] = 100.0
    for index in range(1, n):
        if flat_every is not None and index % flat_every == 0:
            close[index] = close[index - 1]
        else:
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


def _one_class_frame(n: int = 600, valid_from: int = 10) -> FeatureFrame:
    """Every label is +1, so the first fold's training window is single-class."""
    close = 100.0 * (1.001 ** np.arange(n))
    X = np.zeros((n, len(FEATURE_COLUMNS)), dtype=np.float32)
    ts = np.array(
        [np.datetime64("2026-01-05T00:00", "us") + np.timedelta64(i, "m") for i in range(n)]
    )
    return FeatureFrame(
        X=X,
        ts=ts,
        ohlc=np.column_stack((close, close + 0.01, close - 0.01, close)),
        spread=np.full(n, 0.2),
        atr=np.full(n, 0.5),
        valid_from=valid_from,
        columns=list(FEATURE_COLUMNS),
    )


def _flatten_close(frame: FeatureFrame, start: int, end: int) -> FeatureFrame:
    """Make closes ``[start, end]`` constant so those labels become flat."""
    close = frame.ohlc[:, 3].copy()
    close[start : end + 1] = close[start]
    ohlc = np.column_stack((close, close + 0.01, close - 0.01, close))
    return FeatureFrame(
        X=frame.X.copy(),
        ts=frame.ts,
        ohlc=ohlc,
        spread=frame.spread,
        atr=frame.atr,
        valid_from=frame.valid_from,
        columns=frame.columns,
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
    spec = _spec(purge_bars=4)
    config = CVConfig(n_splits=4, purge_bars=3, embargo_bars=2, min_train_rows=20)
    result = cross_validate(frame, symbol="XAUUSD", timeframe="M1", spec=spec, config=config)

    bundle = _bundle(frame, spec)
    positions = np.flatnonzero(bundle.split == 0)
    train_lo, train_hi = int(positions[0]), int(positions[-1]) + 1
    # The fold gap honours spec.purge_bars too, not just CVConfig.
    gap = max(config.purge_bars, spec.purge_bars, spec.horizon) + config.embargo_bars

    assert len(result.folds) == config.n_splits
    previous_end: int | None = None
    for fold in result.folds:
        assert train_lo <= fold.train_start < fold.train_end <= fold.test_start < fold.test_end
        assert fold.test_end <= train_hi
        assert fold.train_start == train_lo
        assert fold.train_rows == fold.train_end - fold.train_start
        assert fold.test_rows == fold.test_end - fold.test_start
        assert fold.test_start - fold.train_end >= gap
        assert np.all(bundle.split[fold.train_start : fold.train_end] == 0)
        assert np.all(bundle.split[fold.test_start : fold.test_end] == 0)
        if previous_end is not None:
            assert fold.test_start == previous_end
        previous_end = fold.test_end

    assert result.folds[-1].test_end == train_hi
    assert [fold.fold for fold in result.folds] == list(range(config.n_splits))


def test_report_can_reproduce_the_run() -> None:
    frame = _learnable_frame()
    spec = _spec(seed=7, groups=("price", "momentum"))
    result = cross_validate(
        frame,
        symbol="XAUUSD",
        timeframe="M1",
        spec=spec,
        config=CVConfig(n_splits=3, purge_bars=2, embargo_bars=1, min_train_rows=20),
    )
    payload = result.to_dict()

    assert payload["spec"]["seed"] == 7
    assert payload["spec"]["model"] == "logistic"
    assert payload["spec"]["horizon"] == spec.horizon
    assert payload["spec"]["include_flat"] is False
    assert tuple(payload["spec"]["groups"]) == spec.groups
    assert payload["columns"] == list(result.columns)
    assert result.columns == _bundle(frame, spec).manifest.columns


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


def test_split_counts_match_ml_train_and_count_flat_separately() -> None:
    frame = _learnable_frame(flat_every=10)
    bundle = _bundle(frame, _spec())
    flat = bundle.y_direction == 0
    dropped = cross_validate(
        frame,
        symbol="XAUUSD",
        timeframe="M1",
        spec=_spec(),
        config=CVConfig(n_splits=4, purge_bars=1, embargo_bars=1, min_train_rows=20),
    )
    assert dropped.split_counts["flat_dropped"] == int(flat.sum()) > 0
    assert dropped.split_counts["train"] == int(((bundle.split == 0) & ~flat).sum())
    assert dropped.split_counts["train"] < bundle.rows("train")

    included = cross_validate(
        frame,
        symbol="XAUUSD",
        timeframe="M1",
        spec=_spec(include_flat=True),
        config=CVConfig(n_splits=4, purge_bars=1, embargo_bars=1, min_train_rows=20),
    )
    assert included.split_counts["flat_dropped"] == 0
    assert included.split_counts["train"] == bundle.rows("train")
    assert any(metrics["rows"] > 0 for metrics in included.fold_metrics)


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
    assert first.spec == second.spec
    assert first.columns == second.columns
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


def test_purge_that_erases_the_training_window_raises() -> None:
    with pytest.raises(ValueError, match="leaves no training rows"):
        cross_validate(
            _learnable_frame(),
            symbol="XAUUSD",
            timeframe="M1",
            spec=_spec(),
            config=CVConfig(n_splits=4, purge_bars=10_000, embargo_bars=0, min_train_rows=20),
        )


def test_single_class_training_window_raises_before_fit() -> None:
    with pytest.raises(ValueError, match="single class"):
        cross_validate(
            _one_class_frame(),
            symbol="XAUUSD",
            timeframe="M1",
            spec=_spec(),
            config=CVConfig(n_splits=2, purge_bars=1, embargo_bars=0, min_train_rows=20),
        )


def test_missing_test_class_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    class _OneClassOutputModel:
        classes_ = np.array([-1])

        def fit(self, X: np.ndarray, y: np.ndarray) -> _OneClassOutputModel:
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.ones((X.shape[0], 1))

    monkeypatch.setattr(cv_module, "build_model", lambda name, seed=0: _OneClassOutputModel())
    with pytest.raises(ValueError, match="absent from the training window"):
        cross_validate(
            _learnable_frame(),
            symbol="XAUUSD",
            timeframe="M1",
            spec=_spec(),
            config=CVConfig(n_splits=2, purge_bars=1, embargo_bars=0, min_train_rows=20),
        )


def test_empty_test_block_after_dropping_flat_rows_raises() -> None:
    frame = _learnable_frame()
    spec = _spec()
    config = CVConfig(n_splits=4, purge_bars=1, embargo_bars=0, min_train_rows=20)
    probe = cross_validate(frame, symbol="XAUUSD", timeframe="M1", spec=spec, config=config)

    # Flatten the second fold's test block. The first fold still trains and
    # scores normally; the second sees only flat labels and must reject.
    block = probe.folds[1]
    flattened = _flatten_close(
        frame,
        frame.valid_from + block.test_start,
        frame.valid_from + block.test_end,
    )
    with pytest.raises(ValueError, match="empty test block after dropping flat rows"):
        cross_validate(flattened, symbol="XAUUSD", timeframe="M1", spec=spec, config=config)


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
        CVConfig(n_splits=1)
    with pytest.raises(ValueError, match="purge_bars"):
        CVConfig(purge_bars=-1)
    with pytest.raises(ValueError, match="embargo_bars"):
        CVConfig(embargo_bars=-1)
    with pytest.raises(ValueError, match="min_train_rows"):
        CVConfig(min_train_rows=0)
