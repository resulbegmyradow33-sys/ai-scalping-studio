"""``ml train`` records a validation precision gate without applying it."""

from __future__ import annotations

import numpy as np
import pytest

from scalping.features.build import FEATURE_COLUMNS, FeatureFrame
from scalping.ml.train import TrainSpec, load_result, save_result, train_classifier


def _learnable_frame(n: int = 700, valid_from: int = 10) -> FeatureFrame:
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


def _spec(**overrides) -> TrainSpec:
    values = {
        "model": "logistic",
        "horizon": 1,
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "purge_bars": 1,
        "embargo_bars": 1,
    }
    return TrainSpec(**{**values, **overrides})  # type: ignore[arg-type]


def test_rejection_gate_is_selected_on_validation_and_saved(tmp_path) -> None:
    result = train_classifier(
        _learnable_frame(), symbol="XAUUSD", timeframe="M1", spec=_spec(reject_precision=0.6)
    )

    gate = result.rejection
    assert gate is not None
    assert gate["satisfied"] is True
    assert gate["target_precision"] == 0.6
    assert gate["precision"] >= 0.6
    assert 0.0 <= gate["threshold"] <= 1.0
    assert gate["accepted"] > 0
    assert result.to_dict()["rejection"] == gate

    directory = tmp_path / "model"
    save_result(result, directory)
    loaded = load_result(directory)
    assert loaded.rejection == gate
    assert loaded.spec.reject_precision == 0.6


def test_rejection_spec_guard_rails() -> None:
    with pytest.raises(ValueError, match="reject_precision"):
        _spec(reject_precision=0.0)
    with pytest.raises(ValueError, match="reject_precision"):
        _spec(reject_precision=1.5)
    with pytest.raises(ValueError, match="reject_min_coverage"):
        _spec(reject_precision=0.6, reject_min_coverage=1.5)
    with pytest.raises(ValueError, match="requires reject_precision"):
        _spec(reject_min_coverage=0.5)


def test_evaluation_can_use_the_saved_rejection_gate() -> None:
    from scalping.config import Settings
    from scalping.ml.evaluate import evaluate_model

    features = _learnable_frame()
    result = train_classifier(
        features, symbol="XAUUSD", timeframe="M1", spec=_spec(reject_precision=0.6)
    )
    assert result.rejection is not None and result.rejection["satisfied"] is True

    evaluation = evaluate_model(
        result,
        features,
        Settings(),
        split="validation",
        window=60,
        compute_regime=False,
        use_rejection_gate=True,
    )

    assert evaluation.threshold == result.rejection["threshold"]
