"""Classical ML baseline: fitting, metrics, persistence."""

from __future__ import annotations

import numpy as np
import pytest

from scalping.config import Settings
from scalping.features.build import FEATURE_COLUMNS, FeatureFrame
from scalping.ml.evaluate import evaluate_model
from scalping.ml.models import build_model
from scalping.ml.train import (
    TrainSpec,
    load_model,
    load_result,
    save_result,
    train_classifier,
)


def _learnable_frame(n: int = 700, valid_from: int = 10) -> FeatureFrame:
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


def test_logistic_learns_the_signal_and_reports_validation() -> None:
    result = train_classifier(
        _learnable_frame(),
        symbol="XAUUSD",
        timeframe="M1",
        spec=TrainSpec(
            model="logistic",
            horizon=1,
            train_fraction=0.6,
            validation_fraction=0.2,
            purge_bars=1,
            embargo_bars=1,
        ),
    )
    assert result.validation["accuracy"] > 0.9
    assert result.validation["rows"] > 50
    assert result.baseline["accuracy"] < 0.7
    assert result.validation["log_loss"] < result.baseline["log_loss"]
    assert result.test is None
    assert result.dataset_id.startswith("XAUUSD:M1:h1:")
    assert result.split_counts["purged"] > 0


def test_model_survives_a_save_and_load(tmp_path) -> None:
    result = train_classifier(
        _learnable_frame(),
        symbol="XAUUSD",
        timeframe="M1",
        spec=TrainSpec(model="random_forest", horizon=1, purge_bars=1, embargo_bars=1),
    )
    model_path, metrics_path = save_result(result, tmp_path)
    assert model_path.is_file() and metrics_path.is_file()
    loaded, columns = load_model(model_path)
    assert columns == list(result.columns)
    frame = _learnable_frame()
    a = result.model.predict_proba(frame.X[-5:])
    b = loaded.predict_proba(frame.X[-5:])
    assert np.allclose(a, b)


def test_flat_rows_are_dropped_by_default() -> None:
    n = 700
    signal = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    close = np.empty(n, dtype=np.float64)
    close[0] = 100.0
    for index in range(1, n):
        if index % 10 == 0:
            close[index] = close[index - 1]  # exact flat return on this bar
        else:
            close[index] = close[index - 1] * (1.0 + 0.002 * signal[index - 1])
    X = np.zeros((n, len(FEATURE_COLUMNS)), dtype=np.float32)
    X[:, 0] = signal
    ts = np.array(
        [np.datetime64("2026-01-05T00:00", "us") + np.timedelta64(i, "m") for i in range(n)]
    )
    frame = FeatureFrame(
        X=X,
        ts=ts,
        ohlc=np.column_stack((close, close + 0.01, close - 0.01, close)),
        spread=np.full(n, 0.2),
        atr=np.full(n, 0.5),
        valid_from=10,
        columns=list(FEATURE_COLUMNS),
    )
    result = train_classifier(
        frame,
        symbol="XAUUSD",
        timeframe="M1",
        spec=TrainSpec(model="logistic", horizon=1, purge_bars=1, embargo_bars=1),
    )
    assert result.split_counts["flat_dropped"] > 0
    assert sorted(int(value) for value in result.model.classes_) == [-1, 1]


@pytest.mark.parametrize("model", ["lightgbm", "xgboost"])
def test_boosting_models_learn_the_signal(model: str) -> None:
    result = train_classifier(
        _learnable_frame(),
        symbol="XAUUSD",
        timeframe="M1",
        spec=TrainSpec(model=model, horizon=1, purge_bars=1, embargo_bars=1),
    )
    assert result.validation["accuracy"] > 0.8
    assert sorted(int(value) for value in result.model.classes_) == [-1, 1]


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model("magic")


def test_test_split_is_optional() -> None:
    result = train_classifier(
        _learnable_frame(),
        symbol="XAUUSD",
        timeframe="M1",
        spec=TrainSpec(
            model="logistic",
            horizon=1,
            purge_bars=1,
            embargo_bars=1,
            evaluate_test=True,
        ),
    )
    assert result.test is not None
    assert result.test["rows"] > 0


def test_evaluate_model_trades_in_the_simulator_without_bypassing_risk() -> None:
    frame = _learnable_frame()
    result = train_classifier(
        frame,
        symbol="XAUUSD",
        timeframe="M1",
        spec=TrainSpec(model="logistic", horizon=1, purge_bars=1, embargo_bars=1),
    )
    evaluation = evaluate_model(
        result,
        frame,
        Settings(),
        split="validation",
        threshold=0.55,
        exit_threshold=0.55,
        stress=(2.0,),
    )
    assert evaluation.metrics["trades"] > 0
    assert evaluation.hold["trades"] == 0
    keys = {"trades", "net_profit", "profit_factor", "expectancy_r", "max_drawdown"}
    assert keys <= set(evaluation.metrics)
    assert keys <= set(evaluation.stress["x2"])
    assert evaluation.split == "validation"


def test_saved_result_round_trip_keeps_identity(tmp_path) -> None:
    result = train_classifier(
        _learnable_frame(),
        symbol="XAUUSD",
        timeframe="M1",
        spec=TrainSpec(model="logistic", horizon=1, purge_bars=1, embargo_bars=1),
    )
    _, metrics_path = save_result(result, tmp_path / "run")
    loaded = load_result(metrics_path.parent)
    assert loaded.symbol == result.symbol
    assert loaded.timeframe == result.timeframe
    assert loaded.dataset_id == result.dataset_id
    assert loaded.spec == result.spec
    assert load_model(tmp_path / "run" / "model.joblib")[1] == list(result.columns)
