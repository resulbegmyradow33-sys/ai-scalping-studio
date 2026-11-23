"""Shared evaluator: hand-checked metrics, checkpoint scoring and the rule."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from scalping.data.synthetic import SyntheticProvider
from scalping.datasets.builder import build_dataset
from scalping.features.build import build_features
from scalping.nn import (
    WindowConfig,
    WindowedDataset,
    build_windowed_dataset,
    classification_metrics,
    compare_predictors,
    evaluate_checkpoint,
    momentum_rule_labels,
    train_classifier,
)
from scalping.nn.train import TrainConfig

WINDOW = 8
FEATURES = 3
START = datetime(2026, 1, 5, tzinfo=UTC)


def toy_dataset(seed: int = 3) -> WindowedDataset:
    rng = np.random.default_rng(seed)

    def split(rows: int):
        X = rng.normal(size=(rows, WINDOW, FEATURES)).astype(np.float32)
        y = np.where(X[:, :, 0].sum(axis=1) > 0, 1, -1).astype(np.int64)
        ts = np.arange(rows).astype("datetime64[m]")
        return X, y, ts

    return WindowedDataset(
        *split(160),
        *split(60),
        *split(40),
        mean=np.zeros(FEATURES, dtype=np.float32),
        std=np.ones(FEATURES, dtype=np.float32),
        columns=("a", "b", "c"),
        dataset_id="toy:w8",
        window=WINDOW,
    )


def tiny_checkpoint(tmp_path):
    data = toy_dataset()
    config = TrainConfig(
        model="mlp",
        epochs=3,
        batch_size=32,
        learning_rate=5e-3,
        hidden=16,
        dropout=0.0,
        device="cpu",
        seed=1,
    )
    checkpoint = tmp_path / "model.pt"
    train_classifier(data, config, checkpoint=checkpoint)
    return data, checkpoint


def test_metrics_match_hand_calculation() -> None:
    metrics = classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), classes=(0, 1))

    assert metrics.rows == 4
    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.balanced_accuracy == pytest.approx(0.75)
    assert metrics.confusion == ((1, 1), (0, 2))
    assert metrics.per_class[0]["precision"] == pytest.approx(1.0)
    assert metrics.per_class[0]["recall"] == pytest.approx(0.5)
    assert metrics.per_class[1]["precision"] == pytest.approx(2 / 3)
    assert metrics.per_class[1]["recall"] == pytest.approx(1.0)
    assert metrics.log_loss is None and metrics.brier is None


def test_probability_metrics_and_validation() -> None:
    truth = np.array([0, 1])
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    metrics = classification_metrics(truth, truth, probabilities, classes=(0, 1))

    assert metrics.log_loss == pytest.approx(0.0, abs=1e-9)
    assert metrics.brier == pytest.approx(0.0)

    with pytest.raises(ValueError, match="shape"):
        classification_metrics(truth, truth, probabilities[:, :1], classes=(0, 1))
    with pytest.raises(ValueError, match="sum to 1"):
        classification_metrics(truth, truth, np.full((2, 2), 0.75), classes=(0, 1))
    with pytest.raises(ValueError, match="outside"):
        classification_metrics(np.array([0, 2]), truth, classes=(0, 1))


def test_compare_sorts_by_accuracy() -> None:
    truth = np.array([0, 0, 1, 1])
    rows = compare_predictors(
        truth,
        {
            "weak": (np.array([0, 1, 1, 0]), None),
            "strong": (np.array([0, 0, 1, 1]), None),
        },
        classes=(0, 1),
    )

    assert [row["name"] for row in rows] == ["strong", "weak"]
    assert rows[0]["accuracy"] == pytest.approx(1.0)


def test_checkpoint_evaluation_reports_metrics_not_profit(tmp_path) -> None:
    data, checkpoint = tiny_checkpoint(tmp_path)
    report = evaluate_checkpoint(checkpoint, data, split="test", device="cpu")

    assert report["split"] == "test"
    assert report["trading_claim"] is False
    metrics = report["metrics"]
    assert metrics["rows"] == data.X_test.shape[0]
    assert metrics["classes"] == [-1, 1]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["log_loss"] is not None and metrics["brier"] is not None


def test_empty_split_is_rejected(tmp_path) -> None:
    data, checkpoint = tiny_checkpoint(tmp_path)
    empty = replace(
        data,
        X_test=np.empty((0, WINDOW, FEATURES), dtype=np.float32),
        y_test=np.empty(0, dtype=np.int64),
        ts_test=np.empty(0, dtype="datetime64[m]"),
    )

    with pytest.raises(ValueError, match="no windows"):
        evaluate_checkpoint(checkpoint, empty, split="test", device="cpu")


def test_rule_labels_align_with_window_timestamps() -> None:
    frame = SyntheticProvider(seed=7).fetch("XAUUSD", START, START + timedelta(days=2))
    bundle = build_dataset(build_features(frame), horizon=1, purge_bars=1)
    data = build_windowed_dataset(bundle, WindowConfig(window=20))

    labels = momentum_rule_labels(bundle, data.ts_test)

    assert labels.shape == data.ts_test.shape
    assert set(np.unique(labels).tolist()).issubset({-1, 0, 1})
    assert labels.dtype == np.int64
