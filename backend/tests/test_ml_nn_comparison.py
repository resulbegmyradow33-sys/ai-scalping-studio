"""Phase 5 × Phase 6 × rule comparison on one shared test slice.

The three predictors are scored with the same ``classification_metrics`` /
``compare_predictors`` contract and on the exact rows a network saw. This is the
integration the Phase 6 board calls "shared evaluator against classical ML and
the rule".

Smoke integration, not a like-for-like training run: the logistic model here is
fit directly on the split matrix and keeps flat labels, while ``ml train`` drops
flat-zero rows by default. The point is the shared scoring path, not the score.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from scalping.data.synthetic import SyntheticProvider
from scalping.datasets.builder import build_dataset
from scalping.features.build import build_features
from scalping.ml.models import build_model as build_ml_model
from scalping.nn import (
    WindowConfig,
    build_windowed_dataset,
    compare_predictors,
    momentum_rule_labels,
    predict_checkpoint,
    train_classifier,
)
from scalping.nn.train import TrainConfig


def make_bundle():
    frame = SyntheticProvider(seed=7).fetch(
        "XAUUSD", datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 7, tzinfo=UTC)
    )
    return build_dataset(build_features(frame), horizon=1, purge_bars=1)


def aligned_rows(bundle, ts: np.ndarray) -> np.ndarray:
    rows = np.searchsorted(bundle.ts, ts)
    assert np.all(bundle.ts[rows] == ts)
    return rows


def test_ml_rule_and_network_are_scored_on_the_same_rows(tmp_path) -> None:
    bundle = make_bundle()
    windowed = build_windowed_dataset(bundle, WindowConfig(window=20, clip=5.0))
    assert windowed.counts()["test"] > 0

    # 1) Phase 5: logistic regression on the same split boundaries.
    rows = aligned_rows(bundle, windowed.ts_test)
    train_rows = bundle.split == 0
    estimator = build_ml_model("logistic", seed=3)
    estimator.fit(bundle.X[train_rows], bundle.y_direction[train_rows])
    ml_proba = estimator.predict_proba(bundle.X[rows])
    ml_classes = tuple(int(value) for value in estimator.classes_)
    ml_labels = np.array(ml_classes, dtype=np.int64)[ml_proba.argmax(axis=1)]

    # 2) Rule: the CLI's ret_5 momentum baseline on the same timestamps.
    rule_labels = momentum_rule_labels(bundle, windowed.ts_test)

    # 3) Phase 6: a tiny network trained on the windowed splits.
    checkpoint = tmp_path / "mlp.pt"
    train_classifier(
        windowed,
        TrainConfig(
            model="mlp",
            epochs=2,
            batch_size=64,
            hidden=16,
            dropout=0.0,
            device="cpu",
            seed=1,
        ),
        checkpoint=checkpoint,
    )
    nn_proba, nn_labels, y_test, ts_test, _meta = predict_checkpoint(
        checkpoint, windowed, split="test", device="cpu"
    )
    assert np.array_equal(ts_test, windowed.ts_test)

    classes = tuple(sorted({int(value) for value in y_test.tolist()}))
    assert set(ml_classes) == set(classes), "train and test must share the class set"
    assert {int(value) for value in nn_labels.tolist()}.issubset(set(classes))
    assert nn_proba.shape[1] == len(classes)

    table = compare_predictors(
        y_test,
        {
            "ml:logistic": (ml_labels, ml_proba),
            "rule:ret_5": (rule_labels, None),
            "nn:mlp": (nn_labels, nn_proba),
        },
        classes=classes,
    )

    assert {row["name"] for row in table} == {"ml:logistic", "rule:ret_5", "nn:mlp"}
    assert all(row["rows"] == windowed.ts_test.shape[0] for row in table)
    for row in table:
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["balanced_accuracy"] <= 1.0
    # Probabilistic predictors report calibration metrics; the rule is labels only.
    by_name = {row["name"]: row for row in table}
    assert by_name["ml:logistic"]["log_loss"] is not None
    assert by_name["nn:mlp"]["log_loss"] is not None
    assert by_name["rule:ret_5"]["log_loss"] is None
    assert table == sorted(table, key=lambda row: (-row["accuracy"], row["name"]))
