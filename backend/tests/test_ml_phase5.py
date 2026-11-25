"""Phase 5 additions: SGD incremental path, anomaly scoring, importances.

These tests pin the boundaries that matter: the two task types never mix, SGD
chunking is deterministic, the anomaly route has no direction probability and no
promotion path, and tree importances are aligned with the fitted columns.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scalping.datasets.builder import build_dataset
from scalping.features.build import FEATURE_COLUMNS, FeatureFrame
from scalping.ml.anomaly import (
    AnomalySpec,
    detect_anomalies,
    score_anomaly_split,
)
from scalping.ml.incremental import incremental_fit
from scalping.ml.models import (
    ALL_MODEL_NAMES,
    ANOMALY_MODEL_NAMES,
    MODEL_NAMES,
    MODEL_TASKS,
    build_anomaly_model,
    build_model,
    feature_importances,
    is_promotion_candidate,
    model_task,
)
from scalping.ml.train import TrainSpec, train_classifier

_TREE_MODELS = ("random_forest", "extra_trees", "lightgbm", "xgboost")


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
    return FeatureFrame(
        X=X,
        ts=ts,
        ohlc=np.column_stack((close, close + 0.01, close - 0.01, close)),
        spread=np.full(n, 0.2),
        atr=np.full(n, 0.5),
        valid_from=valid_from,
        columns=list(FEATURE_COLUMNS),
    )


def _variable_frame(n: int = 600, valid_from: int = 10) -> FeatureFrame:
    """Varying returns and features, so anomaly scores and labels are not flat."""
    rng = np.random.default_rng(4)
    returns = rng.normal(0.0, 0.0012, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    X = rng.normal(size=(n, len(FEATURE_COLUMNS))).astype(np.float32)
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


# --- registry ---------------------------------------------------------------


def test_registry_lists_cover_both_task_types() -> None:
    assert set(MODEL_NAMES).isdisjoint(ANOMALY_MODEL_NAMES)
    assert set(ALL_MODEL_NAMES) == set(MODEL_TASKS)
    assert "sgd" in MODEL_NAMES
    assert "isolation_forest" in ANOMALY_MODEL_NAMES
    assert all(model_task(name) == "direction" for name in MODEL_NAMES)
    assert model_task("isolation_forest") == "anomaly"
    assert all(is_promotion_candidate(name) for name in MODEL_NAMES)
    assert is_promotion_candidate("isolation_forest") is False


def test_unknown_and_anomaly_names_are_rejected_by_the_classifier_factory() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model("magic")
    with pytest.raises(ValueError, match="anomaly detector"):
        build_model("isolation_forest")
    with pytest.raises(ValueError, match="unknown anomaly model"):
        build_anomaly_model("matrix_profile")
    with pytest.raises(ValueError, match="unknown model"):
        model_task("magic")


# --- SGD --------------------------------------------------------------------


def test_sgd_trains_as_a_direction_classifier() -> None:
    result = train_classifier(
        _learnable_frame(),
        symbol="XAUUSD",
        timeframe="M1",
        spec=TrainSpec(model="sgd", horizon=1, purge_bars=1, embargo_bars=1, seed=3),
    )
    assert sorted(int(value) for value in result.model.classes_) == [-1, 1]
    assert result.validation["accuracy"] > 0.9
    assert result.validation["log_loss"] < result.baseline["log_loss"]


def test_sgd_batch_and_incremental_fits_are_deterministic() -> None:
    frame = _learnable_frame()
    spec = TrainSpec(model="sgd", horizon=1, purge_bars=1, embargo_bars=1, seed=11)
    batch_a = train_classifier(frame, symbol="XAUUSD", timeframe="M1", spec=spec)
    batch_b = train_classifier(frame, symbol="XAUUSD", timeframe="M1", spec=spec)
    assert np.array_equal(batch_a.model.coef_, batch_b.model.coef_)

    chunked = replace(spec, incremental_chunks=4)
    inc_a = train_classifier(frame, symbol="XAUUSD", timeframe="M1", spec=chunked)
    inc_b = train_classifier(frame, symbol="XAUUSD", timeframe="M1", spec=chunked)
    assert np.array_equal(inc_a.model.coef_, inc_b.model.coef_)
    assert inc_a.validation == inc_b.validation


def test_sgd_incremental_metrics_track_the_batch_fit() -> None:
    """Same seed/data: deterministic, and close on the synthetic signal.

    The two paths visit rows in a different order, so this is a tolerance check,
    not an equality claim. Larger real-data gaps are documented in
    ``docs/classical-ml.md`` and must be measured, not assumed away.
    """
    frame = _learnable_frame()
    spec = TrainSpec(model="sgd", horizon=1, purge_bars=1, embargo_bars=1, seed=5)
    batch = train_classifier(frame, symbol="XAUUSD", timeframe="M1", spec=spec)
    inc = train_classifier(
        frame,
        symbol="XAUUSD",
        timeframe="M1",
        spec=replace(spec, incremental_chunks=4),
    )
    assert abs(batch.validation["accuracy"] - inc.validation["accuracy"]) <= 0.02
    assert abs(batch.validation["log_loss"] - inc.validation["log_loss"]) <= 0.05
    assert list(batch.model.classes_) == list(inc.model.classes_)


def test_incremental_fit_keeps_the_class_set_fixed() -> None:
    X = np.arange(40, dtype=np.float32).reshape(20, 2)
    y = np.array([-1, 1] * 10, dtype=np.int8)
    # The second chunk is all +1: the class set still comes from the full y.
    model = incremental_fit(build_model("sgd", seed=0), X, y, chunks=2)
    assert list(model.classes_) == [-1, 1]
    assert model.n_features_in_ == 2


def test_incremental_fit_rejects_bad_inputs() -> None:
    X = np.arange(40, dtype=np.float32).reshape(20, 2)
    y = np.array([-1, 1] * 10, dtype=np.int8)
    with pytest.raises(ValueError, match="outside classes"):
        incremental_fit(build_model("sgd", seed=0), X, y, chunks=2, classes=np.array([-1, 2]))
    with pytest.raises(ValueError, match="chunks"):
        incremental_fit(build_model("sgd", seed=0), X, y, chunks=0)
    with pytest.raises(ValueError, match="exceeds"):
        incremental_fit(build_model("sgd", seed=0), X, y, chunks=21)
    with pytest.raises(ValueError, match="no partial_fit"):
        incremental_fit(build_model("random_forest", seed=0), X, y, chunks=2)
    with pytest.raises(ValueError, match="no partial_fit"):
        train_classifier(
            _learnable_frame(),
            symbol="XAUUSD",
            timeframe="M1",
            spec=TrainSpec(
                model="logistic",
                horizon=1,
                purge_bars=1,
                embargo_bars=1,
                incremental_chunks=2,
            ),
        )


# --- importances ------------------------------------------------------------


@pytest.mark.parametrize("model", _TREE_MODELS)
def test_feature_importances_are_aligned_and_finite(model: str) -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(240, 7)).astype(np.float32)
    y = np.where(X[:, 0] > 0, 1, -1).astype(np.int8)
    fitted = build_model(model, seed=0).fit(X, y)
    importances = feature_importances(fitted)
    assert importances is not None
    assert importances.shape == (X.shape[1],)
    assert np.isfinite(importances).all()
    assert (importances >= 0.0).all()


def test_feature_importances_are_none_when_unsupported_or_unfitted() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 5)).astype(np.float32)
    y = np.where(X[:, 0] > 0, 1, -1).astype(np.int8)
    for model in ("logistic", "sgd", "hist_gradient"):
        assert feature_importances(build_model(model, seed=0).fit(X, y)) is None
    assert feature_importances(build_model("random_forest", seed=0)) is None
    assert feature_importances(object()) is None


# --- anomaly ----------------------------------------------------------------


def _anomaly_report(**kwargs):
    defaults = {
        "symbol": "XAUUSD",
        "timeframe": "M1",
        "spec": AnomalySpec(horizon=1, purge_bars=1, embargo_bars=1, seed=2, quantile=0.95),
        "splits": ("validation", "test"),
    }
    defaults.update(kwargs)
    return detect_anomalies(_variable_frame(), **defaults)


def test_anomaly_threshold_comes_from_the_training_quantile() -> None:
    report = _anomaly_report()
    detector = report.detector
    assert detector.threshold == pytest.approx(float(np.quantile(detector.train_scores, 0.95)))
    train_rate = float((detector.train_scores > detector.threshold).mean())
    assert abs(train_rate - 0.05) < 0.02


def test_anomaly_scores_are_finite_on_every_scored_split() -> None:
    report = _anomaly_report()
    for split in ("validation", "test"):
        score = report.splits[split]
        assert score.rows == report.split_counts[split] > 0
        assert 0.0 <= score.anomaly_rate <= 1.0
        assert np.isfinite(
            [
                score.threshold,
                score.score_mean,
                score.score_std,
                score.score_min,
                score.score_max,
            ]
        ).all()
        assert score.score_min <= score.score_mean <= score.score_max

    scores = report.detector.score(_variable_frame().X[:16])
    assert scores.shape == (16,)
    assert np.isfinite(scores).all()
    flags = report.detector.flag(_variable_frame().X[:16])
    assert flags.shape == (16,)
    assert flags.dtype == np.bool_


def test_anomaly_separation_is_reported_only_when_labels_exist() -> None:
    labelled = _anomaly_report(magnitude_quantile=0.9)
    separation = labelled.splits["validation"].separation
    assert separation is not None
    assert separation["metric"] == "roc_auc"
    assert separation["auc"] is None or 0.0 <= separation["auc"] <= 1.0
    assert 0.0 < separation["positive_rate"] < 1.0
    assert labelled.magnitude_cutoff is not None
    assert labelled.magnitude_cutoff > 0.0

    unlabelled = _anomaly_report(magnitude_quantile=None)
    assert unlabelled.splits["validation"].separation is None
    assert unlabelled.magnitude_cutoff is None


def test_single_class_labels_report_no_auc() -> None:
    report = _anomaly_report(magnitude_quantile=None)
    frame = _variable_frame()
    bundle = build_dataset(
        frame,
        symbol="XAUUSD",
        timeframe="M1",
        groups=AnomalySpec().groups,
        horizon=1,
        purge_bars=1,
        embargo_bars=1,
    )
    rows = int(bundle.mask("validation").sum())
    score = score_anomaly_split(
        report.detector,
        bundle,
        split="validation",
        labels=np.ones(rows, dtype=bool),
    )
    assert score.separation is not None
    assert score.separation["auc"] is None
    assert score.separation["reason"] == "single-class labels"


def test_anomaly_cannot_be_a_promotion_candidate_or_a_direction_model() -> None:
    report = _anomaly_report()
    assert report.task == "anomaly"
    assert report.promotion_eligible is False
    assert report.detector.task == "anomaly"
    assert report.detector.promotion_eligible is False
    assert not hasattr(report.detector.model, "predict_proba")
    assert report.to_dict()["promotion_eligible"] is False

    with pytest.raises(ValueError, match="anomaly"):
        train_classifier(
            _learnable_frame(),
            symbol="XAUUSD",
            timeframe="M1",
            spec=TrainSpec(model="isolation_forest", horizon=1, purge_bars=1, embargo_bars=1),
        )


def test_anomaly_spec_rejects_an_out_of_range_quantile() -> None:
    with pytest.raises(ValueError, match="quantile"):
        AnomalySpec(quantile=1.0)
    with pytest.raises(ValueError, match="magnitude_quantile"):
        detect_anomalies(
            _variable_frame(),
            symbol="XAUUSD",
            timeframe="M1",
            spec=AnomalySpec(horizon=1),
            magnitude_quantile=1.0,
        )
