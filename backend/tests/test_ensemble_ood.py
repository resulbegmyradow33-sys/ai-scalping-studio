"""Out-of-distribution scoring and the decision-engine gate."""

from __future__ import annotations

import numpy as np
import pytest

from scalping.ensemble import OODConfig, OODDetector
from scalping.ensemble.decision import REASON_OOD, DecisionEngine


def training(n: int = 400, features: int = 4, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, features))


def test_detects_a_far_point_and_keeps_training_rows() -> None:
    X = training()
    detector = OODDetector().fit(X)

    scores = detector.score(X)
    mask = detector.is_ood(X)

    assert scores.shape == (X.shape[0],)
    assert detector.threshold is not None and detector.threshold > 0
    assert mask.mean() < 0.05
    assert detector.is_ood(np.full((1, X.shape[1]), 50.0)).tolist() == [True]
    assert float(detector.score(np.full((1, X.shape[1]), 50.0))[0]) > detector.threshold


def test_scoring_is_deterministic_and_shape_checked() -> None:
    X = training()
    first = OODDetector().fit(X)
    second = OODDetector().fit(X)

    np.testing.assert_array_equal(first.score(X[:10]), second.score(X[:10]))

    with pytest.raises(ValueError, match="features"):
        first.score(np.zeros((2, X.shape[1] + 1)))
    with pytest.raises(ValueError, match="2-D"):
        first.score(np.zeros(3))
    with pytest.raises(ValueError, match="non-finite"):
        first.score(np.full((1, X.shape[1]), np.nan))


def test_fit_guard_rails() -> None:
    with pytest.raises(ValueError, match="at least"):
        OODDetector().fit(np.zeros((10, 3)))
    with pytest.raises(ValueError, match="empty"):
        OODDetector().fit(np.empty((0, 3)))
    with pytest.raises(ValueError, match="quantile"):
        OODConfig(quantile=0.4)
    with pytest.raises(ValueError, match="shrinkage"):
        OODConfig(shrinkage=0.0)
    with pytest.raises(ValueError, match="min_rows"):
        OODConfig(min_rows=2)


def test_unfitted_detector_raises() -> None:
    detector = OODDetector()

    with pytest.raises(ValueError, match="not fitted"):
        detector.score(training(40))
    with pytest.raises(ValueError, match="not fitted"):
        detector.is_ood(training(40))
    with pytest.raises(ValueError, match="not fitted"):
        detector.to_dict()


def test_metadata_describes_the_fit() -> None:
    X = training()
    metadata = OODDetector().fit(X).to_dict()

    assert metadata["n_features"] == X.shape[1]
    assert metadata["training_rows"] == X.shape[0]
    assert metadata["quantile"] == pytest.approx(0.99)
    assert metadata["threshold"] > 0


def test_decision_engine_blocks_new_entries_when_unknown() -> None:
    engine = DecisionEngine(entry_threshold=0.55, classes=(-1, 0, 1), model_version="test")
    probabilities = (0.05, 0.15, 0.80)

    known = engine.decide(probabilities, disagreement=0.10)
    unknown = engine.decide(probabilities, disagreement=0.10, unknown=True)

    assert known.action == "BUY"
    assert unknown.action == "NO_TRADE"
    assert unknown.reason == REASON_OOD


def test_unknown_does_not_force_an_exit_on_an_open_position() -> None:
    engine = DecisionEngine(entry_threshold=0.55, classes=(-1, 0, 1))
    probabilities = (0.05, 0.15, 0.80)

    decision = engine.decide(
        probabilities, disagreement=0.10, position="long", unknown=True
    )

    assert decision.action == "HOLD"
    assert decision.reason == "hold_long"
