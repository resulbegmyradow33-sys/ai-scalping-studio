"""Temperature scaling: fitting, refusals, and the ``ml train`` report."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from scalping.features.build import FEATURE_COLUMNS, FeatureFrame
from scalping.ml.calibration import apply_temperature, fit_temperature
from scalping.ml.train import (
    TrainSpec,
    load_result,
    save_result,
    train_classifier,
)

CLASSES = (-1, 1)


def _binary_probabilities(logits: np.ndarray) -> np.ndarray:
    p = 1.0 / (1.0 + np.exp(-logits))
    return np.column_stack((1.0 - p, p))


def _calibrated_sample(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Labels drawn from the true probability, so ``T = 1`` is calibrated."""
    rng = np.random.default_rng(seed)
    base = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    logits = np.tile(base, n // base.size + 1)[:n]
    truth = 1.0 / (1.0 + np.exp(-logits))
    labels = np.where(rng.random(n) < truth, 1, -1)
    return labels, logits


def _log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    picked = np.where(y_true == 1, probabilities[:, 1], probabilities[:, 0])
    return float(-np.mean(np.log(np.clip(picked, 1e-12, 1.0))))


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
    return FeatureFrame(
        X=X,
        ts=ts,
        ohlc=np.column_stack((close, close + 0.01, close - 0.01, close)),
        spread=np.full(n, 0.2),
        atr=np.full(n, 0.5),
        valid_from=valid_from,
        columns=list(FEATURE_COLUMNS),
    )


def test_overconfident_fit_softens_and_underconfident_fit_sharpens() -> None:
    labels, logits = _calibrated_sample()

    over = fit_temperature(labels, _binary_probabilities(2.0 * logits), CLASSES)
    under = fit_temperature(labels, _binary_probabilities(0.5 * logits), CLASSES)

    assert 1.6 < over.temperature < 2.6
    assert 0.35 < under.temperature < 0.65
    for fit in (over, under):
        assert fit.log_loss_after < fit.log_loss_before
        assert fit.brier_after < fit.brier_before
        assert fit.ece_after < fit.ece_before
        assert fit.improved is True
        assert fit.rows == labels.size


def test_temperature_one_is_the_identity() -> None:
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    assert np.allclose(apply_temperature(probabilities, 1.0), probabilities, atol=1e-12)

    # A perfectly calibrated, perfectly flat forecast has no temperature
    # signal: the fit must keep T = 1 instead of drifting to a grid edge.
    labels = np.array([1, -1] * 500)
    flat = np.column_stack((np.full(1000, 0.5), np.full(1000, 0.5)))
    assert fit_temperature(labels, flat, CLASSES).temperature == 1.0


def test_log_loss_improves_monotonically_on_a_hand_case() -> None:
    labels = np.array([1] * 6 + [-1] * 4 + [1] * 5 + [-1] * 5)
    probabilities = np.column_stack(
        [np.r_[np.full(10, 0.1), np.full(10, 0.4)], np.r_[np.full(10, 0.9), np.full(10, 0.6)]]
    )
    fit = fit_temperature(labels, probabilities, CLASSES)
    before = _log_loss(labels, probabilities)
    after = _log_loss(labels, apply_temperature(probabilities, fit.temperature))

    assert after < before
    # Walking from the identity toward the fitted temperature must never make
    # log loss worse on this case.
    path = np.geomspace(1.0, fit.temperature, 25)
    losses = [_log_loss(labels, apply_temperature(probabilities, t)) for t in path]
    assert all(later <= earlier + 1e-12 for earlier, later in itertools.pairwise(losses))

    # The fitted value is the best point on a dense sweep.
    candidates = (
        np.geomspace(0.1, 1.0, 200) if fit.temperature < 1.0 else np.geomspace(1.0, 5.0, 200)
    )
    sweep = min(_log_loss(labels, apply_temperature(probabilities, t)) for t in candidates)
    assert after <= sweep + 1e-9


def test_fit_is_deterministic() -> None:
    labels, logits = _calibrated_sample(n=1500, seed=7)
    probabilities = _binary_probabilities(1.8 * logits)
    first = fit_temperature(labels, probabilities, CLASSES)
    second = fit_temperature(labels, probabilities, CLASSES)
    assert first.temperature == second.temperature
    assert first.to_dict() == second.to_dict()


def test_non_finite_and_mismatched_inputs_are_refused() -> None:
    labels = np.array([-1, 1, -1, 1])
    good = _binary_probabilities(np.array([-1.0, 1.0, -0.5, 0.5]))

    nan = good.copy()
    nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_temperature(labels, nan, CLASSES)

    infinite = good.copy()
    infinite[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        fit_temperature(labels, infinite, CLASSES)

    with pytest.raises(ValueError, match="same rows"):
        fit_temperature(labels[:3], good, CLASSES)

    with pytest.raises(ValueError, match="columns"):
        fit_temperature(labels, np.full((4, 3), 1.0 / 3.0), CLASSES)

    with pytest.raises(ValueError, match="not in classes"):
        fit_temperature(np.array([-1, 1, 0, 1]), good, CLASSES)

    with pytest.raises(ValueError, match="sum to 1"):
        fit_temperature(labels, good * 0.5, CLASSES)

    with pytest.raises(ValueError, match="positive"):
        apply_temperature(good, 0.0)


def test_calibration_never_fits_on_the_test_split() -> None:
    frame = _learnable_frame()
    base = TrainSpec(
        model="logistic",
        horizon=1,
        purge_bars=1,
        embargo_bars=1,
        evaluate_test=True,
        seed=3,
    )
    raw = train_classifier(frame, symbol="XAUUSD", timeframe="M1", spec=base)
    calibrated = train_classifier(
        frame,
        symbol="XAUUSD",
        timeframe="M1",
        spec=TrainSpec(
            model="logistic",
            horizon=1,
            purge_bars=1,
            embargo_bars=1,
            evaluate_test=True,
            seed=3,
            calibrate="temperature",
        ),
    )

    assert calibrated.calibration is not None
    assert calibrated.calibration["fitted_on"] == "validation"
    assert calibrated.calibration["rows"] == calibrated.split_counts["validation"]
    # Raw validation and the whole test block are identical: calibration only
    # added a side report and never re-scored either split.
    assert calibrated.validation == raw.validation
    assert calibrated.test == raw.test
    assert calibrated.test is not None
    assert calibrated.test["calibration"]["rows"] == calibrated.split_counts["test"]


def test_report_fields_survive_save_and_load(tmp_path) -> None:
    result = train_classifier(
        _learnable_frame(),
        symbol="XAUUSD",
        timeframe="M1",
        spec=TrainSpec(
            model="logistic",
            horizon=1,
            purge_bars=1,
            embargo_bars=1,
            calibrate="temperature",
        ),
    )
    _, metrics_path = save_result(result, tmp_path / "run")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert payload["spec"]["calibrate"] == "temperature"
    calibration = payload["calibration"]
    assert calibration["method"] == "temperature"
    assert calibration["fitted_on"] == "validation"
    assert isinstance(calibration["temperature"], float)
    assert calibration["temperature"] > 0.0
    assert set(calibration["before"]) == {"log_loss", "brier", "ece", "mce"}
    assert set(calibration["after"]) == {"log_loss", "brier", "ece", "mce"}
    assert calibration["after"]["log_loss"] <= calibration["before"]["log_loss"]

    loaded = load_result(metrics_path.parent)
    assert loaded.spec.calibrate == "temperature"
    assert loaded.calibration == result.calibration
    assert "calibration" in result.to_dict()


def test_unknown_calibration_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown calibration"):
        train_classifier(
            _learnable_frame(),
            symbol="XAUUSD",
            timeframe="M1",
            spec=TrainSpec(
                model="logistic",
                horizon=1,
                purge_bars=1,
                embargo_bars=1,
                calibrate="platt",
            ),
        )
