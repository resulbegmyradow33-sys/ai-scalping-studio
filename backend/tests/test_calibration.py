"""Reliability bins and expected calibration error."""

from __future__ import annotations

import numpy as np

from scalping.ml.calibration import (
    calibration_summary,
    expected_calibration_error,
    reliability_curve,
)

CLASSES = (-1, 1)


def _probabilities(p_up: float, n: int = 1000) -> np.ndarray:
    p = np.full(n, p_up, dtype=np.float64)
    return np.column_stack((1.0 - p, p))


def test_perfect_calibration_has_zero_ece() -> None:
    n = 1000
    p = np.full(n, 0.5)
    y = np.where(np.arange(n) % 2 == 0, 1, -1)
    probabilities = np.column_stack((1.0 - p, p))
    assert expected_calibration_error(y, probabilities, CLASSES) < 0.02
    rows = reliability_curve(y, probabilities, CLASSES)
    assert len(rows) == 1
    assert rows[0]["count"] == n


def test_overconfidence_is_visible() -> None:
    n = 1000
    y = np.where(np.arange(n) % 2 == 0, 1, -1)
    ece = expected_calibration_error(y, _probabilities(0.9, n), CLASSES)
    assert ece > 0.35
    summary = calibration_summary(y, _probabilities(0.9, n), CLASSES)
    assert summary["ece"] > 0.35
    assert summary["mce"] >= summary["ece"]


def test_missing_positive_class_returns_empty_curve() -> None:
    y = np.array([-1, -1, -1])
    probabilities = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert reliability_curve(y, probabilities, (2, 3)) == []
    assert expected_calibration_error(y, probabilities, (2, 3)) == 0.0


def test_mce_ignores_tiny_bins() -> None:
    n = 100
    p = np.full(n, 0.5)
    y = np.where(np.arange(n) % 2 == 0, 1, -1)
    probabilities = np.vstack([np.column_stack((1 - p, p)), [[0.01, 0.99]]])
    y = np.concatenate([y, [-1]])
    summary = calibration_summary(y, probabilities, CLASSES, min_bin_count=20)
    assert summary["mce"] < 0.05
    assert summary["ece"] < 0.02
