"""Precision-target rejection gate: selection, ties and guard rails."""

from __future__ import annotations

import numpy as np
import pytest

from scalping.ml.rejection import (
    RejectionGate,
    accept_mask,
    rejection_summary,
    select_precision_threshold,
)

CLASSES = (0, 1)


def test_selects_the_lowest_threshold_that_holds_precision() -> None:
    probabilities = np.asarray(
        [[0.1, 0.9], [0.2, 0.8], [0.6, 0.4], [0.48, 0.52]], dtype=np.float64
    )
    labels = np.asarray([1, 1, 0, 0])

    choice = select_precision_threshold(
        probabilities, labels, target_precision=0.9, classes=CLASSES
    )

    assert choice.satisfied is True
    assert choice.threshold == pytest.approx(0.60)
    assert choice.precision == 1.0
    assert choice.coverage == 0.75
    assert choice.accepted == 3


def test_min_coverage_can_make_the_gate_impossible() -> None:
    probabilities = np.asarray(
        [[0.1, 0.9], [0.2, 0.8], [0.6, 0.4], [0.48, 0.52]], dtype=np.float64
    )
    labels = np.asarray([1, 1, 0, 0])

    choice = select_precision_threshold(
        probabilities,
        labels,
        target_precision=0.9,
        min_coverage=0.9,
        classes=CLASSES,
    )

    assert choice.satisfied is False
    assert choice.threshold is None
    assert choice.accepted == 0


def test_boundary_ties_enter_together() -> None:
    probabilities = np.asarray([[0.2, 0.8], [0.8, 0.2]], dtype=np.float64)
    labels = np.asarray([1, 1])

    choice = select_precision_threshold(
        probabilities, labels, target_precision=0.5, classes=CLASSES
    )

    # 0.80 is accepted as one group: precision 0.5 over two rows.
    assert choice.threshold == pytest.approx(0.80)
    assert choice.accepted == 2
    assert choice.precision == 0.5


def test_summary_and_mask_on_the_same_threshold() -> None:
    probabilities = np.asarray(
        [[0.1, 0.9], [0.2, 0.8], [0.6, 0.4]], dtype=np.float64
    )
    labels = np.asarray([1, 1, 0])

    summary = rejection_summary(probabilities, labels, 0.8, classes=CLASSES)
    mask = accept_mask(probabilities, 0.8)

    assert summary.accepted == 2
    assert summary.precision == 1.0
    assert summary.coverage == pytest.approx(2 / 3)
    assert mask.tolist() == [True, True, False]


def test_gate_wraps_selection_and_acceptance() -> None:
    probabilities = np.asarray(
        [[0.1, 0.9], [0.2, 0.8], [0.6, 0.4]], dtype=np.float64
    )
    labels = np.asarray([1, 1, 0])
    choice = select_precision_threshold(
        probabilities, labels, target_precision=0.9, classes=CLASSES
    )

    gate = RejectionGate(threshold=choice.threshold or 0.0, validation=choice)

    assert gate.accepts(np.asarray([[0.1, 0.9], [0.45, 0.55]])).tolist() == [True, False]
    assert gate.summary(probabilities, labels, classes=CLASSES).accepted == 3


def test_guard_rails() -> None:
    probabilities = np.asarray([[0.1, 0.9], [0.6, 0.4]], dtype=np.float64)
    labels = np.asarray([1, 0])
    with pytest.raises(ValueError, match="target_precision"):
        select_precision_threshold(probabilities, labels, target_precision=0)
    with pytest.raises(ValueError, match="min_coverage"):
        select_precision_threshold(
            probabilities, labels, target_precision=0.9, min_coverage=1.5
        )
    with pytest.raises(ValueError, match="match the class list"):
        select_precision_threshold(
            probabilities, labels, target_precision=0.9, classes=(0, 1, 2)
        )
    with pytest.raises(ValueError, match="outside the class list"):
        select_precision_threshold(
            probabilities, np.asarray([1, 2]), target_precision=0.9, classes=CLASSES
        )
    with pytest.raises(ValueError, match="threshold"):
        accept_mask(probabilities, 1.5)
    with pytest.raises(ValueError, match="positive sum"):
        select_precision_threshold(
            np.zeros((2, 2)), labels, target_precision=0.9, classes=CLASSES
        )
