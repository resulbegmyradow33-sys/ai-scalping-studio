"""Probability-threshold rejection gate selected on validation rows.

Calibration answers "does 70% mean 70%". A rejection gate answers a different
question: below which confidence should the model not be allowed to trade?
This module chooses the *lowest* top-class probability threshold whose accepted
rows still reach a requested precision, subject to a minimum coverage. Selecting
on validation (never test) keeps the gate out of the evaluation sample.

The sweep accepts a whole tie group at once, so rows sharing the boundary
probability are never split: the reported precision is what a live gate at that
threshold would actually see.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np

__all__ = [
    "RejectionGate",
    "ThresholdChoice",
    "accept_mask",
    "rejection_summary",
    "select_precision_threshold",
]

DEFAULT_CLASSES: Final[tuple[int, ...]] = (-1, 0, 1)


@dataclass(frozen=True, slots=True)
class ThresholdChoice:
    """The selected top-probability threshold and its validation profile.

    ``threshold`` is ``None`` when no threshold satisfied both
    ``target_precision`` and ``min_coverage``; accepted/precision/coverage are
    then zero. ``satisfied`` makes that outcome explicit instead of silently
    returning an unsafe gate.
    """

    threshold: float | None
    precision: float
    coverage: float
    accepted: int
    total: int
    target_precision: float
    min_coverage: float
    satisfied: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RejectionGate:
    """A fixed threshold plus the validation statistics used to choose it."""

    threshold: float
    validation: ThresholdChoice

    def accepts(self, probabilities: Any) -> np.ndarray:
        return accept_mask(probabilities, self.threshold)

    def summary(self, probabilities: Any, labels: Any, *, classes: tuple[int, ...] = DEFAULT_CLASSES):
        return rejection_summary(
            probabilities, labels, self.threshold, classes=classes
        )


def _prepare(
    probabilities: Any, labels: Any, classes: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty (n, k>=2) matrix")
    if matrix.shape[1] != len(classes):
        raise ValueError("probability columns must match the class list")
    if not np.isfinite(matrix).all():
        raise ValueError("probabilities must be finite")
    if (matrix < 0.0).any():
        raise ValueError("probabilities must be non-negative")
    row_sums = matrix.sum(axis=1, keepdims=True)
    if (row_sums <= 0.0).any():
        raise ValueError("every probability row must have a positive sum")
    matrix = matrix / row_sums

    label_array = np.asarray(labels)
    if label_array.ndim != 1 or label_array.shape[0] != matrix.shape[0]:
        raise ValueError("labels must be one value per probability row")
    index_of = {label: column for column, label in enumerate(classes)}
    try:
        columns = np.asarray([index_of[int(label)] for label in label_array])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("labels contain a class outside the class list") from error
    return matrix, label_array, columns


def accept_mask(probabilities: Any, threshold: float) -> np.ndarray:
    """Rows whose top probability is at or above ``threshold``."""
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty (n, k>=2) matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("probabilities must be finite")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return matrix.max(axis=1) >= float(threshold)


def rejection_summary(
    probabilities: Any,
    labels: Any,
    threshold: float,
    *,
    classes: tuple[int, ...] = DEFAULT_CLASSES,
) -> ThresholdChoice:
    """Precision/coverage of a fixed threshold on the rows it is given."""
    matrix, _, columns = _prepare(probabilities, labels, classes)
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    top = matrix.max(axis=1)
    predicted = matrix.argmax(axis=1)
    accepted = top >= float(threshold)
    count = int(accepted.sum())
    correct = int((predicted[accepted] == columns[accepted]).sum())
    total = int(matrix.shape[0])
    precision = correct / count if count else 0.0
    return ThresholdChoice(
        threshold=float(threshold),
        precision=precision,
        coverage=count / total,
        accepted=count,
        total=total,
        target_precision=0.0,
        min_coverage=0.0,
        satisfied=count > 0,
    )


def select_precision_threshold(
    probabilities: Any,
    labels: Any,
    *,
    target_precision: float,
    min_coverage: float = 0.0,
    classes: tuple[int, ...] = DEFAULT_CLASSES,
) -> ThresholdChoice:
    """Lowest top-probability threshold meeting precision and coverage floors.

    Rows are ranked by top probability; a tie group enters together. The sweep
    walks groups from most to least confident and remembers the largest accepted
    set that still reaches ``target_precision`` (and ``min_coverage``).
    """
    if not 0.0 < float(target_precision) <= 1.0:
        raise ValueError("target_precision must be in (0, 1]")
    if not 0.0 <= float(min_coverage) <= 1.0:
        raise ValueError("min_coverage must be in [0, 1]")

    matrix, _, columns = _prepare(probabilities, labels, classes)
    predicted = matrix.argmax(axis=1)
    top = matrix.max(axis=1)
    correct = (predicted == columns).astype(np.int64)

    order = np.argsort(-top, kind="stable")
    ranked_top = top[order]
    ranked_correct = correct[order]
    cumulative = np.cumsum(ranked_correct)

    best: ThresholdChoice | None = None
    # The last index of each equal-probability run is a group boundary.
    boundaries = np.flatnonzero(np.r_[ranked_top[1:] != ranked_top[:-1], True])
    total = int(matrix.shape[0])
    for end in boundaries:
        accepted = int(end) + 1
        precision = float(cumulative[end]) / accepted
        coverage = accepted / total
        if precision < float(target_precision) or coverage < float(min_coverage):
            continue
        best = ThresholdChoice(
            threshold=float(ranked_top[end]),
            precision=precision,
            coverage=coverage,
            accepted=accepted,
            total=total,
            target_precision=float(target_precision),
            min_coverage=float(min_coverage),
            satisfied=True,
        )

    if best is not None:
        return best
    return ThresholdChoice(
        threshold=None,
        precision=0.0,
        coverage=0.0,
        accepted=0,
        total=total,
        target_precision=float(target_precision),
        min_coverage=float(min_coverage),
        satisfied=False,
    )
