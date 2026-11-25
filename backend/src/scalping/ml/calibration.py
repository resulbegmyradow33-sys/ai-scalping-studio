"""Probability calibration diagnostics and temperature scaling.

A softmax/probability output is not a confidence guarantee. These helpers
measure whether "70% up" actually happens about 70% of the time, using the
positive class (+1) and fixed-width probability bins. ``fit_temperature`` fits
one deterministic multiclass temperature on the rows it is given (validation,
never test) and reports log loss, Brier, ECE and MCE before and after it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

__all__ = [
    "TemperatureFit",
    "apply_temperature",
    "calibration_summary",
    "expected_calibration_error",
    "fit_temperature",
    "reliability_curve",
]

DEFAULT_BINS: Final[int] = 10

#: Probabilities are clipped before taking logs so an exact 0/1 prediction does
#: not become an infinite logit. The clip is far below any meaningful probability.
_TINY: Final[float] = 1e-12
#: Search range for the fitted temperature. Outside this range the model is so
#: broken that a monotone rescaling cannot say anything useful.
MIN_TEMPERATURE: Final[float] = 0.05
MAX_TEMPERATURE: Final[float] = 20.0
_GRID_POINTS: Final[int] = 201


def _positive_column(classes: tuple[int, ...], positive: int) -> int | None:
    try:
        return classes.index(positive)
    except ValueError:
        return None


def reliability_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: tuple[int, ...],
    *,
    positive: int = 1,
    bins: int = DEFAULT_BINS,
) -> list[dict[str, Any]]:
    """Bin the predicted probability of ``positive`` and compare with outcomes."""
    if bins < 2:
        raise ValueError("bins must be >= 2")
    column = _positive_column(classes, positive)
    if column is None:
        return []
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("probabilities must be (rows, classes)")
    if probabilities.shape[0] != y_true.shape[0]:
        raise ValueError("y_true and probabilities must have the same rows")
    p = np.clip(probabilities[:, column].astype(np.float64), 0.0, 1.0)
    hit = (y_true == positive).astype(np.float64)
    index = np.minimum((p * bins).astype(np.int64), bins - 1)
    rows: list[dict[str, Any]] = []
    for bucket in range(bins):
        mask = index == bucket
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin": bucket,
                "lower": bucket / bins,
                "upper": (bucket + 1) / bins,
                "count": count,
                "mean_probability": float(p[mask].mean()),
                "observed_rate": float(hit[mask].mean()),
            }
        )
    return rows


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: tuple[int, ...],
    *,
    positive: int = 1,
    bins: int = DEFAULT_BINS,
) -> float:
    """Weighted absolute gap between predicted and observed positive rates."""
    rows = reliability_curve(y_true, probabilities, classes, positive=positive, bins=bins)
    total = sum(int(row["count"]) for row in rows)
    if total == 0:
        return 0.0
    return float(
        sum(
            int(row["count"])
            / total
            * abs(float(row["mean_probability"]) - float(row["observed_rate"]))
            for row in rows
        )
    )


def calibration_summary(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: tuple[int, ...],
    *,
    positive: int = 1,
    bins: int = DEFAULT_BINS,
    min_bin_count: int = 20,
) -> dict[str, Any]:
    """ECE, a guarded MCE, and the full reliability table.

    MCE ignores bins with fewer than ``min_bin_count`` rows: a single
    overconfident row must not become the headline calibration error. ECE still
    weights every row, so that row contributes only its 1/N share.
    """
    rows = reliability_curve(y_true, probabilities, classes, positive=positive, bins=bins)
    total = sum(int(row["count"]) for row in rows)
    gaps = [
        abs(float(row["mean_probability"]) - float(row["observed_rate"]))
        for row in rows
        if int(row["count"]) >= min_bin_count
    ]
    return {
        "bins": rows,
        "ece": expected_calibration_error(
            y_true, probabilities, classes, positive=positive, bins=bins
        ),
        "mce": float(max(gaps)) if gaps else 0.0,
        "mce_min_bin_count": min_bin_count,
        "rows": total,
    }


@dataclass(frozen=True, slots=True)
class TemperatureFit:
    """One fitted temperature and the same-split metrics before/after it."""

    temperature: float
    rows: int
    classes: tuple[int, ...]
    log_loss_before: float
    log_loss_after: float
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float
    mce_before: float
    mce_after: float
    improved: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "temperature",
            "temperature": self.temperature,
            "rows": self.rows,
            "classes": list(self.classes),
            "improved": self.improved,
            "before": {
                "log_loss": self.log_loss_before,
                "brier": self.brier_before,
                "ece": self.ece_before,
                "mce": self.mce_before,
            },
            "after": {
                "log_loss": self.log_loss_after,
                "brier": self.brier_after,
                "ece": self.ece_after,
                "mce": self.mce_after,
            },
        }


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("probabilities must be a non-empty (rows, classes) matrix")
    if not np.isfinite(array).all():
        raise ValueError("probabilities must be finite")
    if (array < 0.0).any():
        raise ValueError("probabilities must be non-negative")
    row_sums = array.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("probability rows must sum to 1")
    return array


def _validate_labels(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: tuple[int, ...],
) -> tuple[np.ndarray, dict[int, int]]:
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("classes must be a non-empty tuple of unique labels")
    if probabilities.shape[1] != len(classes):
        raise ValueError(
            f"probabilities have {probabilities.shape[1]} columns but {len(classes)} classes"
        )
    labels = np.asarray(y_true)
    if labels.ndim != 1 or labels.shape[0] != probabilities.shape[0]:
        raise ValueError("y_true and probabilities must have the same rows")
    column_of = {int(label): index for index, label in enumerate(classes)}
    if not np.isin(labels, np.asarray(classes)).all():
        raise ValueError("y_true contains a label that is not in classes")
    return labels, column_of


def _logits(probabilities: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probabilities, _TINY, 1.0))


def _softmax_scaled(logits: np.ndarray, temperature: float) -> np.ndarray:
    """``softmax(logits / T)``; the row constant in ``log p`` cancels."""
    scaled_logits = logits / temperature
    scaled_logits -= scaled_logits.max(axis=1, keepdims=True)
    scaled = np.exp(scaled_logits)
    scaled /= scaled.sum(axis=1, keepdims=True)
    return scaled


def _scaled(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    return _softmax_scaled(_logits(probabilities), temperature)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale class probabilities by one temperature ``T > 0``.

    This is exactly temperature scaling on the model's logits: for
    ``p = softmax(z)``, normalising ``p ** (1 / T)`` equals ``softmax(z / T)``.
    ``T > 1`` softens an overconfident model, ``T < 1`` sharpens an
    underconfident one, and ``T = 1`` is the identity (up to float round-off).
    """
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a finite positive number")
    array = _validate_probabilities(probabilities)
    return _scaled(array, float(temperature))


def _label_columns(labels: np.ndarray, column_of: dict[int, int]) -> np.ndarray:
    return np.fromiter((column_of[int(label)] for label in labels), dtype=np.int64)


def _nll_probabilities(
    probabilities: np.ndarray,
    columns: np.ndarray,
) -> float:
    picked = probabilities[np.arange(columns.shape[0]), columns]
    return float(-np.mean(np.log(np.clip(picked, _TINY, 1.0))))


def _nll(
    labels: np.ndarray,
    probabilities: np.ndarray,
    column_of: dict[int, int],
) -> float:
    return _nll_probabilities(probabilities, _label_columns(labels, column_of))


def _brier(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: tuple[int, ...],
) -> float:
    one_hot = np.zeros_like(probabilities)
    for index, label in enumerate(classes):
        one_hot[labels == label, index] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _minimise_nll(
    labels: np.ndarray,
    probabilities: np.ndarray,
    column_of: dict[int, int],
    *,
    min_temperature: float,
    max_temperature: float,
) -> float:
    """Deterministic 1-D search over ``log T``; no scipy, no randomness.

    A coarse grid brackets the optimum (and includes ``T = 1`` exactly), then a
    golden-section pass refines the bracket. Ties are broken toward ``T = 1``
    so an already calibrated model keeps the identity transform instead of
    drifting to a grid edge.
    """
    logits = _logits(probabilities)
    columns = _label_columns(labels, column_of)

    def objective(log_temperature: float) -> float:
        scaled = _softmax_scaled(logits, math.exp(log_temperature))
        return _nll_probabilities(scaled, columns)

    lower, upper = math.log(min_temperature), math.log(max_temperature)
    grid = np.linspace(lower, upper, _GRID_POINTS)
    values = np.asarray([objective(float(u)) for u in grid])
    best_value = float(values.min())
    candidates = np.flatnonzero(values <= best_value + 1e-12)
    # ``candidates`` is non-empty; pick the one closest to log(1) = 0.
    best_index = int(candidates[np.argmin(np.abs(grid[candidates]))])
    best_u = float(grid[best_index])
    if values[best_index] > best_value + 1e-12:  # pragma: no cover - defensive
        best_u = float(grid[int(np.argmin(values))])

    lo = float(grid[max(best_index - 1, 0)])
    hi = float(grid[min(best_index + 1, _GRID_POINTS - 1)])
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - ratio * (hi - lo)
    d = lo + ratio * (hi - lo)
    fc = objective(c)
    fd = objective(d)
    for _ in range(128):
        if fc <= fd:
            hi, d, fd = d, c, fc
            c = hi - ratio * (hi - lo)
            fc = objective(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + ratio * (hi - lo)
            fd = objective(d)
        if hi - lo < 1e-10:
            break
    refined = (lo + hi) / 2.0
    refined_value = objective(refined)
    # Only leave the grid for a real gain. A flat objective (perfectly
    # calibrated data) then keeps the ``T = 1`` grid point instead of drifting
    # to a bracket edge on float round-off.
    if refined_value < best_value - 1e-12:
        best_u = refined
    return min(max(math.exp(best_u), min_temperature), max_temperature)


def fit_temperature(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: tuple[int, ...],
    *,
    min_temperature: float = MIN_TEMPERATURE,
    max_temperature: float = MAX_TEMPERATURE,
) -> TemperatureFit:
    """Fit one multiclass temperature by minimising log loss.

    The fit is deterministic and uses only the rows passed in: it must be
    called on validation rows, never on the test split. Refuses non-finite
    probabilities, rows/classes mismatches and unknown labels instead of
    quietly dropping them.
    """
    if not math.isfinite(min_temperature) or not math.isfinite(max_temperature):
        raise ValueError("temperature bounds must be finite")
    if not 0.0 < min_temperature < max_temperature:
        raise ValueError("temperature bounds must satisfy 0 < min < max")
    array = _validate_probabilities(probabilities)
    labels, column_of = _validate_labels(y_true, array, classes)

    temperature = _minimise_nll(
        labels,
        array,
        column_of,
        min_temperature=float(min_temperature),
        max_temperature=float(max_temperature),
    )

    after = _scaled(array, temperature)
    before_summary = calibration_summary(labels, array, classes)
    after_summary = calibration_summary(labels, after, classes)
    log_loss_before = _nll(labels, array, column_of)
    log_loss_after = _nll(labels, after, column_of)
    return TemperatureFit(
        temperature=temperature,
        rows=int(labels.shape[0]),
        classes=tuple(int(label) for label in classes),
        log_loss_before=log_loss_before,
        log_loss_after=log_loss_after,
        brier_before=_brier(labels, array, classes),
        brier_after=_brier(labels, after, classes),
        ece_before=float(before_summary["ece"]),
        ece_after=float(after_summary["ece"]),
        mce_before=float(before_summary["mce"]),
        mce_after=float(after_summary["mce"]),
        improved=log_loss_after < log_loss_before,
    )
