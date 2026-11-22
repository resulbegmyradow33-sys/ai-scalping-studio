"""Probability calibration diagnostics.

A softmax/probability output is not a confidence guarantee. These helpers
measure whether "70% up" actually happens about 70% of the time, using the
positive class (+1) and fixed-width probability bins.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

__all__ = ["calibration_summary", "expected_calibration_error", "reliability_curve"]

DEFAULT_BINS: Final[int] = 10


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
    rows = reliability_curve(
        y_true, probabilities, classes, positive=positive, bins=bins
    )
    total = sum(int(row["count"]) for row in rows)
    if total == 0:
        return 0.0
    return float(
        sum(
            int(row["count"]) / total
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
    rows = reliability_curve(
        y_true, probabilities, classes, positive=positive, bins=bins
    )
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
