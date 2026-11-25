"""Out-of-distribution scoring for ensemble decision rows.

A model can be confident on a row that looks nothing like its training data.
This detector fits a diagonal-shrunk Gaussian on the **training features** and
scores a new row by its Mahalanobis distance; rows above the training quantile
are flagged ``unknown``.

The score is descriptive: it does not predict returns and it cannot place an
order. In :class:`~scalping.ensemble.decision.DecisionEngine` an unknown row may
block a **new** entry, never force an exit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["OODConfig", "OODDetector"]


@dataclass(frozen=True, slots=True)
class OODConfig:
    """Training quantile and covariance guard rails."""

    quantile: float = 0.99
    shrinkage: float = 1e-6
    min_rows: int = 30

    def __post_init__(self) -> None:
        if not 0.5 < self.quantile < 1.0:
            raise ValueError("quantile must be in (0.5, 1)")
        if not 0.0 < self.shrinkage <= 1e-1:
            raise ValueError("shrinkage must be in (0, 0.1]")
        if self.min_rows < 3:
            raise ValueError("min_rows must be >= 3")


def _matrix(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D (rows, features), got shape {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


class OODDetector:
    """Fit on training features, score new rows with a robust Mahalanobis distance."""

    def __init__(self, config: OODConfig | None = None) -> None:
        self.config = OODConfig() if config is None else config
        self._mean: np.ndarray | None = None
        self._inverse: np.ndarray | None = None
        self._threshold: float | None = None
        self._rows = 0

    @property
    def fitted(self) -> bool:
        return self._mean is not None

    @property
    def n_features(self) -> int:
        return 0 if self._mean is None else int(self._mean.size)

    @property
    def threshold(self) -> float | None:
        return self._threshold

    def fit(self, X: object) -> OODDetector:
        """Fit the centre, shrunk covariance and the training score quantile."""
        matrix = _matrix(X, name="X")
        if matrix.shape[0] < self.config.min_rows:
            raise ValueError(
                f"need at least {self.config.min_rows} training rows, got {matrix.shape[0]}"
            )
        mean = matrix.mean(axis=0)
        centred = matrix - mean
        covariance = np.cov(centred, rowvar=False)
        covariance = np.atleast_2d(covariance)
        scale = float(np.trace(covariance)) / max(1, covariance.shape[0])
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        covariance = covariance + np.eye(covariance.shape[0]) * self.config.shrinkage * scale
        self._mean = mean
        self._inverse = np.linalg.pinv(covariance)
        self._rows = int(matrix.shape[0])
        scores = self._scores(matrix)
        self._threshold = float(np.quantile(scores, self.config.quantile))
        return self

    def _scores(self, matrix: np.ndarray) -> np.ndarray:
        if self._mean is None or self._inverse is None:
            raise ValueError("detector is not fitted; call fit() first")
        if matrix.shape[1] != self._mean.size:
            raise ValueError(
                f"expected {self._mean.size} features, got {matrix.shape[1]}"
            )
        centred = matrix - self._mean
        return np.einsum("ij,jk,ik->i", centred, self._inverse, centred)

    def score(self, X: object) -> np.ndarray:
        """Mahalanobis scores for new rows (higher is more unusual)."""
        matrix = _matrix(X, name="X")
        return self._scores(matrix)

    def is_ood(self, X: object) -> np.ndarray:
        """Boolean mask for rows above the fitted training quantile."""
        if self._threshold is None:
            raise ValueError("detector is not fitted; call fit() first")
        return self.score(X) > self._threshold

    def to_dict(self) -> dict[str, Any]:
        """Metadata for a report; no model parameters are persisted here."""
        if self._threshold is None:
            raise ValueError("detector is not fitted; call fit() first")
        return {
            "quantile": self.config.quantile,
            "threshold": self._threshold,
            "n_features": self.n_features,
            "training_rows": self._rows,
            "shrinkage": self.config.shrinkage,
        }
