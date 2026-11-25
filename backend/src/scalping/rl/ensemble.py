"""Policy averaging for seed ensembles.

A seed ensemble averages the members' action distributions before taking the
argmax. Averaging probabilities is safer than majority-voting actions: a member
that is uncertain contributes proportion, and the result stays a proper
distribution the decision layer can calibrate.

This module is the pure combining rule plus a thin callable wrapper. Training N
seeds, loading their run directories and comparing the ensemble against each
member on the simulator is a separate evaluation slice.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Final

import numpy as np

__all__ = ["PolicyEnsemble", "average_probabilities"]

Policy = Callable[[np.ndarray], Sequence[float]]


def _probabilities(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} probabilities are empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} probabilities contain non-finite values")
    if (array < 0.0).any():
        raise ValueError(f"{name} probabilities contain negative values")
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError(f"{name} probabilities must sum to a positive value")
    return array / total


def average_probabilities(
    distributions: Sequence[Sequence[float]],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Weighted mean of action distributions, normalised.

    All members must have the same length. Weights are optional, non-negative
    and need a positive sum; the default is equal weight.
    """
    if not distributions:
        raise ValueError("need at least one member distribution")
    matrix = np.stack(
        [_probabilities(values, name=f"member {index}") for index, values in enumerate(distributions)]
    )
    if weights is None:
        mean = matrix.mean(axis=0)
    else:
        if len(weights) != len(distributions):
            raise ValueError(
                f"weights must match members ({len(distributions)}), got {len(weights)}"
            )
        vector = np.asarray(weights, dtype=np.float64).reshape(-1)
        if not np.isfinite(vector).all() or (vector < 0.0).any():
            raise ValueError("weights must be finite and non-negative")
        total = float(vector.sum())
        if total <= 0.0:
            raise ValueError("weights must sum to a positive value")
        mean = (matrix * (vector / total)[:, None]).sum(axis=0)
    return mean / mean.sum()


class PolicyEnsemble:
    """Average callable policies that return action probabilities."""

    def __init__(
        self,
        policies: Sequence[Policy],
        *,
        weights: Sequence[float] | None = None,
    ) -> None:
        if not policies:
            raise ValueError("need at least one policy")
        if not all(callable(policy) for policy in policies):
            raise TypeError("every ensemble member must be callable")
        if weights is not None and len(weights) != len(policies):
            raise ValueError(
                f"weights must match members ({len(policies)}), got {len(weights)}"
            )
        self.policies: Final[tuple[Policy, ...]] = tuple(policies)
        self.weights = None if weights is None else tuple(float(value) for value in weights)

    def probabilities(self, observation: np.ndarray) -> np.ndarray:
        """Combined distribution for one observation."""
        return average_probabilities(
            [policy(observation) for policy in self.policies], self.weights
        )

    def act(self, observation: np.ndarray) -> int:
        """Argmax of the combined distribution; ties go to the lowest index."""
        return int(np.argmax(self.probabilities(observation)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "members": len(self.policies),
            "weights": None if self.weights is None else list(self.weights),
        }
