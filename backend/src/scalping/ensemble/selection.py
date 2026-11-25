"""Keep the simpler model unless the ensemble earns its complexity.

`compare_to_best_member` reports the scores; this module makes the promotion
choice explicit and auditable. Ties go to the single member: an ensemble must
**strictly** improve on the best member by ``min_gain`` before it is kept.

The choice is descriptive. It does not promote a model into the live path; that
still requires the simulator, a cost stress and the promotion gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Final, Literal

__all__ = ["PromotionChoice", "select_simpler_winner"]

Metric = Literal["accuracy", "balanced_accuracy", "log_loss"]
_METRICS: Final[tuple[str, ...]] = ("accuracy", "balanced_accuracy", "log_loss")
_HIGHER_IS_BETTER: Final[frozenset[str]] = frozenset({"accuracy", "balanced_accuracy"})


@dataclass(frozen=True, slots=True)
class PromotionChoice:
    """The kept name, the reason and the ensemble's gain on the chosen metric."""

    choice: str
    reason: str
    metric: str
    ensemble_gain: float
    best_member: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "reason": self.reason,
            "metric": self.metric,
            "ensemble_gain": self.ensemble_gain,
            "best_member": self.best_member,
        }


def _value(row: dict[str, Any], metric: str) -> float:
    if metric not in row or row[metric] is None:
        raise ValueError(f"row {row.get('name')!r} has no {metric!r} value")
    value = float(row[metric])
    if not isfinite(value):
        raise ValueError(f"row {row.get('name')!r} has a non-finite {metric!r}")
    return value


def select_simpler_winner(
    table: list[dict[str, Any]],
    *,
    metric: Metric = "accuracy",
    min_gain: float = 0.0,
) -> PromotionChoice:
    """Choose the ensemble only when it beats the best member by ``min_gain``.

    ``table`` is the output of
    :func:`~scalping.ensemble.core.compare_to_best_member`: one row per member
    plus one row named ``ensemble``. For ``log_loss`` a lower value is better;
    for accuracy metrics a higher value is better.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")
    if min_gain < 0.0:
        raise ValueError("min_gain must be >= 0")
    if not table:
        raise ValueError("comparison table is empty")
    names = [row.get("name") for row in table]
    if names.count("ensemble") != 1:
        raise ValueError("comparison table must contain exactly one 'ensemble' row")
    members = [row for row in table if row.get("name") != "ensemble"]
    if not members:
        raise ValueError("comparison table has no member rows")

    ensemble = next(row for row in table if row.get("name") == "ensemble")
    direction = 1.0 if metric in _HIGHER_IS_BETTER else -1.0
    best_member = max(members, key=lambda row: direction * _value(row, metric))
    ensemble_value = _value(ensemble, metric)
    member_value = _value(best_member, metric)
    gain = direction * (ensemble_value - member_value)

    if gain > min_gain:
        return PromotionChoice(
            choice="ensemble",
            reason="ensemble_beats_best_member",
            metric=metric,
            ensemble_gain=gain,
            best_member=str(best_member["name"]),
        )
    return PromotionChoice(
        choice=str(best_member["name"]),
        reason="keep_simpler_member",
        metric=metric,
        ensemble_gain=gain,
        best_member=str(best_member["name"]),
    )
