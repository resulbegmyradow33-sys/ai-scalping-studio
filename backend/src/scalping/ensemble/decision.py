"""Turn ensemble probabilities into an explicit trading decision.

The engine is a **pure function of its inputs**: probabilities, disagreement,
the current position and the configured thresholds. It does not read a clock, a
broker or a risk model. ``ts`` and ``model_version`` are metadata copied into the
decision log, not inputs to the rule.

It cannot bypass risk or the simulator. ``BUY``/``SELL`` here mean "the signal
is strong enough to ask for an entry", not "an order was placed". Position
sizing, stops, cost vetoes and the paper ledger live in later layers; an open
position is passed in as ``position`` and the engine only says whether to keep
it. A ``NO_TRADE`` always carries a machine-readable reason, and every decision
carries one, so a log can explain every row without guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Literal

import numpy as np

__all__ = ["ACTIONS", "Action", "Decision", "DecisionEngine", "Position"]

Action = Literal["BUY", "SELL", "HOLD", "NO_TRADE"]
Position = Literal["long", "short"]

ACTIONS: Final[tuple[str, ...]] = ("BUY", "SELL", "HOLD", "NO_TRADE")

#: Reasons the engine can emit. Kept as plain strings so a log query is simple.
REASON_DISAGREEMENT: Final[str] = "disagreement_above_cap"
REASON_BELOW_ENTRY: Final[str] = "below_entry_threshold"
REASON_FLAT_CLASS: Final[str] = "flat_class"
REASON_LOW_EDGE: Final[str] = "edge_below_min_edge"
REASON_NON_DIRECTIONAL: Final[str] = "non_directional_class"
REASON_HOLD_LONG: Final[str] = "hold_long"
REASON_HOLD_SHORT: Final[str] = "hold_short"
REASON_EXIT_LONG: Final[str] = "exit_long"
REASON_EXIT_SHORT: Final[str] = "exit_short"
REASON_ENTRY_LONG: Final[str] = "entry_long"
REASON_ENTRY_SHORT: Final[str] = "entry_short"
REASON_OOD: Final[str] = "out_of_distribution"


@dataclass(frozen=True, slots=True)
class Decision:
    """One decision-log row.

    ``probabilities`` is aligned with ``classes``. :meth:`to_dict` keys the
    probabilities by class label so the log is readable without the class tuple.
    """

    action: Action
    probabilities: tuple[float, ...]
    classes: tuple[int, ...]
    disagreement: float
    regime: str | None
    reason: str
    model_version: str
    ts: str

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action {self.action!r}; choose from {ACTIONS}")
        if not self.reason or not self.reason.strip():
            raise ValueError("every decision needs a non-empty reason")
        if len(self.probabilities) != len(self.classes):
            raise ValueError("probabilities and classes must have the same length")
        values = np.asarray(self.probabilities, dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError("decision probabilities must be finite and non-negative")
        if not np.isclose(float(values.sum()), 1.0, atol=1e-6):
            raise ValueError("decision probabilities must sum to 1")
        if not 0.0 <= float(self.disagreement) <= 1.0:
            raise ValueError("disagreement must be in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        """A JSON-ready row for the decision log."""
        return {
            "ts": self.ts,
            "action": self.action,
            "classes": list(self.classes),
            "probabilities": {
                str(label): float(probability)
                for label, probability in zip(self.classes, self.probabilities, strict=True)
            },
            "disagreement": float(self.disagreement),
            "regime": self.regime,
            "reason": self.reason,
            "model_version": self.model_version,
        }


def _normalise(probabilities: object, classes: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if values.size != len(classes):
        raise ValueError(
            f"expected {len(classes)} probabilities for classes {classes}, got {values.size}"
        )
    if not np.isfinite(values).all():
        raise ValueError("probabilities contain non-finite values")
    if (values < 0.0).any():
        raise ValueError("probabilities contain negative values")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("probabilities must sum to a positive value")
    return values / total


def _timestamp(value: str | datetime | None) -> str:
    if value is None:
        return datetime.now(UTC).isoformat(timespec="seconds")
    if isinstance(value, datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return moment.astimezone(UTC).isoformat(timespec="seconds")
    return str(value)


class DecisionEngine:
    """Threshold rules for ``BUY``/``SELL``/``HOLD``/``NO_TRADE``.

    Entry requires all of: disagreement at or below the cap, top probability at
    or above ``entry_threshold``, a directional top class, and a top-two gap of
    at least ``min_edge``. An open position is kept as ``HOLD`` while the held
    class is still the argmax and its probability is at or above
    ``exit_threshold``; otherwise the engine emits ``NO_TRADE`` with an exit
    reason. Flipping a position is left to the risk layer.

    The disagreement cap gates **new** risk only. With an open position the
    score is logged but does not force an exit: an ensemble disagreement is not
    a risk model.
    """

    def __init__(
        self,
        *,
        entry_threshold: float = 0.55,
        exit_threshold: float = 0.45,
        max_disagreement: float = 0.35,
        min_edge: float = 0.05,
        classes: tuple[int, ...] = (-1, 0, 1),
        model_version: str = "unspecified",
    ) -> None:
        if not 0.0 < entry_threshold <= 1.0:
            raise ValueError("entry_threshold must be in (0, 1]")
        if not 0.0 <= exit_threshold < entry_threshold:
            raise ValueError("exit_threshold must be in [0, entry_threshold)")
        if not 0.0 <= max_disagreement <= 1.0:
            raise ValueError("max_disagreement must be in [0, 1]")
        if not 0.0 <= min_edge <= 1.0:
            raise ValueError("min_edge must be in [0, 1]")
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError("classes must be at least two unique labels")
        self.entry_threshold = float(entry_threshold)
        self.exit_threshold = float(exit_threshold)
        self.max_disagreement = float(max_disagreement)
        self.min_edge = float(min_edge)
        self.classes = tuple(int(value) for value in classes)
        self.model_version = str(model_version)

    def decide(
        self,
        probabilities: object,
        *,
        disagreement: float,
        regime: str | None = None,
        position: Position | None = None,
        ts: str | datetime | None = None,
        model_version: str | None = None,
        unknown: bool = False,
    ) -> Decision:
        """Return one decision; never emits a decision without a reason.

        ``disagreement`` is required. A default of zero would let a caller who
        forgot the ensemble score silently bypass the cap and write a false
        ``0.0`` into the decision log. ``unknown`` is the OOD flag: it blocks a
        new entry but never forces an exit on an open position.
        """
        if position not in (None, "long", "short"):
            raise ValueError(f"unknown position {position!r}; choose long, short or None")
        score = float(disagreement)
        if not 0.0 <= score <= 1.0:
            raise ValueError("disagreement must be in [0, 1]")
        values = _normalise(probabilities, self.classes)
        stamp = _timestamp(ts)
        version = self.model_version if model_version is None else str(model_version)

        def emit(action: Action, reason: str) -> Decision:
            return Decision(
                action=action,
                probabilities=tuple(float(value) for value in values),
                classes=self.classes,
                disagreement=score,
                regime=regime,
                reason=reason,
                model_version=version,
                ts=stamp,
            )

        if position is not None:
            held = 1 if position == "long" else -1
            if held not in self.classes:
                raise ValueError(f"classes {self.classes} do not contain the held side {held}")
            held_probability = float(values[self.classes.index(held)])
            is_top = int(np.argmax(values)) == self.classes.index(held)
            if is_top and held_probability >= self.exit_threshold:
                return emit("HOLD", REASON_HOLD_LONG if held == 1 else REASON_HOLD_SHORT)
            return emit("NO_TRADE", REASON_EXIT_LONG if held == 1 else REASON_EXIT_SHORT)

        if unknown:
            return emit("NO_TRADE", REASON_OOD)
        if score > self.max_disagreement:
            return emit("NO_TRADE", REASON_DISAGREEMENT)

        order = np.argsort(values)
        top_index = int(order[-1])
        top = float(values[top_index])
        edge = top - float(values[int(order[-2])])
        if top < self.entry_threshold:
            return emit("NO_TRADE", REASON_BELOW_ENTRY)
        if 0 in self.classes and top_index == self.classes.index(0):
            return emit("NO_TRADE", REASON_FLAT_CLASS)
        if edge < self.min_edge:
            return emit("NO_TRADE", REASON_LOW_EDGE)

        label = self.classes[top_index]
        if label > 0:
            return emit("BUY", REASON_ENTRY_LONG)
        if label < 0:
            return emit("SELL", REASON_ENTRY_SHORT)
        return emit("NO_TRADE", REASON_NON_DIRECTIONAL)
