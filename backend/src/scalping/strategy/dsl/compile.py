"""Interpreter for the strategy DSL.

Compilation is interpretation, not code generation. :func:`compile_policy`
validates the spec and returns a :class:`Policy` whose ``decide`` method reads
one row plus a caller-owned history. There is no I/O, no randomness and no
module-level mutable state, so replaying the same rows produces the same
decisions.

Missing or non-finite inputs are three-valued: an unknown value propagates
through ``And``/``Or``/``Not`` and never counts as a true condition. A strategy
cannot accidentally fire on a NaN.

Precedence in :meth:`Policy.decide` is fixed:

1. ``NO_TRADE`` when the no-trade condition is true;
2. ``CLOSE`` when a held position reached its holding limit;
3. ``CLOSE`` when any exit condition is true;
4. ``BUY``/``SELL`` for long/short entries (a simultaneous long and short is a
   ``NO_TRADE`` conflict rather than an arbitrary choice);
5. ``HOLD`` otherwise.

``evaluate_series`` returns the position-free action per bar. It does not open
or close positions, so the holding limit appears only through ``decide`` when
the caller supplies a position and ``bars_held``.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from scalping.features.build import FEATURE_COLUMNS
from scalping.strategy.dsl.nodes import (
    SIDES,
    And,
    Between,
    Change,
    Compare,
    Const,
    CrossAbove,
    CrossBelow,
    Fall,
    Feature,
    Not,
    Or,
    Rise,
    StrategySpec,
)
from scalping.strategy.dsl.validate import validate

__all__ = [
    "ACTIONS",
    "REASONS",
    "Decision",
    "Policy",
    "PolicyState",
    "compile_policy",
]

ACTIONS: Final[tuple[str, ...]] = ("HOLD", "BUY", "SELL", "CLOSE", "NO_TRADE")

REASONS: Final[tuple[str, ...]] = (
    "hold",
    "entry_long",
    "entry_short",
    "exit",
    "holding_limit",
    "no_trade",
    "conflicting_entry",
)

_COMPARATORS: Final[dict[str, Callable[[float, float], bool]]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}

# Rows are ``name -> value``; a value that cannot be read is ``None``.
_Row = Mapping[str, float]
_Truth = bool | None
_History = Sequence[_Row]


@dataclass(frozen=True, slots=True)
class Decision:
    """One bar's policy output.

    ``version`` is the DSL spec version, not a code release. ``holding_limit``
    repeats the policy's limit so an execution layer can enforce it even when it
    tracks positions itself.
    """

    action: str
    reason: str
    version: int
    holding_limit: int | None = None

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action {self.action!r}; allowed: {', '.join(ACTIONS)}")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("Decision.reason must be a non-empty string")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("Decision.version must be an int")
        if self.holding_limit is not None and (
            isinstance(self.holding_limit, bool) or not isinstance(self.holding_limit, int)
        ):
            raise ValueError("Decision.holding_limit must be an int or None")


@dataclass(slots=True)
class PolicyState:
    """Caller-owned replay state.

    ``history`` holds **previous** rows, oldest first; the current row is passed
    separately to :meth:`Policy.decide` and is not stored there. ``position`` is
    optional: when set, the holding limit can close it. ``bars_held`` counts
    completed bars since entry.
    """

    history: list[_Row] = field(default_factory=list)
    position: str | None = None
    bars_held: int = 0
    max_history: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.history, list):
            raise TypeError("PolicyState.history must be a list")
        if self.position is not None and self.position not in SIDES:
            raise ValueError(f"PolicyState.position must be None or one of {SIDES}")
        if isinstance(self.bars_held, bool) or not isinstance(self.bars_held, int):
            raise TypeError("PolicyState.bars_held must be an int")
        if self.bars_held < 0:
            raise ValueError("PolicyState.bars_held must be >= 0")
        if isinstance(self.max_history, bool) or not isinstance(self.max_history, int):
            raise TypeError("PolicyState.max_history must be an int")
        if self.max_history < 0:
            raise ValueError("PolicyState.max_history must be >= 0")

    def push(self, row: _Row) -> None:
        """Append a completed row and trim to ``max_history`` when set."""
        self.history.append(dict(row))
        if self.max_history and len(self.history) > self.max_history:
            del self.history[: len(self.history) - self.max_history]


def _lookup(row: _Row, name: str) -> float | None:
    if name not in row:
        return None
    value = row[name]
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _number(node: object, bars: Sequence[_Row], index: int) -> float | None:
    """Evaluate a numeric node at ``index``; ``None`` means unknown."""
    if isinstance(node, Feature):
        if index < 0:
            return None
        return _lookup(bars[index], node.name)
    if isinstance(node, Const):
        return float(node.value)
    if isinstance(node, Change):
        now = _number(node.source, bars, index)
        then = _number(node.source, bars, index - node.window)
        if now is None or then is None:
            return None
        return now - then
    raise TypeError(f"{type(node).__name__} is not a numeric DSL node")


def _and(left: _Truth, right: _Truth) -> _Truth:
    if left is False or right is False:
        return False
    if left is None or right is None:
        return None
    return True


def _or(left: _Truth, right: _Truth) -> _Truth:
    if left is True or right is True:
        return True
    if left is None or right is None:
        return None
    return False


def _truth(node: object, bars: Sequence[_Row], index: int) -> _Truth:
    """Evaluate a condition at ``index`` with three-valued logic."""
    if isinstance(node, Compare):
        left = _number(node.left, bars, index)
        right = _number(node.right, bars, index)
        if left is None or right is None:
            return None
        return _COMPARATORS[node.op](left, right)
    if isinstance(node, And):
        return _and(_truth(node.left, bars, index), _truth(node.right, bars, index))
    if isinstance(node, Or):
        return _or(_truth(node.left, bars, index), _truth(node.right, bars, index))
    if isinstance(node, Not):
        value = _truth(node.operand, bars, index)
        return None if value is None else not value
    if isinstance(node, Between):
        number = _number(node.value, bars, index)
        low = _number(node.low, bars, index)
        high = _number(node.high, bars, index)
        if number is None or low is None or high is None:
            return None
        return low <= number <= high
    if isinstance(node, (CrossAbove, CrossBelow)):
        left_now = _number(node.left, bars, index)
        right_now = _number(node.right, bars, index)
        left_then = _number(node.left, bars, index - node.window)
        right_then = _number(node.right, bars, index - node.window)
        if None in (left_now, right_now, left_then, right_then):
            return None
        assert left_now is not None and right_now is not None
        assert left_then is not None and right_then is not None
        if isinstance(node, CrossAbove):
            return left_now > right_now and left_then <= right_then
        return left_now < right_now and left_then >= right_then
    if isinstance(node, (Rise, Fall)):
        now = _number(node.source, bars, index)
        then = _number(node.source, bars, index - node.window)
        if now is None or then is None:
            return None
        return now > then if isinstance(node, Rise) else now < then
    raise TypeError(f"{type(node).__name__} is not a boolean DSL node")


@dataclass(frozen=True, slots=True)
class Policy:
    """A validated, immutable strategy ready to decide bar by bar."""

    spec: StrategySpec

    def __post_init__(self) -> None:
        validate(self.spec)

    @property
    def version(self) -> int:
        """DSL spec version carried into every decision."""
        return self.spec.spec_version

    def new_state(self) -> PolicyState:
        """A state pre-trimmed to the spec's maximum lookback."""
        return PolicyState(max_history=self.spec.max_lookback)

    def decide(self, row: _Row, state: PolicyState) -> Decision:
        """Decide one bar from ``row`` and the state's previous rows."""
        if not isinstance(state, PolicyState):
            raise TypeError(f"state must be a PolicyState, got {type(state).__name__}")
        bars: list[_Row] = [*state.history, row]
        index = len(bars) - 1
        limit = None if self.spec.holding is None else self.spec.holding.bars

        if (
            self.spec.no_trade is not None
            and _truth(self.spec.no_trade.condition, bars, index) is True
        ):
            return Decision("NO_TRADE", "no_trade", self.version, limit)

        if state.position is not None and limit is not None and state.bars_held >= limit:
            return Decision("CLOSE", "holding_limit", self.version, limit)

        for exit_rule in self.spec.exits:
            if _truth(exit_rule.condition, bars, index) is True:
                return Decision("CLOSE", "exit", self.version, limit)

        long_entry = any(
            entry.side == "long" and _truth(entry.condition, bars, index) is True
            for entry in self.spec.entries
        )
        short_entry = any(
            entry.side == "short" and _truth(entry.condition, bars, index) is True
            for entry in self.spec.entries
        )
        if long_entry and short_entry:
            return Decision("NO_TRADE", "conflicting_entry", self.version, limit)
        if long_entry:
            return Decision("BUY", "entry_long", self.version, limit)
        if short_entry:
            return Decision("SELL", "entry_short", self.version, limit)
        return Decision("HOLD", "hold", self.version, limit)

    def evaluate_series(self, frame: object) -> np.ndarray:
        """Position-free actions for every row of ``frame``.

        ``frame`` is either a
        :class:`~scalping.features.build.FeatureFrame`-like object with ``X`` and
        ``columns``, or a mapping of column name to a 1-D sequence. Missing
        feature names make the conditions that need them unknown (false at the
        top level), never an exception.
        """
        columns, matrix = _frame_arrays(frame)
        rows = len(matrix)
        history: list[_Row] = []
        actions: list[str] = []
        for index in range(rows):
            row = {name: float(matrix[index, column]) for column, name in enumerate(columns)}
            state = PolicyState(history=history)
            actions.append(self.decide(row, state).action)
            history.append(row)
            if len(history) > self.spec.max_lookback:
                del history[0 : len(history) - self.spec.max_lookback]
        return np.asarray(actions, dtype=object)


def _frame_arrays(frame: object) -> tuple[list[str], np.ndarray]:
    """Normalise a FeatureFrame-like object or a column mapping."""
    if hasattr(frame, "X"):
        raw = frame.X
        matrix = np.asarray(raw, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"frame.X must be 2-D, got shape {matrix.shape}")
        raw_columns = getattr(frame, "columns", None)
        columns = list(raw_columns) if raw_columns else []
        if columns and len(columns) != matrix.shape[1]:
            raise ValueError(
                f"frame.columns has {len(columns)} names for {matrix.shape[1]} columns"
            )
        if not columns:
            if matrix.shape[1] != len(FEATURE_COLUMNS):
                raise ValueError(
                    "frame has no column names and its width does not match the "
                    f"{len(FEATURE_COLUMNS)} causal features"
                )
            columns = list(FEATURE_COLUMNS)
        return columns, matrix

    if isinstance(frame, Mapping):
        names = list(frame)
        if not names:
            raise ValueError("frame mapping must contain at least one column")
        arrays: list[np.ndarray] = []
        for name in names:
            array = np.asarray(frame[name], dtype=np.float64)
            if array.ndim != 1:
                raise ValueError(f"column {name!r} must be 1-D, got shape {array.shape}")
            arrays.append(array)
        length = len(arrays[0])
        for name, array in zip(names, arrays, strict=True):
            if len(array) != length:
                raise ValueError(f"column {name!r} has length {len(array)}, expected {length}")
        matrix = np.column_stack(arrays) if arrays else np.zeros((0, 0), dtype=np.float64)
        return names, matrix

    raise TypeError("frame must be a FeatureFrame-like object with X/columns or a column mapping")


def compile_policy(spec: StrategySpec) -> Policy:
    """Validate ``spec`` and return an immutable interpreter for it."""
    validate(spec)
    return Policy(spec)
