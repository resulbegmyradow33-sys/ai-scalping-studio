"""Structural and type validation for strategy specs.

Validation is the safety gate between generated data and the interpreter. It
checks the whole tree, not just the root:

* only the declared node classes are accepted (an arbitrary object, a callable
  or a ``dict`` is rejected);
* every operand has the right kind (number vs condition);
* operators, sides and finite-number rules hold;
* every window is a positive backward offset within ``spec.max_lookback``, so no
  node can reach a bar after the decision bar;
* there is at least one entry and one exit;
* depth and node-count caps hold.

``validate`` is intentionally cheap and side-effect free. It never imports user
code and never invokes a dynamic expression evaluator; it only walks dataclasses.
"""

from __future__ import annotations

import math
from typing import Final

from scalping.strategy.dsl.nodes import (
    COMPARISON_OPERATORS,
    DSL_SPEC_VERSION,
    FEATURE_NAMES,
    MAX_ABSOLUTE_LOOKBACK,
    MAX_DEPTH,
    MAX_HOLDING_BARS,
    MAX_NODES,
    SIDES,
    And,
    AtrMultiple,
    Between,
    Change,
    Compare,
    Const,
    CrossAbove,
    CrossBelow,
    DslValidationError,
    EntryRule,
    ExitRule,
    Fall,
    Feature,
    FixedFraction,
    HoldingLimit,
    Not,
    NoTrade,
    Or,
    Rise,
    StrategySpec,
)

__all__ = ["validate"]

_NUMBER: Final[str] = "number"
_CONDITION: Final[str] = "condition"

_EXPRESSIONS: Final[tuple[type, ...]] = (
    Feature,
    Const,
    Compare,
    And,
    Or,
    Not,
    Between,
    CrossAbove,
    CrossBelow,
    Rise,
    Fall,
    Change,
)


class _Budget:
    """Mutable traversal budget shared by one validation call."""

    __slots__ = ("nodes",)

    def __init__(self) -> None:
        self.nodes = 0

    def tick(self, label: str) -> None:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise DslValidationError(f"{label}: node count exceeds the cap of {MAX_NODES}")


def _check_window(window: object, label: str, max_lookback: int) -> None:
    if isinstance(window, bool) or not isinstance(window, int):
        raise DslValidationError(f"{label} must be an int, got {type(window).__name__}")
    if window < 1:
        # A zero or negative offset would read the decision bar or a future one.
        raise DslValidationError(f"{label} must be >= 1; negative offsets would read the future")
    if window > max_lookback:
        raise DslValidationError(f"{label} = {window} exceeds max_lookback = {max_lookback}")


def _check_holding(bars: object, label: str) -> None:
    if isinstance(bars, bool) or not isinstance(bars, int):
        raise DslValidationError(f"{label} must be an int, got {type(bars).__name__}")
    if not 1 <= bars <= MAX_HOLDING_BARS:
        raise DslValidationError(f"{label} must be in [1, {MAX_HOLDING_BARS}], got {bars}")


def _walk(node: object, depth: int, budget: _Budget, max_lookback: int) -> str:
    """Validate one expression subtree and return its kind."""
    budget.tick(type(node).__name__)
    if depth > MAX_DEPTH:
        raise DslValidationError(f"expression depth exceeds the cap of {MAX_DEPTH}")
    if isinstance(node, Feature):
        if node.name not in FEATURE_NAMES:
            raise DslValidationError(f"unknown feature {node.name!r}")
        return _NUMBER
    if isinstance(node, Const):
        if not isinstance(node.value, float) or not math.isfinite(node.value):
            raise DslValidationError("Const.value must be a finite float")
        return _NUMBER
    if isinstance(node, Change):
        if _walk(node.source, depth + 1, budget, max_lookback) != _NUMBER:
            raise DslValidationError("Change.source must be numeric")
        _check_window(node.window, "Change.window", max_lookback)
        return _NUMBER
    if isinstance(node, Compare):
        if node.op not in COMPARISON_OPERATORS:
            raise DslValidationError(f"unknown comparison operator {node.op!r}")
        if _walk(node.left, depth + 1, budget, max_lookback) != _NUMBER:
            raise DslValidationError("Compare.left must be numeric")
        if _walk(node.right, depth + 1, budget, max_lookback) != _NUMBER:
            raise DslValidationError("Compare.right must be numeric")
        return _CONDITION
    if isinstance(node, (And, Or)):
        if _walk(node.left, depth + 1, budget, max_lookback) != _CONDITION:
            raise DslValidationError(f"{type(node).__name__}.left must be a condition")
        if _walk(node.right, depth + 1, budget, max_lookback) != _CONDITION:
            raise DslValidationError(f"{type(node).__name__}.right must be a condition")
        return _CONDITION
    if isinstance(node, Not):
        if _walk(node.operand, depth + 1, budget, max_lookback) != _CONDITION:
            raise DslValidationError("Not.operand must be a condition")
        return _CONDITION
    if isinstance(node, Between):
        for field in ("value", "low", "high"):
            if _walk(getattr(node, field), depth + 1, budget, max_lookback) != _NUMBER:
                raise DslValidationError(f"Between.{field} must be numeric")
        return _CONDITION
    if isinstance(node, (CrossAbove, CrossBelow)):
        for field in ("left", "right"):
            if _walk(getattr(node, field), depth + 1, budget, max_lookback) != _NUMBER:
                raise DslValidationError(f"{type(node).__name__}.{field} must be numeric")
        _check_window(node.window, f"{type(node).__name__}.window", max_lookback)
        return _CONDITION
    if isinstance(node, (Rise, Fall)):
        if _walk(node.source, depth + 1, budget, max_lookback) != _NUMBER:
            raise DslValidationError(f"{type(node).__name__}.source must be numeric")
        _check_window(node.window, f"{type(node).__name__}.window", max_lookback)
        return _CONDITION
    if callable(node):
        raise DslValidationError(f"callable objects are not DSL nodes (got {type(node).__name__})")
    raise DslValidationError(
        f"unknown DSL node {type(node).__name__}; expected one of "
        f"{', '.join(cls.__name__ for cls in _EXPRESSIONS)}"
    )


def _check_condition(node: object, label: str, budget: _Budget, max_lookback: int) -> None:
    if _walk(node, 1, budget, max_lookback) != _CONDITION:
        raise DslValidationError(f"{label} must be a boolean condition, got a number")


def validate(spec: object) -> None:
    """Raise :class:`DslValidationError` unless ``spec`` is a valid strategy."""
    if not isinstance(spec, StrategySpec):
        raise DslValidationError(f"root must be a StrategySpec, got {type(spec).__name__}")
    if isinstance(spec.spec_version, bool) or not isinstance(spec.spec_version, int):
        raise DslValidationError("spec_version must be an int")
    if spec.spec_version != DSL_SPEC_VERSION:
        raise DslValidationError(
            f"unsupported spec_version {spec.spec_version}; expected {DSL_SPEC_VERSION}"
        )
    if isinstance(spec.max_lookback, bool) or not isinstance(spec.max_lookback, int):
        raise DslValidationError("max_lookback must be an int")
    if not 1 <= spec.max_lookback <= MAX_ABSOLUTE_LOOKBACK:
        raise DslValidationError(
            f"max_lookback must be in [1, {MAX_ABSOLUTE_LOOKBACK}], got {spec.max_lookback}"
        )
    if not isinstance(spec.entries, tuple):
        raise DslValidationError("entries must be a tuple of EntryRule")
    if not isinstance(spec.exits, tuple):
        raise DslValidationError("exits must be a tuple of ExitRule")
    if not spec.entries:
        raise DslValidationError("at least one entry rule is required")
    if not spec.exits:
        raise DslValidationError("at least one exit rule is required")

    budget = _Budget()
    budget.tick("strategy")

    for index, entry in enumerate(spec.entries):
        budget.tick("entry")
        if not isinstance(entry, EntryRule):
            raise DslValidationError(f"entries[{index}] must be an EntryRule")
        if entry.side not in SIDES:
            raise DslValidationError(f"entries[{index}].side must be one of {SIDES}")
        _check_condition(entry.condition, f"entries[{index}].condition", budget, spec.max_lookback)
    for index, exit_rule in enumerate(spec.exits):
        budget.tick("exit")
        if not isinstance(exit_rule, ExitRule):
            raise DslValidationError(f"exits[{index}] must be an ExitRule")
        _check_condition(
            exit_rule.condition, f"exits[{index}].condition", budget, spec.max_lookback
        )
    if spec.holding is not None:
        budget.tick("holding")
        if not isinstance(spec.holding, HoldingLimit):
            raise DslValidationError("holding must be a HoldingLimit or None")
        _check_holding(spec.holding.bars, "holding.bars")
    if spec.no_trade is not None:
        budget.tick("no_trade")
        if not isinstance(spec.no_trade, NoTrade):
            raise DslValidationError("no_trade must be a NoTrade or None")
        _check_condition(spec.no_trade.condition, "no_trade.condition", budget, spec.max_lookback)
    if spec.sizing is not None:
        budget.tick("sizing")
        if not isinstance(spec.sizing, (FixedFraction, AtrMultiple)):
            raise DslValidationError("sizing must be FixedFraction, AtrMultiple or None")
