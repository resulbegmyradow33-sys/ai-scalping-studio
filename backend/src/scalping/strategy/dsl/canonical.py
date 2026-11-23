"""Canonical identity and structural cost of a strategy.

Canonicalisation is the only place a strategy becomes a string. It follows the
same rules as the discovery protocol's ``canonical_json``: sorted keys, no
whitespace, ASCII escapes, ``allow_nan=False``. Structural field order and a
dict's insertion order therefore cannot produce two identities for one strategy,
and a float written as ``1`` or ``1.0`` normalises through the node
constructors.

``complexity`` is the search's structural penalty, not a measure of quality. It
combines node count, tree depth and a lookback term so a wide tree with a long
horizon cannot hide inside the same cap as a compact rule.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, Final

from scalping.strategy.dsl.nodes import (
    DSL_SPEC_VERSION,
    And,
    Between,
    Change,
    Compare,
    Const,
    CrossAbove,
    CrossBelow,
    DslValidationError,
    Fall,
    Feature,
    Not,
    Or,
    Rise,
    StrategySpec,
)

__all__ = [
    "canonical_dict",
    "canonical_digest",
    "canonical_hash",
    "canonical_json",
    "complexity",
]

_LOOKBACK_BUCKET: Final[int] = 10


def _normalise(value: object) -> object:
    """Recursively normalise JSON-safe values; reject non-finite floats."""
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DslValidationError(f"non-finite number {value!r} cannot be canonical")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    raise DslValidationError(f"value of type {type(value).__name__} is not JSON-safe")


def canonical_dict(spec: StrategySpec) -> dict[str, Any]:
    """The strategy as a canonical, JSON-safe payload."""
    if not isinstance(spec, StrategySpec):
        raise DslValidationError(f"expected a StrategySpec, got {type(spec).__name__}")
    if spec.spec_version != DSL_SPEC_VERSION:
        raise DslValidationError(f"unsupported spec_version {spec.spec_version}")
    payload = _normalise(spec.to_dict())
    assert isinstance(payload, dict)
    return payload


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Canonical spelling: sorted keys, compact separators, no NaN/inf."""
    if not isinstance(payload, Mapping):
        raise DslValidationError("canonical JSON needs a mapping")
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_digest(payload: Mapping[str, Any]) -> str:
    """Lowercase sha256 hex of :func:`canonical_json`."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def canonical_hash(spec: StrategySpec) -> str:
    """sha256 identity of the strategy, equal to hashing its canonical dict."""
    return canonical_digest(canonical_dict(spec))


def _expression_cost(node: object) -> tuple[int, int]:
    """``(nodes, depth)`` of one expression tree."""
    if isinstance(node, (Feature, Const)):
        return 1, 1
    if isinstance(node, Change):
        nodes, depth = _expression_cost(node.source)
        return nodes + 1, depth + 1
    if isinstance(node, Compare):
        left_nodes, left_depth = _expression_cost(node.left)
        right_nodes, right_depth = _expression_cost(node.right)
        return left_nodes + right_nodes + 1, max(left_depth, right_depth) + 1
    if isinstance(node, (And, Or)):
        left_nodes, left_depth = _expression_cost(node.left)
        right_nodes, right_depth = _expression_cost(node.right)
        return left_nodes + right_nodes + 1, max(left_depth, right_depth) + 1
    if isinstance(node, Not):
        nodes, depth = _expression_cost(node.operand)
        return nodes + 1, depth + 1
    if isinstance(node, Between):
        costs = [_expression_cost(part) for part in (node.value, node.low, node.high)]
        return sum(cost[0] for cost in costs) + 1, max(cost[1] for cost in costs) + 1
    if isinstance(node, (CrossAbove, CrossBelow)):
        costs = [_expression_cost(node.left), _expression_cost(node.right)]
        return sum(cost[0] for cost in costs) + 1, max(cost[1] for cost in costs) + 1
    if isinstance(node, (Rise, Fall)):
        nodes, depth = _expression_cost(node.source)
        return nodes + 1, depth + 1
    raise DslValidationError(f"unknown DSL node {type(node).__name__}")


def complexity(spec: StrategySpec) -> int:
    """Node count + depth + lookback penalty (a non-negative int).

    Formula::

        nodes   = 1 (root) + sum(1 + expression nodes per rule) + section nodes
        depth   = deepest expression depth
        penalty = ceil(max_lookback / 10)
        cost    = nodes + depth + penalty

    The value is deterministic and bounded by the validation caps, so it is safe
    to compare against ``SearchConfig.complexity_cap``.
    """
    if not isinstance(spec, StrategySpec):
        raise DslValidationError(f"expected a StrategySpec, got {type(spec).__name__}")
    nodes = 1
    depth = 1
    for entry in spec.entries:
        expression_nodes, expression_depth = _expression_cost(entry.condition)
        nodes += 1 + expression_nodes
        depth = max(depth, expression_depth)
    for exit_rule in spec.exits:
        expression_nodes, expression_depth = _expression_cost(exit_rule.condition)
        nodes += 1 + expression_nodes
        depth = max(depth, expression_depth)
    if spec.holding is not None:
        nodes += 1
    if spec.no_trade is not None:
        expression_nodes, expression_depth = _expression_cost(spec.no_trade.condition)
        nodes += 1 + expression_nodes
        depth = max(depth, expression_depth)
    if spec.sizing is not None:
        nodes += 1
    penalty = (spec.max_lookback + _LOOKBACK_BUCKET - 1) // _LOOKBACK_BUCKET
    return nodes + depth + penalty
