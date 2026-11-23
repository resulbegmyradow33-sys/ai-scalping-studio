"""Safe, versioned strategy DSL for autonomous discovery.

The package exposes four layers:

``nodes``
    Frozen dataclasses for the typed AST. Strategies are data.
``validate``
    The structural/type gate. Unknown nodes, unknown fields, bad operators,
    non-finite numbers, out-of-bounds windows, excessive depth and future
    references are rejected.
``canonical``
    Canonical JSON, sha256 identity and the complexity cost.
``compile``
    ``compile_policy(spec)`` returns an interpreter with ``decide`` and
    ``evaluate_series``. No generated code is executed.
``mutate``
    Deterministic ``random_spec``/``mutate``/``crossover`` operators.

The DSL defines a bounded search space. A valid spec is not evidence of a
trading edge; see ``docs/strategy-dsl.md``.
"""

from __future__ import annotations

from scalping.strategy.dsl.canonical import (
    canonical_dict,
    canonical_digest,
    canonical_hash,
    canonical_json,
    complexity,
)
from scalping.strategy.dsl.compile import (
    ACTIONS,
    REASONS,
    Decision,
    Policy,
    PolicyState,
    compile_policy,
)
from scalping.strategy.dsl.mutate import (
    StrategyFactory,
    crossover,
    mutate,
    random_spec,
)
from scalping.strategy.dsl.nodes import (
    COMPARISON_OPERATORS,
    DSL_SPEC_VERSION,
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
from scalping.strategy.dsl.validate import validate

__all__ = [
    "ACTIONS",
    "COMPARISON_OPERATORS",
    "DSL_SPEC_VERSION",
    "MAX_ABSOLUTE_LOOKBACK",
    "MAX_DEPTH",
    "MAX_HOLDING_BARS",
    "MAX_NODES",
    "REASONS",
    "SIDES",
    "And",
    "AtrMultiple",
    "Between",
    "Change",
    "Compare",
    "Const",
    "CrossAbove",
    "CrossBelow",
    "Decision",
    "DslValidationError",
    "EntryRule",
    "ExitRule",
    "Fall",
    "Feature",
    "FixedFraction",
    "HoldingLimit",
    "NoTrade",
    "Not",
    "Or",
    "Policy",
    "PolicyState",
    "Rise",
    "StrategyFactory",
    "StrategySpec",
    "canonical_dict",
    "canonical_digest",
    "canonical_hash",
    "canonical_json",
    "compile_policy",
    "complexity",
    "crossover",
    "mutate",
    "random_spec",
    "validate",
]
