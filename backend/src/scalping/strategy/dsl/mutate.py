"""Deterministic generative operators over strategy specs.

These operators are the search space, not a strategy library. Every function
takes a :class:`numpy.random.Generator`, consumes it in a fixed order and
returns data. Nothing is generated as source code, so mutation cannot smuggle
arbitrary Python into the process.

The output is always validated before it is returned:

* :func:`random_spec` constructs only valid primitives;
* :func:`mutate` retries a bounded number of times and falls back to the valid
  parent rather than returning an invalid child;
* :func:`crossover` returns ``None`` when the combined rules cannot fit the
  structural caps, matching the discovery ``Factory`` protocol.

:class:`StrategyFactory` bundles the three operators and ``validate`` with the
same method names as :class:`scalping.discovery.protocols.Factory`, so wiring
it into the search loop is a constructor argument, not an adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, fields, replace
from typing import Any, Final, cast

import numpy as np

from scalping.strategy.dsl.nodes import (
    COMPARISON_OPERATORS,
    DSL_SPEC_VERSION,
    FEATURE_NAMES,
    MAX_ABSOLUTE_LOOKBACK,
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
    Expr,
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
    "StrategyFactory",
    "crossover",
    "mutate",
    "random_spec",
]

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

_ATOMS: Final[tuple[type, ...]] = (
    Compare,
    CrossAbove,
    CrossBelow,
    Rise,
    Fall,
    Between,
)

_OPERATORS: Final[tuple[str, ...]] = ("<", "<=", ">", ">=", "==", "!=")
assert frozenset(_OPERATORS) == COMPARISON_OPERATORS

_CONST_FACTORS: Final[tuple[float, ...]] = (0.5, 0.8, 1.25, 1.5, 2.0)

_PATH = tuple[str, ...]


def _require_rng(rng: object) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be numpy.random.Generator, got {type(rng).__name__}")
    return rng


def _pool(feature_pool: Iterable[str] | None) -> tuple[str, ...]:
    names = FEATURE_NAMES if feature_pool is None else tuple(feature_pool)
    if not names:
        raise ValueError("feature_pool must contain at least one feature")
    for name in names:
        if name not in FEATURE_NAMES:
            raise ValueError(f"unknown feature in feature_pool: {name!r}")
    return names


def _check_bounds(max_lookback: object, max_depth: object) -> tuple[int, int]:
    if isinstance(max_lookback, bool) or not isinstance(max_lookback, int):
        raise TypeError("max_lookback must be an int")
    if not 1 <= max_lookback <= MAX_ABSOLUTE_LOOKBACK:
        raise ValueError(f"max_lookback must be in [1, {MAX_ABSOLUTE_LOOKBACK}]")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int):
        raise TypeError("max_depth must be an int")
    if not 1 <= max_depth <= 12:
        raise ValueError("max_depth must be in [1, 12]")
    return max_lookback, max_depth


def _window(rng: np.random.Generator, max_lookback: int) -> int:
    return int(rng.integers(1, max_lookback + 1))


def _feature(rng: np.random.Generator, pool: tuple[str, ...]) -> Feature:
    return Feature(pool[int(rng.integers(0, len(pool)))])


def _constant(rng: np.random.Generator) -> Const:
    return Const(round(float(rng.uniform(-2.0, 2.0)), 4))


def _number(
    rng: np.random.Generator, pool: tuple[str, ...], max_lookback: int
) -> Feature | Const | Change:
    """A numeric leaf or a one-step difference; never a boolean."""
    roll = float(rng.random())
    if roll < 0.15:
        return _constant(rng)
    if roll < 0.30:
        return Change(_feature(rng, pool), _window(rng, max_lookback))
    return _feature(rng, pool)


def _series(rng: np.random.Generator, pool: tuple[str, ...], max_lookback: int) -> Feature | Change:
    """A numeric series suitable for rise/fall; constants are excluded."""
    if rng.random() < 0.8:
        return _feature(rng, pool)
    return Change(_feature(rng, pool), _window(rng, max_lookback))


def _atomic(rng: np.random.Generator, pool: tuple[str, ...], max_lookback: int) -> Expr:
    """One comparison, cross, range or rise/fall condition."""
    kind = int(rng.integers(0, 6))
    if kind == 0:
        return Compare(
            _number(rng, pool, max_lookback),
            _OPERATORS[int(rng.integers(0, len(_OPERATORS)))],
            _constant(rng),
        )
    if kind == 1:
        return Compare(
            _number(rng, pool, max_lookback),
            _OPERATORS[int(rng.integers(0, len(_OPERATORS)))],
            _number(rng, pool, max_lookback),
        )
    if kind == 2:
        return CrossAbove(
            _number(rng, pool, max_lookback),
            _number(rng, pool, max_lookback),
            _window(rng, max_lookback),
        )
    if kind == 3:
        return CrossBelow(
            _number(rng, pool, max_lookback),
            _number(rng, pool, max_lookback),
            _window(rng, max_lookback),
        )
    if kind == 4:
        first = float(_constant(rng).value)
        second = float(_constant(rng).value)
        low, high = sorted((first, second))
        return Between(_feature(rng, pool), Const(low), Const(high))
    source = _series(rng, pool, max_lookback)
    window = _window(rng, max_lookback)
    if rng.random() < 0.5:
        return Rise(source, window)
    return Fall(source, window)


def _condition(
    rng: np.random.Generator,
    depth: int,
    pool: tuple[str, ...],
    max_lookback: int,
) -> Expr:
    """A random condition no deeper than ``depth`` nodes."""
    if depth <= 1 or rng.random() < 0.4:
        return _atomic(rng, pool, max_lookback)
    kind = int(rng.integers(0, 4))
    if kind == 0:
        return And(
            _condition(rng, depth - 1, pool, max_lookback),
            _condition(rng, depth - 1, pool, max_lookback),
        )
    if kind == 1:
        return Or(
            _condition(rng, depth - 1, pool, max_lookback),
            _condition(rng, depth - 1, pool, max_lookback),
        )
    if kind == 2:
        return Not(_condition(rng, depth - 1, pool, max_lookback))
    return _atomic(rng, pool, max_lookback)


def _sizing(rng: np.random.Generator) -> FixedFraction | AtrMultiple | None:
    roll = float(rng.random())
    if roll < 0.34:
        return None
    if roll < 0.67:
        return FixedFraction(round(float(rng.uniform(0.05, 0.5)), 3))
    return AtrMultiple(round(float(rng.uniform(0.25, 3.0)), 3))


def random_spec(
    rng: np.random.Generator,
    *,
    max_lookback: int = 20,
    max_depth: int = 3,
    feature_pool: Iterable[str] | None = None,
    spec_version: int = DSL_SPEC_VERSION,
) -> StrategySpec:
    """Draw one valid strategy from the grammar.

    The same generator state always produces the same spec. ``max_depth``
    bounds condition depth; ``max_lookback`` bounds every backward window.
    """
    generator = _require_rng(rng)
    max_lookback, max_depth = _check_bounds(max_lookback, max_depth)
    pool = _pool(feature_pool)
    if spec_version != DSL_SPEC_VERSION:
        raise DslValidationError(f"unsupported spec_version {spec_version}")

    first_side = "long" if generator.random() < 0.7 else "short"
    entries: list[EntryRule] = [
        EntryRule(first_side, _condition(generator, max_depth, pool, max_lookback))
    ]
    if generator.random() < 0.4:
        opposite = "short" if first_side == "long" else "long"
        entries.append(EntryRule(opposite, _condition(generator, max_depth, pool, max_lookback)))

    exits: list[ExitRule] = [ExitRule(_condition(generator, max_depth, pool, max_lookback))]
    if generator.random() < 0.35:
        exits.append(ExitRule(_condition(generator, max_depth, pool, max_lookback)))

    no_trade = None
    if generator.random() < 0.4:
        no_trade = NoTrade(_condition(generator, max_depth, pool, max_lookback))

    spec = StrategySpec(
        entries=tuple(entries),
        exits=tuple(exits),
        holding=HoldingLimit(_window(generator, max_lookback)),
        no_trade=no_trade,
        sizing=_sizing(generator),
        max_lookback=max_lookback,
        spec_version=spec_version,
    )
    validate(spec)
    return spec


def _children(node: Expr) -> Iterator[tuple[str, Expr]]:
    for field in fields(cast(Any, node)):
        value = getattr(node, field.name)
        if isinstance(value, _EXPRESSIONS):
            yield field.name, cast(Expr, value)


def _paths_to(node: Expr, kinds: tuple[type, ...]) -> list[_PATH]:
    """Paths from ``node`` to every descendant of one of ``kinds``."""
    paths: list[_PATH] = []

    def walk(current: Expr, path: _PATH) -> None:
        if isinstance(current, kinds):
            paths.append(path)
        for name, child in _children(current):
            walk(child, (*path, name))

    walk(node, ())
    return paths


def _replace(node: Expr, path: _PATH, transform: Callable[[Expr], Expr]) -> Expr:
    if not path:
        return transform(node)
    name = path[0]
    child = getattr(node, name)
    new_child = _replace(child, path[1:], transform)
    return replace(cast(Any, node), **{name: new_child})


def _perturb(value: float, rng: np.random.Generator) -> float:
    factor = _CONST_FACTORS[int(rng.integers(0, len(_CONST_FACTORS)))]
    candidate = round(value * factor + float(rng.uniform(-0.05, 0.05)), 4)
    if candidate == value:
        candidate = round(value + 0.1, 4)
    return 0.0 if candidate == 0.0 else candidate


def _perturb_const(node: Expr, rng: np.random.Generator) -> Const:
    if not isinstance(node, Const):
        raise TypeError("expected a Const node")
    return Const(_perturb(node.value, rng))


def _mutate_condition(
    condition: Expr,
    rng: np.random.Generator,
    pool: tuple[str, ...],
    max_lookback: int,
) -> Expr:
    """Rewrite one condition subtree, always within the same caps."""
    if rng.random() < 0.2:
        return _condition(rng, 3, pool, max_lookback)

    constants = _paths_to(condition, (Const,))
    if constants and rng.random() < 0.5:
        path = constants[int(rng.integers(0, len(constants)))]
        return _replace(condition, path, lambda node: _perturb_const(node, rng))

    atoms = _paths_to(condition, _ATOMS)
    if atoms and rng.random() < 0.5:
        path = atoms[int(rng.integers(0, len(atoms)))]
        return _replace(condition, path, lambda node: _atomic(rng, pool, max_lookback))

    kind = int(rng.integers(0, 3))
    if kind == 0:
        return Not(condition)
    other = _condition(rng, 2, pool, max_lookback)
    return And(condition, other) if kind == 1 else Or(condition, other)


def _rule_conditions(spec: StrategySpec) -> list[tuple[str, int]]:
    targets: list[tuple[str, int]] = [("entry", index) for index in range(len(spec.entries))]
    targets.extend(("exit", index) for index in range(len(spec.exits)))
    if spec.no_trade is not None:
        targets.append(("no_trade", 0))
    return targets


def _rewrite_rule(
    spec: StrategySpec,
    target: tuple[str, int],
    condition: Expr,
) -> StrategySpec:
    section, index = target
    if section == "entry":
        entries = list(spec.entries)
        entries[index] = replace(entries[index], condition=condition)
        return replace(spec, entries=tuple(entries))
    if section == "exit":
        exits = list(spec.exits)
        exits[index] = replace(exits[index], condition=condition)
        return replace(spec, exits=tuple(exits))
    if section == "no_trade" and spec.no_trade is not None:
        return replace(spec, no_trade=replace(spec.no_trade, condition=condition))
    raise DslValidationError(f"unknown rule target {target!r}")


def _attempt_mutate(
    spec: StrategySpec,
    rng: np.random.Generator,
    pool: tuple[str, ...],
    max_depth: int,
) -> StrategySpec:
    max_lookback = spec.max_lookback
    operation = int(rng.integers(0, 10))
    if operation in (0, 1):
        targets = _rule_conditions(spec)
        target = targets[int(rng.integers(0, len(targets)))]
        if target[0] == "entry":
            condition = spec.entries[target[1]].condition
        elif target[0] == "exit":
            condition = spec.exits[target[1]].condition
        else:
            assert spec.no_trade is not None
            condition = spec.no_trade.condition
        return _rewrite_rule(spec, target, _mutate_condition(condition, rng, pool, max_lookback))
    if operation == 2:
        return replace(spec, holding=HoldingLimit(_window(rng, max_lookback)))
    if operation == 3:
        if spec.no_trade is not None and rng.random() < 0.25:
            return replace(spec, no_trade=None)
        return replace(spec, no_trade=NoTrade(_condition(rng, max_depth, pool, max_lookback)))
    if operation == 4:
        return replace(spec, sizing=_sizing(rng))
    if operation == 5:
        if len(spec.entries) >= 4:
            return spec
        side = SIDES[int(rng.integers(0, len(SIDES)))]
        entries = (*spec.entries, EntryRule(side, _condition(rng, max_depth, pool, max_lookback)))
        return replace(spec, entries=entries)
    if operation == 6:
        if len(spec.exits) >= 4:
            return spec
        exits = (*spec.exits, ExitRule(_condition(rng, max_depth, pool, max_lookback)))
        return replace(spec, exits=exits)
    if operation == 7 and len(spec.entries) > 1:
        remove = int(rng.integers(0, len(spec.entries)))
        entries = tuple(rule for index, rule in enumerate(spec.entries) if index != remove)
        return replace(spec, entries=entries)
    if operation == 8 and len(spec.exits) > 1:
        remove = int(rng.integers(0, len(spec.exits)))
        exits = tuple(rule for index, rule in enumerate(spec.exits) if index != remove)
        return replace(spec, exits=exits)
    if operation == 9 and max_lookback < MAX_ABSOLUTE_LOOKBACK:
        grown = min(
            MAX_ABSOLUTE_LOOKBACK,
            max_lookback + int(rng.integers(1, 11)),
        )
        return replace(spec, max_lookback=grown)
    return spec


def mutate(
    spec: StrategySpec,
    rng: np.random.Generator,
    *,
    max_depth: int = 3,
    feature_pool: Iterable[str] | None = None,
) -> StrategySpec:
    """One valid child of ``spec``.

    A failed structural attempt (depth or node cap) is retried with the next
    generator draw. After a bounded number of attempts the parent is returned
    unchanged, which is still valid. The parent object is never modified.
    """
    generator = _require_rng(rng)
    validate(spec)
    pool = _pool(feature_pool)
    _, max_depth = _check_bounds(spec.max_lookback, max_depth)
    for _ in range(8):
        try:
            child = _attempt_mutate(spec, generator, pool, max_depth)
            validate(child)
        except (TypeError, ValueError):
            continue
        return child
    return spec


def _merge_rules[Rule: (EntryRule, ExitRule)](
    first: tuple[Rule, ...],
    second: tuple[Rule, ...],
    rng: np.random.Generator,
) -> tuple[Rule, ...]:
    """Per-slot random choice from two non-empty rule tuples."""
    count = max(len(first), len(second))
    merged: list[Rule] = []
    for index in range(count):
        options: list[Rule] = []
        if index < len(first):
            options.append(first[index])
        if index < len(second):
            options.append(second[index])
        merged.append(options[int(rng.integers(0, len(options)))])
    return tuple(merged)


def crossover(
    a: StrategySpec,
    b: StrategySpec,
    rng: np.random.Generator,
) -> StrategySpec | None:
    """A child mixing rules from two specs, or ``None`` when it cannot fit.

    The child's ``max_lookback`` is the larger parent value, so windows inherited
    from either parent remain inside their common bound. Structural caps are
    re-checked; exceeding them is a failed crossover, not an invalid candidate.
    """
    generator = _require_rng(rng)
    validate(a)
    validate(b)
    if a.spec_version != b.spec_version:
        return None
    entries = cast(tuple[EntryRule, ...], _merge_rules(a.entries, b.entries, generator))
    exits = cast(tuple[ExitRule, ...], _merge_rules(a.exits, b.exits, generator))
    holding = a.holding if generator.random() < 0.5 else b.holding
    no_trade = a.no_trade if generator.random() < 0.5 else b.no_trade
    sizing = a.sizing if generator.random() < 0.5 else b.sizing
    child = StrategySpec(
        entries=entries,
        exits=exits,
        holding=holding,
        no_trade=no_trade,
        sizing=sizing,
        max_lookback=max(a.max_lookback, b.max_lookback),
        spec_version=a.spec_version,
    )
    try:
        validate(child)
    except DslValidationError:
        return None
    return child


@dataclass(frozen=True, slots=True)
class StrategyFactory:
    """Discovery ``Factory`` over :class:`StrategySpec`, without an adapter."""

    max_lookback: int = 20
    max_depth: int = 3
    feature_pool: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        _check_bounds(self.max_lookback, self.max_depth)
        _pool(self.feature_pool)

    def random(self, rng: np.random.Generator) -> StrategySpec:
        return random_spec(
            rng,
            max_lookback=self.max_lookback,
            max_depth=self.max_depth,
            feature_pool=self.feature_pool,
        )

    def mutate(self, candidate: StrategySpec, rng: np.random.Generator) -> StrategySpec:
        if not isinstance(candidate, StrategySpec):
            raise TypeError("StrategyFactory only mutates StrategySpec candidates")
        return mutate(
            candidate,
            rng,
            max_depth=self.max_depth,
            feature_pool=self.feature_pool,
        )

    def crossover(
        self, a: StrategySpec, b: StrategySpec, rng: np.random.Generator
    ) -> StrategySpec | None:
        if not isinstance(a, StrategySpec) or not isinstance(b, StrategySpec):
            raise TypeError("StrategyFactory only crosses StrategySpec candidates")
        return crossover(a, b, rng)

    def validate(self, candidate: StrategySpec) -> None:
        if not isinstance(candidate, StrategySpec):
            raise DslValidationError(f"expected a StrategySpec, got {type(candidate).__name__}")
        validate(candidate)
