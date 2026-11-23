"""Contract tests for the safe strategy DSL.

The tests cover the four promises the discovery engine relies on: the AST can
be serialised and hashed reproducibly, invalid or unsafe trees are rejected,
the interpreter's semantics are causal and explicit, and the generative
operators are deterministic and always return valid specs.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scalping.strategy.dsl import (
    DSL_SPEC_VERSION,
    MAX_ABSOLUTE_LOOKBACK,
    MAX_HOLDING_BARS,
    And,
    AtrMultiple,
    Between,
    Change,
    Compare,
    Const,
    CrossAbove,
    CrossBelow,
    Decision,
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
    PolicyState,
    Rise,
    StrategySpec,
    canonical_dict,
    canonical_hash,
    canonical_json,
    compile_policy,
    complexity,
    crossover,
    mutate,
    random_spec,
    validate,
)

_EXIT_NEVER = Compare(Feature("ret_1"), "<", Const(-100.0))
_DEFAULT_HOLDING = HoldingLimit(5)


def _spec(
    condition: object,
    *,
    side: str = "long",
    exit_condition: object = _EXIT_NEVER,
    holding: HoldingLimit | None = _DEFAULT_HOLDING,
    no_trade: NoTrade | None = None,
    max_lookback: int = 10,
    entries: tuple[EntryRule, ...] | None = None,
    exits: tuple[ExitRule, ...] | None = None,
) -> StrategySpec:
    return StrategySpec(
        entries=entries if entries is not None else (EntryRule(side, condition),),  # type: ignore[arg-type]
        exits=exits if exits is not None else (ExitRule(exit_condition),),  # type: ignore[arg-type]
        holding=holding,
        no_trade=no_trade,
        max_lookback=max_lookback,
    )


def _actions(spec: StrategySpec, frame: dict[str, list[float]]) -> list[str]:
    return [str(action) for action in compile_policy(spec).evaluate_series(frame)]


def _full_tree(depth: int) -> object:
    if depth <= 1:
        return Compare(Feature("ret_1"), ">", Const(0.0))
    return And(_full_tree(depth - 1), _full_tree(depth - 1))


# ---------------------------------------------------------------------------
# Serialisation, canonical identity and complexity
# ---------------------------------------------------------------------------


def test_round_trip_preserves_the_spec() -> None:
    spec = StrategySpec(
        entries=(
            EntryRule(
                "long",
                And(
                    Compare(Feature("ret_1"), ">", Const(0.0)),
                    Not(Compare(Feature("rsi_14"), ">=", Const(0.8))),
                ),
            ),
            EntryRule("short", CrossBelow(Feature("ema_gap_9_21"), Const(0.0), 3)),
        ),
        exits=(
            ExitRule(Compare(Change(Feature("ret_5"), 2), ">", Const(0.0))),
            ExitRule(Between(Feature("spread_norm"), Const(0.0), Const(2.0))),
        ),
        holding=HoldingLimit(7),
        no_trade=NoTrade(Compare(Feature("vol_regime"), "==", Const(1.0))),
        sizing=AtrMultiple(1.5),
        max_lookback=30,
    )

    payload = spec.to_dict()
    text = json.dumps(payload)
    loaded = StrategySpec.from_dict(json.loads(text))

    assert loaded == spec
    assert loaded.to_dict() == payload
    assert loaded.canonical_hash() == spec.canonical_hash()


def test_round_trip_accepts_a_python_literal_payload() -> None:
    payload = {
        "kind": "strategy",
        "spec_version": 1,
        "entries": [
            {
                "kind": "entry",
                "side": "long",
                "condition": {
                    "kind": "compare",
                    "left": {"kind": "feature", "name": "ret_1"},
                    "op": ">",
                    "right": {"kind": "const", "value": 0.0},
                },
            }
        ],
        "exits": [
            {
                "kind": "exit",
                "condition": {
                    "kind": "compare",
                    "left": {"kind": "feature", "name": "ret_1"},
                    "op": "<",
                    "right": {"kind": "const", "value": -1.0},
                },
            }
        ],
        "holding": None,
        "no_trade": None,
        "sizing": None,
        "max_lookback": 10,
    }
    spec = StrategySpec.from_dict(payload)
    validate(spec)
    assert spec.entries[0].condition == Compare(Feature("ret_1"), ">", Const(0.0))


def _reverse_dict_order(value: object) -> object:
    if isinstance(value, dict):
        return {key: _reverse_dict_order(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reverse_dict_order(item) for item in value]
    return value


def test_canonical_hash_ignores_field_and_key_order() -> None:
    spec = _spec(And(Compare(Feature("ret_1"), ">", Const(0.0)), Rise(Feature("ret_5"), 2)))
    reordered = _reverse_dict_order(spec.to_dict())
    assert isinstance(reordered, dict)
    other = StrategySpec.from_dict(reordered)

    assert spec == other
    assert spec.canonical_hash() == other.canonical_hash()
    assert spec.canonical_hash() == canonical_hash(spec)
    assert len(spec.canonical_hash()) == 64


def test_canonical_hash_is_the_sha256_of_canonical_json() -> None:
    import hashlib

    spec = _spec(Compare(Feature("ret_1"), ">", Const(0.25)))
    payload = canonical_dict(spec)
    expected = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    assert spec.canonical_hash() == expected
    # Compact, sorted, ASCII: one strategy has exactly one spelling.
    text = canonical_json(payload)
    assert " " not in text
    assert list(json.loads(text)) == sorted(payload)


def test_canonical_hash_changes_with_a_threshold() -> None:
    first = _spec(Compare(Feature("ret_1"), ">", Const(0.25)))
    second = _spec(Compare(Feature("ret_1"), ">", Const(0.2501)))
    assert first.canonical_hash() != second.canonical_hash()


def test_canonical_hash_normalises_int_and_float_constants() -> None:
    from_int = _spec(Compare(Feature("ret_1"), ">", Const(1)))
    from_float = _spec(Compare(Feature("ret_1"), ">", Const(1.0)))
    assert from_int == from_float
    assert from_int.canonical_hash() == from_float.canonical_hash()


def test_complexity_counts_structure_and_horizon() -> None:
    simple = _spec(Compare(Feature("ret_1"), ">", Const(0.0)), max_lookback=10)
    wider = _spec(
        And(
            Compare(Feature("ret_1"), ">", Const(0.0)),
            Compare(Feature("ret_5"), ">", Const(0.0)),
        ),
        max_lookback=10,
    )
    longer = _spec(Compare(Feature("ret_1"), ">", Const(0.0)), max_lookback=30)

    assert simple.complexity() == complexity(simple) > 0
    assert wider.complexity() > simple.complexity()
    assert longer.complexity() > simple.complexity()
    assert isinstance(simple.complexity(), int)


# ---------------------------------------------------------------------------
# Node and spec validation
# ---------------------------------------------------------------------------


def test_feature_names_are_exactly_the_causal_registry() -> None:
    from scalping.features.build import FEATURE_COLUMNS
    from scalping.strategy.dsl.nodes import FEATURE_NAMES

    assert FEATURE_NAMES == FEATURE_COLUMNS
    assert len(FEATURE_NAMES) == 59


def test_unknown_feature_is_rejected_at_construction() -> None:
    with pytest.raises(DslValidationError, match="unknown feature"):
        Feature("astrology_score")


def test_unknown_comparison_operator_is_rejected() -> None:
    with pytest.raises(DslValidationError, match="unknown comparison operator"):
        Compare(Feature("ret_1"), "~=", Const(0.0))


def test_non_finite_constants_and_sizing_are_rejected() -> None:
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(DslValidationError, match="finite"):
            Const(bad)
    with pytest.raises(DslValidationError, match="finite"):
        FixedFraction(math.inf)
    with pytest.raises(DslValidationError, match="> 0"):
        AtrMultiple(0.0)
    with pytest.raises(DslValidationError, match=r"\(0, 1\]"):
        FixedFraction(0.0)


def test_unknown_node_and_unknown_fields_are_rejected() -> None:
    spec = _spec(Compare(Feature("ret_1"), ">", Const(0.0)))
    payload = spec.to_dict()
    payload["surprise"] = 1
    with pytest.raises(DslValidationError, match="unknown"):
        StrategySpec.from_dict(payload)

    payload = spec.to_dict()
    condition = payload["entries"][0]["condition"]
    assert isinstance(condition, dict)
    condition["future"] = True
    with pytest.raises(DslValidationError, match="unknown"):
        StrategySpec.from_dict(payload)

    payload = spec.to_dict()
    payload["entries"][0]["condition"] = {"kind": "future", "bars": 1}
    with pytest.raises(DslValidationError, match="unknown node kind"):
        StrategySpec.from_dict(payload)

    payload = spec.to_dict()
    payload["entries"][0]["condition"] = {"kind": "const", "value": 1.0, "scale": 2}
    with pytest.raises(DslValidationError, match="unknown"):
        StrategySpec.from_dict(payload)


def test_unknown_objects_and_callables_are_not_nodes() -> None:
    @dataclass(frozen=True)
    class Future:
        bars: int

    with pytest.raises(DslValidationError, match="unknown DSL node"):
        validate(_spec(Future(1)))  # type: ignore[arg-type]

    with pytest.raises(DslValidationError, match="callable"):
        validate(_spec(lambda row: True))  # type: ignore[arg-type]


def test_depth_cap_is_enforced() -> None:
    condition = Compare(Feature("ret_1"), ">", Const(0.0))
    for _ in range(17):
        condition = And(condition, Compare(Feature("ret_5"), ">", Const(0.0)))
    with pytest.raises(DslValidationError, match="depth"):
        validate(_spec(condition))


def test_node_cap_is_enforced() -> None:
    with pytest.raises(DslValidationError, match="node count"):
        validate(_spec(_full_tree(6)))


def test_lookback_bounds_are_enforced() -> None:
    with pytest.raises(DslValidationError, match="exceeds max_lookback"):
        validate(_spec(CrossAbove(Feature("ret_1"), Const(0.0), 11), max_lookback=10))
    with pytest.raises(DslValidationError, match="holding.bars must be in"):
        validate(
            _spec(
                Compare(Feature("ret_1"), ">", Const(0.0)),
                max_lookback=10,
                holding=HoldingLimit(MAX_HOLDING_BARS + 1),
            )
        )
    # A holding horizon is a forward dwell; it is not limited by lookback.
    validate(
        _spec(
            Compare(Feature("ret_1"), ">", Const(0.0)),
            max_lookback=10,
            holding=HoldingLimit(60),
        )
    )
    with pytest.raises(DslValidationError, match="max_lookback must be in"):
        validate(_spec(Compare(Feature("ret_1"), ">", Const(0.0)), max_lookback=0))
    with pytest.raises(DslValidationError, match="max_lookback must be in"):
        validate(
            _spec(
                Compare(Feature("ret_1"), ">", Const(0.0)),
                max_lookback=MAX_ABSOLUTE_LOOKBACK + 1,
            )
        )


def test_at_least_one_entry_and_one_exit_are_required() -> None:
    exit_rule = (ExitRule(_EXIT_NEVER),)
    with pytest.raises(DslValidationError, match="at least one entry"):
        validate(StrategySpec(entries=(), exits=exit_rule, max_lookback=10))
    with pytest.raises(DslValidationError, match="at least one exit"):
        validate(StrategySpec(entries=(EntryRule("long", _EXIT_NEVER),), exits=(), max_lookback=10))


def test_entries_and_exits_must_be_tuples() -> None:
    with pytest.raises(DslValidationError, match="tuple"):
        validate(
            StrategySpec(
                entries=[EntryRule("long", _EXIT_NEVER)],  # type: ignore[arg-type]
                exits=(ExitRule(_EXIT_NEVER),),
                max_lookback=10,
            )
        )


def test_spec_version_is_checked() -> None:
    spec = _spec(Compare(Feature("ret_1"), ">", Const(0.0)))
    bad = StrategySpec(
        entries=spec.entries,
        exits=spec.exits,
        max_lookback=spec.max_lookback,
        spec_version=DSL_SPEC_VERSION + 1,
    )
    with pytest.raises(DslValidationError, match="spec_version"):
        validate(bad)


def test_negative_and_zero_windows_cannot_reference_the_future() -> None:
    for window in (0, -1):
        with pytest.raises(DslValidationError, match=">= 1"):
            CrossAbove(Feature("ret_1"), Const(0.0), window)
        with pytest.raises(DslValidationError, match=">= 1"):
            Rise(Feature("ret_1"), window)
        with pytest.raises(DslValidationError, match=">= 1"):
            Change(Feature("ret_1"), window)


def test_a_number_cannot_be_used_as_a_condition() -> None:
    with pytest.raises(DslValidationError, match="boolean condition"):
        validate(_spec(Const(1.0)))


def test_a_condition_cannot_be_used_as_a_number() -> None:
    condition = And(
        Compare(Feature("ret_1"), ">", Const(0.0)),
        Compare(Feature("ret_5"), ">", Const(0.0)),
    )
    with pytest.raises(DslValidationError, match="must be numeric"):
        validate(_spec(Compare(condition, ">", Const(0.0))))


# ---------------------------------------------------------------------------
# Interpreter semantics
# ---------------------------------------------------------------------------


def test_compare_entry_and_exit_precedence() -> None:
    spec = _spec(
        Compare(Feature("ret_1"), ">", Const(0.01)),
        exit_condition=Compare(Feature("ret_1"), "<", Const(0.0)),
    )
    assert _actions(spec, {"ret_1": [0.0, 0.02, 0.03, -0.01]}) == [
        "HOLD",
        "BUY",
        "BUY",
        "CLOSE",
    ]


def test_cross_above_uses_only_the_backward_window() -> None:
    frame = {"ret_1": [-0.5, 0.2, 0.3]}
    one = _spec(CrossAbove(Feature("ret_1"), Const(0.0), 1), max_lookback=10)
    two = _spec(CrossAbove(Feature("ret_1"), Const(0.0), 2), max_lookback=10)
    assert _actions(one, frame) == ["HOLD", "BUY", "HOLD"]
    assert _actions(two, frame) == ["HOLD", "HOLD", "BUY"]


def test_cross_below_uses_only_the_backward_window() -> None:
    frame = {"ret_1": [0.5, -0.2, -0.3]}
    spec = _spec(CrossBelow(Feature("ret_1"), Const(0.0), 1), side="short")
    assert _actions(spec, frame) == ["HOLD", "SELL", "HOLD"]


def test_rise_fall_and_change() -> None:
    rising = _spec(Rise(Feature("ret_5"), 2))
    falling = _spec(Fall(Feature("ret_5"), 1))
    changing = _spec(Compare(Change(Feature("ret_5"), 2), ">", Const(1.5)))

    assert _actions(rising, {"ret_5": [0.1, 0.0, 0.2]}) == ["HOLD", "HOLD", "BUY"]
    assert _actions(falling, {"ret_5": [0.2, 0.1, 0.0]}) == ["HOLD", "BUY", "BUY"]
    assert _actions(changing, {"ret_5": [1.0, 2.0, 4.0]}) == ["HOLD", "HOLD", "BUY"]


def test_and_or_not_semantics() -> None:
    frame = {"ret_1": [1.0, 1.0, 0.0], "ret_5": [0.0, 1.0, 0.0]}
    positive = Compare(Feature("ret_1"), ">", Const(0.0))
    second = Compare(Feature("ret_5"), ">", Const(0.0))

    assert _actions(_spec(And(positive, Not(second))), frame) == ["BUY", "HOLD", "HOLD"]
    assert _actions(_spec(Or(positive, second)), frame) == ["BUY", "BUY", "HOLD"]
    assert _actions(_spec(Not(positive)), frame) == ["HOLD", "HOLD", "BUY"]


def test_between_is_inclusive() -> None:
    spec = _spec(Between(Feature("spread_norm"), Const(0.5), Const(1.5)))
    assert _actions(spec, {"spread_norm": [0.4, 0.5, 1.0, 1.5, 1.6]}) == [
        "HOLD",
        "BUY",
        "BUY",
        "BUY",
        "HOLD",
    ]


def test_no_trade_has_precedence_over_exit_and_entry() -> None:
    entry = Compare(Feature("ret_1"), ">", Const(0.01))
    exit_rule = Compare(Feature("ret_1"), ">", Const(0.0))
    spec = _spec(
        entry,
        exit_condition=exit_rule,
        no_trade=NoTrade(Compare(Feature("spread_norm"), ">", Const(1.0))),
    )
    decisions = compile_policy(spec).evaluate_series({"ret_1": [0.5], "spread_norm": [2.0]})
    assert list(decisions) == ["NO_TRADE"]

    decision = compile_policy(spec).decide({"ret_1": 0.5, "spread_norm": 2.0}, PolicyState())
    assert decision == Decision("NO_TRADE", "no_trade", DSL_SPEC_VERSION, 5)


def test_holding_limit_closes_an_open_position() -> None:
    spec = _spec(
        Compare(Feature("ret_1"), ">", Const(0.0)),
        holding=HoldingLimit(3),
    )
    policy = compile_policy(spec)
    row = {"ret_1": 1.0}

    held = policy.decide(row, PolicyState(position="long", bars_held=3))
    fresh = policy.decide(row, PolicyState(position="long", bars_held=2))
    flat = policy.decide(row, PolicyState())

    assert held.action == "CLOSE"
    assert held.reason == "holding_limit"
    assert held.holding_limit == 3
    assert fresh.action == "BUY"
    assert flat.action == "BUY"


def test_simultaneous_long_and_short_entries_are_a_no_trade_conflict() -> None:
    condition = Compare(Feature("ret_1"), ">", Const(0.0))
    spec = _spec(
        condition,
        entries=(EntryRule("long", condition), EntryRule("short", condition)),
    )
    decision = compile_policy(spec).decide({"ret_1": 1.0}, PolicyState())
    assert decision.action == "NO_TRADE"
    assert decision.reason == "conflicting_entry"


def test_future_rows_cannot_change_earlier_decisions() -> None:
    spec = _spec(Rise(Feature("ret_1"), 2), max_lookback=4)
    baseline = _actions(spec, {"ret_1": [0.0, 1.0, 2.0, 3.0]})
    altered = _actions(spec, {"ret_1": [0.0, 1.0, 2.0, 99.0]})
    assert baseline[:-1] == altered[:-1]


def test_missing_or_non_finite_values_never_fire_a_negated_condition() -> None:
    spec = _spec(Not(Compare(Feature("rsi_14"), ">", Const(0.5))))
    assert _actions(spec, {"ret_1": [0.0]}) == ["HOLD"]
    assert _actions(spec, {"rsi_14": [math.nan]}) == ["HOLD"]


def test_decide_carries_the_spec_version() -> None:
    spec = _spec(Compare(Feature("ret_1"), ">", Const(0.0)))
    policy = compile_policy(spec)
    decision = policy.decide({"ret_1": 1.0}, PolicyState())
    assert decision.version == DSL_SPEC_VERSION
    assert policy.version == DSL_SPEC_VERSION
    assert decision.reason == "entry_long"


def test_evaluate_series_accepts_a_feature_frame_like_object() -> None:
    @dataclass
    class Frame:
        X: np.ndarray
        columns: list[str]

    spec = _spec(Compare(Feature("ret_1"), ">", Const(0.0)))
    frame = Frame(X=np.array([[0.0], [1.0]], dtype=np.float32), columns=["ret_1"])
    assert [str(action) for action in compile_policy(spec).evaluate_series(frame)] == [
        "HOLD",
        "BUY",
    ]


def test_policy_state_push_trims_to_the_policy_horizon() -> None:
    spec = _spec(Rise(Feature("ret_1"), 2), max_lookback=3, holding=HoldingLimit(3))
    state = compile_policy(spec).new_state()
    for value in (0.0, 1.0, 2.0, 3.0, 4.0):
        state.push({"ret_1": value})
    assert len(state.history) == 3
    assert [row["ret_1"] for row in state.history] == [2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# Safety boundary in source, and deterministic generation
# ---------------------------------------------------------------------------


def test_package_source_has_no_dynamic_execution() -> None:
    import scalping.strategy.dsl as package

    root = Path(package.__file__).parent
    files = sorted(root.glob("*.py"))
    assert files
    forbidden = (
        re.compile(r"\beval\s*\("),
        re.compile(r"\bexec\s*\("),
        re.compile(r"\bcompile\s*\("),
        re.compile(r"__import__"),
    )
    for path in files:
        source = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert not pattern.search(source), f"{path.name} contains {pattern.pattern}"


def test_random_spec_is_deterministic() -> None:
    first = random_spec(np.random.default_rng(7))
    second = random_spec(np.random.default_rng(7))
    assert first == second
    assert first.canonical_hash() == second.canonical_hash()
    validate(first)


def test_mutate_is_deterministic() -> None:
    parent = random_spec(np.random.default_rng(11))
    first = mutate(parent, np.random.default_rng(12))
    second = mutate(parent, np.random.default_rng(12))
    assert first == second
    assert first.canonical_hash() == second.canonical_hash()


def test_crossover_is_deterministic() -> None:
    first = _spec(Compare(Feature("ret_1"), ">", Const(0.0)))
    second = _spec(Compare(Feature("ret_5"), "<", Const(0.0)), side="short")
    a = crossover(first, second, np.random.default_rng(3))
    b = crossover(first, second, np.random.default_rng(3))
    assert a is not None and a == b
    assert a.canonical_hash() == b.canonical_hash()


def test_mutation_outputs_always_validate_and_are_not_the_parent_object() -> None:
    rng = np.random.default_rng(99)
    spec = random_spec(rng)
    hashes = {spec.canonical_hash()}
    for _ in range(50):
        child = mutate(spec, rng)
        validate(child)
        assert child.canonical_hash() != ""
        hashes.add(child.canonical_hash())
        spec = child
    assert len(hashes) > 1


def test_mutation_does_not_modify_the_parent() -> None:
    parent = random_spec(np.random.default_rng(5))
    before = parent.canonical_hash()
    mutate(parent, np.random.default_rng(6))
    assert parent.canonical_hash() == before


def test_random_spec_respects_its_bounds() -> None:
    for seed in range(10):
        spec = random_spec(
            np.random.default_rng(seed),
            max_lookback=6,
            max_depth=2,
        )
        validate(spec)
        assert spec.max_lookback == 6
        assert spec.holding is not None
        assert 1 <= spec.holding.bars <= 6
        for rule in (*spec.entries, *spec.exits):
            assert _max_window(rule.condition) <= 6


def _max_window(node: object) -> int:
    if isinstance(node, (CrossAbove, CrossBelow, Rise, Fall, Change)):
        return max(
            node.window, _max_window(node.source) if isinstance(node, (Rise, Fall, Change)) else 0
        )
    found = [0]
    for field_name in ("left", "right", "operand", "value", "low", "high", "source", "condition"):
        child = getattr(node, field_name, None)
        if child is not None and not isinstance(child, (str, int, float)):
            found.append(_max_window(child))
    return max(found)


def test_strategy_factory_matches_the_discovery_operator_surface() -> None:
    from scalping.strategy.dsl import StrategyFactory

    factory = StrategyFactory(max_lookback=12, max_depth=2)
    parent = factory.random(np.random.default_rng(31))
    factory.validate(parent)
    child = factory.mutate(parent, np.random.default_rng(32))
    factory.validate(child)
    combined = factory.crossover(parent, child, np.random.default_rng(33))
    if combined is not None:
        factory.validate(combined)


def test_generated_payload_has_only_json_scalars() -> None:
    spec = random_spec(np.random.default_rng(21))
    payload: dict[str, Any] = spec.to_dict()
    text = json.dumps(payload, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["kind"] == "strategy"
