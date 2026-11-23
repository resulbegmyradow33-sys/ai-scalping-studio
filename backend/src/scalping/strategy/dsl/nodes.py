"""Typed, frozen nodes of the strategy DSL.

The DSL is a *data* language. A strategy is a tree of these frozen dataclasses;
nothing in the package turns a tree into Python source. Evaluation happens in
:mod:`scalping.strategy.dsl.compile` with a closed interpreter that knows the
node types below and nothing else.

All offsets are strictly positive backward lookbacks. There is deliberately no
``Shift``/``Lag`` node with a signed argument and no arithmetic that could reach
a row after the decision bar, so a future reference is not expressible in the
first place.

Node constructors reject the cheapest mistakes immediately (unknown feature
names, unknown comparison operators, non-finite constants, non-positive
windows). Structural rules that need the whole tree live in
:mod:`scalping.strategy.dsl.validate`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from scalping.features.build import FEATURE_COLUMNS

__all__ = [
    "COMPARISON_OPERATORS",
    "DSL_SPEC_VERSION",
    "MAX_ABSOLUTE_LOOKBACK",
    "MAX_DEPTH",
    "MAX_HOLDING_BARS",
    "MAX_NODES",
    "SIDES",
    "And",
    "AtrMultiple",
    "Between",
    "Change",
    "Compare",
    "Const",
    "CrossAbove",
    "CrossBelow",
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
    "Rise",
    "StrategySpec",
    "expression_from_payload",
    "node_from_payload",
    "strategy_from_payload",
]

#: Version of the serialized representation. Bump only with a migration.
DSL_SPEC_VERSION: Final[int] = 1

#: Absolute ceiling for ``max_lookback``. Keeps one candidate from asking for
#: years of history when a search generates it.
MAX_ABSOLUTE_LOOKBACK: Final[int] = 500

#: Absolute ceiling for a holding horizon. It is a forward dwell, not history,
#: so it is bounded on its own rather than by ``max_lookback``.
MAX_HOLDING_BARS: Final[int] = 500

#: Structural caps enforced by :func:`~scalping.strategy.dsl.validate.validate`.
MAX_NODES: Final[int] = 64
MAX_DEPTH: Final[int] = 16

#: The complete operator whitelist. No operator string is ever dispatched.
COMPARISON_OPERATORS: Final[frozenset[str]] = frozenset({"<", "<=", ">", ">=", "==", "!="})

#: Entry sides.
SIDES: Final[tuple[str, str]] = ("long", "short")

#: Feature names are exactly the causal columns the feature layer produces.
FEATURE_NAMES: Final[tuple[str, ...]] = FEATURE_COLUMNS

_OPERATOR_ORDER: Final[tuple[str, ...]] = ("<", "<=", ">", ">=", "==", "!=")


class DslValidationError(ValueError):
    """A node or spec violates the DSL contract.

    Subclasses :class:`ValueError` so callers that only understand ``ValueError``
    still treat an invalid candidate as data error, never as a crash to retry.
    """


def _finite(value: object, label: str) -> float:
    """Coerce ``value`` to a finite float, rejecting bools and NaN/inf."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DslValidationError(f"{label} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise DslValidationError(f"{label} must be finite, got {number!r}")
    # Normalise negative zero so two spellings cannot produce two hashes.
    return 0.0 if number == 0.0 else number


def _positive_int(value: object, label: str) -> int:
    """Coerce ``value`` to an int >= 1, rejecting bools."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise DslValidationError(f"{label} must be an int, got {type(value).__name__}")
    if value < 1:
        raise DslValidationError(f"{label} must be >= 1, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class Feature:
    """One causal feature column from the feature registry."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise DslValidationError("Feature.name must be a string")
        if self.name not in FEATURE_NAMES:
            raise DslValidationError(
                f"unknown feature {self.name!r}; "
                f"expected one of the {len(FEATURE_NAMES)} causal columns"
            )


@dataclass(frozen=True, slots=True)
class Const:
    """A finite numeric constant."""

    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite(self.value, "Const.value"))


@dataclass(frozen=True, slots=True)
class Compare:
    """Numeric comparison of two numeric expressions."""

    left: Expr
    op: str
    right: Expr

    def __post_init__(self) -> None:
        if self.op not in COMPARISON_OPERATORS:
            raise DslValidationError(
                f"unknown comparison operator {self.op!r}; allowed: {', '.join(_OPERATOR_ORDER)}"
            )


@dataclass(frozen=True, slots=True)
class And:
    """Logical conjunction."""

    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class Or:
    """Logical disjunction."""

    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class Not:
    """Logical negation."""

    operand: Expr


@dataclass(frozen=True, slots=True)
class Between:
    """Inclusive range test: ``low <= value <= high``."""

    value: Expr
    low: Expr
    high: Expr


@dataclass(frozen=True, slots=True)
class CrossAbove:
    """``left`` is above ``right`` now and was at/below it ``window`` bars ago.

    The window is a backward offset in bars, so the comparison never touches the
    current or a future bar beyond ``left``/``right`` at those two instants.
    """

    left: Expr
    right: Expr
    window: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "window", _positive_int(self.window, "CrossAbove.window"))


@dataclass(frozen=True, slots=True)
class CrossBelow:
    """``left`` is below ``right`` now and was at/above it ``window`` bars ago."""

    left: Expr
    right: Expr
    window: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "window", _positive_int(self.window, "CrossBelow.window"))


@dataclass(frozen=True, slots=True)
class Rise:
    """``source`` is higher than it was ``window`` bars ago."""

    source: Expr
    window: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "window", _positive_int(self.window, "Rise.window"))


@dataclass(frozen=True, slots=True)
class Fall:
    """``source`` is lower than it was ``window`` bars ago."""

    source: Expr
    window: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "window", _positive_int(self.window, "Fall.window"))


@dataclass(frozen=True, slots=True)
class Change:
    """Numeric difference ``source[t] - source[t - window]``."""

    source: Expr
    window: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "window", _positive_int(self.window, "Change.window"))


@dataclass(frozen=True, slots=True)
class EntryRule:
    """Open a position on ``side`` when ``condition`` holds."""

    side: str
    condition: Expr

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise DslValidationError(f"EntryRule.side must be one of {SIDES}, got {self.side!r}")


@dataclass(frozen=True, slots=True)
class ExitRule:
    """Close the open position when ``condition`` holds."""

    condition: Expr


@dataclass(frozen=True, slots=True)
class HoldingLimit:
    """Close a position after at most ``bars`` bars."""

    bars: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bars", _positive_int(self.bars, "HoldingLimit.bars"))


@dataclass(frozen=True, slots=True)
class NoTrade:
    """Suppress all entries while ``condition`` holds (NO TRADE has priority)."""

    condition: Expr


@dataclass(frozen=True, slots=True)
class FixedFraction:
    """Request a fixed fraction of equity per trade, in ``(0, 1]``."""

    fraction: float

    def __post_init__(self) -> None:
        value = _finite(self.fraction, "FixedFraction.fraction")
        if not 0.0 < value <= 1.0:
            raise DslValidationError(f"FixedFraction.fraction must be in (0, 1], got {value}")
        object.__setattr__(self, "fraction", value)


@dataclass(frozen=True, slots=True)
class AtrMultiple:
    """Request risk sized as a multiple of ATR, strictly positive."""

    multiple: float

    def __post_init__(self) -> None:
        value = _finite(self.multiple, "AtrMultiple.multiple")
        if value <= 0.0:
            raise DslValidationError(f"AtrMultiple.multiple must be > 0, got {value}")
        object.__setattr__(self, "multiple", value)


Expr = (
    Feature
    | Const
    | Compare
    | And
    | Or
    | Not
    | Between
    | CrossAbove
    | CrossBelow
    | Rise
    | Fall
    | Change
)

Sizing = FixedFraction | AtrMultiple


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """Root of one strategy.

    ``max_lookback`` bounds every backward window, so a search can price the
    required history once per candidate. The holding limit is a forward dwell
    and is bounded separately by :data:`MAX_HOLDING_BARS`.
    """

    entries: tuple[EntryRule, ...]
    exits: tuple[ExitRule, ...]
    holding: HoldingLimit | None = None
    no_trade: NoTrade | None = None
    sizing: Sizing | None = None
    max_lookback: int = 20
    spec_version: int = DSL_SPEC_VERSION

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe payload. Validates nothing; use :func:`validate` for that."""
        return _payload(self)

    @classmethod
    def from_dict(cls, payload: object) -> StrategySpec:
        """Strict payload loader; unknown nodes/fields and invalid specs raise."""
        from scalping.strategy.dsl.validate import validate

        spec = strategy_from_payload(payload)
        validate(spec)
        return spec

    def canonical_dict(self) -> dict[str, Any]:
        """Canonical JSON payload used as the candidate identity."""
        from scalping.strategy.dsl.canonical import canonical_dict

        return canonical_dict(self)

    def canonical_hash(self) -> str:
        """sha256 of the canonical payload."""
        from scalping.strategy.dsl.canonical import canonical_hash

        return canonical_hash(self)

    def complexity(self) -> int:
        """Structural cost used by the discovery complexity cap."""
        from scalping.strategy.dsl.canonical import complexity

        return complexity(self)


def _strict_keys(payload: object, expected: tuple[str, ...], kind: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise DslValidationError(f"{kind} payload must be a mapping, got {type(payload).__name__}")
    unknown = sorted(set(payload) - set(expected), key=repr)
    missing = [key for key in expected if key not in payload]
    if unknown or missing:
        raise DslValidationError(f"{kind} fields mismatch (unknown={unknown}, missing={missing})")
    return payload


def _payload(node: object) -> dict[str, Any]:
    """Convert any node (or a whole spec) to a JSON-safe dict."""
    if isinstance(node, StrategySpec):
        return {
            "kind": "strategy",
            "spec_version": node.spec_version,
            "entries": [_payload(rule) for rule in node.entries],
            "exits": [_payload(rule) for rule in node.exits],
            "holding": None if node.holding is None else _payload(node.holding),
            "no_trade": None if node.no_trade is None else _payload(node.no_trade),
            "sizing": None if node.sizing is None else _payload(node.sizing),
            "max_lookback": node.max_lookback,
        }
    if isinstance(node, Feature):
        return {"kind": "feature", "name": node.name}
    if isinstance(node, Const):
        return {"kind": "const", "value": node.value}
    if isinstance(node, Compare):
        return {
            "kind": "compare",
            "op": node.op,
            "left": _payload(node.left),
            "right": _payload(node.right),
        }
    if isinstance(node, And):
        return {"kind": "and", "left": _payload(node.left), "right": _payload(node.right)}
    if isinstance(node, Or):
        return {"kind": "or", "left": _payload(node.left), "right": _payload(node.right)}
    if isinstance(node, Not):
        return {"kind": "not", "operand": _payload(node.operand)}
    if isinstance(node, Between):
        return {
            "kind": "between",
            "value": _payload(node.value),
            "low": _payload(node.low),
            "high": _payload(node.high),
        }
    if isinstance(node, CrossAbove):
        return {
            "kind": "cross_above",
            "left": _payload(node.left),
            "right": _payload(node.right),
            "window": node.window,
        }
    if isinstance(node, CrossBelow):
        return {
            "kind": "cross_below",
            "left": _payload(node.left),
            "right": _payload(node.right),
            "window": node.window,
        }
    if isinstance(node, Rise):
        return {"kind": "rise", "source": _payload(node.source), "window": node.window}
    if isinstance(node, Fall):
        return {"kind": "fall", "source": _payload(node.source), "window": node.window}
    if isinstance(node, Change):
        return {"kind": "change", "source": _payload(node.source), "window": node.window}
    if isinstance(node, EntryRule):
        return {"kind": "entry", "side": node.side, "condition": _payload(node.condition)}
    if isinstance(node, ExitRule):
        return {"kind": "exit", "condition": _payload(node.condition)}
    if isinstance(node, HoldingLimit):
        return {"kind": "holding_limit", "bars": node.bars}
    if isinstance(node, NoTrade):
        return {"kind": "no_trade", "condition": _payload(node.condition)}
    if isinstance(node, FixedFraction):
        return {"kind": "fixed_fraction", "fraction": node.fraction}
    if isinstance(node, AtrMultiple):
        return {"kind": "atr_multiple", "multiple": node.multiple}
    raise DslValidationError(f"unknown DSL node {type(node).__name__}")


def node_from_payload(payload: object) -> object:
    """Load one node strictly. Unknown kinds and fields raise."""
    if not isinstance(payload, Mapping):
        raise DslValidationError(f"node payload must be a mapping, got {type(payload).__name__}")
    kind = payload.get("kind")
    if not isinstance(kind, str):
        raise DslValidationError("node payload must carry a string 'kind'")
    if kind == "feature":
        body = _strict_keys(payload, ("kind", "name"), "feature")
        return Feature(body["name"])
    if kind == "const":
        body = _strict_keys(payload, ("kind", "value"), "const")
        return Const(body["value"])
    if kind == "compare":
        body = _strict_keys(payload, ("kind", "op", "left", "right"), "compare")
        return Compare(
            expression_from_payload(body["left"]),
            body["op"],
            expression_from_payload(body["right"]),
        )
    if kind in ("and", "or"):
        body = _strict_keys(payload, ("kind", "left", "right"), kind)
        left = expression_from_payload(body["left"])
        right = expression_from_payload(body["right"])
        return And(left, right) if kind == "and" else Or(left, right)
    if kind == "not":
        body = _strict_keys(payload, ("kind", "operand"), "not")
        return Not(expression_from_payload(body["operand"]))
    if kind == "between":
        body = _strict_keys(payload, ("kind", "value", "low", "high"), "between")
        return Between(
            expression_from_payload(body["value"]),
            expression_from_payload(body["low"]),
            expression_from_payload(body["high"]),
        )
    if kind in ("cross_above", "cross_below"):
        body = _strict_keys(payload, ("kind", "left", "right", "window"), kind)
        left = expression_from_payload(body["left"])
        right = expression_from_payload(body["right"])
        window = body["window"]
        if kind == "cross_above":
            return CrossAbove(left, right, window)
        return CrossBelow(left, right, window)
    if kind in ("rise", "fall", "change"):
        body = _strict_keys(payload, ("kind", "source", "window"), kind)
        source = expression_from_payload(body["source"])
        window = body["window"]
        if kind == "rise":
            return Rise(source, window)
        if kind == "fall":
            return Fall(source, window)
        return Change(source, window)
    if kind == "entry":
        body = _strict_keys(payload, ("kind", "side", "condition"), "entry")
        return EntryRule(body["side"], expression_from_payload(body["condition"]))
    if kind == "exit":
        body = _strict_keys(payload, ("kind", "condition"), "exit")
        return ExitRule(expression_from_payload(body["condition"]))
    if kind == "holding_limit":
        body = _strict_keys(payload, ("kind", "bars"), "holding_limit")
        return HoldingLimit(body["bars"])
    if kind == "no_trade":
        body = _strict_keys(payload, ("kind", "condition"), "no_trade")
        return NoTrade(expression_from_payload(body["condition"]))
    if kind == "fixed_fraction":
        body = _strict_keys(payload, ("kind", "fraction"), "fixed_fraction")
        return FixedFraction(body["fraction"])
    if kind == "atr_multiple":
        body = _strict_keys(payload, ("kind", "multiple"), "atr_multiple")
        return AtrMultiple(body["multiple"])
    raise DslValidationError(f"unknown node kind {kind!r}; generated code is not accepted")


def expression_from_payload(payload: object) -> Expr:
    """Load one expression subtree, rejecting rule/spec payloads here."""
    node = node_from_payload(payload)
    if not isinstance(
        node,
        (
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
        ),
    ):
        raise DslValidationError(f"expected an expression node, got {type(node).__name__}")
    return node


def strategy_from_payload(payload: object) -> StrategySpec:
    """Load a whole spec strictly; structural validation is the caller's job."""
    body = _strict_keys(
        payload,
        (
            "kind",
            "spec_version",
            "entries",
            "exits",
            "holding",
            "no_trade",
            "sizing",
            "max_lookback",
        ),
        "strategy",
    )
    if body["kind"] != "strategy":
        raise DslValidationError(f"expected a strategy payload, got kind {body['kind']!r}")
    raw_entries = body["entries"]
    raw_exits = body["exits"]
    if not isinstance(raw_entries, list):
        raise DslValidationError("strategy.entries must be a list in the payload")
    if not isinstance(raw_exits, list):
        raise DslValidationError("strategy.exits must be a list in the payload")
    entries: list[EntryRule] = []
    for item in raw_entries:
        rule = node_from_payload(item)
        if not isinstance(rule, EntryRule):
            raise DslValidationError("strategy.entries may only contain entry rules")
        entries.append(rule)
    exits: list[ExitRule] = []
    for item in raw_exits:
        rule = node_from_payload(item)
        if not isinstance(rule, ExitRule):
            raise DslValidationError("strategy.exits may only contain exit rules")
        exits.append(rule)

    holding_payload = body["holding"]
    holding = None if holding_payload is None else node_from_payload(holding_payload)
    if holding is not None and not isinstance(holding, HoldingLimit):
        raise DslValidationError("strategy.holding must be a holding_limit node")

    no_trade_payload = body["no_trade"]
    no_trade = None if no_trade_payload is None else node_from_payload(no_trade_payload)
    if no_trade is not None and not isinstance(no_trade, NoTrade):
        raise DslValidationError("strategy.no_trade must be a no_trade node")

    sizing_payload = body["sizing"]
    sizing = None if sizing_payload is None else node_from_payload(sizing_payload)
    if sizing is not None and not isinstance(sizing, (FixedFraction, AtrMultiple)):
        raise DslValidationError("strategy.sizing must be fixed_fraction or atr_multiple")

    return StrategySpec(
        entries=tuple(entries),
        exits=tuple(exits),
        holding=holding,
        no_trade=no_trade,
        sizing=sizing,
        max_lookback=body["max_lookback"],
        spec_version=body["spec_version"],
    )
