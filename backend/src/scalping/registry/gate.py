"""Candidate → champion gate: a fixed policy between "trained" and "champion".

A model that finishes training is a candidate. The registry can make any
registered model the champion, so the gate is the only place that decides
whether a candidate *deserves* it. The decision is a pure function of the
evidence the caller passes in; nothing here trains, re-prices or re-simulates.

Every threshold is disabled at ``0`` (or an empty stress list / ``None``
evidence requirement), following the same convention as ``RiskConfig``: a
default :class:`GateCriteria` accepts everything and changes no existing run.
The recommended policy lives in ``docs/local-learning-engine.md``; the code
keeps the values explicit so a caller cannot inherit a threshold by accident.

Boundary rules are deliberate and tested at the exact value:

* ``min_trades``, ``min_expectancy_r``, ``min_profit_factor`` and
  ``min_regime_coverage`` pass **at** the threshold (``>=``).
* ``max_drawdown`` and ``max_ece`` pass **at** the threshold (``<=``).
* a cost-stress run must be **strictly** profitable (``net_profit > 0``).

Missing evidence rejects. A criterion that is enabled but whose input is
absent produces a rejection, never a silent pass: an unmeasured cost stress or
calibration is not the same as a passing one. The gate encodes policy, not
proof — a candidate that passes has cleared the checks someone chose, it has
not been shown to be profitable in the future.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from scalping.registry.db import ModelRegistry, PromotionRecord, RegistryError

__all__ = [
    "GateCriteria",
    "GateDecision",
    "PromotionRefused",
    "evaluate_candidate",
    "promote_candidate",
]


@dataclass(frozen=True, slots=True)
class GateCriteria:
    """Thresholds for one promotion decision; ``0``/empty means "disabled".

    ``max_drawdown`` and ``max_ece`` are fractions in ``[0, 1]``;
    ``min_regime_coverage`` is a fraction of the closed trades whose entry had
    a known regime. ``require_positive_cost_stress`` holds simulator cost
    multipliers; each one must be present in the stress evidence and have a
    strictly positive ``net_profit``.
    """

    min_trades: int = 0
    min_expectancy_r: float = 0.0
    min_profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_ece: float = 0.0
    require_positive_cost_stress: tuple[float, ...] = ()
    min_regime_coverage: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.min_trades, bool) or not isinstance(self.min_trades, int):
            raise TypeError("min_trades must be an int")
        if self.min_trades < 0:
            raise ValueError("min_trades must be >= 0")
        for name in ("min_expectancy_r", "min_profit_factor"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name in ("max_drawdown", "max_ece", "min_regime_coverage"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a fraction in [0, 1]")
        multipliers = self.require_positive_cost_stress
        if isinstance(multipliers, list):  # caller convenience; store a tuple
            multipliers = tuple(multipliers)
            object.__setattr__(self, "require_positive_cost_stress", multipliers)
        for multiplier in multipliers:
            if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool):
                raise TypeError("cost-stress multipliers must be numbers")
            if not math.isfinite(float(multiplier)) or float(multiplier) <= 0.0:
                raise ValueError("cost-stress multipliers must be finite and > 0")

    def is_empty(self) -> bool:
        """True when no criterion is enabled, so every candidate passes."""
        return self == GateCriteria()


@dataclass(frozen=True, slots=True)
class GateDecision:
    """The gate's verdict and why.

    ``rejections`` is one string per failed check, ``"code: message"``, so a
    log line carries both the machine-readable code and a sentence a human can
    act on. :attr:`codes` returns the codes alone.
    """

    accepted: bool
    rejections: tuple[str, ...]
    criteria: GateCriteria
    observed: Mapping[str, Any] = field(default_factory=dict)

    @property
    def codes(self) -> tuple[str, ...]:
        """The rejection codes, in the order the checks ran."""
        return tuple(item.split(":", 1)[0] for item in self.rejections)

    @property
    def summary(self) -> str:
        """One line for the promotion reason: verdict, codes and measured values."""
        headline = "gate accepted" if self.accepted else "gate rejected"
        if self.codes:
            headline += " (" + ", ".join(self.codes) + ")"
        values = [
            f"{name}={value:g}"
            for name, value in sorted(self.observed.items())
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        return headline + (": " + ", ".join(values) if values else "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rejections": list(self.rejections),
            "codes": list(self.codes),
            "summary": self.summary,
            "observed": dict(self.observed),
            "criteria": {
                "min_trades": self.criteria.min_trades,
                "min_expectancy_r": self.criteria.min_expectancy_r,
                "min_profit_factor": self.criteria.min_profit_factor,
                "max_drawdown": self.criteria.max_drawdown,
                "max_ece": self.criteria.max_ece,
                "require_positive_cost_stress": list(self.criteria.require_positive_cost_stress),
                "min_regime_coverage": self.criteria.min_regime_coverage,
            },
        }


class PromotionRefused(RegistryError):
    """The gate rejected the candidate; the champion is unchanged."""


def _number(metrics: Mapping[str, Any], key: str) -> float | None:
    """A metric as a float, or ``None`` when it is absent or NaN."""
    value = metrics.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _reject(rejections: list[str], code: str, message: str) -> None:
    rejections.append(f"{code}: {message}")


def _stress_label(multiplier: float) -> str:
    return f"x{multiplier:g}"


def _stress_net_profit(stress: Mapping[str, Any] | None, multiplier: float) -> float | None:
    """``net_profit`` of one stress run, or ``None`` when the evidence is absent."""
    if stress is None:
        return None
    label = _stress_label(multiplier)
    entry = stress.get(label)
    if entry is None:
        return None
    if not isinstance(entry, Mapping):
        return None
    return _number(entry, "net_profit")


def _calibration_ece(calibration: Mapping[str, Any] | float | None) -> float | None:
    """ECE from a calibration summary dict or a bare number; ``None`` if absent."""
    if calibration is None:
        return None
    if isinstance(calibration, Mapping):
        return _number(calibration, "ece")
    if isinstance(calibration, bool):
        return None
    try:
        number = float(calibration)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _regime_coverage(regime: Mapping[str, Any] | None, trades: float | None) -> float | None:
    """Fraction of closed trades with a known regime label.

    ``regime`` is the ``EvaluationResult.regime`` shape: ``rows`` excludes the
    ``unknown`` bucket and ``unmatched_trades`` counts it. A missing or empty
    table is missing evidence, not zero coverage. The headline trade count is
    used as the denominator when available so a table that claims more trades
    than the run cannot inflate coverage.
    """
    if regime is None:
        return None
    rows = regime.get("rows")
    unmatched = regime.get("unmatched_trades")
    if not isinstance(rows, (list, tuple)) or unmatched is None:
        return None
    try:
        assigned = sum(int(row.get("trades", 0)) for row in rows if isinstance(row, Mapping))
        unmatched_count = int(unmatched)
    except (TypeError, ValueError):
        return None
    total = assigned + unmatched_count
    if total <= 0:
        return None
    if trades is not None and trades > 0:
        total = max(total, int(trades))
    return min(1.0, assigned / total)


def evaluate_candidate(
    criteria: GateCriteria,
    metrics: Mapping[str, Any],
    stress: Mapping[str, Mapping[str, Any]] | None = None,
    calibration: Mapping[str, Any] | float | None = None,
    regime: Mapping[str, Any] | None = None,
) -> GateDecision:
    """Apply ``criteria`` to one candidate's evidence.

    ``metrics`` is the dict from ``EvaluationResult.metrics``; ``stress`` maps
    ``"x<multiplier>"`` labels (as ``EvaluationResult.stress`` spells them) to
    the same dict shape. ``calibration`` is either a
    ``calibration_summary`` dict (``"ece"`` is read) or the ECE itself, and
    ``regime`` is ``EvaluationResult.regime``.
    """
    if not isinstance(criteria, GateCriteria):
        raise TypeError("criteria must be a GateCriteria")
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    if stress is not None and not isinstance(stress, Mapping):
        raise TypeError("stress must be a mapping or None")
    if calibration is not None and not isinstance(calibration, (Mapping, int, float)):
        raise TypeError("calibration must be a mapping, a number or None")
    if regime is not None and not isinstance(regime, Mapping):
        raise TypeError("regime must be a mapping or None")

    rejections: list[str] = []
    observed: dict[str, Any] = {}
    #: The headline count, used as the regime-coverage denominator when known.
    trades = _number(metrics, "trades")

    if criteria.min_trades > 0:
        raw_trades = trades
        if raw_trades is None:
            _reject(rejections, "missing_metric", "'trades' is required by min_trades")
        else:
            observed["trades"] = raw_trades
            if raw_trades < criteria.min_trades:
                _reject(
                    rejections,
                    "min_trades",
                    f"{raw_trades:g} closed trades is below the required {criteria.min_trades}",
                )

    if criteria.min_expectancy_r > 0.0:
        value = _number(metrics, "expectancy_r")
        if value is None:
            _reject(
                rejections,
                "missing_metric",
                "'expectancy_r' is required by min_expectancy_r",
            )
        else:
            observed["expectancy_r"] = value
            if value < criteria.min_expectancy_r:
                _reject(
                    rejections,
                    "min_expectancy_r",
                    f"expectancy {value:g}R is below the required {criteria.min_expectancy_r:g}R",
                )

    if criteria.min_profit_factor > 0.0:
        value = _number(metrics, "profit_factor")
        if value is None:
            _reject(
                rejections,
                "missing_metric",
                "'profit_factor' is required by min_profit_factor",
            )
        else:
            observed["profit_factor"] = value
            if value < criteria.min_profit_factor:
                _reject(
                    rejections,
                    "min_profit_factor",
                    f"profit factor {value:g} is below the required {criteria.min_profit_factor:g}",
                )

    if criteria.max_drawdown > 0.0:
        value = _number(metrics, "max_drawdown")
        if value is None:
            _reject(
                rejections,
                "missing_metric",
                "'max_drawdown' is required by max_drawdown",
            )
        else:
            observed["max_drawdown"] = value
            if value > criteria.max_drawdown:
                _reject(
                    rejections,
                    "max_drawdown",
                    f"drawdown {value:g} exceeds the allowed {criteria.max_drawdown:g}",
                )

    if criteria.max_ece > 0.0:
        ece = _calibration_ece(calibration)
        if ece is None:
            _reject(
                rejections,
                "missing_calibration",
                "'ece' is required by max_ece and was not supplied",
            )
        else:
            observed["ece"] = ece
            if ece > criteria.max_ece:
                _reject(
                    rejections,
                    "max_ece",
                    f"ECE {ece:g} exceeds the allowed {criteria.max_ece:g}",
                )

    for multiplier in criteria.require_positive_cost_stress:
        net = _stress_net_profit(stress, multiplier)
        if net is None:
            _reject(
                rejections,
                "missing_cost_stress",
                f"cost stress {_stress_label(multiplier)} has no net_profit evidence",
            )
            continue
        observed[f"stress_{_stress_label(multiplier)}_net_profit"] = net
        if net <= 0.0:
            _reject(
                rejections,
                "cost_stress",
                f"cost stress {_stress_label(multiplier)} loses {abs(net):g} (net_profit {net:g})",
            )

    if criteria.min_regime_coverage > 0.0:
        coverage = _regime_coverage(regime, trades)
        if coverage is None:
            _reject(
                rejections,
                "missing_regime",
                "'regime' coverage is required by min_regime_coverage and could not be measured",
            )
        else:
            observed["regime_coverage"] = coverage
            if coverage < criteria.min_regime_coverage:
                _reject(
                    rejections,
                    "min_regime_coverage",
                    f"regime coverage {coverage:.3f} is below the required "
                    f"{criteria.min_regime_coverage:.3f}",
                )

    return GateDecision(
        accepted=not rejections,
        rejections=tuple(rejections),
        criteria=criteria,
        observed=observed,
    )


def promote_candidate(
    registry: ModelRegistry,
    model_id: str,
    decision: GateDecision,
    *,
    scope: str = "global",
    reason: str | None = None,
    force: bool = False,
) -> PromotionRecord:
    """Promote ``model_id`` only through a gate decision.

    An accepted decision promotes normally. A rejected one raises
    :class:`PromotionRefused` unless ``force=True``, and a forced promotion
    requires a non-empty human ``reason``; both the word ``forced`` and the
    gate summary are written into the promotion reason, so the audit trail
    never hides that the policy was overridden. ``ModelRegistry.promote`` keeps
    its own signature and remains the primitive.
    """
    if not isinstance(decision, GateDecision):
        raise TypeError("decision must be a GateDecision")
    if reason is not None:
        if not isinstance(reason, str):
            raise TypeError("reason must be a str")
        human = reason.strip()
    else:
        human = ""
    if not decision.accepted and not force:
        raise PromotionRefused(
            f"candidate {model_id!r} was rejected by the gate: " + "; ".join(decision.rejections)
        )
    if force and not human:
        raise ValueError("a forced promotion requires a non-empty reason")

    summary = decision.summary
    if force:
        text = f"forced: {human}; {summary}"
    elif human:
        text = f"{human}; {summary}"
    else:
        text = summary
    return registry.promote(model_id, text, scope=scope)
