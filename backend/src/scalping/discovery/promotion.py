"""Fail-closed, versioned promotion policy over a candidate evidence bundle.

Discovery produces **candidates**. A candidate becomes the paper champion only
when a named, versioned policy accepts its evidence. This module is that policy:
it reads a :class:`~scalping.discovery.evidence.CandidateEvidence` bundle, checks
identity, sample, stability, cost stress, search exposure and — when a champion
already exists — a same-window improvement, and returns a
:class:`PromotionDecision` that the archive can store.

Why a separate type
-------------------
The registry gate (:class:`scalping.registry.gate.GateCriteria`) and the
exploratory archive promotion compare one measured metric. That is enough to
rank candidates during search and far too little to select what trades. A
promotion policy is deliberately *not* default-constructible into a no-op:

* ``policy_id`` must be non-empty and is not the reserved
  :data:`~scalping.discovery.archive.MANUAL_OVERRIDE_POLICY_ID`;
* ``version`` must be ``>= 1``;
* ``min_folds`` must be at least :data:`~scalping.discovery.evidence.MIN_EVIDENCE_FOLDS`;
* ``min_profit_factor`` must be greater than ``1``;
* drawdown, positive-fold and regime fractions must be in ``(0, 1]``;
* at least one cost-stress multiplier above ``1.0`` is required;
* complexity and trial caps must be positive.

:data:`PROMOTION_V1` is the explicit starting policy. There is no empty policy
and no ``force`` parameter anywhere in this module.

Fail-closed rules
-----------------
Every missing, stale, edited or mismatched input is a rejection, never a pass.
The checks run in a fixed order and each failure appends a
:class:`PromotionRejection` with a stable ``code``, a human ``message`` and the
``observed``/``required`` values. A decision is ``accepted`` only when the
rejection list is empty, the bundle checksum re-derives and every required
section exists.

Two comparisons are deliberately separate from the candidate's own numbers:

* **Champion evidence.** When a champion exists, the challenger is compared on
  the *same* dataset id, dataset checksum, code version, symbol/timeframe and
  fold boundaries. A different window is a rejection, not a normalisation
  problem. The champion's evidence checksum is itself re-derived.
* **Manual override.** :func:`manual_override` records a human decision in a
  separate :class:`ManualOverride` whose ``policy_id`` is ``"manual-override"``.
  It keeps the original policy rejections, has no ``accepted`` flag and cannot
  be passed to :func:`promote_with_evidence`. It is an audit record, not
  validation evidence.

Policy identity
---------------
The evidence bundle records an ``EvidencePolicy`` section (folds, purge/embargo,
stress multipliers, confidence, window) but no policy id/version. The applied
policy's identity is therefore checked where it can be: ``PromotionPolicy``
refuses an empty id or a version below 1, the ``PromotionDecision`` records both,
and the archive refuses a promotion with an empty/zero policy identity. The
checksum binds the numbers; the decision binds the policy that judged them. A
future bundle that records a policy id will be compared to the applied policy.

What acceptance means
---------------------
Acceptance is *policy compliance*: the bundle passed the checks this project
chose. It is not proof of an edge, not out-of-sample validation of future
returns and not a profitability claim. The locked final OOS section is still
missing from the bundle, so ``require_locked_oos`` defaults to ``False``; a
policy that sets it ``True`` rejects every bundle until that section exists.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final

from scalping.discovery.archive import (
    MANUAL_OVERRIDE_POLICY_ID,
    ArchiveError,
    ChampionRecord,
    DiscoveryArchive,
)
from scalping.discovery.evidence import (
    EVIDENCE_SCHEMA_VERSION,
    MIN_EVIDENCE_FOLDS,
    CandidateEvidence,
    EvidencePolicy,
    evidence_checksum,
)

__all__ = [
    "MANUAL_OVERRIDE_POLICY_ID",
    "PROMOTION_V1",
    "ManualOverride",
    "PromotionDecision",
    "PromotionPolicy",
    "PromotionRefused",
    "PromotionRejection",
    "evaluate_promotion",
    "manual_override",
    "promote_with_evidence",
]

#: Version of a manual-override record's own shape, not of a promotion policy.
MANUAL_OVERRIDE_VERSION: Final[int] = 1

#: Fields whose absence makes a bundle unusable before any threshold is read.
_REQUIRED_IDENTITY: Final[tuple[str, ...]] = (
    "spec_hash",
    "dataset_id",
    "dataset_checksum",
    "code_version",
    "symbol",
    "timeframe",
)

#: Fixed labels the policy compares against stored stress multipliers.
_MULTIPLIER_TOLERANCE: Final[float] = 1e-9

#: Float-noise tolerance for an R-difference boundary (0.25 - 0.20 != 0.05).
_COMPARISON_TOLERANCE: Final[float] = 1e-9


class PromotionRefused(ArchiveError):
    """The decision does not permit promotion; the champion is unchanged."""


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    """One named, versioned promotion policy; every field is a real threshold.

    The constructor refuses a degenerate policy, so a caller cannot inherit a
    permissive default. ``require_cost_stress`` multipliers are absolute
    multipliers of the base cost (``1.5`` means "one and a half times") and must
    appear in every fold's stress evidence. ``max_trials`` caps both the
    candidate's own ``trial_count`` and the recorded ``search_budget``.
    ``min_improvement_r`` is the minimum expectancy improvement over the
    champion, in R, and must be positive.

    ``require_locked_oos`` is ``False`` by default because the current evidence
    schema has no locked-OOS section. Setting it ``True`` is honest and makes
    every promotion reject until that section is added; it is not a switch to
    flip casually.
    """

    policy_id: str
    version: int
    min_folds: int
    min_trades: int
    min_expectancy_r: float
    min_profit_factor: float
    max_drawdown: float
    min_positive_fold_fraction: float
    min_regime_coverage: float
    min_ci_lower_r: float
    require_cost_stress: tuple[float, ...]
    max_complexity: int
    max_trials: int
    min_improvement_r: float
    require_locked_oos: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str):
            raise TypeError(f"policy_id must be a str, got {type(self.policy_id).__name__}")
        policy_id = self.policy_id.strip()
        if not policy_id:
            raise ValueError("policy_id must not be empty")
        if policy_id == MANUAL_OVERRIDE_POLICY_ID:
            raise ValueError(
                f"{MANUAL_OVERRIDE_POLICY_ID!r} is reserved for manual override records "
                "and cannot name a promotion policy"
            )
        object.__setattr__(self, "policy_id", policy_id)

        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version < 1:
            raise ValueError(f"version must be >= 1, got {self.version}")

        for name in ("min_folds", "min_trades", "max_complexity", "max_trials"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if self.min_folds < MIN_EVIDENCE_FOLDS:
            raise ValueError(
                f"min_folds must be at least {MIN_EVIDENCE_FOLDS}; promotion evidence "
                "without three folds says nothing about stability"
            )
        for name in ("min_trades", "max_complexity", "max_trials"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")

        for name in (
            "min_expectancy_r",
            "min_profit_factor",
            "max_drawdown",
            "min_positive_fold_fraction",
            "min_regime_coverage",
            "min_ci_lower_r",
            "min_improvement_r",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number, got {type(value).__name__}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.min_profit_factor <= 1.0:
            raise ValueError(
                f"min_profit_factor must be > 1, got {self.min_profit_factor}; a policy "
                "that accepts break-even is not a promotion policy"
            )
        if not 0.0 < self.max_drawdown <= 1.0:
            raise ValueError(f"max_drawdown must be in (0, 1], got {self.max_drawdown}")
        for name in ("min_positive_fold_fraction", "min_regime_coverage"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}")
        if self.min_improvement_r <= 0.0:
            raise ValueError(
                f"min_improvement_r must be > 0, got {self.min_improvement_r}; a tie with "
                "the champion is not an improvement"
            )

        multipliers = self.require_cost_stress
        if isinstance(multipliers, list):  # caller convenience; store a tuple
            multipliers = tuple(multipliers)
            object.__setattr__(self, "require_cost_stress", multipliers)
        if not multipliers:
            raise ValueError("require_cost_stress must name at least one multiplier")
        labels: list[str] = []
        for multiplier in multipliers:
            if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)):
                raise TypeError(f"cost-stress multipliers must be numbers, got {multiplier!r}")
            if not math.isfinite(float(multiplier)) or float(multiplier) <= 1.0:
                raise ValueError(
                    f"cost-stress multipliers must be finite and > 1, got {multiplier!r}; "
                    "a multiplier of 1.0 is the base run, not a stress"
                )
            labels.append(f"x{float(multiplier):g}")
        if len(set(labels)) != len(labels):
            raise ValueError("require_cost_stress multipliers must be distinct")

        if not isinstance(self.require_locked_oos, bool):
            raise TypeError("require_locked_oos must be a bool")


#: The documented starting policy. Values are policy, not statistics: they encode
#: how much evidence this project demands before a DSL spec may trade.
#:
#: * three folds — fewer cannot show stability;
#: * 60 trades — roughly 20 per fold, enough for a pooled mean to mean anything;
#: * +0.05R expectancy and 1.20 profit factor — a margin over a break-even,
#:   cost-paying policy rather than a rounding error;
#: * 25% worst-fold drawdown — the policy cap used by the risk discussion, not a
#:   claim that a real account would survive it;
#: * 2/3 positive folds and 85% regime coverage — a majority of windows must pay
#:   and almost every trade must be attributable to a regime;
#: * bootstrap lower bound strictly above 0R — the interval, not the point
#:   estimate, must exclude zero;
#: * 1.5x and 2.0x cost stress — the multipliers the evidence builder is expected
#:   to have measured;
#: * 40 structural nodes and 500 trials — bounded complexity and bounded
#:   multiple-testing exposure;
#: * +0.05R over the champion — a new champion must improve, not tie.
PROMOTION_V1: Final[PromotionPolicy] = PromotionPolicy(
    policy_id="promotion-v1",
    version=1,
    min_folds=3,
    min_trades=60,
    min_expectancy_r=0.05,
    min_profit_factor=1.20,
    max_drawdown=0.25,
    min_positive_fold_fraction=2.0 / 3.0,
    min_regime_coverage=0.85,
    min_ci_lower_r=0.0,
    require_cost_stress=(1.5, 2.0),
    max_complexity=40,
    max_trials=500,
    min_improvement_r=0.05,
    require_locked_oos=False,
)


@dataclass(frozen=True, slots=True)
class PromotionRejection:
    """One failed check: stable code, human message and the two numbers."""

    code: str
    message: str
    observed: float | str | Sequence[int] | None = None
    required: float | str | Sequence[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "observed": self.observed,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """The policy verdict, attributable to one policy version and one bundle.

    The constructor enforces the invariant the archive relies on: an accepted
    decision cannot carry a rejection or lack an evidence checksum, and a
    decision can never claim the reserved manual-override policy id.
    """

    accepted: bool
    rejections: tuple[PromotionRejection, ...]
    policy_id: str
    version: int
    evidence_checksum: str | None
    champion_checksum: str | None
    candidate_spec_hash: str | None
    champion_spec_hash: str | None
    dataset_id: str | None
    code_version: str | None
    observed: Mapping[str, Any] = field(default_factory=dict, compare=False)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise TypeError(f"accepted must be a bool, got {type(self.accepted).__name__}")
        if not isinstance(self.rejections, tuple):
            raise TypeError("rejections must be a tuple of PromotionRejection")
        for rejection in self.rejections:
            if not isinstance(rejection, PromotionRejection):
                raise TypeError("rejections must be PromotionRejection instances")
        if self.accepted and self.rejections:
            raise ValueError("an accepted decision cannot carry rejections")
        if self.accepted and not self.evidence_checksum:
            raise ValueError("an accepted decision requires an evidence checksum")
        if self.policy_id == MANUAL_OVERRIDE_POLICY_ID:
            raise ValueError("a promotion decision cannot claim the manual-override policy")

    @property
    def codes(self) -> tuple[str, ...]:
        """Rejection codes in the order the checks ran."""
        return tuple(rejection.code for rejection in self.rejections)

    @property
    def summary(self) -> str:
        """One audit-friendly line: verdict, policy and codes."""
        headline = "accepted" if self.accepted else "rejected"
        text = f"policy {self.policy_id} v{self.version} {headline}"
        if self.codes:
            text += " (" + ", ".join(self.codes) + ")"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rejections": [rejection.to_dict() for rejection in self.rejections],
            "codes": list(self.codes),
            "summary": self.summary,
            "policy_id": self.policy_id,
            "version": self.version,
            "evidence_checksum": self.evidence_checksum,
            "champion_checksum": self.champion_checksum,
            "candidate_spec_hash": self.candidate_spec_hash,
            "champion_spec_hash": self.champion_spec_hash,
            "dataset_id": self.dataset_id,
            "code_version": self.code_version,
            "observed": dict(self.observed),
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class ManualOverride:
    """A human override record; deliberately **not** policy validation.

    ``policy_id`` is fixed to :data:`MANUAL_OVERRIDE_POLICY_ID`, the original
    policy id/version are kept beside it and the original rejections are copied
    verbatim. There is no ``accepted`` field and no conversion to a
    :class:`PromotionDecision`; :func:`promote_with_evidence` refuses this type
    outright, so an override can be stored for audit but can never masquerade as
    a passed policy.
    """

    original_policy_id: str
    original_policy_version: int
    reason: str
    owner: str
    rejections: tuple[PromotionRejection, ...]
    evidence_checksum: str | None
    candidate_spec_hash: str | None
    created_at: str
    policy_id: str = MANUAL_OVERRIDE_POLICY_ID
    policy_version: int = MANUAL_OVERRIDE_VERSION
    is_policy_evidence: bool = False

    def __post_init__(self) -> None:
        if self.policy_id != MANUAL_OVERRIDE_POLICY_ID:
            raise ValueError(f"policy_id is fixed to {MANUAL_OVERRIDE_POLICY_ID!r}")
        for name in ("original_policy_id", "reason", "owner"):
            _text(getattr(self, name), name)
        if self.is_policy_evidence:
            raise ValueError("a manual override is never policy evidence")

    @property
    def note(self) -> str:
        """The sentence a report must print next to an override."""
        return "manual override: recorded for audit, not validation evidence"

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "original_policy_id": self.original_policy_id,
            "original_policy_version": self.original_policy_version,
            "reason": self.reason,
            "owner": self.owner,
            "rejections": [rejection.to_dict() for rejection in self.rejections],
            "evidence_checksum": self.evidence_checksum,
            "candidate_spec_hash": self.candidate_spec_hash,
            "created_at": self.created_at,
            "is_policy_evidence": self.is_policy_evidence,
            "note": self.note,
        }


# --- evaluation helpers --------------------------------------------------------


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a str, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def _reject(
    rejections: list[PromotionRejection],
    code: str,
    message: str,
    *,
    observed: float | str | Sequence[int] | None = None,
    required: float | str | Sequence[int] | None = None,
) -> None:
    rejections.append(
        PromotionRejection(code=code, message=message, observed=observed, required=required)
    )


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _stress_present(fold_stress: Sequence[Any], multiplier: float) -> bool:
    return any(
        math.isclose(float(row.multiplier), multiplier, rel_tol=0.0, abs_tol=_MULTIPLIER_TOLERANCE)
        for row in fold_stress
    )


def _locked_oos(evidence: CandidateEvidence) -> bool:
    """A locked-OOS section is present when it names a digest and some trades.

    The current bundle has no such section; the check reads the serialised
    payload so a later schema addition is recognised without changing this
    policy's call sites.
    """
    payload = evidence.to_dict().get("locked_oos")
    if not isinstance(payload, Mapping):
        return False
    checksum = payload.get("checksum")
    trades = payload.get("trades")
    return (
        isinstance(checksum, str)
        and bool(checksum.strip())
        and isinstance(trades, int)
        and not isinstance(trades, bool)
        and trades > 0
    )


def _fold_boundaries(evidence: CandidateEvidence) -> tuple[tuple[int, int, int, int, int], ...]:
    return tuple(
        (
            fold.fold,
            fold.train_start_row,
            fold.train_end_row,
            fold.test_start_row,
            fold.test_end_row,
        )
        for fold in evidence.folds
    )


def _check_bundle(
    evidence: CandidateEvidence,
    policy: PromotionPolicy,
    rejections: list[PromotionRejection],
    observed: dict[str, Any],
) -> None:
    """Identity, structure, sample and threshold checks on the candidate alone."""
    computed = evidence_checksum(evidence)
    if evidence.checksum != computed:
        _reject(
            rejections,
            "checksum_mismatch",
            "the bundle checksum does not re-derive from its payload",
            observed=evidence.checksum[:12] if evidence.checksum else "",
            required=computed[:12],
        )
    else:
        observed["evidence_checksum"] = computed

    if evidence.schema_version != EVIDENCE_SCHEMA_VERSION:
        _reject(
            rejections,
            "schema_unsupported",
            "the bundle layout version is not the one this policy reads",
            observed=evidence.schema_version,
            required=EVIDENCE_SCHEMA_VERSION,
        )

    for name in _REQUIRED_IDENTITY:
        value = getattr(evidence, name)
        if not isinstance(value, str) or not value.strip():
            _reject(
                rejections,
                "missing_section",
                f"identity field {name} is empty; the bundle cannot be attributed",
                observed=name,
                required="non-empty",
            )

    if not isinstance(evidence.policy, EvidencePolicy):
        _reject(
            rejections,
            "missing_section",
            "the bundle has no evidence policy section",
            observed=type(evidence.policy).__name__,
            required="EvidencePolicy",
        )
    elif evidence.policy.bootstrap_samples < 1:
        _reject(
            rejections,
            "missing_section",
            "the bundle records no bootstrap samples",
            observed=evidence.policy.bootstrap_samples,
            required=1,
        )

    folds = evidence.folds
    aggregate = evidence.aggregate
    if not folds:
        _reject(
            rejections,
            "insufficient_folds",
            "the bundle has no folds",
            observed=0,
            required=policy.min_folds,
        )
        return
    if aggregate is None:
        _reject(
            rejections,
            "missing_section",
            "the bundle has no aggregate section",
            observed="absent",
            required="aggregate",
        )
        return

    if len(folds) < policy.min_folds:
        _reject(
            rejections,
            "insufficient_folds",
            f"{len(folds)} fold(s) is below the policy minimum",
            observed=len(folds),
            required=policy.min_folds,
        )
    if [fold.fold for fold in folds] != list(range(1, len(folds) + 1)):
        _reject(
            rejections,
            "inconsistent_evidence",
            "fold numbers are not the contiguous sequence 1..n",
            observed=[fold.fold for fold in folds][:8],
            required="1..n",
        )
    previous_end: int | None = None
    for fold in folds:
        if fold.trades < 0:
            _reject(
                rejections,
                "inconsistent_evidence",
                f"fold {fold.fold} reports a negative trade count",
                observed=fold.trades,
                required=">= 0",
            )
        if fold.test_end_row <= fold.test_start_row or fold.train_end_row > fold.test_start_row:
            _reject(
                rejections,
                "inconsistent_evidence",
                f"fold {fold.fold} has an empty or overlapping train/test range",
                observed=fold.fold,
                required="train_end <= test_start < test_end",
            )
        if previous_end is not None and fold.test_start_row != previous_end:
            _reject(
                rejections,
                "inconsistent_evidence",
                f"fold {fold.fold} does not start where the previous fold ended",
                observed=fold.test_start_row,
                required=previous_end,
            )
        previous_end = fold.test_end_row

    fold_trades = sum(int(fold.trades) for fold in folds)
    if fold_trades != int(aggregate.total_trades):
        _reject(
            rejections,
            "inconsistent_evidence",
            "aggregate total_trades does not equal the sum of fold trades",
            observed=int(aggregate.total_trades),
            required=fold_trades,
        )
    positive = sum(1 for fold in folds if fold.net_profit > 0.0)
    if positive != int(aggregate.positive_folds):
        _reject(
            rejections,
            "inconsistent_evidence",
            "aggregate positive_folds does not match the folds",
            observed=int(aggregate.positive_folds),
            required=positive,
        )
    expected_fraction = positive / len(folds)
    if abs(float(aggregate.positive_fold_fraction) - expected_fraction) > 1e-9:
        _reject(
            rejections,
            "inconsistent_evidence",
            "aggregate positive_fold_fraction does not match the folds",
            observed=float(aggregate.positive_fold_fraction),
            required=expected_fraction,
        )

    total_trades = _finite(aggregate.total_trades)
    if total_trades is None:
        _reject(rejections, "missing_section", "aggregate total_trades is not finite")
    else:
        observed["total_trades"] = int(total_trades)
        if total_trades < policy.min_trades:
            _reject(
                rejections,
                "insufficient_trades",
                f"{int(total_trades)} pooled trades is below the policy minimum",
                observed=int(total_trades),
                required=policy.min_trades,
            )

    for name, code in (
        ("expectancy_r", "min_expectancy_r"),
        ("profit_factor", "min_profit_factor"),
        ("max_drawdown", "max_drawdown"),
        ("positive_fold_fraction", "min_positive_fold_fraction"),
        ("regime_coverage", "min_regime_coverage"),
        ("expectancy_r_ci95_low", "min_ci_lower_r"),
    ):
        if _finite(getattr(aggregate, name)) is None:
            _reject(
                rejections,
                "missing_section",
                f"aggregate {name} is missing or not finite",
                observed="absent",
                required=name,
            )

    expectancy = _finite(aggregate.expectancy_r)
    if expectancy is not None:
        observed["expectancy_r"] = expectancy
        if expectancy < policy.min_expectancy_r:
            _reject(
                rejections,
                "min_expectancy_r",
                f"pooled expectancy {expectancy:g}R is below the policy minimum",
                observed=expectancy,
                required=policy.min_expectancy_r,
            )

    profit_factor = _finite(aggregate.profit_factor)
    if profit_factor is not None:
        observed["profit_factor"] = profit_factor
        if profit_factor < policy.min_profit_factor:
            _reject(
                rejections,
                "min_profit_factor",
                f"pooled profit factor {profit_factor:g} is below the policy minimum",
                observed=profit_factor,
                required=policy.min_profit_factor,
            )

    drawdown = _finite(aggregate.max_drawdown)
    if drawdown is not None:
        observed["max_drawdown"] = drawdown
        if drawdown > policy.max_drawdown:
            _reject(
                rejections,
                "max_drawdown",
                f"worst-fold drawdown {drawdown:g} exceeds the policy maximum",
                observed=drawdown,
                required=policy.max_drawdown,
            )

    positive_fraction = _finite(aggregate.positive_fold_fraction)
    if positive_fraction is not None:
        observed["positive_fold_fraction"] = positive_fraction
        if positive_fraction < policy.min_positive_fold_fraction:
            _reject(
                rejections,
                "min_positive_fold_fraction",
                f"{positive_fraction:.3f} of folds are positive; the policy requires more",
                observed=positive_fraction,
                required=policy.min_positive_fold_fraction,
            )

    coverage = _finite(aggregate.regime_coverage)
    if coverage is not None:
        observed["regime_coverage"] = coverage
        if coverage < policy.min_regime_coverage:
            _reject(
                rejections,
                "min_regime_coverage",
                f"regime coverage {coverage:.3f} is below the policy minimum",
                observed=coverage,
                required=policy.min_regime_coverage,
            )

    ci_low = _finite(aggregate.expectancy_r_ci95_low)
    if ci_low is not None:
        observed["expectancy_r_ci95_low"] = ci_low
        if ci_low <= policy.min_ci_lower_r:
            _reject(
                rejections,
                "min_ci_lower_r",
                (
                    f"bootstrap lower bound {ci_low:g}R does not exceed the policy floor; "
                    "the interval still includes the threshold"
                ),
                observed=ci_low,
                required=f"> {policy.min_ci_lower_r:g}",
            )

    missing_stress: list[str] = []
    for fold in folds:
        for multiplier in policy.require_cost_stress:
            if not _stress_present(fold.stress, float(multiplier)):
                missing_stress.append(f"fold{fold.fold}:x{float(multiplier):g}")
    if missing_stress:
        _reject(
            rejections,
            "missing_cost_stress",
            "the bundle is missing required cost-stress rows",
            observed=",".join(missing_stress[:8]),
            required=",".join(f"x{float(value):g}" for value in policy.require_cost_stress),
        )

    if evidence.complexity > policy.max_complexity:
        _reject(
            rejections,
            "max_complexity",
            f"structural complexity {evidence.complexity} exceeds the policy cap",
            observed=evidence.complexity,
            required=policy.max_complexity,
        )

    if evidence.trial_count > policy.max_trials:
        _reject(
            rejections,
            "trial_exposure",
            f"this candidate took {evidence.trial_count} trials; the cap is a multiple-testing guard",
            observed=evidence.trial_count,
            required=policy.max_trials,
        )
    if evidence.search_budget > policy.max_trials:
        _reject(
            rejections,
            "search_exposure",
            f"the search budget of {evidence.search_budget} exceeds the cap",
            observed=evidence.search_budget,
            required=policy.max_trials,
        )

    if policy.require_locked_oos and not _locked_oos(evidence):
        _reject(
            rejections,
            "missing_locked_oos",
            "the policy requires a locked out-of-sample section and the bundle has none",
            observed="absent",
            required="locked_oos",
        )


def _check_champion(
    evidence: CandidateEvidence,
    policy: PromotionPolicy,
    champion_evidence: CandidateEvidence | None,
    champion_spec_hash: str | None,
    rejections: list[PromotionRejection],
    observed: dict[str, Any],
) -> None:
    """Compare a challenger with the champion on identical evidence geometry."""
    if champion_evidence is None and champion_spec_hash is None:
        return
    if champion_evidence is None:
        _reject(
            rejections,
            "champion_evidence_missing",
            "a champion spec was named but its evidence bundle was not supplied",
            observed="absent",
            required="CandidateEvidence",
        )
        return
    if not champion_spec_hash or not champion_spec_hash.strip():
        _reject(
            rejections,
            "champion_identity_missing",
            "champion evidence was supplied without a champion spec hash",
            observed="absent",
            required="sha256",
        )

    if evidence.aggregate is None or not evidence.folds:
        _reject(
            rejections,
            "champion_improvement",
            "the candidate bundle has no comparable aggregate",
            observed="absent",
            required="aggregate",
        )
        return

    if champion_evidence.checksum != evidence_checksum(champion_evidence):
        _reject(
            rejections,
            "champion_checksum_mismatch",
            "the champion bundle checksum does not re-derive",
            observed=champion_evidence.checksum[:12] if champion_evidence.checksum else "",
            required=evidence_checksum(champion_evidence)[:12],
        )
    if champion_spec_hash and champion_evidence.spec_hash != champion_spec_hash:
        _reject(
            rejections,
            "champion_spec_mismatch",
            "the champion bundle is not for the named champion spec",
            observed=champion_evidence.spec_hash[:12],
            required=champion_spec_hash[:12],
        )
    if champion_evidence.spec_hash == evidence.spec_hash:
        _reject(
            rejections,
            "champion_same_spec",
            "the candidate is the champion's own spec; there is nothing to improve on",
            observed=evidence.spec_hash[:12],
            required="a different spec",
        )

    if champion_evidence.dataset_id != evidence.dataset_id:
        _reject(
            rejections,
            "champion_dataset_mismatch",
            "candidate and champion were measured on different datasets",
            observed=evidence.dataset_id,
            required=champion_evidence.dataset_id,
        )
    if champion_evidence.dataset_checksum != evidence.dataset_checksum:
        _reject(
            rejections,
            "champion_dataset_checksum_mismatch",
            "candidate and champion were measured on different dataset builds",
            observed=evidence.dataset_checksum[:12],
            required=champion_evidence.dataset_checksum[:12],
        )
    if champion_evidence.code_version != evidence.code_version:
        _reject(
            rejections,
            "champion_code_version_mismatch",
            "candidate and champion were produced by different code versions",
            observed=evidence.code_version,
            required=champion_evidence.code_version,
        )
    if (
        champion_evidence.symbol != evidence.symbol
        or champion_evidence.timeframe != evidence.timeframe
    ):
        _reject(
            rejections,
            "champion_scope_mismatch",
            "candidate and champion cover different symbols or timeframes",
            observed=f"{evidence.symbol}:{evidence.timeframe}",
            required=f"{champion_evidence.symbol}:{champion_evidence.timeframe}",
        )
    if _fold_boundaries(champion_evidence) != _fold_boundaries(evidence):
        _reject(
            rejections,
            "champion_folds_mismatch",
            "candidate and champion do not share identical fold boundaries",
            observed=len(_fold_boundaries(evidence)),
            required=len(_fold_boundaries(champion_evidence)),
        )

    champion_expectancy = _finite(champion_evidence.aggregate.expectancy_r)
    candidate_expectancy = _finite(evidence.aggregate.expectancy_r)
    if champion_expectancy is None or candidate_expectancy is None:
        _reject(
            rejections,
            "champion_improvement",
            "an improvement cannot be measured without both expectations",
            observed=candidate_expectancy,
            required=champion_expectancy,
        )
    else:
        delta = candidate_expectancy - champion_expectancy
        observed["champion_delta_r"] = delta
        # R values are floats: 0.25 - 0.20 is 0.04999999999999999. A 1e-9R
        # tolerance keeps the documented "at least min_improvement_r" boundary
        # from failing on representation noise without accepting a real tie.
        if delta < policy.min_improvement_r - _COMPARISON_TOLERANCE:
            _reject(
                rejections,
                "champion_improvement",
                (
                    f"expectancy improves by {delta:g}R over the champion; "
                    f"the policy requires at least {policy.min_improvement_r:g}R"
                ),
                observed=delta,
                required=policy.min_improvement_r,
            )


# --- public API ----------------------------------------------------------------


def evaluate_promotion(
    evidence: CandidateEvidence | None,
    policy: PromotionPolicy,
    *,
    champion_evidence: CandidateEvidence | None = None,
    champion_spec_hash: str | None = None,
) -> PromotionDecision:
    """Apply ``policy`` to one evidence bundle; every doubt is a rejection.

    ``champion_evidence`` and ``champion_spec_hash`` are either both absent (no
    champion yet) or both present. A run with an accepted decision has cleared
    the policy's checks; it has not been shown to be profitable.
    """
    if not isinstance(policy, PromotionPolicy):
        raise TypeError(f"policy must be a PromotionPolicy, got {type(policy).__name__}")
    if champion_evidence is not None and not isinstance(champion_evidence, CandidateEvidence):
        raise TypeError(
            "champion_evidence must be a CandidateEvidence or None, "
            f"got {type(champion_evidence).__name__}"
        )
    if champion_spec_hash is not None and not isinstance(champion_spec_hash, str):
        raise TypeError("champion_spec_hash must be a str or None")

    rejections: list[PromotionRejection] = []
    observed: dict[str, Any] = {}

    if not isinstance(evidence, CandidateEvidence):
        code = "missing_evidence" if evidence is None else "invalid_evidence"
        _reject(
            rejections,
            code,
            "promotion needs a CandidateEvidence bundle",
            observed="absent" if evidence is None else type(evidence).__name__,
            required="CandidateEvidence",
        )
        return PromotionDecision(
            accepted=False,
            rejections=tuple(rejections),
            policy_id=policy.policy_id,
            version=policy.version,
            evidence_checksum=None,
            champion_checksum=None,
            candidate_spec_hash=None,
            champion_spec_hash=champion_spec_hash,
            dataset_id=None,
            code_version=None,
            observed=observed,
        )

    _check_bundle(evidence, policy, rejections, observed)
    _check_champion(
        evidence,
        policy,
        champion_evidence,
        champion_spec_hash,
        rejections,
        observed,
    )

    checksum = evidence_checksum(evidence)
    checksum_ok = evidence.checksum == checksum
    accepted = not rejections and checksum_ok
    return PromotionDecision(
        accepted=accepted,
        rejections=tuple(rejections),
        policy_id=policy.policy_id,
        version=policy.version,
        evidence_checksum=checksum if checksum_ok else None,
        champion_checksum=None if champion_evidence is None else champion_evidence.checksum,
        candidate_spec_hash=evidence.spec_hash or None,
        champion_spec_hash=champion_spec_hash,
        dataset_id=evidence.dataset_id or None,
        code_version=evidence.code_version or None,
        observed=observed,
    )


def manual_override(
    evidence: CandidateEvidence | None,
    policy: PromotionPolicy,
    *,
    reason: str,
    owner: str,
) -> ManualOverride:
    """Record a human override of a policy rejection; never validation evidence.

    The evidence is evaluated again so the record carries the exact policy
    rejections the owner overrode. An override of already-passing evidence is
    refused: there is nothing to override, and recording one would blur the
    distinction between a durable result and a human decision.
    """
    if not isinstance(policy, PromotionPolicy):
        raise TypeError(f"policy must be a PromotionPolicy, got {type(policy).__name__}")
    reason = _text(reason, "reason")
    owner = _text(owner, "owner")

    decision = evaluate_promotion(evidence, policy)
    if decision.accepted:
        raise ValueError("the evidence already satisfies the policy; no override is needed")

    checksum = None
    spec_hash = None
    if isinstance(evidence, CandidateEvidence):
        try:
            checksum = evidence_checksum(evidence)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            checksum = None
        spec_hash = evidence.spec_hash or None
    return ManualOverride(
        original_policy_id=policy.policy_id,
        original_policy_version=policy.version,
        reason=reason,
        owner=owner,
        rejections=decision.rejections,
        evidence_checksum=checksum,
        candidate_spec_hash=spec_hash,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def promote_with_evidence(
    archive: DiscoveryArchive,
    candidate_id: str,
    decision: PromotionDecision,
    *,
    scope: str,
    reason: str,
) -> ChampionRecord:
    """Promote only an accepted policy decision, storing its identity.

    A rejected decision, a manual override and a candidate whose id is not the
    evidence's spec hash are refused before the archive is touched. The archive
    receives the policy id/version, both checksums and the candidate spec hash,
    so the audit row states which policy allowed the change.
    """
    if not isinstance(archive, DiscoveryArchive):
        raise TypeError(f"archive must be a DiscoveryArchive, got {type(archive).__name__}")
    if not isinstance(decision, PromotionDecision):
        raise TypeError(
            "a promotion needs a PromotionDecision; a ManualOverride is not policy evidence"
        )
    candidate_id = _text(candidate_id, "candidate_id")
    scope = _text(scope, "scope")
    reason = _text(reason, "reason")
    if not decision.accepted:
        raise PromotionRefused(
            f"candidate {candidate_id!r} was rejected by policy "
            f"{decision.policy_id} v{decision.version}: " + "; ".join(decision.codes)
        )
    if decision.evidence_checksum is None:  # pragma: no cover - constructor enforces
        raise PromotionRefused("an accepted decision has no evidence checksum")
    if decision.candidate_spec_hash != candidate_id:
        raise PromotionRefused(
            f"the evidence is for spec {decision.candidate_spec_hash}, "
            f"not candidate {candidate_id}; refusing a detached promotion"
        )
    if decision.dataset_id is None:  # pragma: no cover - accepted bundle has one
        raise PromotionRefused("an accepted decision has no dataset id")

    return archive.promote_evidence(
        candidate_id,
        scope=scope,
        reason=reason,
        policy_id=decision.policy_id,
        policy_version=decision.version,
        evidence_checksum=decision.evidence_checksum,
        dataset_id=decision.dataset_id,
        candidate_spec_hash=decision.candidate_spec_hash,
        champion_checksum=decision.champion_checksum,
        detail={
            "decision_summary": decision.summary,
            "observed": dict(decision.observed),
            "rejection_codes": [],
        },
    )
