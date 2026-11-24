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
returns and not a profitability claim. The production policy requires a locked
final OOS section, so **no promotion is accepted until the held-out holdout slice
attaches one**. Every other check can pass and the decision still rejects with
``missing_locked_oos``; that is the intended fail-closed state.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from typing import Any, Final

from scalping.discovery.archive import (
    MANUAL_OVERRIDE_POLICY_ID,
    ArchiveError,
    ChampionRecord,
    DiscoveryArchive,
)
from scalping.discovery.evidence import (
    MIN_EVIDENCE_FOLDS,
    CandidateEvidence,
    EvidenceError,
    EvidencePolicy,
    SplitBoundaries,
    evidence_checksum,
    evidence_from_dict,
    verify_evidence,
)
from scalping.discovery.protocols import canonical_json

__all__ = [
    "MANUAL_OVERRIDE_POLICY_ID",
    "PROMOTION_V1",
    "EvidenceIdentity",
    "ManualOverride",
    "PromotionDecision",
    "PromotionPolicy",
    "PromotionRefused",
    "PromotionRejection",
    "evaluate_promotion",
    "manual_override",
    "promote_manual_override",
    "promote_with_evidence",
]

#: Version of a manual-override record's own shape, not of a promotion policy.
MANUAL_OVERRIDE_VERSION: Final[int] = 1

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

    ``require_locked_oos`` defaults to ``True`` and production
    :data:`PROMOTION_V1` keeps it ``True``: an evidence bundle without a
    ``locked_oos`` section is rejected, so **nothing promotes until the locked
    holdout slice attaches that section**. That is the correct fail-closed state,
    not a temporary bug. Setting it ``False`` requires an explicit, named
    non-production policy and must never be used for the paper champion.

    ``max_selection_fold_share`` caps the fraction of folds whose test block
    overlaps the dataset's validation window. Those folds were inside the slice
    the search selected on, so a bundle made mostly of them is closer to
    in-sample evidence than to walk-forward evidence. ``0`` disables the check
    explicitly; the default ``0.4`` requires a clear majority of folds to sit
    outside the selection window (one of three is fine, two of four is not).

    ``min_bootstrap_confidence`` and ``min_bootstrap_samples`` prevent a caller
    from weakening the confidence gate: a bundle with 1 sample or a 50% interval
    is not promotion evidence. ``max_stress_degradation`` bounds how much of the
    base expectancy a stressed rerun may lose: at ``2.0`` a stressed fold must
    keep at least half of the base expectancy, and every required stress must
    also stay profitable with a profit factor above 1.
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
    min_bootstrap_confidence: float = 0.95
    min_bootstrap_samples: int = 500
    max_stress_degradation: float = 2.0
    require_locked_oos: bool = True
    max_selection_fold_share: float = 0.4

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

        confidence = self.min_bootstrap_confidence
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError(
                f"min_bootstrap_confidence must be a number, got {type(confidence).__name__}"
            )
        if not math.isfinite(float(confidence)) or not 0.95 <= float(confidence) < 1.0:
            raise ValueError(
                "min_bootstrap_confidence must be in [0.95, 1); a weaker interval is not "
                f"promotion evidence, got {confidence!r}"
            )
        if isinstance(self.min_bootstrap_samples, bool) or not isinstance(
            self.min_bootstrap_samples, int
        ):
            raise TypeError("min_bootstrap_samples must be an int")
        if self.min_bootstrap_samples < 500:
            raise ValueError(
                f"min_bootstrap_samples must be >= 500, got {self.min_bootstrap_samples}"
            )

        degradation = self.max_stress_degradation
        if isinstance(degradation, bool) or not isinstance(degradation, (int, float)):
            raise TypeError(
                f"max_stress_degradation must be a number, got {type(degradation).__name__}"
            )
        if not math.isfinite(float(degradation)) or float(degradation) < 1.0:
            raise ValueError(f"max_stress_degradation must be finite and >= 1, got {degradation!r}")

        share = self.max_selection_fold_share
        if isinstance(share, bool) or not isinstance(share, (int, float)):
            raise TypeError(
                f"max_selection_fold_share must be a number, got {type(share).__name__}"
            )
        if not math.isfinite(float(share)) or not 0.0 <= float(share) <= 1.0:
            raise ValueError(
                f"max_selection_fold_share must be in [0, 1] (0 disables it), got {share!r}"
            )

    @property
    def digest(self) -> str:
        """sha256 of every policy field; the audit row names the exact policy."""
        payload = {item.name: getattr(self, item.name) for item in fields(self)}
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


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
#: * at most 40% of folds inside the validation/selection window — a majority of
#:   the evidence must sit outside the slice the search chose on;
#: * a 95% interval from at least 500 bootstrap draws — a weaker interval is not
#:   evidence;
#: * stressed folds must stay profitable, keep PF > 1 and retain at least half the
#:   base expectancy;
#: * a locked final OOS section — **nothing promotes until the holdout slice
#:   attaches it**, by design;
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
    min_bootstrap_confidence=0.95,
    min_bootstrap_samples=500,
    max_stress_degradation=2.0,
    require_locked_oos=True,
    max_selection_fold_share=0.4,
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
class EvidenceIdentity:
    """The execution universe a promotion is expected to belong to.

    The verifier compares these with the bundle's fields, so an old code version,
    dataset build or settings digest is rejected as stale instead of silently
    accepted because the bundle agrees with itself.
    """

    code_version: str
    dataset_checksum: str
    settings_digest: str

    def __post_init__(self) -> None:
        for name in ("code_version", "dataset_checksum", "settings_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


#: Module-private identity of a genuine decision. Constructing the sentinel
#: elsewhere is impossible, and the digest half makes ``dataclasses.replace`` of
#: a valid decision fail unless the payload is unchanged.
_DECISION_SEAL: Final[object] = object()


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """The policy verdict, attributable to one policy version and one bundle.

    The constructor enforces the invariant the archive relies on: an accepted
    decision cannot carry a rejection or lack an evidence checksum, and a
    decision can never claim the reserved manual-override policy id.

    ``_seal`` is not a public toggle. A decision only exists if
    :func:`evaluate_promotion` built it: a hand-constructed ``accepted=True``
    object raises, and ``dataclasses.replace`` of a real decision with a changed
    field raises because the seal's content digest no longer matches.
    """

    accepted: bool
    rejections: tuple[PromotionRejection, ...]
    policy_id: str
    version: int
    policy_digest: str
    evidence_checksum: str | None
    champion_checksum: str | None
    candidate_spec_hash: str | None
    champion_spec_hash: str | None
    dataset_id: str | None
    dataset_checksum: str | None
    settings_digest: str | None
    code_version: str | None
    observed: Mapping[str, Any] = field(default_factory=dict, compare=False)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    _seal: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is _DECISION_SEAL:
            object.__setattr__(self, "_seal", (_DECISION_SEAL, self._content_digest()))
        elif not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _DECISION_SEAL
            and self._seal[1] == self._content_digest()
        ):
            raise ValueError("a PromotionDecision must be produced by evaluate_promotion")
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
        if self.accepted and not (self.dataset_checksum and self.settings_digest):
            raise ValueError(
                "an accepted decision requires a dataset checksum and a settings digest"
            )
        if self.policy_id == MANUAL_OVERRIDE_POLICY_ID:
            raise ValueError("a promotion decision cannot claim the manual-override policy")

    def _content_digest(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

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
            "policy_digest": self.policy_digest,
            "evidence_checksum": self.evidence_checksum,
            "champion_checksum": self.champion_checksum,
            "candidate_spec_hash": self.candidate_spec_hash,
            "champion_spec_hash": self.champion_spec_hash,
            "dataset_id": self.dataset_id,
            "dataset_checksum": self.dataset_checksum,
            "settings_digest": self.settings_digest,
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
    dataset_id: str | None = None
    policy_digest: str | None = None
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
            "dataset_id": self.dataset_id,
            "policy_digest": self.policy_digest,
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


def _stress_row(fold_stress: Sequence[Any], multiplier: float) -> Any | None:
    for row in fold_stress:
        if math.isclose(
            float(row.multiplier), multiplier, rel_tol=0.0, abs_tol=_MULTIPLIER_TOLERANCE
        ):
            return row
    return None


def _locked_oos(evidence: CandidateEvidence, policy: PromotionPolicy) -> bool:
    """True when a valid locked-holdout section is attached to this bundle.

    The current evidence builder does not add the section; the locked-holdout
    slice attaches it after one authorised evaluation. Presence alone is not
    enough: the section must name a digest, a positive trade count and the same
    dataset, spec, code version, settings and policy as the promotion. That
    stops an unrelated or hand-written section from unlocking promotion.
    """
    payload = evidence.to_dict().get("locked_oos")
    if not isinstance(payload, Mapping):
        return False
    checksum = payload.get("checksum")
    trades = payload.get("trades")
    if not (
        isinstance(checksum, str)
        and len(checksum) == 64
        and all(character in "0123456789abcdef" for character in checksum)
        and isinstance(trades, int)
        and not isinstance(trades, bool)
        and trades > 0
    ):
        return False
    for name, expected in (
        ("dataset_id", evidence.dataset_id),
        ("dataset_checksum", evidence.dataset_checksum),
        ("spec_hash", evidence.spec_hash),
        ("code_version", evidence.code_version),
        ("settings_digest", evidence.settings_digest),
        ("policy_id", policy.policy_id),
        ("policy_version", policy.version),
    ):
        if payload.get(name) != expected:
            return False
    return True


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


def _selection_folds(evidence: CandidateEvidence) -> tuple[int, ...]:
    """Folds whose test block intersects the dataset's validation window.

    The search selects on the validation split, so a fold that overlaps it is
    selection evidence: useful, but not independent walk-forward evidence.
    Half-open intersection, so a fold that ends exactly where validation starts
    is not counted.
    """
    splits = evidence.splits
    return tuple(
        fold.fold
        for fold in evidence.folds
        if fold.test_start_row < splits.validation_end_row
        and fold.test_end_row > splits.validation_start_row
    )


def _verify(
    evidence: CandidateEvidence,
    rejections: list[PromotionRejection],
    observed: dict[str, Any],
    expected_identity: EvidenceIdentity | None,
) -> bool:
    """Run the evidence module's verifier against an expected identity.

    ``verify_evidence`` is the single owner of checksum, schema, identity and
    required-section rules. With ``expected_identity`` the comparison is against
    the *current* code version, dataset build and settings digest, so a stale
    bundle is rejected as ``stale_*`` instead of passing because it agrees with
    itself. Without one, the bundle is checked for internal self-consistency
    only; the production entry point always supplies it.
    """
    if expected_identity is None:
        dataset_checksum = evidence.dataset_checksum
        code_version = evidence.code_version
        settings_digest = evidence.settings_digest
    else:
        dataset_checksum = expected_identity.dataset_checksum
        code_version = expected_identity.code_version
        settings_digest = expected_identity.settings_digest
    try:
        verify_evidence(
            evidence,
            dataset_id=evidence.dataset_id,
            dataset_checksum=dataset_checksum,
            spec_hash=evidence.spec_hash,
            code_version=code_version,
            settings_digest=settings_digest,
        )
    except EvidenceError as exc:
        _reject(rejections, exc.code, str(exc).removeprefix(f"{exc.code}: "))
        return False
    except (TypeError, ValueError) as exc:
        # canonical_json refuses NaN/inf, so an unverifiable payload lands here.
        _reject(rejections, "checksum_mismatch", f"the bundle payload is not canonical: {exc}")
        return False
    observed["evidence_checksum"] = evidence.checksum
    return True


def _check_bundle(
    evidence: CandidateEvidence,
    policy: PromotionPolicy,
    rejections: list[PromotionRejection],
    observed: dict[str, Any],
    expected_identity: EvidenceIdentity | None,
) -> None:
    """Identity, structure, sample and threshold checks on the candidate alone."""
    _verify(evidence, rejections, observed, expected_identity)

    if not isinstance(evidence.splits, SplitBoundaries):
        _reject(
            rejections,
            "missing_section",
            "the bundle has no dataset split boundaries",
            observed=type(evidence.splits).__name__,
            required="SplitBoundaries",
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

    if isinstance(evidence.policy, EvidencePolicy) and (
        not evidence.policy.cost_stress
        or not any(float(value) > 1.0 for value in evidence.policy.cost_stress)
    ):
        _reject(
            rejections,
            "missing_cost_stress",
            "the bundle's evidence policy has no genuine cost-stress multiplier above 1.0",
            observed="none > 1.0",
            required="> 1.0",
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

    if isinstance(evidence.splits, SplitBoundaries):
        splits = evidence.splits
        if (
            splits.train_start_row < 0
            or splits.train_end_row < splits.train_start_row
            or splits.validation_start_row < splits.train_end_row
            or splits.validation_end_row <= splits.validation_start_row
            or splits.test_start_row < splits.validation_end_row
            or splits.test_end_row < splits.test_start_row
        ):
            _reject(
                rejections,
                "fold_geometry",
                "dataset split boundaries are not ordered train <= validation < test",
                observed=", ".join(
                    str(value)
                    for value in (
                        splits.train_start_row,
                        splits.train_end_row,
                        splits.validation_start_row,
                        splits.validation_end_row,
                        splits.test_start_row,
                        splits.test_end_row,
                    )
                ),
                required="train_start <= train_end <= validation_start < validation_end <= test_start <= test_end",
            )
        if folds[0].test_start_row < splits.train_start_row:
            _reject(
                rejections,
                "fold_geometry",
                "the first fold starts before the dataset train range",
                observed=folds[0].test_start_row,
                required=splits.train_start_row,
            )
        if folds[-1].test_end_row != splits.validation_end_row:
            _reject(
                rejections,
                "fold_geometry",
                "the folds do not end at the validation boundary; a gap or a test-split fold is hidden",
                observed=folds[-1].test_end_row,
                required=splits.validation_end_row,
            )

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
        if isinstance(evidence.splits, SplitBoundaries):
            if fold.train_start_row != evidence.splits.train_start_row:
                _reject(
                    rejections,
                    "fold_geometry",
                    f"fold {fold.fold} does not start its training window at the dataset boundary",
                    observed=fold.train_start_row,
                    required=evidence.splits.train_start_row,
                )
            if fold.test_start_row < fold.train_start_row:
                _reject(
                    rejections,
                    "fold_geometry",
                    f"fold {fold.fold} tests before its own training start",
                    observed=fold.test_start_row,
                    required=f">= {fold.train_start_row}",
                )
            if fold.test_end_row > evidence.splits.validation_end_row:
                _reject(
                    rejections,
                    "fold_geometry",
                    f"fold {fold.fold} reaches past the validation boundary into the test split",
                    observed=fold.test_end_row,
                    required=f"<= {evidence.splits.validation_end_row}",
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

    if isinstance(evidence.splits, SplitBoundaries) and policy.max_selection_fold_share > 0.0:
        selection = _selection_folds(evidence)
        share = len(selection) / len(folds)
        observed["selection_fold_share"] = share
        observed["selection_folds"] = selection
        if share > policy.max_selection_fold_share + _COMPARISON_TOLERANCE:
            _reject(
                rejections,
                "selection_fold_share",
                (
                    f"{len(selection)} of {len(folds)} folds overlap the validation/selection "
                    "window; the policy requires more independent folds"
                ),
                observed=share,
                required=policy.max_selection_fold_share,
            )

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

    samples = aggregate.bootstrap_samples
    if isinstance(samples, bool) or not isinstance(samples, int):
        _reject(
            rejections,
            "missing_section",
            "aggregate bootstrap_samples is missing or not an int",
            observed=repr(samples),
            required=f">= {policy.min_bootstrap_samples}",
        )
    else:
        observed["bootstrap_samples"] = samples
        if samples < policy.min_bootstrap_samples:
            _reject(
                rejections,
                "min_bootstrap_samples",
                f"{samples} bootstrap samples is below the policy minimum",
                observed=samples,
                required=policy.min_bootstrap_samples,
            )
    confidence = _finite(aggregate.bootstrap_confidence)
    if confidence is None:
        _reject(
            rejections,
            "missing_section",
            "aggregate bootstrap_confidence is missing or not finite",
            observed="absent",
            required=f">= {policy.min_bootstrap_confidence:g}",
        )
    else:
        observed["bootstrap_confidence"] = confidence
        if confidence < policy.min_bootstrap_confidence or confidence >= 1.0:
            _reject(
                rejections,
                "min_bootstrap_confidence",
                f"bootstrap confidence {confidence:g} is below the policy minimum or degenerate",
                observed=confidence,
                required=f">= {policy.min_bootstrap_confidence:g} and < 1",
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
    weak_stress: list[str] = []
    degraded_stress: list[str] = []
    for fold in folds:
        for multiplier in policy.require_cost_stress:
            row = _stress_row(fold.stress, float(multiplier))
            if row is None:
                missing_stress.append(f"fold{fold.fold}:x{float(multiplier):g}")
                continue
            expectancy = _finite(row.expectancy_r)
            profit_factor = _finite(row.profit_factor)
            if expectancy is None or profit_factor is None:
                weak_stress.append(f"fold{fold.fold}:x{float(multiplier):g}:missing")
                continue
            if expectancy <= 0.0 or profit_factor <= 1.0:
                weak_stress.append(
                    f"fold{fold.fold}:x{float(multiplier):g}:{expectancy:g}R/PF{profit_factor:g}"
                )
            base = _finite(fold.expectancy_r)
            floor = 0.0 if base is None or base <= 0.0 else base / policy.max_stress_degradation
            if expectancy < floor:
                degraded_stress.append(
                    f"fold{fold.fold}:x{float(multiplier):g}:{expectancy:g}R<{floor:g}R"
                )
    if missing_stress:
        _reject(
            rejections,
            "missing_cost_stress",
            "the bundle is missing required cost-stress rows",
            observed=",".join(missing_stress[:8]),
            required=",".join(f"x{float(value):g}" for value in policy.require_cost_stress),
        )
    if weak_stress:
        _reject(
            rejections,
            "cost_stress",
            "a required stress rerun is not profitable or its profit factor is not above 1",
            observed=",".join(weak_stress[:8]),
            required="expectancy_r > 0 and profit_factor > 1",
        )
    if degraded_stress:
        _reject(
            rejections,
            "stress_degradation",
            "a required stress rerun loses more expectancy than the policy allows",
            observed=",".join(degraded_stress[:8]),
            required=f"at least base/{policy.max_stress_degradation:g}",
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

    if policy.require_locked_oos and not _locked_oos(evidence, policy):
        _reject(
            rejections,
            "missing_locked_oos",
            "the policy requires a locked out-of-sample section and the bundle has none",
            observed="absent",
            required="locked_oos",
        )


def _fold_identity(evidence: CandidateEvidence) -> tuple[Any, ...] | None:
    """The complete geometry a same-window comparison needs, or ``None``."""
    if not evidence.folds or not isinstance(evidence.splits, SplitBoundaries):
        return None
    identity: list[Any] = []
    for fold in evidence.folds:
        values = (
            fold.fold,
            fold.train_start_row,
            fold.train_end_row,
            fold.test_start_row,
            fold.test_end_row,
            fold.train_start_ts,
            fold.train_end_ts,
            fold.test_start_ts,
            fold.test_end_ts,
            fold.purge_bars,
            fold.embargo_bars,
        )
        if any(isinstance(value, str) and not value.strip() for value in values):
            return None
        identity.append(values)
    return (
        evidence.splits.train_start_row,
        evidence.splits.train_end_row,
        evidence.splits.validation_start_row,
        evidence.splits.validation_end_row,
        evidence.splits.test_start_row,
        evidence.splits.test_end_row,
        tuple(identity),
        len(evidence.folds),
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
            "a champion exists but its evidence bundle was not supplied",
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

    missing: list[str] = []
    for side, bundle in (("candidate", evidence), ("champion", champion_evidence)):
        for name in ("dataset_checksum", "settings_digest", "code_version"):
            value = getattr(bundle, name)
            if not isinstance(value, str) or not value.strip():
                missing.append(f"{side}.{name}")
        if not bundle.folds:
            missing.append(f"{side}.folds")
        if not isinstance(bundle.splits, SplitBoundaries):
            missing.append(f"{side}.splits")
        if not isinstance(bundle.policy, EvidencePolicy):
            missing.append(f"{side}.policy")
    candidate_identity = _fold_identity(evidence)
    champion_identity = _fold_identity(champion_evidence)
    if candidate_identity is None:
        missing.append("candidate.fold_identity")
    if champion_identity is None:
        missing.append("champion.fold_identity")
    if missing:
        _reject(
            rejections,
            "incomparable_evidence",
            "candidate and champion do not carry the identity a comparison needs",
            observed=",".join(missing[:8]),
            required="dataset_checksum, settings_digest, code_version, splits, folds, fold identity",
        )
        return

    try:
        champion_computed = evidence_checksum(champion_evidence)
    except (TypeError, ValueError) as exc:
        _reject(
            rejections,
            "champion_invalid_evidence",
            f"the stored champion bundle is not canonical: {exc}",
        )
        return
    if champion_evidence.checksum != champion_computed:
        _reject(
            rejections,
            "champion_checksum_mismatch",
            "the champion bundle checksum does not re-derive",
            observed=champion_evidence.checksum[:12] if champion_evidence.checksum else "",
            required=champion_computed[:12],
        )
        return

    try:
        verify_evidence(
            champion_evidence,
            dataset_id=champion_evidence.dataset_id,
            dataset_checksum=champion_evidence.dataset_checksum,
            spec_hash=champion_evidence.spec_hash,
            code_version=champion_evidence.code_version,
            settings_digest=champion_evidence.settings_digest,
        )
    except EvidenceError as exc:
        _reject(
            rejections,
            "champion_invalid_evidence",
            f"the stored champion bundle failed verification: {exc.code}",
            observed=exc.code,
            required="a verified CandidateEvidence",
        )
        return
    except (TypeError, ValueError) as exc:
        _reject(
            rejections,
            "champion_invalid_evidence",
            f"the stored champion bundle is not canonical: {exc}",
        )
        return

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
    if champion_evidence.settings_digest != evidence.settings_digest:
        _reject(
            rejections,
            "champion_settings_mismatch",
            "candidate and champion were priced under different execution settings",
            observed=evidence.settings_digest[:12],
            required=champion_evidence.settings_digest[:12],
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

    if champion_evidence.splits != evidence.splits:
        _reject(
            rejections,
            "champion_splits_mismatch",
            "candidate and champion were cut from different dataset split boundaries",
            observed=repr(evidence.splits),
            required=repr(champion_evidence.splits),
        )
    if champion_identity != candidate_identity:
        _reject(
            rejections,
            "champion_folds_mismatch",
            "candidate and champion do not share identical fold geometry (rows, timestamps, purge/embargo)",
            observed=len(evidence.folds),
            required=len(champion_evidence.folds),
        )
    if isinstance(evidence.policy, EvidencePolicy) and isinstance(
        champion_evidence.policy, EvidencePolicy
    ):
        if set(evidence.policy.cost_stress) != set(champion_evidence.policy.cost_stress):
            _reject(
                rejections,
                "champion_cost_stress_mismatch",
                "candidate and champion were stressed with different cost multipliers",
                observed=",".join(f"x{value:g}" for value in evidence.policy.cost_stress),
                required=",".join(f"x{value:g}" for value in champion_evidence.policy.cost_stress),
            )
        if (
            evidence.aggregate.bootstrap_samples != champion_evidence.aggregate.bootstrap_samples
            or evidence.aggregate.bootstrap_confidence
            != champion_evidence.aggregate.bootstrap_confidence
            or evidence.policy.bootstrap_samples != champion_evidence.policy.bootstrap_samples
            or evidence.policy.bootstrap_confidence != champion_evidence.policy.bootstrap_confidence
        ):
            _reject(
                rejections,
                "champion_bootstrap_mismatch",
                "candidate and champion used different bootstrap sample counts or confidence",
                observed=(
                    f"{evidence.aggregate.bootstrap_samples}/"
                    f"{evidence.aggregate.bootstrap_confidence:g}"
                ),
                required=(
                    f"{champion_evidence.aggregate.bootstrap_samples}/"
                    f"{champion_evidence.aggregate.bootstrap_confidence:g}"
                ),
            )
        if evidence.policy.window != champion_evidence.policy.window:
            _reject(
                rejections,
                "champion_window_mismatch",
                "candidate and champion used different observation windows",
                observed=evidence.policy.window,
                required=champion_evidence.policy.window,
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
    expected_identity: EvidenceIdentity | None = None,
) -> PromotionDecision:
    """Apply ``policy`` to one evidence bundle; every doubt is a rejection.

    ``champion_evidence`` and ``champion_spec_hash`` are either both absent (no
    champion yet) or both present. ``expected_identity`` compares the bundle with
    the *current* code version, dataset checksum and settings digest; production
    callers always pass it, and omitting it only checks internal consistency.

    A run with an accepted decision has cleared the policy's checks; it has not
    been shown to be profitable.
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
    if expected_identity is not None and not isinstance(expected_identity, EvidenceIdentity):
        raise TypeError(
            "expected_identity must be an EvidenceIdentity or None, "
            f"got {type(expected_identity).__name__}"
        )

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
            policy_digest=policy.digest,
            evidence_checksum=None,
            champion_checksum=None,
            candidate_spec_hash=None,
            champion_spec_hash=champion_spec_hash,
            dataset_id=None,
            dataset_checksum=None,
            settings_digest=None,
            code_version=None,
            observed=observed,
            _seal=_DECISION_SEAL,
        )

    _check_bundle(evidence, policy, rejections, observed, expected_identity)
    _check_champion(
        evidence,
        policy,
        champion_evidence,
        champion_spec_hash,
        rejections,
        observed,
    )

    accepted = not rejections
    return PromotionDecision(
        accepted=accepted,
        rejections=tuple(rejections),
        policy_id=policy.policy_id,
        version=policy.version,
        policy_digest=policy.digest,
        evidence_checksum=observed.get("evidence_checksum"),
        champion_checksum=None if champion_evidence is None else champion_evidence.checksum,
        candidate_spec_hash=evidence.spec_hash or None,
        champion_spec_hash=champion_spec_hash,
        dataset_id=evidence.dataset_id or None,
        dataset_checksum=evidence.dataset_checksum or None,
        settings_digest=evidence.settings_digest or None,
        code_version=evidence.code_version or None,
        observed=observed,
        _seal=_DECISION_SEAL,
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
    dataset_id = None
    if isinstance(evidence, CandidateEvidence):
        try:
            checksum = evidence_checksum(evidence)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            checksum = None
        spec_hash = evidence.spec_hash or None
        dataset_id = evidence.dataset_id or None
    return ManualOverride(
        original_policy_id=policy.policy_id,
        original_policy_version=policy.version,
        reason=reason,
        owner=owner,
        rejections=decision.rejections,
        evidence_checksum=checksum,
        candidate_spec_hash=spec_hash,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        dataset_id=dataset_id,
        policy_digest=policy.digest,
    )


def promote_with_evidence(
    archive: DiscoveryArchive,
    candidate_id: str,
    evidence: CandidateEvidence,
    policy: PromotionPolicy,
    *,
    scope: str,
    reason: str,
    expected_identity: EvidenceIdentity,
) -> ChampionRecord:
    """Evaluate ``policy`` internally and promote only if it accepts.

    There is no accepted-decision argument to pass and no public archive method
    to bypass. The champion is read from the archive; when one exists its stored
    evidence is the comparison baseline, and a champion without stored evidence
    is a rejection (``champion_evidence_missing``). A first promotion records
    explicitly that there was no baseline.
    """
    if not isinstance(archive, DiscoveryArchive):
        raise TypeError(f"archive must be a DiscoveryArchive, got {type(archive).__name__}")
    if not isinstance(evidence, CandidateEvidence):
        raise TypeError(f"evidence must be a CandidateEvidence, got {type(evidence).__name__}")
    if not isinstance(policy, PromotionPolicy):
        raise TypeError(f"policy must be a PromotionPolicy, got {type(policy).__name__}")
    if not isinstance(expected_identity, EvidenceIdentity):
        raise TypeError(
            f"expected_identity must be an EvidenceIdentity, got {type(expected_identity).__name__}"
        )
    candidate_id = _text(candidate_id, "candidate_id")
    scope = _text(scope, "scope")
    reason = _text(reason, "reason")

    champion = archive.champion(scope)
    champion_id = None if champion is None else champion.candidate.candidate_id
    champion_evidence: CandidateEvidence | None = None
    if champion is not None:
        record = archive.champion_evidence(scope)
        if record is not None:
            try:
                loaded = evidence_from_dict(record.payload)
            except EvidenceError as exc:
                raise ArchiveError(
                    f"stored champion evidence {record.checksum[:12]} cannot be rebuilt: {exc}"
                ) from exc
            if evidence_checksum(loaded) != record.checksum:
                raise ArchiveError(
                    f"stored champion evidence {record.checksum[:12]} does not re-derive"
                )
            if loaded.spec_hash != champion_id:
                raise ArchiveError(
                    f"stored champion evidence is for spec {loaded.spec_hash[:12]}, "
                    f"not champion {champion_id}"
                )
            champion_evidence = loaded

    decision = evaluate_promotion(
        evidence,
        policy,
        champion_evidence=champion_evidence,
        champion_spec_hash=champion_id,
        expected_identity=expected_identity,
    )
    if not decision.accepted:
        raise PromotionRefused(
            f"candidate {candidate_id!r} was rejected by policy "
            f"{decision.policy_id} v{decision.version}: " + "; ".join(decision.codes)
        )
    if decision.candidate_spec_hash != candidate_id:
        raise PromotionRefused(
            f"the evidence is for spec {decision.candidate_spec_hash}, "
            f"not candidate {candidate_id}; refusing a detached promotion"
        )
    if decision.evidence_checksum is None:  # pragma: no cover - constructor enforces
        raise PromotionRefused("an accepted decision has no evidence checksum")

    return archive._promote_evidence(
        candidate_id,
        scope=scope,
        reason=reason,
        policy_id=decision.policy_id,
        policy_version=decision.version,
        policy_digest=decision.policy_digest,
        evidence_checksum=decision.evidence_checksum,
        dataset_id=evidence.dataset_id,
        dataset_checksum=evidence.dataset_checksum,
        settings_digest=evidence.settings_digest,
        code_version=evidence.code_version,
        candidate_spec_hash=evidence.spec_hash,
        evidence_payload=evidence.to_dict(),
        champion_checksum=decision.champion_checksum,
        champion_candidate_id=champion_id,
        detail={
            "decision_summary": decision.summary,
            "observed": dict(decision.observed),
            "rejection_codes": [],
        },
    )


def promote_manual_override(
    archive: DiscoveryArchive,
    candidate_id: str,
    override: ManualOverride,
    *,
    scope: str,
    reason: str,
) -> ChampionRecord:
    """Store a human override as a ``mode = 'manual'`` champion.

    It is deliberately not an evidence promotion: the policy did not accept the
    bundle, the record keeps the original rejections, and
    :class:`~scalping.discovery.champion.ChampionProvider` refuses it for paper.
    """
    if not isinstance(archive, DiscoveryArchive):
        raise TypeError(f"archive must be a DiscoveryArchive, got {type(archive).__name__}")
    if not isinstance(override, ManualOverride):
        raise TypeError(f"override must be a ManualOverride, got {type(override).__name__}")
    candidate_id = _text(candidate_id, "candidate_id")
    scope = _text(scope, "scope")
    reason = _text(reason, "reason")
    if override.candidate_spec_hash != candidate_id:
        raise PromotionRefused(
            f"the override is for spec {override.candidate_spec_hash}, "
            f"not candidate {candidate_id}; refusing a detached override"
        )
    if override.dataset_id is None:
        raise PromotionRefused("the override has no dataset id")
    champion = archive.champion(scope)
    champion_id = None if champion is None else champion.candidate.candidate_id
    return archive._promote_manual(
        candidate_id,
        scope=scope,
        reason=reason,
        candidate_spec_hash=override.candidate_spec_hash or candidate_id,
        dataset_id=override.dataset_id,
        evidence_checksum=override.evidence_checksum,
        champion_candidate_id=champion_id,
        detail={
            "override": override.to_dict(),
            "decision_summary": "manual override; not policy evidence",
        },
    )
