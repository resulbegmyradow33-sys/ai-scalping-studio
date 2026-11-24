"""Promotion policy: fail-closed checks, champion comparison and archive audit.

The evidence bundles here are built by hand, not simulated. These tests exercise
the *policy* arithmetic and identity logic; the fold geometry and checksum of a
real bundle are covered by ``test_discovery_evidence.py``. A synthetic bundle is
enough to prove that every threshold, every stale identity and every missing
section is a rejection and that no path accepts a decision without a checksum.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from discovery_fakes import ToyCandidate
from scalping.discovery.archive import (
    MANUAL_OVERRIDE_POLICY_ID,
    ArchiveError,
    DiscoveryArchive,
)
from scalping.discovery.evidence import (
    EVIDENCE_SCHEMA_VERSION,
    AggregateEvidence,
    CandidateEvidence,
    EvidencePolicy,
    FoldEvidence,
    RegimeEvidence,
    SplitBoundaries,
    StressEvidence,
    evidence_checksum,
)
from scalping.discovery.promotion import (
    PROMOTION_V1,
    PromotionDecision,
    PromotionRefused,
    evaluate_promotion,
    manual_override,
    promote_with_evidence,
)
from scalping.discovery.protocols import TrialResult

SPEC_HASH = "a" * 64
CHAMPION_HASH = "b" * 64
DATASET_ID = "validation:XAUUSD:M1:abcd1234"
DATASET_CHECKSUM = "c" * 64
SETTINGS_DIGEST = "d" * 64
CODE_VERSION = "test-promotion-2026-11-24"
SCOPE = "XAUUSD:M1"
START = datetime(2026, 1, 5, tzinfo=UTC)

_UNSET = object()


def _ts(minutes: int) -> str:
    return (START + timedelta(minutes=minutes)).isoformat()


def fold(
    index: int,
    *,
    trades: int = 25,
    net_profit: float = 1.0,
    expectancy_r: float = 0.25,
    profit_factor: float = 1.5,
    max_drawdown: float = 0.05,
    stress: tuple[float, ...] = (1.5, 2.0),
    shift: int = 0,
) -> FoldEvidence:
    test_start = 100 + shift + (index - 1) * 50
    test_end = test_start + 50
    train_end = test_start - 5
    return FoldEvidence(
        fold=index,
        train_start_row=0,
        train_end_row=train_end,
        test_start_row=test_start,
        test_end_row=test_end,
        train_rows=train_end,
        test_rows=50,
        train_start_ts=_ts(0),
        train_end_ts=_ts(test_start - 1),
        test_start_ts=_ts(test_start),
        test_end_ts=_ts(test_end - 1),
        purge_bars=3,
        embargo_bars=2,
        trades=trades,
        net_profit=net_profit,
        profit_factor=profit_factor,
        win_rate=0.5,
        expectancy_r=expectancy_r,
        max_drawdown=max_drawdown,
        regimes=(
            RegimeEvidence(
                regime="trend_up",
                trades=trades,
                net_profit=net_profit,
                win_rate=0.5,
                expectancy_r=expectancy_r,
                total_r=expectancy_r * trades,
            ),
        ),
        stress=tuple(
            StressEvidence(
                multiplier=multiplier,
                trades=trades,
                net_profit=net_profit,
                profit_factor=profit_factor,
                max_drawdown=max_drawdown,
                expectancy_r=expectancy_r,
            )
            for multiplier in stress
        ),
    )


def default_folds(*, shift: int = 0) -> tuple[FoldEvidence, ...]:
    return tuple(fold(index, shift=shift) for index in (1, 2, 3))


def default_splits(*, validation_start: int = 200) -> SplitBoundaries:
    """Folds tile rows 100-250; validation starts at 200 so only fold 3 overlaps."""
    return SplitBoundaries(
        train_start_row=0,
        train_end_row=validation_start,
        validation_start_row=validation_start,
        validation_end_row=260,
        test_start_row=260,
        test_end_row=300,
    )


def aggregate_for(
    folds: tuple[FoldEvidence, ...],
    **overrides: Any,
) -> AggregateEvidence:
    positive = sum(1 for item in folds if item.net_profit > 0.0)
    total_trades = sum(item.trades for item in folds)
    values: dict[str, Any] = {
        "total_trades": total_trades,
        "total_r": 0.25 * total_trades,
        "expectancy_r": 0.25,
        "profit_factor": 1.5,
        "max_drawdown": 0.10,
        "positive_folds": positive,
        "positive_fold_fraction": positive / len(folds) if folds else 0.0,
        "expectancy_r_ci95_low": 0.10,
        "expectancy_r_ci95_high": 0.40,
        "total_r_ci95_low": 1.0,
        "total_r_ci95_high": 4.0,
        "bootstrap_samples": 200,
        "bootstrap_confidence": 0.95,
        "bootstrap_method": "iid-bootstrap",
        "regime_coverage": 0.95,
        "regime_trades": (),
    }
    values.update(overrides)
    return AggregateEvidence(**values)


def evidence_policy() -> EvidencePolicy:
    return EvidencePolicy(
        folds=3,
        purge_bars=3,
        embargo_bars=2,
        min_trades_per_fold=2,
        cost_stress=(1.5, 2.0),
        regime_coverage_min=0.85,
        bootstrap_samples=200,
        bootstrap_confidence=0.95,
        window=60,
    )


def reseal(evidence: CandidateEvidence, **changes: Any) -> CandidateEvidence:
    """Apply changes and recompute the checksum, as a legitimate builder would."""
    draft = replace(evidence, checksum="", **changes)
    return replace(draft, checksum=evidence_checksum(draft))


def make_evidence(
    *,
    folds: tuple[FoldEvidence, ...] | None = None,
    aggregate: AggregateEvidence | None | object = _UNSET,
    splits: SplitBoundaries | None = None,
    **overrides: Any,
) -> CandidateEvidence:
    items = default_folds() if folds is None else tuple(folds)
    summary = aggregate_for(items) if aggregate is _UNSET else aggregate
    values: dict[str, Any] = {
        "spec_hash": SPEC_HASH,
        "spec_version": 1,
        "complexity": 10,
        "dataset_id": DATASET_ID,
        "dataset_checksum": DATASET_CHECKSUM,
        "symbol": "XAUUSD",
        "timeframe": "M1",
        "code_version": CODE_VERSION,
        "settings_digest": SETTINGS_DIGEST,
        "seed": 0,
        "trial_count": 10,
        "search_budget": 100,
        "splits": default_splits() if splits is None else splits,
        "policy": evidence_policy(),
        "folds": items,
        "aggregate": summary,
        "checksum": "",
        "created_at": "2026-11-24T00:00:00+00:00",
    }
    values.update(overrides)
    bundle = CandidateEvidence(**values)
    return bundle if values["checksum"] else reseal(bundle)


def codes(decision: PromotionDecision) -> tuple[str, ...]:
    return decision.codes


# --- policy construction ------------------------------------------------------


def test_promotion_v1_is_explicit_and_versioned() -> None:
    policy = PROMOTION_V1
    assert policy.policy_id == "promotion-v1"
    assert policy.version == 1
    assert policy.min_folds >= 3
    assert policy.min_trades > 0
    assert policy.min_profit_factor > 1.0
    assert 0.0 < policy.max_drawdown <= 1.0
    assert 0.0 < policy.min_positive_fold_fraction <= 1.0
    assert 0.0 < policy.min_regime_coverage <= 1.0
    assert policy.require_cost_stress
    assert all(multiplier > 1.0 for multiplier in policy.require_cost_stress)
    assert policy.max_complexity > 0
    assert policy.max_trials > 0
    assert policy.min_improvement_r > 0.0
    assert policy.require_locked_oos is False
    assert 0.0 < policy.max_selection_fold_share <= 1.0


@pytest.mark.parametrize(
    "changes",
    [
        {"policy_id": ""},
        {"policy_id": "   "},
        {"policy_id": MANUAL_OVERRIDE_POLICY_ID},
        {"version": 0},
        {"min_folds": 2},
        {"min_trades": 0},
        {"min_profit_factor": 1.0},
        {"max_drawdown": 0.0},
        {"min_positive_fold_fraction": 0.0},
        {"min_regime_coverage": 0.0},
        {"require_cost_stress": ()},
        {"require_cost_stress": (2.0, 1.0)},
        {"require_cost_stress": (2.0, 2.0)},
        {"max_complexity": 0},
        {"max_trials": 0},
        {"min_improvement_r": 0.0},
        {"require_cost_stress": (1.0,)},
        {"max_selection_fold_share": -0.1},
        {"max_selection_fold_share": 1.1},
    ],
)
def test_degenerate_policies_are_refused(changes: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(PROMOTION_V1, **changes)


def test_there_is_no_force_parameter() -> None:
    for function in (evaluate_promotion, manual_override, promote_with_evidence):
        assert "force" not in inspect.signature(function).parameters


# --- accepted path ------------------------------------------------------------


def test_a_complete_bundle_is_accepted_and_attributed() -> None:
    bundle = make_evidence()
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert decision.accepted is True
    assert decision.rejections == ()
    assert decision.policy_id == "promotion-v1"
    assert decision.version == 1
    assert decision.evidence_checksum == bundle.checksum
    assert decision.dataset_checksum == DATASET_CHECKSUM
    assert decision.settings_digest == SETTINGS_DIGEST
    assert decision.champion_checksum is None
    assert decision.candidate_spec_hash == SPEC_HASH
    assert decision.dataset_id == DATASET_ID
    assert decision.summary.startswith("policy promotion-v1 v1 accepted")


# --- candidate rejections -----------------------------------------------------


def test_insufficient_folds_are_rejected() -> None:
    bundle = make_evidence(folds=(fold(1, trades=30), fold(2, trades=30)))
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert decision.accepted is False
    assert "insufficient_folds" in codes(decision)


def test_insufficient_trades_are_rejected() -> None:
    bundle = make_evidence(folds=tuple(fold(index, trades=5) for index in (1, 2, 3)))
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert "insufficient_trades" in codes(decision)
    assert decision.evidence_checksum == bundle.checksum


def test_low_expectancy_is_rejected() -> None:
    folds = default_folds()
    bundle = make_evidence(folds=folds, aggregate=aggregate_for(folds, expectancy_r=0.01))
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert "min_expectancy_r" in codes(decision)
    assert decision.rejections[0].observed == pytest.approx(0.01)


def test_low_profit_factor_is_rejected() -> None:
    folds = default_folds()
    bundle = make_evidence(folds=folds, aggregate=aggregate_for(folds, profit_factor=1.1))
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert "min_profit_factor" in codes(decision)


def test_deep_drawdown_is_rejected() -> None:
    folds = default_folds()
    bundle = make_evidence(folds=folds, aggregate=aggregate_for(folds, max_drawdown=0.30))
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert "max_drawdown" in codes(decision)


def test_too_few_positive_folds_are_rejected() -> None:
    folds = (
        fold(1),
        fold(2, net_profit=-1.0),
        fold(3, net_profit=-1.0),
    )
    bundle = make_evidence(folds=folds, aggregate=aggregate_for(folds))
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert "min_positive_fold_fraction" in codes(decision)


def test_low_regime_coverage_is_rejected() -> None:
    folds = default_folds()
    bundle = make_evidence(folds=folds, aggregate=aggregate_for(folds, regime_coverage=0.5))
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert "min_regime_coverage" in codes(decision)


def test_a_confidence_interval_that_touches_zero_is_rejected() -> None:
    folds = default_folds()
    bundle = make_evidence(folds=folds, aggregate=aggregate_for(folds, expectancy_r_ci95_low=0.0))
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert "min_ci_lower_r" in codes(decision)
    assert decision.rejections[0].required == "> 0"


def test_a_missing_stress_multiplier_is_rejected() -> None:
    folds = tuple(fold(index, stress=(1.5,)) for index in (1, 2, 3))
    bundle = make_evidence(folds=folds)
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert "missing_cost_stress" in codes(decision)
    assert "x2" in str(decision.rejections[0].required)


def test_too_complex_a_spec_is_rejected() -> None:
    bundle = make_evidence(complexity=PROMOTION_V1.max_complexity + 1)
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert "max_complexity" in codes(decision)
    assert decision.rejections[0].observed == PROMOTION_V1.max_complexity + 1


def test_trial_and_search_exposure_are_capped() -> None:
    trials = make_evidence(
        trial_count=PROMOTION_V1.max_trials + 1,
        search_budget=PROMOTION_V1.max_trials + 10,
    )
    assert "trial_exposure" in codes(evaluate_promotion(trials, PROMOTION_V1))

    search = make_evidence(search_budget=PROMOTION_V1.max_trials + 1)
    decision = evaluate_promotion(search, PROMOTION_V1)
    assert "search_exposure" in codes(decision)
    assert "trial_exposure" not in codes(decision)


def test_locked_oos_is_required_when_the_policy_says_so() -> None:
    policy = replace(PROMOTION_V1, require_locked_oos=True)
    decision = evaluate_promotion(make_evidence(), policy)

    assert "missing_locked_oos" in codes(decision)


def test_a_missing_bundle_is_rejected_not_raised() -> None:
    decision = evaluate_promotion(None, PROMOTION_V1)
    assert decision.accepted is False
    assert codes(decision) == ("missing_evidence",)


def test_a_wrong_object_is_rejected_not_raised() -> None:
    decision = evaluate_promotion("not a bundle", PROMOTION_V1)  # type: ignore[arg-type]
    assert decision.accepted is False
    assert codes(decision) == ("invalid_evidence",)


# --- stale / edited evidence --------------------------------------------------


def test_a_tampered_checksum_is_rejected() -> None:
    bundle = replace(make_evidence(), checksum="0" * 64)
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert "checksum_mismatch" in codes(decision)
    assert decision.accepted is False


def test_an_unknown_schema_is_rejected() -> None:
    bundle = make_evidence(schema_version=EVIDENCE_SCHEMA_VERSION + 1)
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert "schema_mismatch" in codes(decision)


def test_a_missing_settings_digest_is_rejected() -> None:
    bundle = make_evidence(settings_digest="")
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert "missing_section" in codes(decision)


def test_a_missing_dataset_checksum_is_rejected() -> None:
    bundle = make_evidence(dataset_checksum="")
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert "missing_section" in codes(decision)


def test_selection_folds_over_the_share_cap_are_rejected() -> None:
    bundle = make_evidence(splits=default_splits(validation_start=150))
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert "selection_fold_share" in codes(decision)
    assert decision.observed["selection_folds"] == (2, 3)


def test_a_zero_selection_share_disables_the_check() -> None:
    policy = replace(PROMOTION_V1, max_selection_fold_share=0.0)
    bundle = make_evidence(splits=default_splits(validation_start=150))
    decision = evaluate_promotion(bundle, policy)

    assert "selection_fold_share" not in codes(decision)
    assert decision.accepted is True


def test_a_no_op_bundle_stress_policy_is_rejected() -> None:
    policy = replace(
        evidence_policy(),
        cost_stress=(1.0,),
    )
    bundle = make_evidence(policy=policy)
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert "missing_cost_stress" in codes(decision)


def test_an_empty_identity_is_rejected() -> None:
    for field in ("spec_hash", "dataset_id", "dataset_checksum", "code_version"):
        bundle = make_evidence(**{field: ""})
        decision = evaluate_promotion(bundle, PROMOTION_V1)
        assert "missing_section" in codes(decision), field


def test_a_missing_aggregate_is_rejected() -> None:
    bundle = make_evidence(aggregate=None)
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert "missing_section" in codes(decision)


def test_internally_inconsistent_evidence_is_rejected() -> None:
    folds = default_folds()
    bundle = make_evidence(folds=folds, aggregate=aggregate_for(folds, total_trades=999))
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert "inconsistent_evidence" in codes(decision)


def test_an_empty_fold_list_is_rejected() -> None:
    bundle = make_evidence(folds=(), aggregate=aggregate_for(()))
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert "insufficient_folds" in codes(decision)


# --- champion comparison ------------------------------------------------------


def test_champion_must_use_the_same_dataset() -> None:
    champion = make_evidence(spec_hash=CHAMPION_HASH, dataset_id="validation:XAGUSD:M1:ffff")
    decision = evaluate_promotion(
        make_evidence(),
        PROMOTION_V1,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_dataset_mismatch" in codes(decision)


def test_champion_must_use_the_same_dataset_checksum() -> None:
    champion = make_evidence(spec_hash=CHAMPION_HASH, dataset_checksum="d" * 64)
    decision = evaluate_promotion(
        make_evidence(),
        PROMOTION_V1,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_dataset_checksum_mismatch" in codes(decision)


def test_champion_must_use_the_same_code_version() -> None:
    champion = make_evidence(spec_hash=CHAMPION_HASH, code_version="test-other")
    decision = evaluate_promotion(
        make_evidence(),
        PROMOTION_V1,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_code_version_mismatch" in codes(decision)


def test_champion_must_use_the_same_settings() -> None:
    champion = make_evidence(spec_hash=CHAMPION_HASH, settings_digest="e" * 64)
    decision = evaluate_promotion(
        make_evidence(),
        PROMOTION_V1,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_settings_mismatch" in codes(decision)


def test_a_champion_without_comparable_identity_is_incomparable() -> None:
    champion = make_evidence(spec_hash=CHAMPION_HASH, settings_digest="")
    decision = evaluate_promotion(
        make_evidence(),
        PROMOTION_V1,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "incomparable_evidence" in codes(decision)
    assert "champion.settings_digest" in str(decision.rejections[-1].observed)


def test_a_candidate_without_settings_is_incomparable_when_a_champion_exists() -> None:
    champion = make_evidence(spec_hash=CHAMPION_HASH)
    decision = evaluate_promotion(
        make_evidence(settings_digest=""),
        PROMOTION_V1,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "incomparable_evidence" in codes(decision)
    assert "candidate.settings_digest" in str(decision.rejections[-1].observed)


def test_champion_must_use_identical_fold_boundaries() -> None:
    champion = make_evidence(spec_hash=CHAMPION_HASH, folds=default_folds(shift=5))
    decision = evaluate_promotion(
        make_evidence(),
        PROMOTION_V1,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_folds_mismatch" in codes(decision)


def test_champion_bundle_must_match_the_named_spec() -> None:
    champion = make_evidence(spec_hash=CHAMPION_HASH)
    decision = evaluate_promotion(
        make_evidence(),
        PROMOTION_V1,
        champion_evidence=champion,
        champion_spec_hash="e" * 64,
    )

    assert "champion_spec_mismatch" in codes(decision)


def test_a_champion_that_is_the_same_spec_is_rejected() -> None:
    champion = make_evidence()
    decision = evaluate_promotion(
        make_evidence(),
        PROMOTION_V1,
        champion_evidence=champion,
        champion_spec_hash=SPEC_HASH,
    )

    assert "champion_same_spec" in codes(decision)


def test_champion_named_without_evidence_is_rejected() -> None:
    decision = evaluate_promotion(make_evidence(), PROMOTION_V1, champion_spec_hash=CHAMPION_HASH)
    assert "champion_evidence_missing" in codes(decision)


def test_champion_evidence_without_a_hash_is_rejected() -> None:
    decision = evaluate_promotion(
        make_evidence(), PROMOTION_V1, champion_evidence=make_evidence(spec_hash=CHAMPION_HASH)
    )
    assert "champion_identity_missing" in codes(decision)


def test_improvement_over_the_champion_is_strict() -> None:
    folds = default_folds()
    champion = make_evidence(
        spec_hash=CHAMPION_HASH,
        folds=folds,
        aggregate=aggregate_for(folds, expectancy_r=0.20),
    )

    tie = evaluate_promotion(
        make_evidence(spec_hash=SPEC_HASH),
        PROMOTION_V1,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )
    # Default candidate expectancy is 0.25, the champion is 0.20: exactly the
    # policy minimum, which passes; a 0.24 candidate is only +0.04R.
    assert tie.accepted is True, tie.codes

    below = make_evidence(
        spec_hash=SPEC_HASH,
        folds=folds,
        aggregate=aggregate_for(folds, expectancy_r=0.24),
    )
    weak = evaluate_promotion(
        below,
        PROMOTION_V1,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )
    assert "champion_improvement" in codes(weak)

    equal = evaluate_promotion(
        make_evidence(
            spec_hash=SPEC_HASH,
            folds=folds,
            aggregate=aggregate_for(folds, expectancy_r=0.20),
        ),
        PROMOTION_V1,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )
    assert "champion_improvement" in codes(equal)


def test_a_stale_champion_checksum_is_rejected() -> None:
    champion = replace(make_evidence(spec_hash=CHAMPION_HASH), checksum="f" * 64)
    decision = evaluate_promotion(
        make_evidence(),
        PROMOTION_V1,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_checksum_mismatch" in codes(decision)


# --- manual override ----------------------------------------------------------


def rejected_bundle() -> CandidateEvidence:
    folds = tuple(fold(index, trades=5) for index in (1, 2, 3))
    return make_evidence(folds=folds)


def test_manual_override_keeps_the_original_rejections() -> None:
    bundle = rejected_bundle()
    override = manual_override(
        bundle,
        PROMOTION_V1,
        reason="owner accepts the risk for a paper-only run",
        owner="resul",
    )

    assert override.policy_id == "manual-override"
    assert override.policy_id != PROMOTION_V1.policy_id
    assert override.original_policy_id == PROMOTION_V1.policy_id
    assert override.original_policy_version == PROMOTION_V1.version
    assert override.rejections
    assert {item.code for item in override.rejections} == set(
        codes(evaluate_promotion(bundle, PROMOTION_V1))
    )
    assert override.evidence_checksum == bundle.checksum
    assert override.is_policy_evidence is False
    assert not hasattr(override, "accepted")
    assert "not validation evidence" in override.note


@pytest.mark.parametrize(
    ("reason", "owner"),
    [("", "resul"), ("   ", "resul"), ("owner override", ""), ("owner override", "   ")],
)
def test_manual_override_requires_a_reason_and_owner(reason: str, owner: str) -> None:
    with pytest.raises(ValueError):
        manual_override(rejected_bundle(), PROMOTION_V1, reason=reason, owner=owner)


def test_manual_override_is_refused_when_the_policy_already_accepts() -> None:
    with pytest.raises(ValueError, match="already satisfies"):
        manual_override(
            make_evidence(),
            PROMOTION_V1,
            reason="nothing to override",
            owner="resul",
        )


def test_manual_override_cannot_be_promoted_as_evidence(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path)
    override = manual_override(
        rejected_bundle(),
        PROMOTION_V1,
        reason="paper-only risk acceptance",
        owner="resul",
    )

    with pytest.raises(TypeError, match="ManualOverride"):
        promote_with_evidence(
            archive,
            candidate.canonical_hash(),
            override,  # type: ignore[arg-type]
            scope=SCOPE,
            reason="override",
        )


# --- archive integration ------------------------------------------------------


def archive_with_candidate(tmp_path: Any) -> tuple[DiscoveryArchive, ToyCandidate]:
    archive = DiscoveryArchive(tmp_path / "discovery.sqlite")
    candidate = ToyCandidate((1, 2))
    archive.add_candidate(candidate, code_version=CODE_VERSION)
    archive.record_trial(
        candidate.canonical_hash(),
        TrialResult(metrics={"expectancy_r": 0.25, "trades": 75.0}, accepted=True),
        dataset_id=DATASET_ID,
        code_version=CODE_VERSION,
        seed=0,
    )
    return archive, candidate


def accepted_decision_for(candidate: ToyCandidate) -> PromotionDecision:
    decision = evaluate_promotion(make_evidence(spec_hash=candidate.canonical_hash()), PROMOTION_V1)
    assert decision.accepted
    return decision


def test_archive_promotion_stores_the_policy_and_checksum(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path)
    decision = accepted_decision_for(candidate)

    champion = promote_with_evidence(
        archive,
        candidate.canonical_hash(),
        decision,
        scope=SCOPE,
        reason="passed promotion-v1 on the validation folds",
    )

    assert champion.candidate.candidate_id == candidate.canonical_hash()
    assert "promotion-v1 v1" in champion.reason
    assert decision.evidence_checksum is not None
    assert decision.evidence_checksum[:12] in champion.reason

    history = archive.champion_history(SCOPE)
    assert len(history) == 1
    assert history[0].metric is None  # no metric-only path
    assert history[0].value is None
    assert history[0].previous_candidate_id is None

    recorded = archive.decisions(candidate.canonical_hash())[-1]
    assert recorded.decision == "promoted"
    detail = recorded.detail
    assert detail is not None
    assert detail["mode"] == "evidence"
    assert detail["policy_id"] == PROMOTION_V1.policy_id
    assert detail["policy_version"] == PROMOTION_V1.version
    assert detail["evidence_checksum"] == decision.evidence_checksum
    assert detail["candidate_spec_hash"] == candidate.canonical_hash()
    assert detail["dataset_id"] == DATASET_ID
    assert detail["dataset_checksum"] == DATASET_CHECKSUM
    assert detail["settings_digest"] == SETTINGS_DIGEST


def test_archive_refuses_a_rejected_decision(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path)
    decision = evaluate_promotion(rejected_bundle(), PROMOTION_V1)
    assert decision.accepted is False

    with pytest.raises(PromotionRefused):
        promote_with_evidence(
            archive,
            candidate.canonical_hash(),
            decision,
            scope=SCOPE,
            reason="should not pass",
        )

    assert archive.champion(SCOPE) is None
    assert archive.champion_history(SCOPE) == ()
    assert archive.decisions() == ()


def test_archive_refuses_evidence_for_another_candidate(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path)
    other = ToyCandidate((3, 4))
    archive.add_candidate(other, code_version=CODE_VERSION)
    decision = accepted_decision_for(candidate)

    with pytest.raises(PromotionRefused, match="detached promotion"):
        promote_with_evidence(
            archive,
            other.canonical_hash(),
            decision,
            scope=SCOPE,
            reason="wrong candidate",
        )


def test_archive_has_no_metric_only_promote(tmp_path: Any) -> None:
    archive, _ = archive_with_candidate(tmp_path)
    assert not hasattr(archive, "promote")
    assert hasattr(archive, "promote_exploratory")
    assert hasattr(archive, "promote_evidence")


def test_archive_rejects_manual_override_identity(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path)

    with pytest.raises(ArchiveError, match="manual override"):
        archive.promote_evidence(
            candidate.canonical_hash(),
            scope=SCOPE,
            reason="override",
            policy_id=MANUAL_OVERRIDE_POLICY_ID,
            policy_version=1,
            evidence_checksum="a" * 64,
            dataset_id=DATASET_ID,
            dataset_checksum=DATASET_CHECKSUM,
            settings_digest=SETTINGS_DIGEST,
            candidate_spec_hash=candidate.canonical_hash(),
        )


def test_archive_rejects_missing_provenance(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path)

    with pytest.raises(ValueError):
        archive.promote_evidence(
            candidate.canonical_hash(),
            scope=SCOPE,
            reason="no checksum",
            policy_id="promotion-v1",
            policy_version=1,
            evidence_checksum="",
            dataset_id=DATASET_ID,
            dataset_checksum=DATASET_CHECKSUM,
            settings_digest=SETTINGS_DIGEST,
            candidate_spec_hash=candidate.canonical_hash(),
        )
    with pytest.raises(ValueError):
        archive.promote_evidence(
            candidate.canonical_hash(),
            scope=SCOPE,
            reason="bad version",
            policy_id="promotion-v1",
            policy_version=0,
            evidence_checksum="a" * 64,
            dataset_id=DATASET_ID,
            dataset_checksum=DATASET_CHECKSUM,
            settings_digest=SETTINGS_DIGEST,
            candidate_spec_hash=candidate.canonical_hash(),
        )
    with pytest.raises(ArchiveError, match="scope"):
        archive.promote_evidence(
            candidate.canonical_hash(),
            scope="XAGUSD:M1",
            reason="wrong market",
            policy_id="promotion-v1",
            policy_version=1,
            evidence_checksum="a" * 64,
            dataset_id=DATASET_ID,
            dataset_checksum=DATASET_CHECKSUM,
            settings_digest=SETTINGS_DIGEST,
            candidate_spec_hash=candidate.canonical_hash(),
        )


def test_exploratory_promotion_is_still_available_and_separate(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path)

    champion = archive.promote_exploratory(
        candidate.canonical_hash(),
        scope=SCOPE,
        reason="research ablation only",
    )
    assert champion.metrics["expectancy_r"] == pytest.approx(0.25)
    assert "exploratory" in type(archive).promote_exploratory.__doc__.splitlines()[0].lower()
