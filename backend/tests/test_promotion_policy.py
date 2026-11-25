"""Promotion policy: fail-closed checks, champion comparison and archive audit.

The evidence bundles here are built by hand, not simulated. These tests exercise
the *policy* arithmetic and identity logic; the fold geometry and checksum of a
real bundle are covered by ``test_discovery_evidence.py``. A synthetic bundle is
enough to prove that every threshold, every stale identity and every missing
section is a rejection, and that no path accepts a decision without a sealed
policy evaluation.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from discovery_fakes import ToyCandidate
from promotion_support import promote_for_tests, registered_test_policies
from scalping.discovery.archive import (
    SCHEMA_VERSION,
    ArchiveError,
    DiscoveryArchive,
)
from scalping.discovery.champion import (
    ChampionProvider,
    NoEvidenceChampion,
)
from scalping.discovery.evidence import (
    EVIDENCE_SCHEMA_VERSION,
    AggregateEvidence,
    CandidateEvidence,
    EvidencePolicy,
    FoldEvidence,
    LockedOOSEvidence,
    RegimeEvidence,
    SplitBoundaries,
    StressEvidence,
    evidence_checksum,
    evidence_from_dict,
    locked_oos_checksum,
)
from scalping.discovery.promotion import (
    PRODUCTION_POLICIES,
    PROMOTION_V1,
    EvidenceIdentity,
    ExpectedIdentity,
    PromotionDecision,
    PromotionRefused,
    evaluate_promotion,
    manual_override,
    promote_manual_override,
    promote_with_evidence,
)
from scalping.discovery.protocols import TrialResult
from scalping.strategy.dsl import (
    Compare,
    Const,
    EntryRule,
    ExitRule,
    HoldingLimit,
    StrategySpec,
)

SPEC_HASH = "a" * 64
CHAMPION_HASH = "b" * 64
DATASET_ID = "validation:XAUUSD:M1:abcd1234"
DATASET_CHECKSUM = "c" * 64
SETTINGS_DIGEST = "d" * 64
CODE_VERSION = "test-promotion-2026-11-24"
SCOPE = "XAUUSD:M1"
START = datetime(2026, 1, 5, tzinfo=UTC)
IDENTITY = EvidenceIdentity(
    code_version=CODE_VERSION,
    dataset_checksum=DATASET_CHECKSUM,
    settings_digest=SETTINGS_DIGEST,
)
#: A non-production policy for tests that must produce an accepted decision.
#: It is not PROMOTION_V1: the production policy requires a locked OOS section.
TEST_POLICY = replace(
    PROMOTION_V1,
    policy_id="test-policy",
    require_locked_oos=False,
)

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
    stress_expectancy_r: float | None = None,
    stress_profit_factor: float | None = None,
    shift: int = 0,
    ts_shift: int = 0,
    purge_bars: int = 3,
    embargo_bars: int = 2,
) -> FoldEvidence:
    test_start = 100 + shift + (index - 1) * 50
    test_end = test_start + 50
    train_end = test_start - (purge_bars + embargo_bars)
    stressed_r = expectancy_r if stress_expectancy_r is None else stress_expectancy_r
    stressed_pf = profit_factor if stress_profit_factor is None else stress_profit_factor
    return FoldEvidence(
        fold=index,
        train_start_row=0,
        train_end_row=train_end,
        test_start_row=test_start,
        test_end_row=test_end,
        train_rows=train_end,
        test_rows=50,
        train_start_ts=_ts(0 + ts_shift),
        train_end_ts=_ts(test_start - 1 + ts_shift),
        test_start_ts=_ts(test_start + ts_shift),
        test_end_ts=_ts(test_end - 1 + ts_shift),
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
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
                profit_factor=stressed_pf,
                max_drawdown=max_drawdown,
                expectancy_r=stressed_r,
            )
            for multiplier in stress
        ),
    )


def default_folds(*, shift: int = 0) -> tuple[FoldEvidence, ...]:
    return tuple(fold(index, shift=shift) for index in (1, 2, 3))


def default_splits(*, validation_start: int = 200) -> SplitBoundaries:
    """Folds test rows 100-250; only fold 3 overlaps validation [200, 250)."""
    return SplitBoundaries(
        train_start_row=0,
        train_end_row=validation_start,
        validation_start_row=validation_start,
        validation_end_row=250,
        test_start_row=250,
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
        "bootstrap_samples": 1000,
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
        bootstrap_samples=1000,
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
        # A policy-arithmetic fixture: it claims a real dataset explicitly so
        # the provenance gate is not what the assertion is about. A test that
        # exercises that gate overrides these.
        "source": "test:fixture",
        "synthetic": False,
    }
    values.update(overrides)
    bundle = CandidateEvidence(**values)
    return bundle if values["checksum"] else reseal(bundle)


def codes(decision: PromotionDecision) -> tuple[str, ...]:
    return decision.codes


# --- policy construction ------------------------------------------------------


def test_promotion_v1_is_explicit_and_locked() -> None:
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
    assert policy.require_locked_oos is True
    assert 0.0 < policy.max_selection_fold_share <= 1.0
    assert policy.min_bootstrap_confidence >= 0.95
    assert policy.min_bootstrap_samples >= 500
    assert policy.max_stress_degradation >= 1.0
    assert len(policy.digest) == 64


@pytest.mark.parametrize(
    "changes",
    [
        {"policy_id": ""},
        {"policy_id": "   "},
        {"policy_id": "manual-override"},
        {"version": 0},
        {"min_folds": 2},
        {"min_trades": 0},
        {"min_profit_factor": 1.0},
        {"max_drawdown": 0.0},
        {"min_positive_fold_fraction": 0.0},
        {"min_regime_coverage": 0.0},
        {"require_cost_stress": ()},
        {"require_cost_stress": (2.0, 1.0)},
        {"require_cost_stress": (1.0,)},
        {"require_cost_stress": (2.0, 2.0)},
        {"max_complexity": 0},
        {"max_trials": 0},
        {"min_improvement_r": 0.0},
        {"min_bootstrap_confidence": 0.90},
        {"min_bootstrap_confidence": 1.0},
        {"min_bootstrap_samples": 499},
        {"max_stress_degradation": 0.5},
        {"max_selection_fold_share": -0.1},
        {"max_selection_fold_share": 0.0},
        {"max_selection_fold_share": 1.1},
    ],
)
def test_degenerate_policies_are_refused(changes: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(PROMOTION_V1, **changes)


def test_there_is_no_force_parameter() -> None:
    for function in (
        evaluate_promotion,
        manual_override,
        promote_with_evidence,
        promote_manual_override,
    ):
        assert "force" not in inspect.signature(function).parameters
    assert "decision" not in inspect.signature(promote_with_evidence).parameters


# --- B1: production policy needs the locked OOS section -----------------------


def test_production_policy_rejects_without_locked_oos() -> None:
    bundle = make_evidence()
    decision = evaluate_promotion(bundle, PROMOTION_V1)

    assert decision.accepted is False
    assert "missing_locked_oos" in codes(decision)


def test_the_test_policy_is_the_only_accepting_path_in_these_tests() -> None:
    decision = evaluate_promotion(make_evidence(), TEST_POLICY)

    assert decision.accepted is True
    assert decision.policy_id == "test-policy"
    assert decision.policy_digest == TEST_POLICY.digest


def locked_section(evidence: CandidateEvidence, policy: Any) -> LockedOOSEvidence:
    from scalping.discovery.promotion import _base_evidence_checksum

    splits = evidence.splits
    base = LockedOOSEvidence(
        checksum="0" * 64,
        evidence_checksum=_base_evidence_checksum(evidence),
        dataset_id=evidence.dataset_id,
        dataset_checksum=evidence.dataset_checksum,
        symbol=evidence.symbol,
        timeframe=evidence.timeframe,
        source=evidence.source,
        synthetic=evidence.synthetic,
        spec_hash=evidence.spec_hash,
        spec_version=evidence.spec_version,
        complexity=evidence.complexity,
        code_version=evidence.code_version,
        settings_digest=evidence.settings_digest,
        policy_id=policy.policy_id,
        policy_version=policy.version,
        test_start_row=splits.test_start_row,
        test_end_row=splits.test_end_row,
        test_rows=splits.test_end_row - splits.test_start_row,
        test_start_ts=_ts(300),
        test_end_ts=_ts(399),
        trades=75,
        net_profit=5.0,
        profit_factor=1.6,
        win_rate=0.5,
        expectancy_r=0.3,
        total_r=22.5,
        max_drawdown=0.1,
        expectancy_r_ci95_low=0.1,
        expectancy_r_ci95_high=0.5,
        total_r_ci95_low=2.0,
        total_r_ci95_high=20.0,
        bootstrap_samples=1000,
        bootstrap_confidence=0.95,
        bootstrap_method="iid-bootstrap",
        regime_coverage=0.95,
        stress=tuple(
            StressEvidence(
                multiplier=multiplier,
                trades=75,
                net_profit=4.0,
                profit_factor=1.4,
                max_drawdown=0.1,
                expectancy_r=0.2,
            )
            for multiplier in (1.5, 2.0)
        ),
        created_at="2026-11-24T00:00:00+00:00",
    )
    return replace(base, checksum=locked_oos_checksum(base))


def test_a_locked_oos_section_is_a_real_dataclass_field_and_unlocks() -> None:
    policy = replace(TEST_POLICY, policy_id="locked-test", require_locked_oos=True)
    base = make_evidence()
    section = locked_section(base, policy)
    bundle = reseal(base, locked_oos=section)

    decision = evaluate_promotion(bundle, policy)

    assert decision.accepted is True, decision.codes
    restored = evidence_from_dict(bundle.to_dict())
    assert restored.locked_oos == section
    assert evidence_checksum(restored) == bundle.checksum


def reseal_locked(section: LockedOOSEvidence, **changes: Any) -> LockedOOSEvidence:
    draft = replace(section, **changes)
    return replace(draft, checksum=locked_oos_checksum(draft))


def locked_bundle(section: LockedOOSEvidence) -> CandidateEvidence:
    return reseal(make_evidence(), locked_oos=section)


def test_a_fabricated_locked_checksum_is_rejected() -> None:
    policy = replace(TEST_POLICY, policy_id="locked-test", require_locked_oos=True)
    base = make_evidence()
    section = locked_section(base, policy)
    forged = replace(section, checksum="f" * 64)

    decision = evaluate_promotion(locked_bundle(forged), policy)

    assert "locked_oos_checksum" in codes(decision)


def test_a_locked_section_for_another_bundle_is_rejected() -> None:
    policy = replace(TEST_POLICY, policy_id="locked-test", require_locked_oos=True)
    base = make_evidence()
    section = reseal_locked(locked_section(base, policy), evidence_checksum="f" * 64)

    decision = evaluate_promotion(locked_bundle(section), policy)

    assert "locked_oos_evidence_checksum" in codes(decision)


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"test_rows": 10}, "locked_oos_geometry"),
        ({"test_start_row": 0}, "locked_oos_geometry"),
        ({"test_start_ts": "2026-01-01T00:00:00+00:00"}, "locked_oos_geometry"),
        ({"test_start_ts": "not-a-timestamp"}, "locked_oos_geometry"),
        ({"expectancy_r": -5.0}, "min_expectancy_r"),
        ({"profit_factor": 0.1}, "min_profit_factor"),
        ({"max_drawdown": 0.99}, "max_drawdown"),
        ({"regime_coverage": 0.1}, "min_regime_coverage"),
        ({"expectancy_r_ci95_low": -0.1}, "min_ci_lower_r"),
        ({"expectancy_r_ci95_low": 0.0}, "min_ci_lower_r"),
        ({"trades": 1}, "insufficient_trades"),
        ({"bootstrap_samples": 1}, "min_bootstrap_samples"),
        ({"bootstrap_confidence": 0.5}, "min_bootstrap_confidence"),
        ({"complexity": 999}, "max_complexity"),
    ],
)
def test_bad_locked_metrics_or_geometry_are_rejected(changes: dict[str, Any], code: str) -> None:
    policy = replace(TEST_POLICY, policy_id="locked-test", require_locked_oos=True)
    base = make_evidence()
    section = reseal_locked(locked_section(base, policy), **changes)

    decision = evaluate_promotion(locked_bundle(section), policy)

    assert decision.accepted is False
    assert code in codes(decision)


def test_a_locked_ci_exactly_at_the_floor_is_rejected_but_above_passes() -> None:
    policy = replace(TEST_POLICY, policy_id="locked-test", require_locked_oos=True)
    base = make_evidence()
    section = locked_section(base, policy)

    at_floor = reseal_locked(section, expectancy_r_ci95_low=0.0)
    decision = evaluate_promotion(locked_bundle(at_floor), policy)
    assert "min_ci_lower_r" in codes(decision)
    assert decision.accepted is False

    just_above = reseal_locked(section, expectancy_r_ci95_low=0.01)
    decision = evaluate_promotion(locked_bundle(just_above), policy)
    assert decision.accepted is True, decision.codes


def test_catastrophic_locked_stress_is_rejected() -> None:
    policy = replace(TEST_POLICY, policy_id="locked-test", require_locked_oos=True)
    base = make_evidence()
    section = locked_section(base, policy)
    bad_stress = tuple(replace(row, expectancy_r=-5.0, profit_factor=0.1) for row in section.stress)
    section = reseal_locked(section, stress=bad_stress)

    decision = evaluate_promotion(locked_bundle(section), policy)

    assert "cost_stress" in codes(decision)
    assert "stress_degradation" in codes(decision)


def test_a_locked_section_from_another_policy_does_not_unlock() -> None:
    policy = replace(TEST_POLICY, policy_id="locked-test", require_locked_oos=True)
    base = make_evidence()
    section = locked_section(base, replace(policy, policy_id="other-policy"))
    bundle = reseal(base, locked_oos=section)

    decision = evaluate_promotion(bundle, policy)

    assert decision.accepted is False
    assert "locked_oos_identity" in codes(decision)


# --- accepted path ------------------------------------------------------------


def test_a_complete_bundle_is_accepted_and_attributed() -> None:
    bundle = make_evidence()
    decision = evaluate_promotion(bundle, TEST_POLICY, expected_identity=IDENTITY)

    assert decision.accepted is True
    assert decision.rejections == ()
    assert decision.policy_id == "test-policy"
    assert decision.version == 1
    assert decision.policy_digest == TEST_POLICY.digest
    assert decision.evidence_checksum == bundle.checksum
    assert decision.dataset_checksum == DATASET_CHECKSUM
    assert decision.settings_digest == SETTINGS_DIGEST
    assert decision.champion_checksum is None
    assert decision.candidate_spec_hash == SPEC_HASH
    assert decision.dataset_id == DATASET_ID


# --- B3: forged decisions and public bypass -----------------------------------


def test_a_hand_built_decision_is_refused() -> None:
    with pytest.raises(ValueError, match="evaluate_promotion"):
        PromotionDecision(
            accepted=True,
            rejections=(),
            policy_id="test-policy",
            version=1,
            policy_digest=TEST_POLICY.digest,
            evidence_checksum=SPEC_HASH,
            champion_checksum=None,
            candidate_spec_hash=SPEC_HASH,
            champion_spec_hash=None,
            dataset_id=DATASET_ID,
            dataset_checksum=DATASET_CHECKSUM,
            settings_digest=SETTINGS_DIGEST,
            code_version=CODE_VERSION,
        )


def test_a_sealed_decision_cannot_be_replaced_into_an_acceptance() -> None:
    rejected = evaluate_promotion(
        make_evidence(folds=tuple(fold(i, trades=5) for i in (1, 2, 3))), TEST_POLICY
    )
    assert rejected.accepted is False

    with pytest.raises(ValueError, match="evaluate_promotion"):
        replace(rejected, accepted=True)


def test_the_archive_has_no_public_evidence_promotion() -> None:
    assert not hasattr(DiscoveryArchive, "promote_evidence")
    assert hasattr(DiscoveryArchive, "_promote_evidence")


def test_a_forged_seal_is_refused() -> None:
    import scalping.discovery.promotion as promotion_module

    assert not hasattr(promotion_module, "_DECISION_SEAL")
    assert not hasattr(promotion_module, "_SEALER")
    with pytest.raises(ValueError, match="evaluate_promotion"):
        PromotionDecision(
            accepted=True,
            rejections=(),
            policy_id="test-policy",
            version=1,
            policy_digest=TEST_POLICY.digest,
            evidence_checksum=SPEC_HASH,
            champion_checksum=None,
            candidate_spec_hash=SPEC_HASH,
            champion_spec_hash=None,
            dataset_id=DATASET_ID,
            dataset_checksum=DATASET_CHECKSUM,
            settings_digest=SETTINGS_DIGEST,
            code_version=CODE_VERSION,
            _seal="0" * 64,
        )


def test_a_subclass_cannot_inject_a_locked_oos_section() -> None:
    base = make_evidence()

    class ForgedEvidence(CandidateEvidence):
        def to_dict(self) -> dict[str, Any]:
            payload = super().to_dict()
            payload["locked_oos"] = {"checksum": "a" * 64, "trades": 10}
            return payload

    forged = ForgedEvidence(**{name: getattr(base, name) for name in base.__dataclass_fields__})
    decision = evaluate_promotion(forged, TEST_POLICY)  # type: ignore[arg-type]

    assert decision.accepted is False
    assert "invalid_evidence" in codes(decision)


def test_a_public_call_refuses_a_test_policy(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path)

    with pytest.raises(PromotionRefused, match="non_production_policy"):
        promote_with_evidence(
            archive,
            candidate.canonical_hash(),
            candidate_evidence(candidate),
            TEST_POLICY,
            scope=SCOPE,
            reason="test policy must not pass the public API",
            expected_identity=IDENTITY,
        )
    assert archive.champion(SCOPE) is None


# --- candidate rejections -----------------------------------------------------


def test_insufficient_folds_are_rejected() -> None:
    bundle = make_evidence(folds=(fold(1, trades=30), fold(2, trades=30)))
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert decision.accepted is False
    assert "insufficient_folds" in codes(decision)


def test_insufficient_trades_are_rejected() -> None:
    bundle = make_evidence(folds=tuple(fold(index, trades=5) for index in (1, 2, 3)))
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "insufficient_trades" in codes(decision)
    assert decision.evidence_checksum == bundle.checksum


def test_low_expectancy_is_rejected() -> None:
    folds = default_folds()
    bundle = make_evidence(folds=folds, aggregate=aggregate_for(folds, expectancy_r=0.01))
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "min_expectancy_r" in codes(decision)
    assert decision.rejections[0].observed == pytest.approx(0.01)


def test_low_profit_factor_is_rejected() -> None:
    folds = default_folds()
    bundle = make_evidence(folds=folds, aggregate=aggregate_for(folds, profit_factor=1.1))
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "min_profit_factor" in codes(decision)


def test_deep_drawdown_is_rejected() -> None:
    folds = default_folds()
    bundle = make_evidence(folds=folds, aggregate=aggregate_for(folds, max_drawdown=0.30))
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "max_drawdown" in codes(decision)


def test_too_few_positive_folds_are_rejected() -> None:
    folds = (fold(1), fold(2, net_profit=-1.0), fold(3, net_profit=-1.0))
    bundle = make_evidence(folds=folds, aggregate=aggregate_for(folds))
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "min_positive_fold_fraction" in codes(decision)


def test_low_regime_coverage_is_rejected() -> None:
    folds = default_folds()
    bundle = make_evidence(folds=folds, aggregate=aggregate_for(folds, regime_coverage=0.5))
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "min_regime_coverage" in codes(decision)


def test_a_confidence_interval_that_touches_zero_is_rejected() -> None:
    folds = default_folds()
    bundle = make_evidence(folds=folds, aggregate=aggregate_for(folds, expectancy_r_ci95_low=0.0))
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "min_ci_lower_r" in codes(decision)
    assert decision.rejections[0].required == "> 0"


def test_a_missing_stress_multiplier_is_rejected() -> None:
    folds = tuple(fold(index, stress=(1.5,)) for index in (1, 2, 3))
    bundle = make_evidence(folds=folds)
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "missing_cost_stress" in codes(decision)
    assert "x2" in str(decision.rejections[0].required)


def test_too_complex_a_spec_is_rejected() -> None:
    bundle = make_evidence(complexity=TEST_POLICY.max_complexity + 1)
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "max_complexity" in codes(decision)
    assert decision.rejections[0].observed == TEST_POLICY.max_complexity + 1


def test_trial_and_search_exposure_are_capped() -> None:
    trials = make_evidence(
        trial_count=TEST_POLICY.max_trials + 1,
        search_budget=TEST_POLICY.max_trials + 10,
    )
    assert "trial_exposure" in codes(evaluate_promotion(trials, TEST_POLICY))

    search = make_evidence(search_budget=TEST_POLICY.max_trials + 1)
    decision = evaluate_promotion(search, TEST_POLICY)
    assert "search_exposure" in codes(decision)
    assert "trial_exposure" not in codes(decision)


def test_locked_oos_is_required_when_the_policy_says_so() -> None:
    policy = replace(TEST_POLICY, require_locked_oos=True)
    decision = evaluate_promotion(make_evidence(), policy)

    assert "missing_locked_oos" in codes(decision)


def test_a_missing_bundle_is_rejected_not_raised() -> None:
    decision = evaluate_promotion(None, TEST_POLICY)
    assert decision.accepted is False
    assert codes(decision) == ("missing_evidence",)


def test_a_wrong_object_is_rejected_not_raised() -> None:
    decision = evaluate_promotion("not a bundle", TEST_POLICY)  # type: ignore[arg-type]
    assert decision.accepted is False
    assert codes(decision) == ("invalid_evidence",)


# --- B4: fold geometry --------------------------------------------------------


def test_a_fold_reaching_the_test_split_is_rejected() -> None:
    reaching = replace(
        fold(3, trades=30),
        test_start_row=250,
        test_end_row=300,
        test_start_ts=_ts(250),
        test_end_ts=_ts(299),
    )
    folds = (fold(1, trades=30), fold(2, trades=30), reaching)
    bundle = make_evidence(folds=folds, splits=default_splits())
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "fold_geometry" in codes(decision)
    assert any("250" in str(item.required) for item in decision.rejections)


def test_folds_must_end_at_the_validation_boundary() -> None:
    folds = (fold(1), fold(2), fold(3, shift=-10))
    bundle = make_evidence(folds=folds, splits=default_splits())
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "fold_geometry" in codes(decision)


# --- B5: bootstrap gate -------------------------------------------------------


def test_low_bootstrap_samples_are_rejected() -> None:
    folds = default_folds()
    bundle = make_evidence(folds=folds, aggregate=aggregate_for(folds, bootstrap_samples=1))
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "min_bootstrap_samples" in codes(decision)


def test_a_weak_bootstrap_confidence_is_rejected() -> None:
    folds = default_folds()
    bundle = make_evidence(folds=folds, aggregate=aggregate_for(folds, bootstrap_confidence=0.50))
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "min_bootstrap_confidence" in codes(decision)


def test_a_degenerate_bootstrap_confidence_is_rejected() -> None:
    folds = default_folds()
    bundle = make_evidence(folds=folds, aggregate=aggregate_for(folds, bootstrap_confidence=1.0))
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "min_bootstrap_confidence" in codes(decision)


# --- H3: stress thresholds ----------------------------------------------------


def test_catastrophic_stress_is_rejected() -> None:
    folds = tuple(
        fold(index, stress_expectancy_r=-5.0, stress_profit_factor=0.1) for index in (1, 2, 3)
    )
    bundle = make_evidence(folds=folds)
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "cost_stress" in codes(decision)
    assert "stress_degradation" in codes(decision)


def test_stress_may_not_lose_more_than_the_degradation_cap() -> None:
    folds = tuple(fold(index, expectancy_r=1.0, stress_expectancy_r=0.4) for index in (1, 2, 3))
    bundle = make_evidence(folds=folds)
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "stress_degradation" in codes(decision)

    tolerable = tuple(fold(index, expectancy_r=1.0, stress_expectancy_r=0.6) for index in (1, 2, 3))
    accepted = evaluate_promotion(make_evidence(folds=tolerable), TEST_POLICY)
    assert "stress_degradation" not in codes(accepted)


# --- H1: staleness against the current identity -------------------------------


@pytest.mark.parametrize(
    ("identity", "code"),
    [
        (
            EvidenceIdentity("other-code", DATASET_CHECKSUM, SETTINGS_DIGEST),
            "stale_code_version",
        ),
        (
            EvidenceIdentity(CODE_VERSION, "e" * 64, SETTINGS_DIGEST),
            "stale_dataset_checksum",
        ),
        (
            EvidenceIdentity(CODE_VERSION, DATASET_CHECKSUM, "f" * 64),
            "stale_settings",
        ),
    ],
)
def test_a_stale_bundle_is_rejected(identity: EvidenceIdentity, code: str) -> None:
    decision = evaluate_promotion(make_evidence(), TEST_POLICY, expected_identity=identity)

    assert decision.accepted is False
    assert code in codes(decision)


def test_expected_identity_current_uses_live_code_settings_and_manifest() -> None:
    from scalping.config import Settings
    from scalping.discovery.evidence import settings_digest

    identity = ExpectedIdentity.current(
        dataset_checksum=DATASET_CHECKSUM,
        settings=Settings(),
        code_version=CODE_VERSION,
    )

    assert identity.code_version == CODE_VERSION
    assert identity.dataset_checksum == DATASET_CHECKSUM
    assert identity.settings_digest == settings_digest(Settings())


def test_an_old_trial_is_rejected_against_the_current_identity(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path)
    old_settings = "e" * 64
    old_evidence = candidate_evidence(candidate, settings_digest=old_settings)
    archive.record_trial(
        candidate.canonical_hash(),
        TrialResult(metrics={"expectancy_r": 0.3, "trades": 75.0}, accepted=True),
        dataset_id=DATASET_ID,
        code_version=CODE_VERSION,
        seed=2,
        dataset_checksum=DATASET_CHECKSUM,
        settings_digest=old_settings,
    )
    current = ExpectedIdentity.current(
        dataset_checksum=DATASET_CHECKSUM,
        code_version=CODE_VERSION,
    )

    with pytest.raises((ArchiveError, PromotionRefused)) as exc:
        promote_for_tests(
            archive,
            candidate.canonical_hash(),
            old_evidence,
            TEST_POLICY,
            scope=SCOPE,
            reason="old evidence against the current settings",
            expected_identity=current,
        )
    assert "stale_settings" in str(exc.value)


def test_an_empty_expected_identity_is_refused() -> None:
    with pytest.raises(ValueError):
        EvidenceIdentity("", DATASET_CHECKSUM, SETTINGS_DIGEST)


# --- stored evidence round-trip -----------------------------------------------


def test_evidence_round_trips_through_json() -> None:
    bundle = make_evidence()
    restored = evidence_from_dict(bundle.to_dict())

    assert evidence_checksum(restored) == bundle.checksum
    assert restored == bundle


# --- stale / edited evidence --------------------------------------------------


def test_a_tampered_checksum_is_rejected() -> None:
    bundle = replace(make_evidence(), checksum="0" * 64)
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "checksum_mismatch" in codes(decision)
    assert decision.accepted is False


def test_an_unknown_schema_is_rejected() -> None:
    bundle = make_evidence(schema_version=EVIDENCE_SCHEMA_VERSION + 1)
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "schema_mismatch" in codes(decision)


def test_an_empty_identity_is_rejected() -> None:
    for field in ("spec_hash", "dataset_id", "dataset_checksum", "settings_digest"):
        bundle = make_evidence(**{field: ""})
        decision = evaluate_promotion(bundle, TEST_POLICY)
        assert "missing_section" in codes(decision), field


def test_a_missing_aggregate_is_rejected() -> None:
    bundle = make_evidence(aggregate=None)
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "missing_section" in codes(decision)


def test_internally_inconsistent_evidence_is_rejected() -> None:
    folds = default_folds()
    bundle = make_evidence(folds=folds, aggregate=aggregate_for(folds, total_trades=999))
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "inconsistent_evidence" in codes(decision)


def test_an_empty_fold_list_is_rejected() -> None:
    bundle = make_evidence(folds=(), aggregate=aggregate_for(()))
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "insufficient_folds" in codes(decision)


def test_policy_and_aggregate_bootstrap_must_match() -> None:
    folds = default_folds()
    bundle = make_evidence(
        folds=folds,
        policy=replace(evidence_policy(), bootstrap_samples=900),
    )
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "inconsistent_evidence" in codes(decision)


def test_policy_fold_count_must_match_the_folds() -> None:
    bundle = make_evidence(policy=replace(evidence_policy(), folds=5))
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "inconsistent_evidence" in codes(decision)


def test_policy_cost_stress_must_match_every_fold() -> None:
    bundle = make_evidence(policy=replace(evidence_policy(), cost_stress=(1.5, 3.0)))
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "inconsistent_evidence" in codes(decision)


def test_a_zero_trade_fold_is_inconsistent() -> None:
    folds = (fold(1, trades=30), fold(2, trades=30), fold(3, trades=0))
    bundle = make_evidence(folds=folds, aggregate=aggregate_for(folds))
    decision = evaluate_promotion(bundle, TEST_POLICY)

    assert "inconsistent_evidence" in codes(decision)


# --- champion comparison ------------------------------------------------------


def test_champion_must_use_the_same_dataset() -> None:
    champion = make_evidence(spec_hash=CHAMPION_HASH, dataset_id="validation:XAGUSD:M1:ffff")
    decision = evaluate_promotion(
        make_evidence(),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_dataset_mismatch" in codes(decision)


def test_champion_must_use_the_same_dataset_checksum() -> None:
    champion = make_evidence(spec_hash=CHAMPION_HASH, dataset_checksum="e" * 64)
    decision = evaluate_promotion(
        make_evidence(),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_dataset_checksum_mismatch" in codes(decision)


def test_champion_must_use_the_same_settings() -> None:
    champion = make_evidence(spec_hash=CHAMPION_HASH, settings_digest="e" * 64)
    decision = evaluate_promotion(
        make_evidence(),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_settings_mismatch" in codes(decision)


def test_champion_must_use_the_same_code_version() -> None:
    champion = make_evidence(spec_hash=CHAMPION_HASH, code_version="test-other")
    decision = evaluate_promotion(
        make_evidence(),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_code_version_mismatch" in codes(decision)


def test_champion_must_use_the_same_split_boundaries() -> None:
    champion = make_evidence(spec_hash=CHAMPION_HASH, splits=default_splits(validation_start=190))
    decision = evaluate_promotion(
        make_evidence(),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_splits_mismatch" in codes(decision)


def test_champion_must_use_identical_fold_geometry() -> None:
    champion = make_evidence(spec_hash=CHAMPION_HASH, folds=default_folds(shift=5))
    decision = evaluate_promotion(
        make_evidence(),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_folds_mismatch" in codes(decision)


def test_champion_fold_timestamps_are_part_of_the_identity() -> None:
    champion = make_evidence(
        spec_hash=CHAMPION_HASH,
        folds=tuple(fold(index, ts_shift=9) for index in (1, 2, 3)),
    )
    decision = evaluate_promotion(
        make_evidence(),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_folds_mismatch" in codes(decision)


def test_champion_purge_embargo_are_part_of_the_identity() -> None:
    champion = make_evidence(
        spec_hash=CHAMPION_HASH,
        folds=tuple(fold(index, purge_bars=4, embargo_bars=2) for index in (1, 2, 3)),
    )
    decision = evaluate_promotion(
        make_evidence(),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_folds_mismatch" in codes(decision)


def test_champion_must_use_the_same_cost_stress_set() -> None:
    champion = make_evidence(
        spec_hash=CHAMPION_HASH,
        policy=replace(evidence_policy(), cost_stress=(1.5,)),
    )
    decision = evaluate_promotion(
        make_evidence(),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_cost_stress_mismatch" in codes(decision)


def test_champion_must_use_the_same_bootstrap_settings() -> None:
    folds = default_folds()
    champion = make_evidence(
        spec_hash=CHAMPION_HASH,
        folds=folds,
        aggregate=aggregate_for(folds, bootstrap_samples=800),
    )
    decision = evaluate_promotion(
        make_evidence(),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_bootstrap_mismatch" in codes(decision)


def test_champion_must_use_the_same_observation_window() -> None:
    champion = make_evidence(
        spec_hash=CHAMPION_HASH,
        policy=replace(evidence_policy(), window=30),
    )
    decision = evaluate_promotion(
        make_evidence(),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_window_mismatch" in codes(decision)


def test_champion_bundle_must_match_the_named_spec() -> None:
    champion = make_evidence(spec_hash=CHAMPION_HASH)
    decision = evaluate_promotion(
        make_evidence(),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash="e" * 64,
    )

    assert "champion_spec_mismatch" in codes(decision)


def test_a_champion_that_is_the_same_spec_is_rejected() -> None:
    champion = make_evidence()
    decision = evaluate_promotion(
        make_evidence(),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=SPEC_HASH,
    )

    assert "champion_same_spec" in codes(decision)


def test_champion_named_without_evidence_is_rejected() -> None:
    decision = evaluate_promotion(make_evidence(), TEST_POLICY, champion_spec_hash=CHAMPION_HASH)
    assert "champion_evidence_missing" in codes(decision)


def test_champion_evidence_without_a_hash_is_rejected() -> None:
    decision = evaluate_promotion(
        make_evidence(), TEST_POLICY, champion_evidence=make_evidence(spec_hash=CHAMPION_HASH)
    )
    assert "champion_identity_missing" in codes(decision)


def test_a_champion_without_comparable_identity_is_incomparable() -> None:
    champion = make_evidence(spec_hash=CHAMPION_HASH, settings_digest="")
    decision = evaluate_promotion(
        make_evidence(),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "incomparable_evidence" in codes(decision)
    assert "champion.settings_digest" in str(decision.rejections[-1].observed)


def test_a_candidate_without_settings_is_incomparable_when_a_champion_exists() -> None:
    champion = make_evidence(spec_hash=CHAMPION_HASH)
    decision = evaluate_promotion(
        make_evidence(settings_digest=""),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "incomparable_evidence" in codes(decision)
    assert "candidate.settings_digest" in str(decision.rejections[-1].observed)


def test_an_edited_champion_bundle_is_rejected() -> None:
    champion = replace(make_evidence(spec_hash=CHAMPION_HASH), checksum="f" * 64)
    decision = evaluate_promotion(
        make_evidence(),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_checksum_mismatch" in codes(decision)


def test_champion_without_an_aggregate_is_an_error_not_a_crash() -> None:
    champion = make_evidence(spec_hash=CHAMPION_HASH, aggregate=None)
    decision = evaluate_promotion(
        make_evidence(),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_invalid_evidence" in codes(decision)


def test_improvement_over_the_champion_is_strict() -> None:
    folds = default_folds()
    champion = make_evidence(
        spec_hash=CHAMPION_HASH,
        folds=folds,
        aggregate=aggregate_for(folds, expectancy_r=0.20),
    )

    at_boundary = evaluate_promotion(
        make_evidence(spec_hash=SPEC_HASH),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )
    assert at_boundary.accepted is True, at_boundary.codes

    below = evaluate_promotion(
        make_evidence(
            spec_hash=SPEC_HASH,
            folds=folds,
            aggregate=aggregate_for(folds, expectancy_r=0.24),
        ),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )
    assert "champion_improvement" in codes(below)

    equal = evaluate_promotion(
        make_evidence(
            spec_hash=SPEC_HASH,
            folds=folds,
            aggregate=aggregate_for(folds, expectancy_r=0.20),
        ),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )
    assert "champion_improvement" in codes(equal)


# --- manual override ----------------------------------------------------------


def rejected_bundle() -> CandidateEvidence:
    folds = tuple(fold(index, trades=5) for index in (1, 2, 3))
    return make_evidence(folds=folds)


def test_manual_override_keeps_the_original_rejections() -> None:
    bundle = rejected_bundle()
    override = manual_override(
        bundle,
        TEST_POLICY,
        reason="owner accepts the risk for a paper-only run",
        owner="resul",
    )

    assert override.policy_id == "manual-override"
    assert override.original_policy_id == TEST_POLICY.policy_id
    assert override.original_policy_version == TEST_POLICY.version
    assert override.rejections
    assert {item.code for item in override.rejections} == set(
        codes(evaluate_promotion(bundle, TEST_POLICY))
    )
    assert override.evidence_checksum == bundle.checksum
    assert override.dataset_id == DATASET_ID
    assert override.policy_digest == TEST_POLICY.digest
    assert override.is_policy_evidence is False
    assert not hasattr(override, "accepted")
    assert "not validation evidence" in override.note


@pytest.mark.parametrize(
    ("reason", "owner"),
    [("", "resul"), ("   ", "resul"), ("owner override", ""), ("owner override", "   ")],
)
def test_manual_override_requires_a_reason_and_owner(reason: str, owner: str) -> None:
    with pytest.raises(ValueError):
        manual_override(rejected_bundle(), TEST_POLICY, reason=reason, owner=owner)


def test_manual_override_is_refused_when_the_policy_already_accepts() -> None:
    with pytest.raises(ValueError, match="already satisfies"):
        manual_override(
            make_evidence(),
            TEST_POLICY,
            reason="nothing to override",
            owner="resul",
        )


# --- archive integration ------------------------------------------------------


def archive_with_candidate(
    tmp_path: Any,
    *,
    nodes: tuple[int, ...] = (1, 2),
    accepted: bool = True,
) -> tuple[DiscoveryArchive, ToyCandidate]:
    archive = DiscoveryArchive(tmp_path / "discovery.sqlite")
    candidate = ToyCandidate(nodes)
    archive.add_candidate(candidate, code_version=CODE_VERSION)
    archive.record_trial(
        candidate.canonical_hash(),
        TrialResult(
            metrics={"expectancy_r": 0.25, "trades": 75.0},
            accepted=accepted,
            rejections=() if accepted else ("below_threshold",),
        ),
        dataset_id=DATASET_ID,
        code_version=CODE_VERSION,
        seed=0,
        dataset_checksum=DATASET_CHECKSUM,
        settings_digest=SETTINGS_DIGEST,
    )
    return archive, candidate


def candidate_evidence(candidate: ToyCandidate, **overrides: Any) -> CandidateEvidence:
    return make_evidence(spec_hash=candidate.canonical_hash(), **overrides)


def test_first_evidence_promotion_is_recorded_without_a_baseline(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path)
    evidence = candidate_evidence(candidate)

    champion = promote_for_tests(
        archive,
        candidate.canonical_hash(),
        evidence,
        TEST_POLICY,
        scope=SCOPE,
        reason="passed the test policy on the validation folds",
        expected_identity=IDENTITY,
    )

    assert champion.mode == "evidence"
    assert champion.evidence_checksum == evidence.checksum
    assert champion.candidate.candidate_id == candidate.canonical_hash()
    assert "test-policy v1" in champion.reason

    history = archive.champion_history(SCOPE)
    assert len(history) == 1
    assert history[0].mode == "evidence"
    assert history[0].evidence_checksum == evidence.checksum
    assert history[0].metric is None
    assert history[0].value is None

    recorded = archive.decisions(candidate.canonical_hash())[-1]
    assert recorded.decision == "promoted"
    detail = recorded.detail
    assert detail is not None
    assert detail["mode"] == "evidence"
    assert detail["comparison"] == "first"
    assert detail["champion_candidate_id"] is None
    assert detail["policy_id"] == TEST_POLICY.policy_id
    assert detail["policy_version"] == TEST_POLICY.version
    assert detail["policy_digest"] == TEST_POLICY.digest
    assert detail["evidence_checksum"] == evidence.checksum
    assert detail["dataset_checksum"] == DATASET_CHECKSUM
    assert detail["settings_digest"] == SETTINGS_DIGEST
    assert detail["code_version"] == CODE_VERSION

    stored = archive.evidence_record(evidence.checksum)
    assert stored is not None
    assert stored.candidate_id == candidate.canonical_hash()
    restored = evidence_from_dict(stored.payload)
    assert restored == evidence
    assert evidence_checksum(restored) == evidence.checksum
    assert archive.champion_evidence(SCOPE) == stored


def test_second_promotion_uses_the_stored_champion_evidence(tmp_path: Any) -> None:
    archive, first = archive_with_candidate(tmp_path)
    first_evidence = candidate_evidence(first)
    promote_for_tests(
        archive,
        first.canonical_hash(),
        first_evidence,
        TEST_POLICY,
        scope=SCOPE,
        reason="first champion",
        expected_identity=IDENTITY,
    )

    second = ToyCandidate((1, 2, 3))
    archive.add_candidate(second, code_version=CODE_VERSION)
    archive.record_trial(
        second.canonical_hash(),
        TrialResult(metrics={"expectancy_r": 0.35, "trades": 75.0}, accepted=True),
        dataset_id=DATASET_ID,
        code_version=CODE_VERSION,
        seed=0,
        dataset_checksum=DATASET_CHECKSUM,
        settings_digest=SETTINGS_DIGEST,
    )
    folds = default_folds()
    second_evidence = candidate_evidence(
        second,
        folds=folds,
        aggregate=aggregate_for(folds, expectancy_r=0.35),
    )

    champion = promote_for_tests(
        archive,
        second.canonical_hash(),
        second_evidence,
        TEST_POLICY,
        scope=SCOPE,
        reason="beats the stored champion evidence",
        expected_identity=IDENTITY,
    )

    assert champion.candidate.candidate_id == second.canonical_hash()
    history = archive.champion_history(SCOPE)
    assert [row.mode for row in history] == ["evidence", "evidence"]
    assert history[1].evidence_checksum == second_evidence.checksum
    assert history[1].previous_candidate_id == first.canonical_hash()


def test_second_promotion_without_champion_evidence_is_refused(tmp_path: Any) -> None:
    archive, first = archive_with_candidate(tmp_path)
    promote_for_tests(
        archive,
        first.canonical_hash(),
        candidate_evidence(first),
        TEST_POLICY,
        scope=SCOPE,
        reason="first evidence champion",
        expected_identity=IDENTITY,
    )
    # An evidence-mode champion whose stored evidence pointer is gone must not be
    # silently treated as a first promotion.
    archive.connection.execute("UPDATE champions SET evidence_checksum = NULL")

    second = ToyCandidate((3, 4))
    archive.add_candidate(second, code_version=CODE_VERSION)
    archive.record_trial(
        second.canonical_hash(),
        TrialResult(metrics={"expectancy_r": 0.35, "trades": 75.0}, accepted=True),
        dataset_id=DATASET_ID,
        code_version=CODE_VERSION,
        seed=0,
        dataset_checksum=DATASET_CHECKSUM,
        settings_digest=SETTINGS_DIGEST,
    )
    folds = default_folds()
    evidence = candidate_evidence(
        second,
        folds=folds,
        aggregate=aggregate_for(folds, expectancy_r=0.35),
    )

    with pytest.raises(PromotionRefused, match="champion_evidence_missing"):
        promote_for_tests(
            archive,
            second.canonical_hash(),
            evidence,
            TEST_POLICY,
            scope=SCOPE,
            reason="cannot compare without the baseline",
            expected_identity=IDENTITY,
        )

    assert archive.champion(SCOPE).candidate.candidate_id == first.canonical_hash()  # type: ignore[union-attr]


def test_a_stale_expected_identity_refuses_promotion(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path)
    evidence = candidate_evidence(candidate)

    with pytest.raises(PromotionRefused, match="stale_code_version"):
        promote_for_tests(
            archive,
            candidate.canonical_hash(),
            evidence,
            TEST_POLICY,
            scope=SCOPE,
            reason="stale build",
            expected_identity=EvidenceIdentity("other-code", DATASET_CHECKSUM, SETTINGS_DIGEST),
        )

    assert archive.champion(SCOPE) is None


def test_archive_refuses_a_rejected_policy(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path)
    evidence = candidate_evidence(
        candidate, folds=tuple(fold(index, trades=5) for index in (1, 2, 3))
    )

    with pytest.raises(PromotionRefused):
        promote_for_tests(
            archive,
            candidate.canonical_hash(),
            evidence,
            TEST_POLICY,
            scope=SCOPE,
            reason="should not pass",
            expected_identity=IDENTITY,
        )

    assert archive.champion(SCOPE) is None
    assert archive.champion_history(SCOPE) == ()
    assert archive.decisions() == ()


def test_archive_refuses_evidence_for_another_candidate(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path)
    other = ToyCandidate((3, 4))
    archive.add_candidate(other, code_version=CODE_VERSION)
    evidence = candidate_evidence(candidate)

    with pytest.raises(PromotionRefused, match="detached promotion"):
        promote_for_tests(
            archive,
            other.canonical_hash(),
            evidence,
            TEST_POLICY,
            scope=SCOPE,
            reason="wrong candidate",
            expected_identity=IDENTITY,
        )


def test_archive_requires_an_accepted_trial(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path, accepted=False)
    evidence = candidate_evidence(candidate)

    with pytest.raises(ArchiveError, match="no_accepted_trial"):
        promote_for_tests(
            archive,
            candidate.canonical_hash(),
            evidence,
            TEST_POLICY,
            scope=SCOPE,
            reason="no recorded acceptance",
            expected_identity=IDENTITY,
        )

    assert archive.champion(SCOPE) is None


def test_archive_requires_a_trial_at_all(tmp_path: Any) -> None:
    archive = DiscoveryArchive(tmp_path / "discovery.sqlite")
    candidate = ToyCandidate((5,))
    archive.add_candidate(candidate, code_version=CODE_VERSION)
    evidence = candidate_evidence(candidate)

    with pytest.raises(ArchiveError, match="no_accepted_trial"):
        promote_for_tests(
            archive,
            candidate.canonical_hash(),
            evidence,
            TEST_POLICY,
            scope=SCOPE,
            reason="untested candidate",
            expected_identity=IDENTITY,
        )


def test_archive_scope_must_match_the_dataset(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path)

    with pytest.raises(ArchiveError, match="scope"):
        promote_for_tests(
            archive,
            candidate.canonical_hash(),
            candidate_evidence(candidate),
            TEST_POLICY,
            scope="XAGUSD:M1",
            reason="wrong market",
            expected_identity=IDENTITY,
        )


def test_exploratory_promotion_is_marked_and_refused_by_the_provider(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path)
    champion = archive.promote_exploratory(
        candidate.canonical_hash(),
        scope=SCOPE,
        reason="research ablation only",
    )

    assert champion.mode == "exploratory"
    assert archive.champion_history(SCOPE)[0].mode == "exploratory"
    with pytest.raises(NoEvidenceChampion, match="exploratory"):
        ChampionProvider(archive, scope=SCOPE).champion()


def test_manual_override_promotes_as_manual_and_is_not_paper_evidence(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path, accepted=False)
    evidence = candidate_evidence(
        candidate, folds=tuple(fold(index, trades=5) for index in (1, 2, 3))
    )
    override = manual_override(
        evidence,
        TEST_POLICY,
        reason="paper-only risk acceptance",
        owner="resul",
    )

    champion = promote_manual_override(
        archive,
        candidate.canonical_hash(),
        override,
        scope=SCOPE,
        reason="owner override",
    )

    assert champion.mode == "manual"
    assert archive.champion_history(SCOPE)[-1].mode == "manual"
    with pytest.raises(NoEvidenceChampion, match="manual"):
        ChampionProvider(archive, scope=SCOPE).champion()


def test_manual_override_cannot_be_passed_to_the_evidence_api(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path, accepted=False)
    override = manual_override(
        candidate_evidence(candidate, folds=tuple(fold(index, trades=5) for index in (1, 2, 3))),
        TEST_POLICY,
        reason="paper-only risk acceptance",
        owner="resul",
    )

    with pytest.raises(TypeError):
        promote_for_tests(
            archive,
            candidate.canonical_hash(),
            override,  # type: ignore[arg-type]
            TEST_POLICY,
            scope=SCOPE,
            reason="not evidence",
            expected_identity=IDENTITY,
        )


def test_rollback_preserves_the_restored_mode(tmp_path: Any) -> None:
    archive, first = archive_with_candidate(tmp_path)
    first_evidence = candidate_evidence(first)
    promote_for_tests(
        archive,
        first.canonical_hash(),
        first_evidence,
        TEST_POLICY,
        scope=SCOPE,
        reason="evidence champion",
        expected_identity=IDENTITY,
    )
    second = ToyCandidate((2, 1))
    archive.add_candidate(second, code_version=CODE_VERSION)
    archive.record_trial(
        second.canonical_hash(),
        TrialResult(metrics={"expectancy_r": 0.9, "trades": 75.0}, accepted=True),
        dataset_id=DATASET_ID,
        code_version=CODE_VERSION,
        seed=0,
        dataset_checksum=DATASET_CHECKSUM,
        settings_digest=SETTINGS_DIGEST,
    )
    archive.promote_exploratory(second.canonical_hash(), scope=SCOPE, reason="research")

    restored = archive.rollback(SCOPE, to_candidate_id=first.canonical_hash(), reason="rollback")

    assert restored.mode == "evidence"
    assert restored.evidence_checksum == first_evidence.checksum
    assert archive.champion_history(SCOPE)[-1].mode == "evidence"
    assert archive.champion_evidence(SCOPE) is not None


def test_archive_evidence_schema_and_modes(tmp_path: Any) -> None:
    archive = DiscoveryArchive(tmp_path / "discovery.sqlite")
    assert archive.schema_version == SCHEMA_VERSION == 4
    tables = {
        row[0]
        for row in archive.connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert "evidence" in tables


def test_a_direct_private_archive_call_with_a_forged_payload_is_refused(
    tmp_path: Any,
) -> None:
    archive, candidate = archive_with_candidate(tmp_path)
    evidence = candidate_evidence(candidate)

    from scalping.discovery._evidence_auth import issue

    payload = {
        "candidate_id": candidate.canonical_hash(),
        "evidence_checksum": evidence.checksum,
        "policy_digest": TEST_POLICY.digest,
        "dataset_id": DATASET_ID,
        "dataset_checksum": DATASET_CHECKSUM,
        "settings_digest": SETTINGS_DIGEST,
        "code_version": CODE_VERSION,
        "champion_candidate_id": None,
    }
    with pytest.raises(ArchiveError, match="cannot be rebuilt"):
        archive._promote_evidence(
            candidate.canonical_hash(),
            scope=SCOPE,
            reason="forged payload",
            policy_id=TEST_POLICY.policy_id,
            policy_version=TEST_POLICY.version,
            policy_digest=TEST_POLICY.digest,
            evidence_checksum=evidence.checksum,
            dataset_id=DATASET_ID,
            dataset_checksum=DATASET_CHECKSUM,
            settings_digest=SETTINGS_DIGEST,
            code_version=CODE_VERSION,
            candidate_spec_hash=candidate.canonical_hash(),
            evidence_payload={"not": "an evidence bundle"},
            champion_candidate_id=None,
            authorization=issue(payload),
            authorization_payload=payload,
            policy=TEST_POLICY,
            champion_evidence=None,
            expected_identity=IDENTITY,
            require_production=False,
        )
    assert archive.champion(SCOPE) is None


def test_a_direct_private_archive_call_with_a_forged_capability_is_refused(
    tmp_path: Any,
) -> None:
    archive, candidate = archive_with_candidate(tmp_path)
    evidence = candidate_evidence(candidate)
    payload = {
        "candidate_id": candidate.canonical_hash(),
        "evidence_checksum": evidence.checksum,
        "policy_digest": TEST_POLICY.digest,
        "dataset_id": DATASET_ID,
        "dataset_checksum": DATASET_CHECKSUM,
        "settings_digest": SETTINGS_DIGEST,
        "code_version": CODE_VERSION,
        "champion_candidate_id": None,
    }

    with pytest.raises(ArchiveError, match="unauthorized"):
        archive._promote_evidence(
            candidate.canonical_hash(),
            scope=SCOPE,
            reason="forged capability",
            policy_id=TEST_POLICY.policy_id,
            policy_version=TEST_POLICY.version,
            policy_digest=TEST_POLICY.digest,
            evidence_checksum=evidence.checksum,
            dataset_id=DATASET_ID,
            dataset_checksum=DATASET_CHECKSUM,
            settings_digest=SETTINGS_DIGEST,
            code_version=CODE_VERSION,
            candidate_spec_hash=candidate.canonical_hash(),
            evidence_payload=evidence.to_dict(),
            champion_candidate_id=None,
            authorization="0" * 64,
            authorization_payload=payload,
            policy=TEST_POLICY,
            champion_evidence=None,
            expected_identity=IDENTITY,
            require_production=False,
        )
    assert archive.champion(SCOPE) is None


def test_a_direct_private_archive_call_without_a_capability_is_refused(
    tmp_path: Any,
) -> None:
    archive, candidate = archive_with_candidate(tmp_path)
    evidence = candidate_evidence(candidate)

    with pytest.raises(TypeError, match="authorization"):
        archive._promote_evidence(
            candidate.canonical_hash(),
            scope=SCOPE,
            reason="no capability",
            policy_id=TEST_POLICY.policy_id,
            policy_version=TEST_POLICY.version,
            policy_digest=TEST_POLICY.digest,
            evidence_checksum=evidence.checksum,
            dataset_id=DATASET_ID,
            dataset_checksum=DATASET_CHECKSUM,
            settings_digest=SETTINGS_DIGEST,
            code_version=CODE_VERSION,
            candidate_spec_hash=candidate.canonical_hash(),
            evidence_payload=evidence.to_dict(),
            champion_candidate_id=None,
            policy=TEST_POLICY,
            champion_evidence=None,
            expected_identity=IDENTITY,
            require_production=False,
        )


def test_a_trial_without_the_evidence_checksum_is_stale(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path)
    archive.record_trial(
        candidate.canonical_hash(),
        TrialResult(metrics={"expectancy_r": 0.3, "trades": 75.0}, accepted=True),
        dataset_id=DATASET_ID,
        code_version=CODE_VERSION,
        seed=1,
        dataset_checksum="e" * 64,
        settings_digest=SETTINGS_DIGEST,
    )

    with pytest.raises(ArchiveError, match="stale_trial_evidence"):
        promote_for_tests(
            archive,
            candidate.canonical_hash(),
            candidate_evidence(candidate),
            TEST_POLICY,
            scope=SCOPE,
            reason="stale trial",
            expected_identity=IDENTITY,
        )
    assert archive.champion(SCOPE) is None


def test_a_trial_without_digests_is_stale(tmp_path: Any) -> None:
    archive = DiscoveryArchive(tmp_path / "discovery.sqlite")
    candidate = ToyCandidate((9,))
    archive.add_candidate(candidate, code_version=CODE_VERSION)
    archive.record_trial(
        candidate.canonical_hash(),
        TrialResult(metrics={"expectancy_r": 0.3, "trades": 75.0}, accepted=True),
        dataset_id=DATASET_ID,
        code_version=CODE_VERSION,
        seed=0,
    )

    with pytest.raises(ArchiveError, match="stale_trial_evidence"):
        promote_for_tests(
            archive,
            candidate.canonical_hash(),
            candidate_evidence(candidate),
            TEST_POLICY,
            scope=SCOPE,
            reason="legacy trial",
            expected_identity=IDENTITY,
        )


def test_an_exploratory_champion_is_replaced_as_an_audited_transition(tmp_path: Any) -> None:
    archive, first = archive_with_candidate(tmp_path)
    archive.promote_exploratory(first.canonical_hash(), scope=SCOPE, reason="legacy research")
    previous = archive.champion(SCOPE)
    assert previous is not None and previous.mode == "exploratory"

    second = ToyCandidate((3, 1))
    archive.add_candidate(second, code_version=CODE_VERSION)
    archive.record_trial(
        second.canonical_hash(),
        TrialResult(metrics={"expectancy_r": 0.35, "trades": 75.0}, accepted=True),
        dataset_id=DATASET_ID,
        code_version=CODE_VERSION,
        seed=0,
        dataset_checksum=DATASET_CHECKSUM,
        settings_digest=SETTINGS_DIGEST,
    )
    folds = default_folds()
    evidence = candidate_evidence(
        second,
        folds=folds,
        aggregate=aggregate_for(folds, expectancy_r=0.35),
    )

    champion = promote_for_tests(
        archive,
        second.canonical_hash(),
        evidence,
        TEST_POLICY,
        scope=SCOPE,
        reason="first evidence champion after research mode",
        expected_identity=IDENTITY,
    )

    assert champion.mode == "evidence"
    assert champion.previous_candidate_id == first.canonical_hash()
    detail = archive.decisions(second.canonical_hash())[-1].detail
    assert detail is not None
    assert detail["comparison"] == "replaces_non_evidence"
    assert detail["replaces_non_evidence"]["previous_mode"] == "exploratory"
    assert detail["replaces_non_evidence"]["previous_candidate_id"] == first.canonical_hash()
    assert detail["replaces_non_evidence"]["previous_reason"] == previous.reason


def test_the_provider_refuses_a_champion_from_an_unregistered_policy(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path)
    promote_for_tests(
        archive,
        candidate.canonical_hash(),
        candidate_evidence(candidate),
        TEST_POLICY,
        scope=SCOPE,
        reason="not registered in production",
        expected_identity=IDENTITY,
    )

    with pytest.raises(NoEvidenceChampion, match="allow-list"):
        ChampionProvider(archive, scope=SCOPE).champion()


def test_provider_rejects_a_stored_policy_id_version_mismatch(tmp_path: Any) -> None:
    archive, candidate = archive_with_candidate(tmp_path)
    with registered_test_policies(TEST_POLICY):
        promote_for_tests(
            archive,
            candidate.canonical_hash(),
            candidate_evidence(candidate),
            TEST_POLICY,
            scope=SCOPE,
            reason="registered policy",
            expected_identity=IDENTITY,
        )
        archive.connection.execute("UPDATE champions SET policy_version = 99")

        with pytest.raises(NoEvidenceChampion, match="id/version"):
            ChampionProvider(archive, scope=SCOPE).champion()


def test_the_test_registry_is_cleaned_up() -> None:
    assert TEST_POLICY.digest not in PRODUCTION_POLICIES  # type: ignore[operator]
    with registered_test_policies(TEST_POLICY):
        assert TEST_POLICY.digest in PRODUCTION_POLICIES
    assert TEST_POLICY.digest not in PRODUCTION_POLICIES


def test_provider_accepts_a_real_spec_promoted_with_evidence(
    tmp_path: Any, monkeypatch: Any
) -> None:
    spec = StrategySpec(
        entries=(EntryRule("long", Compare(Const(1.0), ">", Const(0.5))),),
        exits=(ExitRule(Compare(Const(0.0), ">", Const(1.0))),),
        holding=HoldingLimit(10),
        max_lookback=2,
    )
    archive = DiscoveryArchive(tmp_path / "discovery.sqlite")
    candidate_id = archive.add_candidate(spec, code_version=CODE_VERSION).candidate_id
    archive.record_trial(
        candidate_id,
        TrialResult(metrics={"expectancy_r": 0.3, "trades": 70.0}, accepted=True),
        dataset_id=DATASET_ID,
        code_version=CODE_VERSION,
        seed=0,
        dataset_checksum=DATASET_CHECKSUM,
        settings_digest=SETTINGS_DIGEST,
    )
    evidence = make_evidence(spec_hash=candidate_id)

    with registered_test_policies(TEST_POLICY):
        promote_for_tests(
            archive,
            candidate_id,
            evidence,
            TEST_POLICY,
            scope=SCOPE,
            reason="evidence champion",
            expected_identity=IDENTITY,
        )
        champion = ChampionProvider(archive, scope=SCOPE).champion()

    assert champion.mode == "evidence"
    assert champion.evidence_checksum == evidence.checksum
    assert champion.spec.canonical_hash() == candidate_id
