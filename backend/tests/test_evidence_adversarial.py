"""Adversarial tests for the evidence -> promotion -> holdout -> reproduce chain.

Every test here attacks a seam with a malformed, forged or self-inconsistent
input. Passing the happy path is covered by ``test_discovery_evidence.py``,
``test_promotion_policy.py`` and ``test_locked_holdout.py``; this file asks only
one question per test: does the gate refuse the input with a stable code before a
metric is read?

The bundles are hand-built (except the reproduce cases, which replay a real
``build_evidence`` run) so the adversarial shape is explicit and cheap. A
synthetic bundle proves the gate arithmetic, never an edge.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from discovery_fakes import ToyCandidate
from promotion_support import promote_for_tests, registered_test_policies
from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.datasets.artifact import save_dataset_arrays
from scalping.datasets.builder import DatasetBundle, build_dataset
from scalping.discovery._evidence_auth import issue as issue_authorization
from scalping.discovery.archive import ArchiveError, DiscoveryArchive
from scalping.discovery.champion import ChampionProvider, NoEvidenceChampion
from scalping.discovery.evidence import (
    AggregateEvidence,
    CandidateEvidence,
    EvidenceConfig,
    EvidenceError,
    EvidencePolicy,
    FoldEvidence,
    LockedOOSEvidence,
    RegimeEvidence,
    SplitBoundaries,
    StressEvidence,
    build_evidence,
    evidence_checksum,
    locked_oos_checksum,
    verify_evidence,
)
from scalping.discovery.promotion import (
    PRODUCTION_POLICIES,
    PROMOTION_V1,
    ExpectedIdentity,
    PromotionRefused,
    _base_evidence_checksum,
    evaluate_promotion,
    is_production_policy,
    manual_override,
    promote_manual_override,
    promote_with_evidence,
)
from scalping.discovery.protocols import TrialResult
from scalping.discovery.reproduce import (
    ReproduceMismatch,
    dataset_artifact_dir,
    reproduce_bundle,
    spec_sidecar_path,
)
from scalping.discovery.run import code_version as current_code_version
from scalping.features.build import FeatureFrame, build_features
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
CODE_VERSION = "test-adversarial-2026-11-24"
SCOPE = "XAUUSD:M1"
START = datetime(2026, 1, 5, tzinfo=UTC)
IDENTITY = ExpectedIdentity(
    code_version=CODE_VERSION,
    dataset_checksum=DATASET_CHECKSUM,
    settings_digest=SETTINGS_DIGEST,
)

#: A named non-production policy. It cannot be promoted through the public API.
TEST_POLICY = replace(
    PROMOTION_V1,
    policy_id="adversarial-test",
    require_locked_oos=False,
)
LOCKED_POLICY = replace(TEST_POLICY, policy_id="adversarial-locked", require_locked_oos=True)


def _ts(minutes: int) -> str:
    return (START + timedelta(minutes=minutes)).isoformat()


# --- synthetic bundle helpers -------------------------------------------------


def _fold(
    index: int,
    *,
    trades: int = 25,
    net_profit: float = 1.0,
    expectancy_r: float = 0.25,
) -> FoldEvidence:
    test_start = 100 + (index - 1) * 50
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
        profit_factor=1.5,
        win_rate=0.5,
        expectancy_r=expectancy_r,
        max_drawdown=0.05,
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
                profit_factor=1.5,
                max_drawdown=0.05,
                expectancy_r=expectancy_r,
            )
            for multiplier in (1.5, 2.0)
        ),
    )


def _splits() -> SplitBoundaries:
    return SplitBoundaries(
        train_start_row=0,
        train_end_row=200,
        validation_start_row=200,
        validation_end_row=250,
        test_start_row=250,
        test_end_row=300,
    )


def _aggregate(folds: tuple[FoldEvidence, ...], **overrides: Any) -> AggregateEvidence:
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


def _evidence_policy() -> EvidencePolicy:
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


def _make_evidence(
    *,
    folds: tuple[FoldEvidence, ...] | None = None,
    aggregate: AggregateEvidence | None = None,
    **overrides: Any,
) -> CandidateEvidence:
    items = (_fold(1), _fold(2), _fold(3)) if folds is None else folds
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
        "splits": _splits(),
        "policy": _evidence_policy(),
        "folds": items,
        "aggregate": _aggregate(items) if aggregate is None else aggregate,
        "checksum": "",
        "created_at": "2026-11-24T00:00:00+00:00",
    }
    values.update(overrides)
    draft = CandidateEvidence(**values)
    return replace(draft, checksum=evidence_checksum(draft))


def _reseal(evidence: CandidateEvidence, **changes: Any) -> CandidateEvidence:
    draft = replace(evidence, checksum="", **changes)
    return replace(draft, checksum=evidence_checksum(draft))


def _locked_section(
    evidence: CandidateEvidence,
    policy: Any,
    **overrides: Any,
) -> LockedOOSEvidence:
    splits = evidence.splits
    base = LockedOOSEvidence(
        checksum="0" * 64,
        evidence_checksum=_base_evidence_checksum(evidence),
        dataset_id=evidence.dataset_id,
        dataset_checksum=evidence.dataset_checksum,
        symbol=evidence.symbol,
        timeframe=evidence.timeframe,
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
        regimes=(
            RegimeEvidence(
                regime="trend_up",
                trades=75,
                net_profit=5.0,
                win_rate=0.5,
                expectancy_r=0.3,
                total_r=22.5,
            ),
        ),
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
    if overrides:
        base = replace(base, **overrides)
    return replace(base, checksum=locked_oos_checksum(base))


def _reseal_locked(section: LockedOOSEvidence, **changes: Any) -> LockedOOSEvidence:
    draft = replace(section, checksum="0" * 64, **changes)
    return replace(draft, checksum=locked_oos_checksum(draft))


def _force_section(section: LockedOOSEvidence, **changes: Any) -> LockedOOSEvidence:
    """Bypass ``__post_init__`` to model a field edited in stored JSON."""
    draft = replace(section, checksum="0" * 64)
    for name, value in changes.items():
        object.__setattr__(draft, name, value)
    object.__setattr__(draft, "checksum", locked_oos_checksum(draft))
    return draft


def _locked_bundle(
    section: LockedOOSEvidence, base: CandidateEvidence | None = None
) -> CandidateEvidence:
    return _reseal(_make_evidence() if base is None else base, locked_oos=section)


# --- archive helpers ----------------------------------------------------------


def _archive_with_candidate(
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


def _candidate_evidence(candidate: ToyCandidate, **overrides: Any) -> CandidateEvidence:
    return _make_evidence(spec_hash=candidate.canonical_hash(), **overrides)


def _add_trial(archive: DiscoveryArchive, candidate: ToyCandidate) -> None:
    archive.add_candidate(candidate, code_version=CODE_VERSION)
    archive.record_trial(
        candidate.canonical_hash(),
        TrialResult(metrics={"expectancy_r": 0.35, "trades": 75.0}, accepted=True),
        dataset_id=DATASET_ID,
        code_version=CODE_VERSION,
        seed=0,
        dataset_checksum=DATASET_CHECKSUM,
        settings_digest=SETTINGS_DIGEST,
    )


# --- 1. forged bundle objects -------------------------------------------------


def test_a_subclass_injecting_locked_oos_through_to_dict_is_rejected() -> None:
    base = _make_evidence()

    class ForgedEvidence(CandidateEvidence):
        def to_dict(self) -> dict[str, Any]:
            payload = super().to_dict()
            payload["locked_oos"] = {"checksum": "a" * 64, "trades": 75}
            return payload

    forged = ForgedEvidence(**{name: getattr(base, name) for name in base.__dataclass_fields__})

    decision = evaluate_promotion(forged, LOCKED_POLICY)  # type: ignore[arg-type]
    assert decision.accepted is False
    assert decision.codes == ("invalid_evidence",)

    for call in (
        lambda: evidence_checksum(forged),  # type: ignore[arg-type]
        lambda: verify_evidence(
            forged,  # type: ignore[arg-type]
            dataset_id=DATASET_ID,
            dataset_checksum=DATASET_CHECKSUM,
            spec_hash=SPEC_HASH,
            code_version=CODE_VERSION,
            settings_digest=SETTINGS_DIGEST,
        ),
    ):
        with pytest.raises(TypeError, match="CandidateEvidence"):
            call()


def test_a_plain_mapping_that_looks_like_a_bundle_is_rejected() -> None:
    decision = evaluate_promotion(_make_evidence().to_dict(), TEST_POLICY)  # type: ignore[arg-type]

    assert decision.accepted is False
    assert decision.codes == ("invalid_evidence",)


# --- 2. locked OOS section attacks --------------------------------------------


def test_a_forged_locked_checksum_is_rejected() -> None:
    base = _make_evidence()
    section = replace(_locked_section(base, LOCKED_POLICY), checksum="f" * 64)

    decision = evaluate_promotion(_locked_bundle(section, base), LOCKED_POLICY)

    assert decision.accepted is False
    assert "locked_oos_checksum" in decision.codes


def test_a_locked_section_bound_to_another_bundle_is_rejected() -> None:
    base = _make_evidence()
    section = _reseal_locked(_locked_section(base, LOCKED_POLICY), evidence_checksum="f" * 64)

    decision = evaluate_promotion(_locked_bundle(section, base), LOCKED_POLICY)

    assert "locked_oos_evidence_checksum" in decision.codes


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"test_rows": 10}, "locked_oos_geometry"),
        ({"test_start_row": 0}, "locked_oos_geometry"),
        ({"test_start_ts": _ts(399), "test_end_ts": _ts(300)}, "locked_oos_geometry"),
        ({"test_start_ts": "2026-01-01T00:00:00+00:00"}, "locked_oos_geometry"),
        ({"trades": 1}, "insufficient_trades"),
        ({"expectancy_r": -5.0}, "min_expectancy_r"),
        ({"profit_factor": 0.1}, "min_profit_factor"),
        ({"max_drawdown": 0.99}, "max_drawdown"),
        ({"regime_coverage": 0.1}, "min_regime_coverage"),
        ({"expectancy_r_ci95_low": 0.0}, "min_ci_lower_r"),
    ],
)
def test_a_tampered_locked_section_is_rejected_with_its_code(
    changes: dict[str, Any], code: str
) -> None:
    base = _make_evidence()
    section = _reseal_locked(_locked_section(base, LOCKED_POLICY), **changes)

    decision = evaluate_promotion(_locked_bundle(section, base), LOCKED_POLICY)

    assert decision.accepted is False
    assert code in decision.codes


def test_a_locked_section_with_zero_trades_cannot_be_constructed() -> None:
    base = _make_evidence()
    section = _locked_section(base, LOCKED_POLICY)

    with pytest.raises(ValueError, match="at least one trade"):
        replace(section, trades=0)


def test_a_forced_zero_trade_locked_section_is_rejected() -> None:
    base = _make_evidence()
    section = _force_section(_locked_section(base, LOCKED_POLICY), trades=0)

    decision = evaluate_promotion(_locked_bundle(section, base), LOCKED_POLICY)

    assert decision.accepted is False
    assert "insufficient_trades" in decision.codes


def test_a_catastrophic_locked_stress_rerun_is_rejected() -> None:
    base = _make_evidence()
    section = _locked_section(base, LOCKED_POLICY)
    bad = tuple(replace(row, expectancy_r=-5.0, profit_factor=0.1) for row in section.stress)
    section = _reseal_locked(section, stress=bad)

    decision = evaluate_promotion(_locked_bundle(section, base), LOCKED_POLICY)

    assert "cost_stress" in decision.codes
    assert "stress_degradation" in decision.codes


def test_a_locked_ci_exactly_at_zero_is_refused_and_a_hair_above_passes() -> None:
    base = _make_evidence()
    section = _locked_section(base, LOCKED_POLICY)

    at_zero = _reseal_locked(section, expectancy_r_ci95_low=0.0)
    refused = evaluate_promotion(_locked_bundle(at_zero, base), LOCKED_POLICY)
    assert "min_ci_lower_r" in refused.codes
    assert refused.accepted is False

    above = _reseal_locked(section, expectancy_r_ci95_low=1e-9)
    accepted = evaluate_promotion(_locked_bundle(above, base), LOCKED_POLICY)
    assert accepted.accepted is True, accepted.codes


# --- 3. policy registry and public API ----------------------------------------


def test_the_public_api_refuses_an_unregistered_test_policy(tmp_path: Any) -> None:
    archive, candidate = _archive_with_candidate(tmp_path)

    with pytest.raises(PromotionRefused, match="non_production_policy"):
        promote_with_evidence(
            archive,
            candidate.canonical_hash(),
            _candidate_evidence(candidate),
            TEST_POLICY,
            scope=SCOPE,
            reason="a test policy must not reach the production entry point",
            expected_identity=IDENTITY,
        )
    assert archive.champion(SCOPE) is None


def test_a_modified_copy_of_the_production_policy_is_not_production(tmp_path: Any) -> None:
    modified = replace(PROMOTION_V1, min_trades=PROMOTION_V1.min_trades + 1)
    assert is_production_policy(PROMOTION_V1) is True
    assert is_production_policy(modified) is False
    assert modified.digest != PROMOTION_V1.digest

    archive, candidate = _archive_with_candidate(tmp_path)
    with pytest.raises(PromotionRefused, match="non_production_policy"):
        promote_with_evidence(
            archive,
            candidate.canonical_hash(),
            _candidate_evidence(candidate),
            modified,
            scope=SCOPE,
            reason="an edited copy of the production policy is a different policy",
            expected_identity=IDENTITY,
        )
    assert archive.champion(SCOPE) is None


def test_the_production_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        PRODUCTION_POLICIES["forged"] = PROMOTION_V1  # type: ignore[index]
    with pytest.raises(TypeError):
        del PRODUCTION_POLICIES[PROMOTION_V1.digest]  # type: ignore[attr-defined]
    assert not hasattr(PRODUCTION_POLICIES, "update")
    assert PROMOTION_V1.digest in PRODUCTION_POLICIES


def test_the_manual_override_id_cannot_name_a_promotion_policy() -> None:
    with pytest.raises(ValueError, match="manual-override"):
        replace(PROMOTION_V1, policy_id="manual-override")


# --- 4. direct private archive promotion --------------------------------------


def _authorization_payload(
    candidate_id: str,
    evidence: CandidateEvidence,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "evidence_checksum": evidence.checksum,
        "policy_digest": TEST_POLICY.digest,
        "dataset_id": DATASET_ID,
        "dataset_checksum": DATASET_CHECKSUM,
        "settings_digest": SETTINGS_DIGEST,
        "code_version": CODE_VERSION,
        "champion_candidate_id": None,
    }


def _private_promotion(
    archive: DiscoveryArchive,
    candidate_id: str,
    evidence: CandidateEvidence,
    *,
    authorization: str | None,
    payload: dict[str, Any],
    evidence_payload: dict[str, Any] | None = None,
) -> None:
    kwargs: dict[str, Any] = {
        "scope": SCOPE,
        "reason": "direct private call",
        "policy_id": TEST_POLICY.policy_id,
        "policy_version": TEST_POLICY.version,
        "policy_digest": TEST_POLICY.digest,
        "evidence_checksum": evidence.checksum,
        "dataset_id": DATASET_ID,
        "dataset_checksum": DATASET_CHECKSUM,
        "settings_digest": SETTINGS_DIGEST,
        "code_version": CODE_VERSION,
        "candidate_spec_hash": candidate_id,
        "evidence_payload": (evidence.to_dict() if evidence_payload is None else evidence_payload),
        "champion_candidate_id": None,
        "policy": TEST_POLICY,
        "champion_evidence": None,
        "expected_identity": IDENTITY,
        "require_production": False,
    }
    if authorization is not None:
        kwargs["authorization"] = authorization
        kwargs["authorization_payload"] = payload
    archive._promote_evidence(candidate_id, **kwargs)


def test_a_direct_private_promotion_without_a_capability_is_refused(tmp_path: Any) -> None:
    archive, candidate = _archive_with_candidate(tmp_path)
    evidence = _candidate_evidence(candidate)
    payload = _authorization_payload(candidate.canonical_hash(), evidence)

    with pytest.raises(TypeError, match="authorization"):
        _private_promotion(
            archive, candidate.canonical_hash(), evidence, authorization=None, payload=payload
        )

    assert archive.champion(SCOPE) is None


def test_a_direct_private_promotion_with_a_forged_capability_is_refused(tmp_path: Any) -> None:
    archive, candidate = _archive_with_candidate(tmp_path)
    evidence = _candidate_evidence(candidate)
    payload = _authorization_payload(candidate.canonical_hash(), evidence)

    with pytest.raises(ArchiveError, match="unauthorized"):
        _private_promotion(
            archive,
            candidate.canonical_hash(),
            evidence,
            authorization="0" * 64,
            payload=payload,
        )

    assert archive.champion(SCOPE) is None


def test_a_valid_capability_does_not_excuse_a_tampered_payload(tmp_path: Any) -> None:
    archive, candidate = _archive_with_candidate(tmp_path)
    evidence = _candidate_evidence(candidate)
    payload = _authorization_payload(candidate.canonical_hash(), evidence)
    tampered = evidence.to_dict()
    tampered["aggregate"]["expectancy_r"] = 9.99
    # The payload is still shaped like evidence; only its numbers moved.
    with pytest.raises(ArchiveError, match="does not match the payload"):
        _private_promotion(
            archive,
            candidate.canonical_hash(),
            evidence,
            authorization=issue_authorization(payload),
            payload=payload,
            evidence_payload=tampered,
        )

    assert archive.champion(SCOPE) is None


def test_a_valid_capability_does_not_skip_the_policy(tmp_path: Any) -> None:
    archive, candidate = _archive_with_candidate(tmp_path)
    weak = _candidate_evidence(candidate, folds=tuple(_fold(i, trades=5) for i in (1, 2, 3)))
    payload = _authorization_payload(candidate.canonical_hash(), weak)

    with pytest.raises(ArchiveError, match="promotion_rejected"):
        _private_promotion(
            archive,
            candidate.canonical_hash(),
            weak,
            authorization=issue_authorization(payload),
            payload=payload,
        )

    assert archive.champion(SCOPE) is None


# --- 5. second promotion ------------------------------------------------------


def test_a_second_promotion_without_champion_evidence_is_refused(tmp_path: Any) -> None:
    archive, first = _archive_with_candidate(tmp_path)
    promote_for_tests(
        archive,
        first.canonical_hash(),
        _candidate_evidence(first),
        TEST_POLICY,
        scope=SCOPE,
        reason="first champion",
        expected_identity=IDENTITY,
    )
    archive.connection.execute("UPDATE champions SET evidence_checksum = NULL")

    second = ToyCandidate((3, 4))
    _add_trial(archive, second)
    folds = (_fold(1), _fold(2), _fold(3))
    evidence = _candidate_evidence(
        second, folds=folds, aggregate=_aggregate(folds, expectancy_r=0.35)
    )

    with pytest.raises(PromotionRefused, match="champion_evidence_missing"):
        promote_for_tests(
            archive,
            second.canonical_hash(),
            evidence,
            TEST_POLICY,
            scope=SCOPE,
            reason="no baseline to compare against",
            expected_identity=IDENTITY,
        )

    assert archive.champion(SCOPE).candidate.candidate_id == first.canonical_hash()  # type: ignore[union-attr]


def test_re_promoting_the_champion_itself_is_refused(tmp_path: Any) -> None:
    archive, candidate = _archive_with_candidate(tmp_path)
    evidence = _candidate_evidence(candidate)
    promote_for_tests(
        archive,
        candidate.canonical_hash(),
        evidence,
        TEST_POLICY,
        scope=SCOPE,
        reason="first champion",
        expected_identity=IDENTITY,
    )

    with pytest.raises(PromotionRefused, match="champion_same_spec"):
        promote_for_tests(
            archive,
            candidate.canonical_hash(),
            evidence,
            TEST_POLICY,
            scope=SCOPE,
            reason="same spec as the champion",
            expected_identity=IDENTITY,
        )

    assert archive.champion_history(SCOPE)[-1].mode == "evidence"


@pytest.mark.parametrize(
    ("champion_changes", "code"),
    [
        ({"dataset_id": "validation:XAGUSD:M1:ffff"}, "champion_dataset_mismatch"),
        ({"dataset_checksum": "e" * 64}, "champion_dataset_checksum_mismatch"),
        ({"settings_digest": "e" * 64}, "champion_settings_mismatch"),
        ({"code_version": "other-code"}, "champion_code_version_mismatch"),
    ],
)
def test_a_champion_from_another_experiment_is_not_comparable(
    champion_changes: dict[str, Any], code: str
) -> None:
    champion = _make_evidence(spec_hash=CHAMPION_HASH, **champion_changes)
    decision = evaluate_promotion(
        _make_evidence(),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert decision.accepted is False
    assert code in decision.codes


def test_a_champion_on_another_fold_geometry_is_not_comparable() -> None:
    shifted = replace(_fold(3), test_start_row=201, test_end_row=251)
    champion = _make_evidence(
        spec_hash=CHAMPION_HASH,
        folds=(_fold(1), _fold(2), shifted),
    )

    decision = evaluate_promotion(
        _make_evidence(),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert decision.accepted is False
    assert "champion_folds_mismatch" in decision.codes


def test_a_champion_bundle_with_a_forged_checksum_is_not_a_baseline() -> None:
    champion = replace(_make_evidence(spec_hash=CHAMPION_HASH), checksum="f" * 64)

    decision = evaluate_promotion(
        _make_evidence(),
        TEST_POLICY,
        champion_evidence=champion,
        champion_spec_hash=CHAMPION_HASH,
    )

    assert "champion_checksum_mismatch" in decision.codes


# --- 6. exploratory and manual champions --------------------------------------


def test_an_exploratory_champion_is_refused_by_the_provider(tmp_path: Any) -> None:
    archive, candidate = _archive_with_candidate(tmp_path)
    archive.promote_exploratory(candidate.canonical_hash(), scope=SCOPE, reason="research only")

    with pytest.raises(NoEvidenceChampion, match="exploratory"):
        ChampionProvider(archive, scope=SCOPE).champion()


def test_a_manual_champion_is_refused_by_the_provider(tmp_path: Any) -> None:
    archive, candidate = _archive_with_candidate(tmp_path, accepted=False)
    rejected = _candidate_evidence(
        candidate,
        folds=tuple(_fold(i, trades=5) for i in (1, 2, 3)),
    )
    override = manual_override(
        rejected,
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
    with pytest.raises(NoEvidenceChampion, match="manual"):
        ChampionProvider(archive, scope=SCOPE).champion()


def test_replacing_a_legacy_champion_is_an_audited_transition(tmp_path: Any) -> None:
    legacy_spec = StrategySpec(
        entries=(EntryRule("long", Compare(Const(1.0), ">", Const(0.5))),),
        exits=(ExitRule(Compare(Const(0.0), ">", Const(1.0))),),
        holding=HoldingLimit(10),
        max_lookback=2,
    )
    candidate_spec = replace(legacy_spec, holding=HoldingLimit(11))

    archive = DiscoveryArchive(tmp_path / "discovery.sqlite")
    created = archive.add_candidate(legacy_spec, code_version=CODE_VERSION)
    legacy_id = created.candidate_id
    archive.record_trial(
        legacy_id,
        TrialResult(metrics={"expectancy_r": 0.2, "trades": 75.0}, accepted=True),
        dataset_id=DATASET_ID,
        code_version=CODE_VERSION,
        seed=0,
        dataset_checksum=DATASET_CHECKSUM,
        settings_digest=SETTINGS_DIGEST,
    )
    archive.promote_exploratory(legacy_id, scope=SCOPE, reason="legacy research")

    candidate_id = archive.add_candidate(candidate_spec, code_version=CODE_VERSION).candidate_id
    archive.record_trial(
        candidate_id,
        TrialResult(metrics={"expectancy_r": 0.35, "trades": 75.0}, accepted=True),
        dataset_id=DATASET_ID,
        code_version=CODE_VERSION,
        seed=0,
        dataset_checksum=DATASET_CHECKSUM,
        settings_digest=SETTINGS_DIGEST,
    )
    folds = (_fold(1), _fold(2), _fold(3))
    evidence = _make_evidence(
        spec_hash=candidate_id,
        folds=folds,
        aggregate=_aggregate(folds, expectancy_r=0.35),
    )

    with registered_test_policies(TEST_POLICY):
        champion = promote_for_tests(
            archive,
            candidate_id,
            evidence,
            TEST_POLICY,
            scope=SCOPE,
            reason="first governed evidence champion",
            expected_identity=IDENTITY,
        )
        paper = ChampionProvider(archive, scope=SCOPE).champion()

    assert champion.mode == "evidence"
    detail = archive.decisions(candidate_id)[-1].detail
    assert detail is not None
    assert detail["comparison"] == "replaces_non_evidence"
    assert detail["replaces_non_evidence"]["previous_mode"] == "exploratory"
    assert detail["replaces_non_evidence"]["previous_candidate_id"] == legacy_id
    assert paper.mode == "evidence"
    assert paper.evidence_checksum == evidence.checksum


# --- 7. reproduction ----------------------------------------------------------

REPRO_START = datetime(2026, 1, 5, tzinfo=UTC)
REPRO_ROWS = 1600

_ALWAYS = Compare(Const(0.0), "<", Const(1.0))
_NEVER = Compare(Const(0.0), ">", Const(1.0))


def _flat_bars(rows: int = REPRO_ROWS) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [REPRO_START + timedelta(minutes=index) for index in range(rows)],
            "open": [1000.0] * rows,
            "high": [1000.001] * rows,
            "low": [999.999] * rows,
            "close": [1000.0] * rows,
            "bid_close": [1000.0] * rows,
            "ask_close": [1000.2] * rows,
            "spread": [0.2] * rows,
            "volume": [10] * rows,
        }
    )


def _repro_spec() -> StrategySpec:
    return StrategySpec(
        entries=(EntryRule("long", _ALWAYS),),
        exits=(ExitRule(_NEVER),),
        holding=HoldingLimit(1),
        max_lookback=5,
    )


def _repro_config(**overrides: Any) -> EvidenceConfig:
    values: dict[str, Any] = {
        "code_version": current_code_version(),
        "symbol": "XAUUSD",
        "timeframe": "M1",
        "folds": 3,
        "purge_bars": 0,
        "embargo_bars": 0,
        "min_trades_per_fold": 2,
        "cost_stress": (1.5, 2.0),
        "regime_coverage_min": 0.0,
        "bootstrap_samples": 100,
        "bootstrap_confidence": 0.95,
        "seed": 7,
        "search_budget": 50,
        "trial_count": 4,
        "window": 60,
    }
    values.update(overrides)
    return EvidenceConfig(**values)


def _stored_bundle(tmp_path: Path) -> SimpleNamespace:
    """Write a complete, reproducible experiment under ``tmp_path``."""
    store_root = tmp_path / "data"
    bars = _flat_bars()
    ParquetStore(store_root).write("XAUUSD", "M1", bars)

    features: FeatureFrame = build_features(bars)
    bundle: DatasetBundle = build_dataset(
        features,
        symbol="XAUUSD",
        timeframe="M1",
        horizon=1,
        train_fraction=0.6,
        validation_fraction=0.2,
        purge_bars=3,
        embargo_bars=2,
    )
    save_dataset_arrays(dataset_artifact_dir(store_root, bundle.manifest.checksum), bundle)
    evidence = build_evidence(
        bundle,
        _repro_spec(),
        _repro_config(),
        features=features,
        settings=Settings(),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    path = tmp_path / "candidate.evidence.json"
    path.write_text(json.dumps(evidence.to_dict(), indent=2) + "\n", encoding="utf-8")
    spec_sidecar_path(path).write_text(json.dumps(_repro_spec().canonical_dict()), encoding="utf-8")
    return SimpleNamespace(
        store_root=store_root,
        evidence=evidence,
        path=path,
    )


def _reproduce(stored: SimpleNamespace) -> CandidateEvidence | ReproduceMismatch:
    return reproduce_bundle(
        stored.path,
        store_root=stored.store_root,
        code_version=current_code_version(),
    )


def test_a_tampered_reproduce_bundle_fails_its_checksum(tmp_path: Path) -> None:
    stored = _stored_bundle(tmp_path)
    payload = json.loads(stored.path.read_text(encoding="utf-8"))
    payload["aggregate"]["expectancy_r"] = payload["aggregate"]["expectancy_r"] + 0.5
    stored.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvidenceError) as exc:
        _reproduce(stored)

    assert exc.value.code == "checksum_mismatch"


def test_a_tampered_locked_section_fails_its_checksum(tmp_path: Path) -> None:
    stored = _stored_bundle(tmp_path)
    section = _locked_section(stored.evidence, LOCKED_POLICY)
    section = replace(section, evidence_checksum=stored.evidence.checksum)
    section = replace(section, checksum=locked_oos_checksum(section))
    locked = replace(stored.evidence, locked_oos=section, checksum="")
    locked = replace(locked, checksum=evidence_checksum(locked))
    payload = locked.to_dict()
    payload["locked_oos"]["expectancy_r"] = 0.99
    stored.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvidenceError) as exc:
        _reproduce(stored)

    assert exc.value.code == "checksum_mismatch"


def test_a_locked_bundle_reproduces_the_base_checksum_without_the_section(
    tmp_path: Path,
) -> None:
    stored = _stored_bundle(tmp_path)
    section = _locked_section(stored.evidence, LOCKED_POLICY)
    section = replace(section, evidence_checksum=stored.evidence.checksum)
    section = replace(section, checksum=locked_oos_checksum(section))
    locked = replace(stored.evidence, locked_oos=section, checksum="")
    locked = replace(locked, checksum=evidence_checksum(locked))
    stored.path.write_text(json.dumps(locked.to_dict(), indent=2) + "\n", encoding="utf-8")

    result = _reproduce(stored)

    assert isinstance(result, CandidateEvidence)
    assert result.locked_oos is None
    assert result.checksum == stored.evidence.checksum
    assert result.folds == stored.evidence.folds
