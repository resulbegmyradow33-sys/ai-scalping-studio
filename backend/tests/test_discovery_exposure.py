"""Search exposure: archive-derived trial counts and the evidence rewrite.

The archive is the evidence log. These tests make every trial row explicit and
assert the derived numbers, the refusal for an understated claim, and that the
bundle rewrite is a new checksummed object rather than an edit.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from discovery_fakes import ToyCandidate
from scalping.discovery import (
    AggregateEvidence,
    CandidateEvidence,
    DiscoveryArchive,
    EvidencePolicy,
    ExposureError,
    SplitBoundaries,
    TrialResult,
    TrialStats,
    UnknownCandidateError,
    apply_to_evidence,
    evidence_checksum,
    exposure_for,
)

CODE_VERSION = "test-exposure-2026-11-25"
OTHER_CODE_VERSION = "test-exposure-2026-11-24"
DATASET_ID = "validation:XAUUSD:M5:abcd1234"
OTHER_DATASET_ID = "validation:EURUSD:M5:abcd1234"
CREATED_AT = "2026-11-25T10:00:00+00:00"


def _add(archive: DiscoveryArchive, *nodes: int) -> str:
    record = archive.add_candidate(ToyCandidate(nodes), code_version=CODE_VERSION)
    return record.candidate_id


def _record(
    archive: DiscoveryArchive,
    candidate_id: str,
    *,
    dataset_id: str = DATASET_ID,
    code_version: str = CODE_VERSION,
    accepted: bool = True,
    seed: int = 0,
) -> None:
    archive.record_trial(
        candidate_id,
        _result(accepted=accepted),
        dataset_id=dataset_id,
        code_version=code_version,
        seed=seed,
    )


def _result(*, accepted: bool) -> TrialResult:
    return TrialResult(
        metrics={"net_profit": 1.0 if accepted else -1.0},
        accepted=accepted,
        rejections=() if accepted else ("below_threshold",),
    )


def _policy() -> EvidencePolicy:
    return EvidencePolicy(
        folds=3,
        purge_bars=0,
        embargo_bars=0,
        min_trades_per_fold=5,
        cost_stress=(1.5, 2.0),
        regime_coverage_min=1.0,
        bootstrap_samples=100,
        bootstrap_confidence=0.95,
        window=60,
    )


def _aggregate() -> AggregateEvidence:
    return AggregateEvidence(
        total_trades=0,
        total_r=0.0,
        expectancy_r=0.0,
        profit_factor=1.0,
        max_drawdown=0.0,
        positive_folds=0,
        positive_fold_fraction=0.0,
        expectancy_r_ci95_low=0.0,
        expectancy_r_ci95_high=0.0,
        total_r_ci95_low=0.0,
        total_r_ci95_high=0.0,
        bootstrap_samples=100,
        bootstrap_confidence=0.95,
        bootstrap_method="iid",
        regime_coverage=1.0,
        regime_trades=(),
    )


def _bundle(
    spec_hash: str,
    *,
    dataset_id: str = DATASET_ID,
    code_version: str = CODE_VERSION,
    trial_count: int = 1,
    search_budget: int = 100,
    checksum: str | None = None,
) -> CandidateEvidence:
    evidence = CandidateEvidence(
        spec_hash=spec_hash,
        spec_version=1,
        complexity=4,
        dataset_id=dataset_id,
        dataset_checksum="a" * 64,
        symbol="XAUUSD",
        timeframe="M5",
        code_version=code_version,
        settings_digest="b" * 64,
        seed=7,
        trial_count=trial_count,
        search_budget=search_budget,
        splits=SplitBoundaries(0, 1, 1, 2, 2, 3),
        policy=_policy(),
        folds=(),
        aggregate=_aggregate(),
        checksum="",
        created_at=CREATED_AT,
    )
    actual = evidence_checksum(evidence) if checksum is None else checksum
    return replace(evidence, checksum=actual)


def _archive_with_search(tmp_path: Path) -> tuple[DiscoveryArchive, str, str]:
    archive = DiscoveryArchive(tmp_path / "discovery.sqlite")
    anchor = _add(archive, 1, 2)
    rival = _add(archive, 3, 4)
    _record(archive, anchor, accepted=True)
    _record(archive, rival, accepted=False)
    _record(archive, anchor, accepted=True, dataset_id=OTHER_DATASET_ID)
    _record(
        archive,
        rival,
        accepted=True,
        dataset_id=OTHER_DATASET_ID,
        code_version=OTHER_CODE_VERSION,
    )
    return archive, anchor, rival


# --- trial_stats --------------------------------------------------------------


def test_trial_stats_counts_the_whole_search(tmp_path: Path) -> None:
    archive, anchor, _ = _archive_with_search(tmp_path)
    with archive:
        stats = archive.trial_stats(anchor)

    assert isinstance(stats, TrialStats)
    assert stats.total_attempts == 4
    assert stats.accepted == 3
    assert stats.rejected == 1
    assert stats.distinct_candidates == 2
    assert stats.deduped == 0
    assert stats.per_dataset == {DATASET_ID: 2, OTHER_DATASET_ID: 2}
    assert stats.spec_attempts == 2
    assert stats.dataset_id is None
    assert stats.code_version is None


def test_trial_stats_filters_dataset_and_code_version(tmp_path: Path) -> None:
    archive, anchor, _ = _archive_with_search(tmp_path)
    with archive:
        on_dataset = archive.trial_stats(anchor, dataset_id=DATASET_ID)
        on_version = archive.trial_stats(anchor, code_version=CODE_VERSION)

    assert on_dataset.total_attempts == 2
    assert on_dataset.accepted == 1
    assert on_dataset.rejected == 1
    assert on_dataset.distinct_candidates == 2
    assert on_dataset.spec_attempts == 1
    assert on_dataset.per_dataset == {DATASET_ID: 2}

    assert on_version.total_attempts == 3
    assert on_version.dataset_id is None
    assert on_version.code_version == CODE_VERSION
    assert on_version.per_dataset == {DATASET_ID: 2, OTHER_DATASET_ID: 1}


def test_repeated_triple_is_counted_as_deduped(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        anchor = _add(archive, 1, 2)
        _record(archive, anchor, seed=1)
        _record(archive, anchor, seed=2)

        stats = archive.trial_stats(anchor, dataset_id=DATASET_ID, code_version=CODE_VERSION)

    assert stats.total_attempts == 2
    assert stats.distinct_candidates == 1
    assert stats.deduped == 1


def test_known_spec_without_trials_reports_zeros(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        anchor = _add(archive, 1, 2)

        stats = archive.trial_stats(anchor, dataset_id=DATASET_ID, code_version=CODE_VERSION)

    assert stats.total_attempts == 0
    assert stats.accepted == 0
    assert stats.rejected == 0
    assert stats.distinct_candidates == 0
    assert stats.deduped == 0
    assert stats.per_dataset == {}
    assert stats.spec_attempts == 0


def test_unknown_spec_is_refused(tmp_path: Path) -> None:
    with (
        DiscoveryArchive(tmp_path / "discovery.sqlite") as archive,
        pytest.raises(UnknownCandidateError),
    ):
        archive.trial_stats("0" * 64)


def test_trial_stats_is_read_only(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        anchor = _add(archive, 1, 2)
        _record(archive, anchor)
        before = archive.connection.execute("SELECT COUNT(*) FROM trials").fetchone()

        archive.trial_stats(anchor)
        archive.trial_stats(anchor, dataset_id=DATASET_ID, code_version=CODE_VERSION)

        after = archive.connection.execute("SELECT COUNT(*) FROM trials").fetchone()
        assert before == after


def test_trial_stats_is_deterministic(tmp_path: Path) -> None:
    archive, anchor, _ = _archive_with_search(tmp_path)
    with archive:
        first = archive.trial_stats(anchor, code_version=CODE_VERSION)
        second = archive.trial_stats(anchor, code_version=CODE_VERSION)

    assert first == second
    assert first.to_dict() == second.to_dict()


# --- exposure_for -------------------------------------------------------------


def test_exposure_for_uses_the_archive_count(tmp_path: Path) -> None:
    archive, anchor, _ = _archive_with_search(tmp_path)
    with archive:
        exposure = exposure_for(archive, anchor, DATASET_ID, CODE_VERSION)

    assert exposure.trial_count == 2
    assert exposure.search_budget == 2
    assert exposure.distinct_candidates == 2
    assert exposure.accepted == 1
    assert exposure.rejected == 1
    assert exposure.per_dataset == {DATASET_ID: 2}
    assert exposure.to_dict()["trial_count"] == 2


def test_exposure_for_missing_trial_evidence_is_refused(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        anchor = _add(archive, 1, 2)

        with pytest.raises(ExposureError) as caught:
            exposure_for(archive, anchor, DATASET_ID, CODE_VERSION)

    assert caught.value.code == "missing_trial_evidence"


def test_exposure_for_refuses_another_candidates_trial_as_evidence(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        anchor = _add(archive, 1, 2)
        other = _add(archive, 3, 4)
        _record(archive, other)

        with pytest.raises(ExposureError) as caught:
            exposure_for(archive, anchor, DATASET_ID, CODE_VERSION)

    assert caught.value.code == "missing_trial_evidence"


def test_exposure_for_requires_matching_dataset_and_version(tmp_path: Path) -> None:
    archive, anchor, _ = _archive_with_search(tmp_path)
    with archive, pytest.raises(ExposureError) as caught:
        exposure_for(archive, anchor, DATASET_ID, OTHER_CODE_VERSION)

    assert caught.value.code == "missing_trial_evidence"


# --- apply_to_evidence --------------------------------------------------------


def test_understated_claim_is_refused_and_evidence_is_untouched(tmp_path: Path) -> None:
    archive, anchor, _ = _archive_with_search(tmp_path)
    with archive:
        exposure = exposure_for(archive, anchor, DATASET_ID, CODE_VERSION)
        evidence = _bundle(anchor, trial_count=1, search_budget=100)

        with pytest.raises(ExposureError) as caught:
            apply_to_evidence(evidence, exposure)

    assert caught.value.code == "trial_count_understated"
    assert evidence.trial_count == 1
    assert evidence.checksum == _bundle(anchor, trial_count=1, search_budget=100).checksum


def test_apply_replaces_the_claim_and_updates_the_checksum(tmp_path: Path) -> None:
    archive, anchor, _ = _archive_with_search(tmp_path)
    with archive:
        exposure = exposure_for(archive, anchor, DATASET_ID, CODE_VERSION)
        evidence = _bundle(anchor, trial_count=25, search_budget=100)
        updated = apply_to_evidence(evidence, exposure)

    assert updated is not evidence
    assert updated.trial_count == 2
    assert updated.search_budget == 100
    assert updated.created_at == evidence.created_at
    assert updated.checksum != evidence.checksum
    assert updated.checksum == evidence_checksum(updated)
    assert updated.to_dict()["trial_count"] == 2
    # The input bundle is frozen and was not edited.
    assert evidence.trial_count == 25
    assert evidence.checksum == evidence_checksum(evidence)


def test_apply_lifts_an_inconsistent_budget_to_the_archive_count(tmp_path: Path) -> None:
    archive, anchor, _ = _archive_with_search(tmp_path)
    with archive:
        exposure = exposure_for(archive, anchor, DATASET_ID, CODE_VERSION)
        evidence = _bundle(anchor, trial_count=2, search_budget=1)

    updated = apply_to_evidence(evidence, exposure)

    assert updated.trial_count == 2
    assert updated.search_budget == 2


def test_apply_is_idempotent(tmp_path: Path) -> None:
    archive, anchor, _ = _archive_with_search(tmp_path)
    with archive:
        exposure = exposure_for(archive, anchor, DATASET_ID, CODE_VERSION)
        evidence = _bundle(anchor, trial_count=9, search_budget=100)

    once = apply_to_evidence(evidence, exposure)
    twice = apply_to_evidence(once, exposure)

    assert once == twice
    assert once.checksum == twice.checksum


def test_apply_refuses_a_mismatched_exposure(tmp_path: Path) -> None:
    archive, anchor, _ = _archive_with_search(tmp_path)
    with archive:
        exposure = exposure_for(archive, anchor, DATASET_ID, CODE_VERSION)
        evidence = _bundle(anchor, dataset_id=OTHER_DATASET_ID, trial_count=9)

        with pytest.raises(ExposureError) as caught:
            apply_to_evidence(evidence, exposure)

    assert caught.value.code == "exposure_mismatch"


def test_apply_refuses_a_tampered_bundle(tmp_path: Path) -> None:
    archive, anchor, _ = _archive_with_search(tmp_path)
    with archive:
        exposure = exposure_for(archive, anchor, DATASET_ID, CODE_VERSION)
        evidence = _bundle(anchor, trial_count=9)
        tampered = replace(evidence, checksum="0" * 64)

        with pytest.raises(ExposureError) as caught:
            apply_to_evidence(tampered, exposure)

    assert caught.value.code == "checksum_mismatch"
