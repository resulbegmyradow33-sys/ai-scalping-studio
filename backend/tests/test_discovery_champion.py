"""Strategy champion governance in the discovery archive.

These tests pin the properties that make a promoted strategy trustworthy enough
for the paper path:

* one champion per scope, scopes independent;
* a promotion needs evidence: a known candidate, a trial on the scope's dataset,
  an accepted verdict, a measured metric, a strict improvement and a reason;
* ``force`` overrides the comparison only, never the evidence gate, and says so;
* rollback restores a previous champion of the same scope and nothing else;
* every change leaves an exact audit row in ``champion_promotions`` and
  ``decisions``;
* the champion spec round-trips through the DSL validator and its hash still
  matches the archived id;
* a version 1 database migrates to version 3 without losing candidates, trials or
  decisions, and the state survives a reopen.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from promotion_evidence import evidence_for
from promotion_support import promote_for_tests, registered_test_policies
from scalping.discovery import (
    DEFAULT_SCOPE,
    SCHEMA_VERSION,
    ArchiveError,
    DiscoveryArchive,
    TrialResult,
    UnknownCandidateError,
)
from scalping.discovery import archive as archive_module
from scalping.discovery.champion import (
    ChampionProvider,
    InvalidChampionSpec,
    NoChampionError,
    NoEvidenceChampion,
    scope_for,
)
from scalping.discovery.promotion import PROMOTION_V1, EvidenceIdentity
from scalping.discovery.protocols import canonical_json
from scalping.strategy.dsl import (
    Compare,
    Const,
    EntryRule,
    ExitRule,
    HoldingLimit,
    StrategySpec,
)

SYMBOL = "XAUUSD"
TIMEFRAME = "M5"
SCOPE = f"{SYMBOL}:{TIMEFRAME}"
OTHER_SCOPE = "EURUSD:M5"
DATASET = f"validation:{SYMBOL}:{TIMEFRAME}:abcd1234"
OTHER_DATASET = f"validation:{OTHER_SCOPE}:abcd1234"
WRONG_TIMEFRAME_DATASET = "validation:XAUUSD:M15:abcd1234"
CODE_VERSION = "test-2026-11-24"
DATASET_CHECKSUM = "a1" * 32
SETTINGS_DIGEST = "b2" * 32
STAMP = "2026-11-20T10:00:00+00:00"
IDENTITY = EvidenceIdentity(
    code_version=CODE_VERSION,
    dataset_checksum=DATASET_CHECKSUM,
    settings_digest=SETTINGS_DIGEST,
)
TEST_POLICY = replace(PROMOTION_V1, policy_id="test-policy", require_locked_oos=False)


@pytest.fixture(autouse=True)
def _register_test_policy():
    with registered_test_policies(TEST_POLICY):
        yield


def _evidence_mode_champion(
    archive: DiscoveryArchive,
    candidate_id: str,
    *,
    expectancy_r: float = 0.4,
) -> object:
    """Promote a candidate through the evidence API for provider tests."""
    evidence = evidence_for(
        spec_hash=candidate_id,
        dataset_id=DATASET,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        code_version=CODE_VERSION,
        dataset_checksum=DATASET_CHECKSUM,
        settings_digest=SETTINGS_DIGEST,
        expectancy_r=expectancy_r,
    )
    return promote_for_tests(
        archive,
        candidate_id,
        evidence,
        TEST_POLICY,
        scope=SCOPE,
        reason="evidence-mode test champion",
        expected_identity=IDENTITY,
    )


# --------------------------------------------------------------------------- fixtures


def _spec(*, threshold: float = 1.0) -> StrategySpec:
    """A tiny valid spec; the metric lives in the trial, not in the tree."""
    return StrategySpec(
        entries=(EntryRule("long", Compare(Const(threshold), ">", Const(0.5))),),
        exits=(ExitRule(Compare(Const(0.0), ">", Const(1.0))),),
        holding=HoldingLimit(10),
        max_lookback=2,
    )


def _add(archive: DiscoveryArchive, spec: StrategySpec) -> str:
    return archive.add_candidate(spec, code_version=CODE_VERSION).candidate_id


def _trial(
    archive: DiscoveryArchive,
    candidate_id: str,
    value: float | None,
    *,
    accepted: bool = True,
    dataset_id: str = DATASET,
    metric: str = "expectancy_r",
) -> None:
    metrics = {"trades": 120.0}
    if value is not None:
        metrics[metric] = value
    archive.record_trial(
        candidate_id,
        TrialResult(
            metrics=metrics,
            accepted=accepted,
            rejections=()
            if accepted
            else ("min_trades: 3 closed trades is below the required 100",),
        ),
        dataset_id=dataset_id,
        code_version=CODE_VERSION,
        seed=1,
        dataset_checksum=DATASET_CHECKSUM,
        settings_digest=SETTINGS_DIGEST,
    )


def _promotable(archive: DiscoveryArchive, *, value: float, threshold: float = 1.0) -> str:
    candidate_id = _add(archive, _spec(threshold=threshold))
    _trial(archive, candidate_id, value)
    return candidate_id


# -------------------------------------------------------------------- scope algebra


def test_scope_key_and_one_champion_per_scope(tmp_path: Path) -> None:
    assert scope_for(SYMBOL, TIMEFRAME) == SCOPE
    with pytest.raises(ValueError):
        scope_for("  ", TIMEFRAME)

    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        gold = _promotable(archive, value=0.4)
        euro = _add(archive, _spec(threshold=2.0))
        _trial(archive, euro, 0.3, dataset_id=OTHER_DATASET)

        first = archive.promote_exploratory(gold, scope=SCOPE, reason="gold baseline")
        second = archive.promote_exploratory(euro, scope=OTHER_SCOPE, reason="euro baseline")

        assert archive.champion(SCOPE) is not None
        assert archive.champion(SCOPE).candidate.candidate_id == gold  # type: ignore[union-attr]
        assert archive.champion(OTHER_SCOPE).candidate.candidate_id == euro  # type: ignore[union-attr]
        assert first.promotion_id == 1
        assert second.promotion_id == 2
        assert len(archive.champion_history(SCOPE)) == 1
        assert len(archive.champion_history(OTHER_SCOPE)) == 1

        counts = {
            str(row["scope"]): int(row["n"])
            for row in archive.connection.execute(
                "SELECT scope, COUNT(*) AS n FROM champions GROUP BY scope"
            )
        }
        assert counts == {SCOPE: 1, OTHER_SCOPE: 1}

        # The primary key, not application code, is what forbids a second
        # champion in one scope.
        with pytest.raises(sqlite3.IntegrityError):
            archive.connection.execute(
                "INSERT INTO champions (scope, candidate_id, promotion_id, promoted_at, reason)"
                " VALUES (?, ?, 1, ?, 'dup')",
                (SCOPE, euro, STAMP),
            )


def test_dataset_must_name_every_scope_token(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        candidate_id = _add(archive, _spec())
        _trial(archive, candidate_id, 0.4, dataset_id=WRONG_TIMEFRAME_DATASET)
        _trial(archive, candidate_id, 0.4, dataset_id=OTHER_DATASET)

        with pytest.raises(ArchiveError, match="no trial on the dataset of scope"):
            archive.promote_exploratory(candidate_id, scope=SCOPE, reason="wrong market")
        assert archive.champion(SCOPE) is None

        _trial(archive, candidate_id, 0.4)
        promoted = archive.promote_exploratory(candidate_id, scope=SCOPE, reason="right market")
        assert promoted.metrics["expectancy_r"] == pytest.approx(0.4)


# ------------------------------------------------------------------ promote: rules


def test_promotion_replaces_the_champion_and_keeps_the_history(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        first = _promotable(archive, value=0.4)
        second = _promotable(archive, value=0.6, threshold=2.0)

        archive.promote_exploratory(first, scope=SCOPE, reason="first evidence")
        promoted = archive.promote_exploratory(second, scope=SCOPE, reason="better evidence")

        champion = archive.champion(SCOPE)
        assert champion is not None
        assert champion.candidate.candidate_id == second
        assert champion.previous_candidate_id == first
        assert champion.promotion_id == promoted.promotion_id == 2
        assert champion.metrics["expectancy_r"] == pytest.approx(0.6)

        history = archive.champion_history(SCOPE)
        assert [row.candidate_id for row in history] == [first, second]
        assert [row.kind for row in history] == ["promoted", "promoted"]
        assert [row.value for row in history] == [pytest.approx(0.4), pytest.approx(0.6)]
        assert history[1].previous_candidate_id == first

        # The replaced champion stays a candidate with its own trial record.
        assert archive.candidate(first) is not None
        assert archive.trial_count(first) == 1
        rows = archive.connection.execute(
            "SELECT COUNT(*) FROM champions WHERE scope = ?", (SCOPE,)
        ).fetchone()
        assert rows is not None and int(rows[0]) == 1


def test_promotion_refusals_write_nothing(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        known = _add(archive, _spec())
        rejected = _add(archive, _spec(threshold=2.0))
        _trial(archive, rejected, 0.9, accepted=False)

        with pytest.raises(UnknownCandidateError):
            archive.promote_exploratory("0" * 64, scope=SCOPE, reason="ghost")
        with pytest.raises(ArchiveError, match="no trial on the dataset of scope"):
            archive.promote_exploratory(known, scope=SCOPE, reason="no evidence")
        with pytest.raises(ArchiveError, match="rejected by the gate"):
            archive.promote_exploratory(rejected, scope=SCOPE, reason="gate said no")
        # The evidence gate is not forceable; only the comparison is.
        with pytest.raises(ArchiveError, match="rejected by the gate"):
            archive.promote_exploratory(rejected, scope=SCOPE, reason="override", force=True)
        with pytest.raises(ValueError, match="reason must not be empty"):
            archive.promote_exploratory(known, scope=SCOPE, reason="   ")
        with pytest.raises(ValueError, match="metric"):
            archive.promote_exploratory(
                known, scope=SCOPE, reason="bad metric", metric="not a metric"
            )

        assert archive.champion(SCOPE) is None
        assert archive.champion_history(SCOPE) == ()
        assert archive.decisions() == ()


def test_not_better_is_refused_and_force_is_recorded(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        champion_id = _promotable(archive, value=0.5)
        equal = _promotable(archive, value=0.5, threshold=2.0)
        worse = _promotable(archive, value=0.4, threshold=3.0)
        archive.promote_exploratory(champion_id, scope=SCOPE, reason="baseline")

        with pytest.raises(ArchiveError, match="does not strictly beat"):
            archive.promote_exploratory(equal, scope=SCOPE, reason="tie")
        with pytest.raises(ArchiveError, match="does not strictly beat"):
            archive.promote_exploratory(worse, scope=SCOPE, reason="worse")
        with pytest.raises(ValueError, match="forced promotion requires"):
            archive.promote_exploratory(worse, scope=SCOPE, reason=" ", force=True)

        forced = archive.promote_exploratory(
            worse, scope=SCOPE, reason="risk desk override", force=True
        )
        assert forced.reason.startswith("forced: risk desk override; ")
        assert "expectancy_r 0.4 vs 0.5 of " + champion_id[:12] in forced.reason

        decision = archive.decisions(worse)[-1]
        assert decision.decision == "promoted"
        detail = decision.detail
        assert detail is not None
        assert detail["forced"] is True
        assert detail["value"] == pytest.approx(0.4)
        assert detail["previous_value"] == pytest.approx(0.5)
        assert detail["previous_candidate_id"] == champion_id

        assert archive.champion(SCOPE) is not None
        assert archive.champion(SCOPE).candidate.candidate_id == worse  # type: ignore[union-attr]


def test_missing_baseline_is_refused_unless_forced(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        champion_id = _add(archive, _spec())
        _trial(archive, champion_id, 0.5)  # publishes expectancy_r only

        unmeasured = _add(archive, _spec(threshold=4.0))
        _trial(archive, unmeasured, 0.5)
        with pytest.raises(ArchiveError, match=r"does not publish 'profit_factor'"):
            archive.promote_exploratory(
                unmeasured, scope=SCOPE, reason="unmeasured", metric="profit_factor"
            )

        archive.promote_exploratory(champion_id, scope=SCOPE, reason="baseline")

        challenger = _add(archive, _spec(threshold=2.0))
        _trial(archive, challenger, 1.8, metric="profit_factor")

        with pytest.raises(ArchiveError, match=r"champion .* has no 'profit_factor'"):
            archive.promote_exploratory(challenger, scope=SCOPE, reason="x", metric="profit_factor")
        forced = archive.promote_exploratory(
            challenger,
            scope=SCOPE,
            reason="owner accepts an unmeasured comparison",
            metric="profit_factor",
            force=True,
        )
        assert forced.reason.startswith("forced: ")
        history = archive.champion_history(SCOPE)
        assert history[-1].value == pytest.approx(1.8)
        assert history[-1].previous_value is None


# ----------------------------------------------------------------- rollback: rules


def test_rollback_only_to_a_previous_champion(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        first = _promotable(archive, value=0.4)
        second = _promotable(archive, value=0.6, threshold=2.0)
        never = _promotable(archive, value=0.9, threshold=3.0)

        with pytest.raises(ArchiveError, match="has no champion"):
            archive.rollback(SCOPE, to_candidate_id=first, reason="nothing to undo")

        archive.promote_exploratory(first, scope=SCOPE, reason="first evidence")
        archive.promote_exploratory(second, scope=SCOPE, reason="better evidence")

        with pytest.raises(UnknownCandidateError):
            archive.rollback(SCOPE, to_candidate_id="1" * 64, reason="ghost")
        with pytest.raises(ArchiveError, match="already the champion"):
            archive.rollback(SCOPE, to_candidate_id=second, reason="no change")
        with pytest.raises(ArchiveError, match="was never the champion"):
            archive.rollback(SCOPE, to_candidate_id=never, reason="not earned")

        restored = archive.rollback(SCOPE, to_candidate_id=first, reason="decay detected")
        assert restored.candidate.candidate_id == first
        assert restored.previous_candidate_id == second
        assert restored.reason == f"decay detected; rolled back from {second[:12]}"

        history = archive.champion_history(SCOPE)
        assert [row.kind for row in history] == ["promoted", "promoted", "rolled_back"]
        assert history[-1].candidate_id == first
        assert history[-1].previous_candidate_id == second
        assert history[-1].metric is None and history[-1].value is None

        rolled_back = archive.decisions(first)[-1]
        assert rolled_back.decision == "rolled_back"
        assert rolled_back.champion_id == first
        detail = rolled_back.detail
        assert detail is not None
        assert detail["previous_candidate_id"] == second

        # The candidate that was rolled back from is still a candidate, and it
        # is still eligible for a later rollback of its own.
        again = archive.promote_exploratory(second, scope=SCOPE, reason="recovered")
        assert again.candidate.candidate_id == second
        archive.rollback(SCOPE, to_candidate_id=first, reason="decay again")
        assert {row.candidate_id for row in archive.champion_history(SCOPE)} == {first, second}


# ------------------------------------------------------------------- provider seam


def test_champion_spec_round_trips_and_validates(tmp_path: Path) -> None:
    spec = _spec(threshold=1.5)
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        candidate_id = _add(archive, spec)
        _trial(archive, candidate_id, 0.42)
        promoted = _evidence_mode_champion(archive, candidate_id, expectancy_r=0.42)

        provider = ChampionProvider(archive, scope=SCOPE)
        assert provider.scope == SCOPE
        assert provider.spec().canonical_hash() == candidate_id
        assert provider.spec().canonical_dict() == spec.canonical_dict()

        champion = provider.champion()
        assert champion.candidate_id == candidate_id
        assert champion.spec_hash == candidate_id
        assert champion.promotion_id == promoted.promotion_id
        assert champion.promoted_at == promoted.promoted_at
        assert champion.reason == promoted.reason
        assert champion.metrics["expectancy_r"] == pytest.approx(0.42)
        assert champion.mode == "evidence"
        assert champion.to_dict()["spec"] == spec.canonical_dict()

        # An explicit scope argument wins over the constructor default.
        with pytest.raises(NoChampionError):
            provider.spec(OTHER_SCOPE)

    with pytest.raises(NoChampionError, match="no champion in scope"):
        ChampionProvider(DiscoveryArchive(tmp_path / "empty.sqlite")).champion()


def test_provider_refuses_an_exploratory_champion(tmp_path: Path) -> None:
    spec = _spec()
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        candidate_id = _add(archive, spec)
        _trial(archive, candidate_id, 0.3)
        archive.promote_exploratory(candidate_id, scope=SCOPE, reason="research only")

        with pytest.raises(NoEvidenceChampion, match="exploratory"):
            ChampionProvider(archive, scope=SCOPE).champion()


def test_provider_refuses_an_invalid_or_tampered_stored_spec(tmp_path: Path) -> None:
    spec = _spec()
    path = tmp_path / "discovery.sqlite"
    with DiscoveryArchive(path) as archive:
        candidate_id = _add(archive, spec)
        _trial(archive, candidate_id, 0.3)
        _evidence_mode_champion(archive, candidate_id, expectancy_r=0.3)
        other_payload = _spec(threshold=2.0).canonical_dict()

        # A valid JSON object that is not a strategy payload.
        archive.connection.execute(
            "UPDATE candidates SET canonical_json = ? WHERE candidate_id = ?",
            ('{"kind":"strategy"}', candidate_id),
        )
        with pytest.raises(InvalidChampionSpec, match="does not load as a strategy"):
            ChampionProvider(archive).spec(SCOPE)

        # A perfectly valid strategy payload whose hash is not the archived id.
        archive.connection.execute(
            "UPDATE candidates SET canonical_json = ? WHERE candidate_id = ?",
            (canonical_json(other_payload), candidate_id),
        )
        with pytest.raises(InvalidChampionSpec, match="does not match its archived identity"):
            ChampionProvider(archive).spec(SCOPE)

        # No silent fallback: repairing the row restores the same strategy and
        # there is no path that substitutes a different one.
        archive.connection.execute(
            "UPDATE candidates SET canonical_json = ? WHERE candidate_id = ?",
            (canonical_json(spec.canonical_dict()), candidate_id),
        )
        assert ChampionProvider(archive).spec(SCOPE).canonical_hash() == candidate_id


def test_no_hand_written_fallback_when_the_archive_is_empty(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        provider = ChampionProvider(archive)
        with pytest.raises(NoChampionError):
            provider.spec()
        assert provider.scope == DEFAULT_SCOPE


# ------------------------------------------------------------------------ audit


def test_audit_rows_are_exact(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        first = _promotable(archive, value=0.4)
        second = _promotable(archive, value=0.6, threshold=2.0)

        first_reason = "best validation expectancy; first promotion; expectancy_r 0.4"
        second_reason = f"stronger evidence; expectancy_r 0.6 vs 0.4 of {first[:12]}"
        promoted_first = archive.promote_exploratory(
            first, scope=SCOPE, reason="best validation expectancy"
        )
        promoted_second = archive.promote_exploratory(
            second, scope=SCOPE, reason="stronger evidence"
        )
        assert promoted_first.reason == first_reason
        assert promoted_second.reason == second_reason

        champions = [
            tuple(row)
            for row in archive.connection.execute(
                "SELECT scope, candidate_id, promotion_id, promoted_at, reason FROM champions"
            )
        ]
        assert champions == [(SCOPE, second, 2, promoted_second.promoted_at, second_reason)]

        promotions = [
            tuple(row)
            for row in archive.connection.execute(
                """
                SELECT id, scope, kind, candidate_id, previous_candidate_id,
                       metric, value, previous_value, reason, created_at
                FROM champion_promotions ORDER BY id
                """
            )
        ]
        assert promotions == [
            (
                1,
                SCOPE,
                "promoted",
                first,
                None,
                "expectancy_r",
                0.4,
                None,
                first_reason,
                promoted_first.promoted_at,
            ),
            (
                2,
                SCOPE,
                "promoted",
                second,
                first,
                "expectancy_r",
                0.6,
                0.4,
                second_reason,
                promoted_second.promoted_at,
            ),
        ]

        details = [
            canonical_json(
                {
                    "scope": SCOPE,
                    "metric": "expectancy_r",
                    "value": 0.4,
                    "previous_candidate_id": None,
                    "previous_value": None,
                    "forced": False,
                    "promotion_id": 1,
                    "mode": "exploratory",
                }
            ),
            canonical_json(
                {
                    "scope": SCOPE,
                    "metric": "expectancy_r",
                    "value": 0.6,
                    "previous_candidate_id": first,
                    "previous_value": 0.4,
                    "forced": False,
                    "promotion_id": 2,
                    "mode": "exploratory",
                }
            ),
        ]
        decisions = [
            tuple(row)
            for row in archive.connection.execute(
                """
                SELECT id, candidate_id, decision, reason, champion_id, created_at, detail_json
                FROM decisions ORDER BY id
                """
            )
        ]
        assert decisions == [
            (
                1,
                first,
                "promoted",
                first_reason,
                first,
                promoted_first.promoted_at,
                details[0],
            ),
            (
                2,
                second,
                "promoted",
                second_reason,
                second,
                promoted_second.promoted_at,
                details[1],
            ),
        ]

        # The generic decision log still works and keeps its detail JSON.
        extra = archive.record_decision(
            second, "no_change", "champion retained", detail={"candidates": 3}
        )
        assert extra.detail == {"candidates": 3}
        assert archive.decisions()[-1].decision == "no_change"


def test_compare_to_champion_reports_missing_values(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        champion_id = _promotable(archive, value=0.5)
        better_id = _promotable(archive, value=0.7, threshold=2.0)
        silent = _add(archive, _spec(threshold=3.0))
        _trial(archive, silent, None)  # accepted, but publishes no expectancy_r

        with pytest.raises(UnknownCandidateError):
            archive.compare_to_champion("2" * 64, SCOPE)

        no_champion = archive.compare_to_champion(better_id, SCOPE)
        assert no_champion.champion_id is None
        assert no_champion.better is False

        archive.promote_exploratory(champion_id, scope=SCOPE, reason="baseline")

        comparison = archive.compare_to_champion(better_id, SCOPE)
        assert comparison.champion_id == champion_id
        assert comparison.candidate_value == pytest.approx(0.7)
        assert comparison.champion_value == pytest.approx(0.5)
        assert comparison.delta == pytest.approx(0.2)
        assert comparison.better is True

        missing = archive.compare_to_champion(silent, SCOPE)
        assert missing.candidate_value is None
        assert missing.delta is None
        assert missing.better is False
        assert missing.to_dict()["champion_id"] == champion_id


# --------------------------------------------------------------- migration/reopen


def _build_v1_database(path: Path, spec: StrategySpec, value: float) -> str:
    """Write the version 1 layout and one candidate/trial/decision by hand."""
    candidate_id = spec.canonical_hash()
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        conn.execute(archive_module._SCHEMA_VERSION_DDL)
        for statement in archive_module._MIGRATIONS[1]:
            conn.execute(statement)
        conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (1, ?)", (STAMP,))
        conn.execute(
            """
            INSERT INTO candidates (
                candidate_id, canonical_json, complexity, created_at,
                parent_a, parent_b, mutation, code_version, trials
            ) VALUES (?, ?, ?, ?, NULL, NULL, 'random', ?, 1)
            """,
            (
                candidate_id,
                canonical_json(spec.canonical_dict()),
                spec.complexity(),
                STAMP,
                "v1-code",
            ),
        )
        conn.execute(
            """
            INSERT INTO trials (
                candidate_id, dataset_id, code_version, seed,
                metrics_json, accepted, rejections_json, created_at
            ) VALUES (?, ?, ?, 7, ?, 1, '[]', ?)
            """,
            (
                candidate_id,
                DATASET,
                "v1-code",
                canonical_json({"expectancy_r": value, "trades": 120.0}),
                STAMP,
            ),
        )
        conn.execute(
            """
            INSERT INTO decisions (candidate_id, decision, reason, champion_id, created_at)
            VALUES (?, 'rejected', 'v1 note', NULL, ?)
            """,
            (candidate_id, STAMP),
        )
    finally:
        conn.close()
    return candidate_id


def test_schema_v1_migrates_and_keeps_evidence(tmp_path: Path) -> None:
    path = tmp_path / "discovery.sqlite"
    spec = _spec()
    candidate_id = _build_v1_database(path, spec, 0.35)

    with DiscoveryArchive(path) as archive:
        assert archive.schema_version == SCHEMA_VERSION == 4
        assert archive.candidate(candidate_id) is not None
        assert archive.trial_count(candidate_id) == 1
        old = archive.decisions()
        assert [(row.decision, row.reason) for row in old] == [("rejected", "v1 note")]
        assert old[0].detail_json is None
        assert archive.champion(SCOPE) is None
        assert archive.champion_history(SCOPE) == ()

        # The widened CHECK accepts the new outcome without a second migration.
        archive.promote_exploratory(candidate_id, scope=SCOPE, reason="migrated baseline")
        assert archive.champion(SCOPE) is not None
        assert archive.champion(SCOPE).candidate.candidate_id == candidate_id  # type: ignore[union-attr]

    with (
        pytest.raises(sqlite3.IntegrityError),
        DiscoveryArchive(path) as archive,
    ):
        # The version 3 schema really is in place, not just accepted by the API.
        archive.connection.execute(
            "INSERT INTO decisions (candidate_id, decision, reason, created_at)"
            " VALUES (?, 'made_up', 'x', ?)",
            (candidate_id, STAMP),
        )


def test_state_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "discovery.sqlite"
    spec = _spec()
    with DiscoveryArchive(path) as archive:
        candidate_id = _add(archive, spec)
        _trial(archive, candidate_id, 0.41)
        _evidence_mode_champion(archive, candidate_id, expectancy_r=0.41)
        before = archive.champion(SCOPE)
        before_history = [row.to_dict() for row in archive.champion_history(SCOPE)]

    with DiscoveryArchive(path) as archive:
        after = archive.champion(SCOPE)
        assert before is not None and after is not None
        assert after.to_dict() == before.to_dict()
        assert [row.to_dict() for row in archive.champion_history(SCOPE)] == before_history
        assert ChampionProvider(archive, scope=SCOPE).spec().canonical_hash() == candidate_id
        assert len(archive.decisions()) == 1
