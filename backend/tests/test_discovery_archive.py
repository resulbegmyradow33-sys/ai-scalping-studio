"""Discovery archive: schema, dedupe, lineage, trial history and decisions."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from discovery_fakes import ToyCandidate
from scalping.discovery import (
    DECISIONS,
    SCHEMA_VERSION,
    ArchiveError,
    DiscoveryArchive,
    TrialResult,
    UnknownCandidateError,
)

CODE_VERSION = "test-2026-11-23"
DATASET_ID = "validation:XAUUSD:M5:abcd1234"
OTHER_DATASET_ID = "validation:EURUSD:M5:abcd1234"


def _root() -> ToyCandidate:
    return ToyCandidate((1, 2))


def _child(parent: ToyCandidate) -> ToyCandidate:
    return ToyCandidate(parent.nodes + (3,))


def _add(
    archive: DiscoveryArchive,
    candidate: ToyCandidate,
    *,
    parent_a: str | None = None,
    parent_b: str | None = None,
    mutation: str | None = None,
):
    return archive.add_candidate(
        candidate,
        code_version=CODE_VERSION,
        parent_a=parent_a,
        parent_b=parent_b,
        mutation=mutation,
    )


def _trial(metric: float, *, accepted: bool = True) -> TrialResult:
    return TrialResult(
        metrics={"net_profit": metric},
        accepted=accepted,
        rejections=() if accepted else ("below_threshold",),
    )


def test_schema_is_idempotent_and_uses_wal(tmp_path: Path) -> None:
    path = tmp_path / "discovery.sqlite"
    archive = DiscoveryArchive(path)
    assert archive.schema_version == SCHEMA_VERSION
    assert archive.init_schema() == SCHEMA_VERSION
    tables = {
        str(row[0])
        for row in archive.connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {"schema_version", "candidates", "trials", "lineage", "decisions"} <= tables
    mode = archive.connection.execute("PRAGMA journal_mode").fetchone()
    assert mode is not None and str(mode[0]).lower() == "wal"
    archive.close()

    reopened = DiscoveryArchive(path)
    assert reopened.schema_version == SCHEMA_VERSION
    assert len(reopened.decisions()) == 0
    reopened.close()


def test_add_candidate_dedupes_by_canonical_hash(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        first = _add(archive, _root())
        second = _add(archive, ToyCandidate((1, 2)))
        assert first.candidate_id == second.candidate_id
        assert first.to_dict() == second.to_dict()
        count = archive.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()
        assert count is not None and int(count[0]) == 1
        assert archive.dedupe(ToyCandidate((1, 2))) is not None
        assert archive.dedupe(ToyCandidate((9, 9))) is None


def test_candidate_hash_must_match_its_payload(tmp_path: Path) -> None:
    class Liar:
        def canonical_dict(self) -> dict[str, object]:
            return {"format": "toy-v1", "nodes": [1, 2]}

        def canonical_hash(self) -> str:
            return "0" * 64

        def complexity(self) -> int:
            return 2

    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        with pytest.raises(ArchiveError, match="does not match"):
            archive.add_candidate(Liar(), code_version=CODE_VERSION)
        count = archive.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()
        assert count is not None and int(count[0]) == 0


def test_lineage_edges_and_depth(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        root = _add(archive, _root())
        mutated = _add(archive, _child(_root()), parent_a=root.candidate_id, mutation="mutation")
        other = _add(archive, ToyCandidate((4, 4)))
        crossed = _add(
            archive,
            ToyCandidate((1, 4)),
            parent_a=root.candidate_id,
            parent_b=other.candidate_id,
            mutation="crossover",
        )
        grandchild = _add(
            archive,
            ToyCandidate((1, 2, 3, 4)),
            parent_a=mutated.candidate_id,
            mutation="mutation",
        )

        assert set(archive.lineage(crossed.candidate_id)) == {
            (root.candidate_id, "crossover"),
            (other.candidate_id, "crossover"),
        }
        assert archive.lineage(mutated.candidate_id) == ((root.candidate_id, "mutation"),)
        assert archive.lineage(root.candidate_id) == ()
        assert archive.depth(root.candidate_id) == 0
        assert archive.depth(mutated.candidate_id) == 1
        assert archive.depth(grandchild.candidate_id) == 2
        stored = archive.candidate(grandchild.candidate_id)
        assert stored is not None and stored.parent_a == mutated.candidate_id
        assert stored.mutation == "mutation"


def test_lineage_requires_known_parent(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        with pytest.raises(ArchiveError, match="unknown parent"):
            archive.add_candidate(
                _root(),
                code_version=CODE_VERSION,
                parent_a="f" * 64,
                mutation="mutation",
            )
        count = archive.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()
        assert count is not None and int(count[0]) == 0


def test_trial_history_and_counter(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        record = _add(archive, _root())
        assert archive.trial_count(record.candidate_id) == 0
        first = archive.record_trial(
            record.candidate_id,
            _trial(1.5),
            dataset_id=DATASET_ID,
            code_version=CODE_VERSION,
            seed=11,
        )
        second = archive.record_trial(
            record.candidate_id,
            _trial(-0.5, accepted=False),
            dataset_id=DATASET_ID,
            code_version=CODE_VERSION,
            seed=12,
        )
        assert (first.id, second.id) == (1, 2)
        history = archive.history(record.candidate_id)
        assert [trial.id for trial in history] == [1, 2]
        assert [trial.seed for trial in history] == [11, 12]
        assert history[0].metrics == {"net_profit": 1.5}
        assert history[1].accepted is False
        assert history[1].rejections == ("below_threshold",)
        assert archive.trial_count(record.candidate_id) == 2
        stored = archive.candidate(record.candidate_id)
        assert stored is not None and stored.trials == 2
        assert archive.has_trial(record.candidate_id, DATASET_ID)
        assert not archive.has_trial(record.candidate_id, OTHER_DATASET_ID)


def test_trial_lookup_matches_the_manifest_spelling(tmp_path: Path) -> None:
    plain = "XAUUSD:M5:abcd1234"
    prefixed = f"validation:{plain}"
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        record = _add(archive, _root())
        archive.record_trial(
            record.candidate_id,
            _trial(1.5),
            dataset_id=prefixed,
            code_version=CODE_VERSION,
            seed=1,
        )

        assert archive.has_trial(record.candidate_id, plain)
        assert archive.has_trial(record.candidate_id, prefixed)
        assert archive.has_trial(record.candidate_id, plain, code_version=CODE_VERSION)
        assert not archive.has_trial(record.candidate_id, plain, code_version="other-code")
        assert not archive.has_trial(record.candidate_id, f"test:{plain}")
        assert not archive.has_trial(record.candidate_id, "XAUUSD:M15:abcd1234")

        accepted = archive.accepted_trial(record.candidate_id, plain, code_version=CODE_VERSION)
        assert accepted is not None
        assert accepted.dataset_id == prefixed
        assert archive.accepted_trial(record.candidate_id, plain, code_version="other-code") is None


@pytest.mark.parametrize("stored_id", ["", "   ", "validation:"])
def test_malformed_archived_dataset_ids_are_ignored(tmp_path: Path, stored_id: str) -> None:
    """A blank legacy row is a skipped defect, not a ValueError in a lookup."""
    plain = "XAUUSD:M5:abcd1234"
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        record = _add(archive, _root())
        archive.connection.execute(
            """
            INSERT INTO trials (
                candidate_id, dataset_id, code_version, seed,
                metrics_json, accepted, rejections_json, created_at
            ) VALUES (?, ?, ?, 0, '{}', 1, '[]', '2026-11-26T00:00:00+00:00')
            """,
            (record.candidate_id, stored_id, CODE_VERSION),
        )

        assert not archive.has_trial(record.candidate_id, plain)
        stats = archive.trial_stats(record.candidate_id, dataset_id=plain)
        assert stats.total_attempts == 0
        assert archive.accepted_trial(record.candidate_id, plain, code_version=CODE_VERSION) is None

        # A well-formed row after the defect is still found.
        archive.record_trial(
            record.candidate_id,
            _trial(1.0),
            dataset_id=plain,
            code_version=CODE_VERSION,
            seed=2,
        )
        assert archive.has_trial(record.candidate_id, plain)
        accepted = archive.accepted_trial(record.candidate_id, plain, code_version=CODE_VERSION)
        assert accepted is not None and accepted.dataset_id == plain


def test_best_honours_minimum_trials_and_acceptance(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        once = _add(archive, ToyCandidate((1,)))
        twice = _add(archive, ToyCandidate((2, 2)))
        rejected = _add(archive, ToyCandidate((3, 3, 3)))

        archive.record_trial(
            once.candidate_id, _trial(5.0), dataset_id=DATASET_ID, code_version=CODE_VERSION, seed=1
        )
        archive.record_trial(
            twice.candidate_id,
            _trial(3.0),
            dataset_id=DATASET_ID,
            code_version=CODE_VERSION,
            seed=2,
        )
        archive.record_trial(
            twice.candidate_id,
            _trial(4.0),
            dataset_id=DATASET_ID,
            code_version=CODE_VERSION,
            seed=3,
        )
        archive.record_trial(
            rejected.candidate_id,
            _trial(99.0, accepted=False),
            dataset_id=DATASET_ID,
            code_version=CODE_VERSION,
            seed=4,
        )

        best = archive.best("net_profit")
        assert best is not None and best.candidate_id == once.candidate_id
        best_evidence = archive.best("net_profit", minimum_trials=2)
        assert best_evidence is not None and best_evidence.candidate_id == twice.candidate_id
        # A rejected trial with a huge number is not evidence of anything.
        strict = archive.best("net_profit", accepted_only=True)
        assert strict is not None and strict.candidate_id != rejected.candidate_id
        loose = archive.best("net_profit", accepted_only=False)
        assert loose is not None and loose.candidate_id == rejected.candidate_id
        assert archive.best("missing_metric") is None
        with pytest.raises(ValueError, match="minimum_trials"):
            archive.best("net_profit", minimum_trials=0)
        with pytest.raises(ValueError, match="JSON path"):
            archive.best('net"profit')


def test_decision_audit(tmp_path: Path) -> None:
    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        champion = _add(archive, _root())
        candidate = _add(archive, _child(_root()))

        promoted = archive.record_decision(
            candidate.candidate_id,
            "promoted",
            "beats champion after costs",
            champion_id=champion.candidate_id,
        )
        rejected = archive.record_decision(candidate.candidate_id, "rejected", "cost stress fails")
        no_trade = archive.record_decision(
            candidate.candidate_id, "no_trade", "expected edge below costs"
        )

        assert promoted.champion_id == champion.candidate_id
        assert rejected.champion_id is None
        rows = archive.decisions(candidate.candidate_id)
        assert [row.decision for row in rows] == ["promoted", "rejected", "no_trade"]
        assert [row.reason for row in rows] == [
            "beats champion after costs",
            "cost stress fails",
            "expected edge below costs",
        ]
        assert rows[1].id == no_trade.id - 1
        assert len(archive.decisions()) == 3

        with pytest.raises(ValueError, match="decision must be one of"):
            archive.record_decision(candidate.candidate_id, "maybe", "unclear")
        with pytest.raises(UnknownCandidateError):
            archive.record_decision("a" * 64, "rejected", "unknown")


def test_reopen_persists_everything(tmp_path: Path) -> None:
    path = tmp_path / "discovery.sqlite"
    archive = DiscoveryArchive(path)
    root = _add(archive, _root())
    child = _add(archive, _child(_root()), parent_a=root.candidate_id, mutation="mutation")
    archive.record_trial(
        child.candidate_id, _trial(2.0), dataset_id=DATASET_ID, code_version=CODE_VERSION, seed=7
    )
    archive.record_decision(child.candidate_id, "no_change", "champion still better")
    archive.close()

    reopened = DiscoveryArchive(path)
    stored = reopened.candidate(child.candidate_id)
    assert stored is not None
    assert stored.parent_a == root.candidate_id
    assert reopened.trial_count(child.candidate_id) == 1
    assert reopened.history(child.candidate_id)[0].seed == 7
    assert [row.decision for row in reopened.decisions()] == ["no_change"]
    assert reopened.depth(child.candidate_id) == 1
    reopened.close()


def test_foreign_keys_are_enforced(tmp_path: Path) -> None:
    path = tmp_path / "discovery.sqlite"
    with DiscoveryArchive(path) as archive:
        with pytest.raises(UnknownCandidateError):
            archive.record_trial(
                "b" * 64,
                _trial(1.0),
                dataset_id=DATASET_ID,
                code_version=CODE_VERSION,
                seed=1,
            )
        with pytest.raises(UnknownCandidateError):
            archive.history("b" * 64)
        with pytest.raises(UnknownCandidateError):
            archive.depth("b" * 64)

    raw = sqlite3.connect(path)
    raw.execute("PRAGMA foreign_keys = ON")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            raw.execute(
                """
                INSERT INTO trials (
                    candidate_id, dataset_id, code_version, seed,
                    metrics_json, accepted, rejections_json, created_at
                ) VALUES (?, ?, ?, ?, '{}', 0, '[]', '2026-11-23T00:00:00+00:00')
                """,
                ("c" * 64, DATASET_ID, CODE_VERSION, 1),
            )
        with pytest.raises(sqlite3.IntegrityError):
            raw.execute(
                """
                INSERT INTO candidates (
                    candidate_id, canonical_json, complexity, created_at, code_version, trials
                ) VALUES (?, 'not json', 1, '2026-11-23T00:00:00+00:00', ?, 0)
                """,
                ("d" * 64, CODE_VERSION),
            )
    finally:
        raw.close()


def test_decision_vocabulary_is_closed() -> None:
    assert DECISIONS == {"promoted", "rejected", "no_change", "no_trade", "rolled_back"}


def test_closed_archive_rejects_writes(tmp_path: Path) -> None:
    archive = DiscoveryArchive(tmp_path / "discovery.sqlite")
    archive.close()
    with pytest.raises(ArchiveError, match="closed"):
        archive.add_candidate(_root(), code_version=CODE_VERSION)
