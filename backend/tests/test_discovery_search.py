"""Discovery search: determinism, budget, skips, lineage and dataset discipline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from discovery_fakes import FixedFactory, ToyCandidate, ToyEvaluator, ToyFactory
from scalping.discovery import (
    DiscoveryArchive,
    SearchConfig,
    SearchResult,
    TrialResult,
    evolve,
)

DATASET_ID = "validation:XAUUSD:M5:abcd1234"
CODE_VERSION = "test-2026-11-23"


def _config(**overrides: object) -> SearchConfig:
    values: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "code_version": CODE_VERSION,
        "population": 6,
        "generations": 4,
        "mutation_rate": 0.4,
        "crossover_rate": 0.4,
        "max_trials": 60,
        "seed": 13,
        "patience": 3,
        "complexity_cap": 4,
        "novelty_requirement": 1,
        "selection_metric": "net_profit",
    }
    values.update(overrides)
    return SearchConfig(**values)  # type: ignore[arg-type]


def _open(tmp_path: Path, name: str) -> DiscoveryArchive:
    return DiscoveryArchive(tmp_path / f"{name}.sqlite")


def _trials(archive: DiscoveryArchive) -> list[tuple[str, int, str]]:
    rows = archive.connection.execute(
        "SELECT candidate_id, seed, dataset_id FROM trials ORDER BY id"
    ).fetchall()
    return [(str(row[0]), int(row[1]), str(row[2])) for row in rows]


class _BrokenFactory:
    """Every proposal fails the validation gate."""

    def random(self, rng: np.random.Generator) -> ToyCandidate:
        return ToyCandidate((9,))

    def mutate(self, candidate: ToyCandidate, rng: np.random.Generator) -> ToyCandidate:
        return ToyCandidate((9,))

    def crossover(
        self, a: ToyCandidate, b: ToyCandidate, rng: np.random.Generator
    ) -> ToyCandidate | None:
        return ToyCandidate((9,))

    def validate(self, candidate: ToyCandidate) -> None:
        raise ValueError("node 9 is outside the toy grammar")


def test_search_is_deterministic(tmp_path: Path) -> None:
    config = _config()

    with _open(tmp_path, "first") as first_archive:
        first = evolve(ToyFactory(), ToyEvaluator(), first_archive, config)
        first_trials = _trials(first_archive)

    with _open(tmp_path, "second") as second_archive:
        second = evolve(ToyFactory(), ToyEvaluator(), second_archive, config)
        second_trials = _trials(second_archive)

    assert first_trials == second_trials
    assert first.to_dict() == second.to_dict()
    assert first.best is not None and second.best is not None
    assert first.best.canonical_hash() == second.best.canonical_hash()
    assert len({candidate_id for candidate_id, _seed, _dataset in first_trials}) == len(
        first_trials
    )


def test_max_trials_counts_duplicates_exactly(tmp_path: Path) -> None:
    config = _config(
        population=3,
        generations=5,
        max_trials=7,
        patience=None,
        novelty_requirement=0,
    )
    with _open(tmp_path, "archive") as archive:
        evaluator = ToyEvaluator(threshold=0.0)
        result = evolve(FixedFactory(ToyCandidate((1, 2))), evaluator, archive, config)

        assert result.stop_reason == "budget"
        assert result.trials == 1
        assert result.skipped == 6
        assert result.trials + result.skipped == 7
        assert len(evaluator.calls) == 1
        assert archive.trial_count(evaluator.calls[0][0]) == 1
        total = archive.connection.execute("SELECT COUNT(*) FROM trials").fetchone()
        assert total is not None and int(total[0]) == 1


def test_invalid_candidates_are_counted_not_evaluated(tmp_path: Path) -> None:
    config = _config(population=4, generations=5, max_trials=100)
    with _open(tmp_path, "archive") as archive:
        evaluator = ToyEvaluator()
        result = evolve(_BrokenFactory(), evaluator, archive, config)

        assert result.stop_reason == "exhausted"
        assert result.trials == 0
        assert result.skipped == 4
        assert result.rejections == {}
        assert result.best is None
        assert result.generations == 1
        assert evaluator.calls == []


def test_complexity_cap_skips_without_evaluating(tmp_path: Path) -> None:
    config = _config(population=3, generations=5, max_trials=50, complexity_cap=2)
    with _open(tmp_path, "archive") as archive:
        evaluator = ToyEvaluator()
        result = evolve(FixedFactory(ToyCandidate((1, 2, 3))), evaluator, archive, config)

        assert result.stop_reason == "exhausted"
        assert result.trials == 0
        assert result.skipped == 3
        assert evaluator.calls == []


def test_should_stop_is_honoured_mid_generation(tmp_path: Path) -> None:
    polls = 0

    def should_stop() -> bool:
        nonlocal polls
        polls += 1
        return polls > 3

    config = _config(population=10, generations=10, max_trials=100, patience=None)
    with _open(tmp_path, "archive") as archive:
        result = evolve(ToyFactory(), ToyEvaluator(), archive, config, should_stop=should_stop)

        assert result.stop_reason == "stopped"
        assert result.trials + result.skipped == 3
        assert result.generations == 1
        assert polls == 4
        total = archive.connection.execute("SELECT COUNT(*) FROM trials").fetchone()
        assert total is not None and int(total[0]) == result.trials


def test_patience_stops_when_nothing_is_accepted(tmp_path: Path) -> None:
    config = _config(population=4, generations=10, max_trials=100, patience=2)
    with _open(tmp_path, "archive") as archive:
        result = evolve(ToyFactory(), ToyEvaluator(threshold=1e9), archive, config)

        assert result.stop_reason == "patience"
        assert result.generations == 2
        assert result.trials + result.skipped == 8
        assert result.accepted == 0
        assert result.best is None
        assert result.rejections == {"below_threshold": result.trials}


def test_best_matches_the_archive_and_budget_is_reported(tmp_path: Path) -> None:
    config = _config(population=5, generations=3, max_trials=12)
    with _open(tmp_path, "archive") as archive:
        evaluator = ToyEvaluator(threshold=0.0)
        result = evolve(ToyFactory(), evaluator, archive, config)

        assert result.best is not None
        assert result.best_id is not None
        assert result.best_metrics is not None
        assert result.best_metrics["net_profit"] == float(sum(result.best.nodes))
        stored = archive.best("net_profit")
        assert stored is not None and stored.candidate_id == result.best_id
        assert result.trials + result.skipped == 12
        assert result.stop_reason == "budget"

        budgets = [call[3] for call in evaluator.calls]
        assert budgets == sorted(budgets, reverse=True)
        assert budgets[0] == 12
        assert min(budgets) >= 1


def test_trials_are_unique_and_lineage_is_recorded(tmp_path: Path) -> None:
    config = _config(population=6, generations=3, max_trials=18)
    with _open(tmp_path, "archive") as archive:
        evaluator = ToyEvaluator(threshold=0.0)
        result = evolve(ToyFactory(), evaluator, archive, config)

        candidate_ids = [candidate_id for candidate_id, _seed, _dataset in _trials(archive)]
        assert len(candidate_ids) == len(set(candidate_ids))
        assert result.trials == len(candidate_ids)
        assert {call[1] for call in evaluator.calls} == {DATASET_ID}

        duplicated = archive.connection.execute(
            """
            SELECT candidate_id, dataset_id, COUNT(*) FROM trials
            GROUP BY candidate_id, dataset_id HAVING COUNT(*) > 1
            """
        ).fetchall()
        assert duplicated == []

        assert result.lineage_depth >= 1
        children = archive.connection.execute(
            "SELECT candidate_id FROM candidates WHERE parent_a IS NOT NULL"
        ).fetchall()
        assert children
        for row in children:
            assert archive.lineage(str(row[0])) != ()


def test_a_second_run_never_retrials_a_known_candidate(tmp_path: Path) -> None:
    config = _config(population=4, generations=3, max_trials=20)
    with _open(tmp_path, "archive") as archive:
        first = evolve(ToyFactory(), ToyEvaluator(threshold=0.0), archive, config)
        assert first.trials > 0
        before = {candidate_id for candidate_id, _seed, _dataset in _trials(archive)}

        second = evolve(ToyFactory(), ToyEvaluator(threshold=0.0), archive, config)
        trials = _trials(archive)
        new_ids = {candidate_id for candidate_id, _seed, _dataset in trials} - before

        assert second.trials == len(new_ids)
        assert new_ids.isdisjoint(before)
        assert second.skipped >= 4, "the first generation repeats the earlier candidates"
        assert len(trials) == first.trials + second.trials


def test_minimize_ranks_the_lowest_metric(tmp_path: Path) -> None:
    config = _config(population=4, generations=2, max_trials=8, minimize=True)
    with _open(tmp_path, "archive") as archive:
        evaluator = ToyEvaluator(
            threshold=-1e9, score=lambda candidate: float(sum(candidate.nodes))
        )
        result = evolve(ToyFactory(), evaluator, archive, config)
        assert result.best is not None
        assert result.best_metrics is not None
        assert result.best_metrics["net_profit"] == float(sum(result.best.nodes))
        assert result.accepted == result.trials


def test_evaluator_crash_propagates(tmp_path: Path) -> None:
    config = _config(population=2, generations=2, max_trials=10)
    with (
        _open(tmp_path, "archive") as archive,
        pytest.raises(RuntimeError, match="evaluator crash"),
    ):
        evolve(ToyFactory(), ToyEvaluator(crash=True), archive, config)


def test_result_is_json_safe(tmp_path: Path) -> None:
    config = _config(population=4, generations=2, max_trials=8)
    with _open(tmp_path, "archive") as archive:
        result = evolve(ToyFactory(), ToyEvaluator(threshold=0.0), archive, config)
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["dataset_id"] == DATASET_ID
        assert payload["trials"] == result.trials
        assert payload["stop_reason"] in {"budget", "patience", "exhausted", "stopped"}


def test_search_result_rejects_inconsistent_counts() -> None:
    with pytest.raises(ValueError, match="accepted cannot exceed trials"):
        SearchResult(
            dataset_id=DATASET_ID,
            code_version=CODE_VERSION,
            seed=1,
            trials=1,
            skipped=0,
            accepted=2,
        )
    with pytest.raises(ValueError, match="stop_reason"):
        SearchResult(
            dataset_id=DATASET_ID,
            code_version=CODE_VERSION,
            seed=1,
            trials=0,
            skipped=0,
            accepted=0,
            stop_reason="finished",
        )


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="population"):
        _config(population=1)
    with pytest.raises(ValueError, match="mutation_rate \\+ crossover_rate"):
        _config(mutation_rate=0.8, crossover_rate=0.5)
    with pytest.raises(ValueError, match="dataset_id"):
        _config(dataset_id="  ")
    with pytest.raises(ValueError, match="selection_metric"):
        _config(selection_metric='net"profit')
    with pytest.raises(ValueError, match="max_trials"):
        _config(max_trials=0)


def test_trial_result_invariants() -> None:
    with pytest.raises(ValueError, match="accepted trial cannot carry"):
        TrialResult(metrics={"net_profit": 1.0}, accepted=True, rejections=("why",))
    with pytest.raises(ValueError, match="must carry at least one"):
        TrialResult(metrics={"net_profit": 1.0}, accepted=False)
    with pytest.raises(ValueError, match="finite"):
        TrialResult(metrics={"net_profit": float("nan")}, accepted=True)
    with pytest.raises(TypeError, match="must be a number"):
        TrialResult(metrics={"net_profit": True}, accepted=True)
    accepted = TrialResult(metrics={"net_profit": 1}, accepted=True)
    assert accepted.metrics == {"net_profit": 1.0}
