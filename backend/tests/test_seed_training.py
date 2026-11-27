"""Seed training helper: ordering, uniqueness and error propagation."""

from __future__ import annotations

import pytest

from scalping.rl.seed_training import SeedRun, train_seed_runs


def test_runs_each_seed_once_in_order() -> None:
    calls: list[int] = []

    def train(seed: int) -> dict[str, int]:
        calls.append(seed)
        return {"seed": seed}

    runs = train_seed_runs(train, [3, 1, 2])

    assert calls == [3, 1, 2]
    assert [run.seed for run in runs] == [3, 1, 2]
    assert all(isinstance(run, SeedRun) for run in runs)
    assert runs[0].result == {"seed": 3}


def test_duplicate_and_invalid_seeds_are_rejected_before_training() -> None:
    def train(seed: int) -> int:
        return seed

    with pytest.raises(ValueError, match="duplicate seed"):
        train_seed_runs(train, [1, 1])
    with pytest.raises(ValueError, match="at least one"):
        train_seed_runs(train, [])
    with pytest.raises(ValueError, match=">= 0"):
        train_seed_runs(train, [-1])
    with pytest.raises(TypeError, match="seed"):
        train_seed_runs(train, [True])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="callable"):
        train_seed_runs(1, [0])  # type: ignore[arg-type]


def test_a_trainer_error_propagates() -> None:
    def train(seed: int) -> int:
        if seed == 2:
            raise RuntimeError("boom")
        return seed

    with pytest.raises(RuntimeError, match="boom"):
        train_seed_runs(train, [0, 1, 2])
