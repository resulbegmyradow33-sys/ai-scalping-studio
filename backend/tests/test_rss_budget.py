"""Reusable RSS budget: validation, exceeded state and raise."""

from __future__ import annotations

import pytest

from scalping.bench.budget import MemoryBudgetError, RssBudget, process_rss_mb


def test_process_rss_is_plausible() -> None:
    assert process_rss_mb() > 0


def test_exceeded_and_check() -> None:
    budget = RssBudget(max_mb=1)

    assert budget.limit_mb == 1
    assert budget.exceeded is True
    with pytest.raises(MemoryBudgetError, match="exceeds budget"):
        budget.check()


def test_reserve_reduces_the_limit() -> None:
    budget = RssBudget(max_mb=100_000, reserve_mb=1_000)

    assert budget.limit_mb == 99_000
    assert budget.exceeded is False
    budget.check()
    assert budget.to_dict()["limit_mb"] == 99_000


def test_validation() -> None:
    with pytest.raises(ValueError, match="max_mb"):
        RssBudget(max_mb=0)
    with pytest.raises(TypeError, match="max_mb"):
        RssBudget(max_mb=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reserve_mb"):
        RssBudget(max_mb=10, reserve_mb=10)
    with pytest.raises(TypeError, match="reserve_mb"):
        RssBudget(max_mb=10, reserve_mb=True)  # type: ignore[arg-type]
