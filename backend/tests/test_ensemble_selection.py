"""Automatic simpler-winner selection between an ensemble and its members."""

from __future__ import annotations

import pytest

from scalping.ensemble import PromotionChoice, select_simpler_winner


def table(ensemble: float, *members: tuple[str, float], metric: str = "accuracy"):
    rows = [{"name": name, metric: value} for name, value in members]
    rows.append({"name": "ensemble", metric: ensemble})
    return rows


def test_ensemble_is_kept_only_when_it_strictly_wins() -> None:
    choice = select_simpler_winner(table(0.60, ("a", 0.58), ("b", 0.55)))

    assert isinstance(choice, PromotionChoice)
    assert choice.choice == "ensemble"
    assert choice.reason == "ensemble_beats_best_member"
    assert choice.ensemble_gain == pytest.approx(0.02)
    assert choice.best_member == "a"


def test_ties_and_losses_keep_the_best_member() -> None:
    tie = select_simpler_winner(table(0.58, ("a", 0.58), ("b", 0.55)))
    worse = select_simpler_winner(table(0.50, ("a", 0.58), ("b", 0.55)))

    assert (tie.choice, tie.reason) == ("a", "keep_simpler_member")
    assert (worse.choice, worse.reason) == ("a", "keep_simpler_member")
    assert worse.ensemble_gain < 0


def test_min_gain_blocks_a_small_improvement() -> None:
    choice = select_simpler_winner(
        table(0.60, ("a", 0.58)), min_gain=0.03
    )

    assert choice.choice == "a"
    assert choice.reason == "keep_simpler_member"


def test_log_loss_is_lower_better() -> None:
    wins = select_simpler_winner(
        table(0.40, ("a", 0.50), metric="log_loss"), metric="log_loss"
    )
    loses = select_simpler_winner(
        table(0.60, ("a", 0.50), metric="log_loss"), metric="log_loss"
    )

    assert wins.choice == "ensemble"
    assert wins.ensemble_gain == pytest.approx(0.10)
    assert loses.choice == "a"


def test_balanced_accuracy_is_supported() -> None:
    choice = select_simpler_winner(
        table(0.60, ("a", 0.55), metric="balanced_accuracy"), metric="balanced_accuracy"
    )

    assert choice.choice == "ensemble"
    assert choice.metric == "balanced_accuracy"


def test_choice_serialises() -> None:
    choice = select_simpler_winner(table(0.60, ("a", 0.58))).to_dict()

    assert set(choice) == {"choice", "reason", "metric", "ensemble_gain", "best_member"}


def test_invalid_tables_and_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        select_simpler_winner([])
    with pytest.raises(ValueError, match="exactly one"):
        select_simpler_winner([{"name": "a", "accuracy": 0.5}])
    with pytest.raises(ValueError, match="exactly one"):
        select_simpler_winner(
            [{"name": "ensemble", "accuracy": 0.5}, {"name": "ensemble", "accuracy": 0.6}]
        )
    with pytest.raises(ValueError, match="metric"):
        select_simpler_winner(table(0.6, ("a", 0.5)), metric="brier")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_gain"):
        select_simpler_winner(table(0.6, ("a", 0.5)), min_gain=-0.1)
    with pytest.raises(ValueError, match="no 'accuracy'"):
        select_simpler_winner([{"name": "a"}, {"name": "ensemble", "accuracy": 0.5}])
    with pytest.raises(ValueError, match="non-finite"):
        select_simpler_winner(
            [{"name": "a", "accuracy": float("nan")}, {"name": "ensemble", "accuracy": 0.5}]
        )
