"""Multiple-testing selection control: math, refusals, monotonicity and JSON.

The fixtures are deterministic and hand-computable. A 25-trade sample has mean
1, sample standard deviation 1, so the standard error is 0.2 and the t
statistic against zero is exactly 5. That lets the Bonferroni/Sidak thresholds
be checked against published Student-t table values rather than against the
implementation itself.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest

from scalping.discovery.selection import (
    MAX_SEARCH_BUDGET,
    SELECTION_CAVEAT,
    SelectionConfig,
    SelectionError,
    promotion_advice,
    selection_report,
)
from scalping.types import Side, Trade

START = datetime(2026, 1, 5, tzinfo=UTC)


def _trade(r_multiple: float, index: int) -> Trade:
    return Trade(
        entry_ts=START + timedelta(minutes=index),
        exit_ts=START + timedelta(minutes=index + 1),
        side=Side.LONG,
        lot=0.1,
        entry=2000.0,
        exit=2000.0,
        sl=1999.0,
        tp=2001.0,
        pnl=r_multiple * 100.0,
        costs=1.0,
        r_multiple=r_multiple,
        reason="test",
        strategy="test",
    )


def _series(values: list[float]) -> list[Trade]:
    return [_trade(value, index) for index, value in enumerate(values)]


def _unit_sample(mean_r: float = 1.0) -> list[float]:
    """25 values with the requested mean and sample standard deviation 1."""
    deviations = [1.0] * 12 + [0.0] + [-1.0] * 12
    return [mean_r + deviation for deviation in deviations]


def test_bonferroni_and_sidak_alpha_match_hand_values() -> None:
    report = selection_report(
        _series(_unit_sample()), SelectionConfig(trial_count=10, search_budget=20)
    )

    assert report.bonferroni_alpha == pytest.approx(0.005)
    assert report.sidak_alpha == pytest.approx(1.0 - 0.95 ** (1.0 / 10.0))
    # Bonferroni is the more conservative of the two and gates the report.
    assert report.adjusted_alpha == pytest.approx(min(report.bonferroni_alpha, report.sidak_alpha))
    assert report.adjusted_alpha == pytest.approx(report.bonferroni_alpha)


def test_bonferroni_p_value_is_the_trial_count_times_the_naive_p() -> None:
    report = selection_report(
        _series(_unit_sample()), SelectionConfig(trial_count=10, search_budget=20)
    )

    assert report.bonferroni_p_value == pytest.approx(min(1.0, 10 * report.naive_p_value))
    assert report.bonferroni_p_value > report.naive_p_value


def test_single_trial_reduces_to_the_unadjusted_alpha() -> None:
    report = selection_report(
        _series(_unit_sample()), SelectionConfig(trial_count=1, search_budget=1)
    )

    assert report.bonferroni_alpha == pytest.approx(0.05)
    assert report.sidak_alpha == pytest.approx(0.05)
    assert report.adjusted_alpha == pytest.approx(0.05)


def test_significance_threshold_matches_student_t_table_value() -> None:
    report = selection_report(
        _series(_unit_sample()), SelectionConfig(trial_count=1, search_budget=1)
    )

    assert report.trades == 25
    assert report.mean_r == pytest.approx(1.0)
    assert report.std_r == pytest.approx(1.0)
    assert report.standard_error == pytest.approx(0.2)
    assert report.t_statistic == pytest.approx(5.0)
    # One-sided 95% Student-t quantile with 24 degrees of freedom.
    expected = 1.7108820799 * 0.2
    assert report.significance_threshold_r == pytest.approx(expected, rel=1e-9)
    assert report.deflated_threshold_r == pytest.approx(expected, rel=1e-9)
    assert report.naive_p_value == pytest.approx(2.07842817e-05, rel=1e-6)


def test_deflated_threshold_hand_value_for_trials_and_complexity() -> None:
    report = selection_report(
        _series(_unit_sample()),
        SelectionConfig(trial_count=10, search_budget=20, complexity=10),
    )

    # t(0.995, 24) = 2.7969395048, se = 0.2.
    significance = 2.7969395048 * 0.2
    penalty = 0.5 * 0.2 * (math.sqrt(2.0 * math.log(10)) + 0.25 * math.log(11))
    assert report.significance_threshold_r == pytest.approx(significance, rel=1e-11)
    assert report.deflated_threshold_r == pytest.approx(significance + penalty, rel=1e-11)


def test_penalty_is_monotonic_in_trials() -> None:
    trades = _series(_unit_sample())
    penalties = []
    for trial_count in (1, 2, 5, 10, 100):
        report = selection_report(
            trades,
            SelectionConfig(trial_count=trial_count, search_budget=100),
        )
        penalties.append(report.deflated_threshold_r - report.significance_threshold_r)

    assert penalties[0] == pytest.approx(0.0)
    assert all(later >= earlier for earlier, later in pairwise(penalties))
    assert penalties[-1] > penalties[1]


def test_penalty_is_monotonic_in_complexity() -> None:
    trades = _series(_unit_sample())
    penalties = []
    for complexity in (0, 1, 5, 10):
        report = selection_report(
            trades,
            SelectionConfig(trial_count=10, search_budget=100, complexity=complexity),
        )
        penalties.append(report.deflated_threshold_r - report.significance_threshold_r)

    assert all(later >= earlier for earlier, later in pairwise(penalties))
    assert penalties[-1] > penalties[0]


def test_many_trials_do_not_false_pass() -> None:
    report = selection_report(
        _series(_unit_sample()),
        SelectionConfig(trial_count=MAX_SEARCH_BUDGET, search_budget=MAX_SEARCH_BUDGET),
    )

    assert report.trial_count == MAX_SEARCH_BUDGET
    assert report.mean_r == pytest.approx(1.0)
    assert report.deflated_threshold_r > report.mean_r
    assert report.passed is False
    assert report.bonferroni_p_value == pytest.approx(1.0)
    assert promotion_advice(report).recommendation != "accept"


def test_min_effect_is_a_floor_not_a_target() -> None:
    report = selection_report(
        _series(_unit_sample()),
        SelectionConfig(trial_count=1, search_budget=1, min_effect=2.0),
    )

    assert report.deflated_threshold_r == pytest.approx(2.0)
    assert report.passed is False


def test_insufficient_sample_is_refused() -> None:
    with pytest.raises(SelectionError) as caught:
        selection_report(_series(_unit_sample()[:14]), SelectionConfig(folds=3))

    assert caught.value.code == "insufficient_trades"


def test_no_trades_are_refused() -> None:
    with pytest.raises(SelectionError) as caught:
        selection_report([], SelectionConfig())

    assert caught.value.code == "insufficient_trades"


def test_degenerate_zero_variance_sample_is_refused() -> None:
    with pytest.raises(SelectionError) as caught:
        selection_report(_series([1.0] * 25), SelectionConfig())

    assert caught.value.code == "degenerate_sample"


def test_invalid_trial_exposure_is_refused() -> None:
    trades = _series(_unit_sample())

    with pytest.raises(SelectionError) as caught:
        selection_report(trades, SelectionConfig(trial_count=0, search_budget=10))
    assert caught.value.code == "invalid_trial_count"

    with pytest.raises(SelectionError) as caught:
        selection_report(trades, SelectionConfig(trial_count=11, search_budget=10))
    assert caught.value.code == "trial_count_exceeds_budget"

    with pytest.raises(SelectionError) as caught:
        selection_report(trades, SelectionConfig(trial_count=1, search_budget=0))
    assert caught.value.code == "invalid_search_budget"


@pytest.mark.parametrize(
    ("config", "code"),
    [
        (SelectionConfig(family_alpha=float("nan")), "invalid_family_alpha"),
        (SelectionConfig(family_alpha=0.0), "invalid_family_alpha"),
        (SelectionConfig(min_effect=float("inf")), "invalid_min_effect"),
        (SelectionConfig(min_effect=-0.1), "invalid_min_effect"),
        (SelectionConfig(complexity=-1), "invalid_complexity"),
        (SelectionConfig(complexity=1_000_001), "invalid_complexity"),
        (SelectionConfig(folds=0), "invalid_folds"),
        (SelectionConfig(seed=-1), "invalid_seed"),
        (SelectionConfig(seed=2**63), "invalid_seed"),
    ],
)
def test_non_finite_or_out_of_range_config_is_refused(config: SelectionConfig, code: str) -> None:
    with pytest.raises(SelectionError) as caught:
        selection_report(_series(_unit_sample()), config)

    assert caught.value.code == code


def test_non_finite_trade_return_is_refused() -> None:
    trades = _series(_unit_sample())
    trades[3] = _trade(float("nan"), 3)

    with pytest.raises(SelectionError) as caught:
        selection_report(trades, SelectionConfig())

    assert caught.value.code == "non_finite_trade_return"


def test_non_trade_input_is_refused() -> None:
    with pytest.raises(SelectionError) as caught:
        selection_report([1.0, 2.0], SelectionConfig())  # type: ignore[list-item]

    assert caught.value.code == "invalid_trades"


def test_promotion_advice_accept_is_only_a_recommendation() -> None:
    report = selection_report(
        _series(_unit_sample()), SelectionConfig(trial_count=1, search_budget=1)
    )
    advice = promotion_advice(report)

    assert report.passed is True
    assert advice.recommendation == "accept"
    assert advice.passed is True
    assert "policy remains the gate" in " ".join(advice.reasons)
    assert advice.caveat == SELECTION_CAVEAT


def test_promotion_advice_rejects_when_the_effect_is_ruled_out() -> None:
    report = selection_report(
        _series(_unit_sample(-1.0)), SelectionConfig(trial_count=1, search_budget=1)
    )
    advice = promotion_advice(report)

    assert report.passed is False
    assert advice.recommendation == "reject"
    assert "rules out" in " ".join(advice.reasons)


def test_promotion_advice_is_insufficient_when_the_interval_straddles_zero() -> None:
    report = selection_report(
        _series(_unit_sample(0.2)), SelectionConfig(trial_count=1, search_budget=1)
    )
    advice = promotion_advice(report)

    assert report.passed is False
    assert report.naive_ci[0] < 0.0 < report.naive_ci[1]
    assert advice.recommendation == "insufficient_evidence"


def test_promotion_advice_honours_a_stricter_trade_floor() -> None:
    report = selection_report(
        _series(_unit_sample()), SelectionConfig(trial_count=1, search_budget=1)
    )
    advice = promotion_advice(report, min_trades=1000)

    assert advice.recommendation == "insufficient_evidence"
    assert advice.passed is False


def test_promotion_advice_rejects_a_non_report() -> None:
    with pytest.raises(TypeError):
        promotion_advice(object())  # type: ignore[arg-type]


def test_report_is_deterministic() -> None:
    trades = _series(_unit_sample())
    config = SelectionConfig(trial_count=5, search_budget=10, seed=7)

    first = selection_report(trades, config)
    second = selection_report(trades, config)

    assert first.to_dict() == second.to_dict()
    assert first.seed == 7


def test_report_and_advice_are_json_safe() -> None:
    report = selection_report(
        _series(_unit_sample()), SelectionConfig(trial_count=10, search_budget=20, seed=3)
    )

    payload = json.loads(json.dumps(report.to_dict(), allow_nan=False, sort_keys=True))
    assert payload["trades"] == 25
    assert payload["trial_count"] == 10
    assert payload["seed"] == 3
    assert isinstance(payload["naive_ci"], list)
    assert isinstance(payload["fold_trades"], list)

    advice_payload = json.loads(
        json.dumps(promotion_advice(report).to_dict(), allow_nan=False, sort_keys=True)
    )
    assert advice_payload["recommendation"] in {"accept", "reject", "insufficient_evidence"}
