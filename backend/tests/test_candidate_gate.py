"""Candidate → champion gate: thresholds, missing evidence, forced promotion."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from scalping.registry import (
    GateCriteria,
    GateDecision,
    ModelRegistry,
    PromotionRefused,
    RegistryError,
    UnknownModelError,
    evaluate_candidate,
    promote_candidate,
)

DATASET_ID = "XAUUSD:M1:h1:gate00000000"

GOOD_METRICS: dict[str, Any] = {
    "trades": 250,
    "net_profit": 1500.0,
    "profit_factor": 1.6,
    "win_rate": 0.54,
    "expectancy_r": 0.12,
    "max_drawdown": 0.08,
    "total_costs": 120.0,
}
GOOD_STRESS = {"x1.5": {"net_profit": 900.0}, "x2": {"net_profit": 200.0}}
GOOD_CALIBRATION = {"ece": 0.03, "mce": 0.09}
GOOD_REGIME = {"rows": [{"regime": "trend", "trades": 240}], "unmatched_trades": 10}

POLICY = GateCriteria(
    min_trades=100,
    min_expectancy_r=0.05,
    min_profit_factor=1.3,
    max_drawdown=0.25,
    max_ece=0.1,
    require_positive_cost_stress=(1.5,),
    min_regime_coverage=0.9,
)


def _metrics(**overrides: Any) -> dict[str, Any]:
    metrics = dict(GOOD_METRICS)
    metrics.update(overrides)
    return metrics


def _registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(tmp_path / "registry.sqlite3")


@pytest.fixture
def registry(tmp_path: Path) -> Iterator[ModelRegistry]:
    handle = _registry(tmp_path)
    try:
        yield handle
    finally:
        handle.close()


def _seed_models(registry: ModelRegistry, *model_ids: str) -> None:
    registry.register_dataset(
        DATASET_ID,
        symbol="XAUUSD",
        timeframe="M1",
        digest="gate00000000",
        path="/data/gate.parquet",
        rows=100_000,
        synthetic=False,
    )
    for model_id in model_ids:
        registry.register_model(
            model_id,
            dataset_id=DATASET_ID,
            name="lightgbm",
            path=f"/models/{model_id}.pkl",
            metrics={"accuracy": 0.52},
        )


def _accepted() -> GateDecision:
    return evaluate_candidate(POLICY, GOOD_METRICS, GOOD_STRESS, GOOD_CALIBRATION, GOOD_REGIME)


def _rejected() -> GateDecision:
    return evaluate_candidate(
        POLICY,
        _metrics(trades=5, expectancy_r=-0.1, profit_factor=0.5, max_drawdown=0.6),
        {"x1.5": {"net_profit": -20.0}},
        {"ece": 0.4},
        {"rows": [{"regime": "trend", "trades": 1}], "unmatched_trades": 9},
    )


# --------------------------------------------------------------------- criteria


def test_default_criteria_accepts_everything() -> None:
    criteria = GateCriteria()
    assert criteria.is_empty()
    decision = evaluate_candidate(criteria, {})
    assert decision.accepted is True
    assert decision.rejections == ()
    assert decision.codes == ()
    assert decision.summary.startswith("gate accepted")


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"trades": 10}, "min_trades"),
        ({"expectancy_r": -0.01}, "min_expectancy_r"),
        ({"profit_factor": 1.0}, "min_profit_factor"),
        ({"max_drawdown": 0.5}, "max_drawdown"),
    ],
)
def test_metric_rejections(overrides: dict[str, Any], code: str) -> None:
    decision = evaluate_candidate(
        POLICY, _metrics(**overrides), GOOD_STRESS, GOOD_CALIBRATION, GOOD_REGIME
    )
    assert decision.accepted is False
    assert code in decision.codes
    assert any(item.startswith(f"{code}: ") for item in decision.rejections)


def test_calibration_rejection() -> None:
    decision = evaluate_candidate(POLICY, GOOD_METRICS, GOOD_STRESS, {"ece": 0.2}, GOOD_REGIME)
    assert decision.accepted is False
    assert decision.codes == ("max_ece",)
    assert "0.2" in decision.rejections[0]


def test_cost_stress_rejection() -> None:
    decision = evaluate_candidate(
        POLICY,
        GOOD_METRICS,
        {"x1.5": {"net_profit": -0.01}},
        GOOD_CALIBRATION,
        GOOD_REGIME,
    )
    assert decision.accepted is False
    assert decision.codes == ("cost_stress",)
    assert "x1.5" in decision.rejections[0]


def test_regime_coverage_rejection() -> None:
    decision = evaluate_candidate(
        POLICY,
        _metrics(trades=100),
        GOOD_STRESS,
        GOOD_CALIBRATION,
        {"rows": [{"regime": "trend", "trades": 40}], "unmatched_trades": 60},
    )
    assert decision.accepted is False
    assert decision.codes == ("min_regime_coverage",)
    assert "0.400" in decision.rejections[0]


def test_regime_coverage_uses_the_headline_trade_count() -> None:
    # The table claims 100 trades while the headline says 250: the missing 150
    # are unlabelled, so coverage is 40/250, not 40/100.
    decision = evaluate_candidate(
        POLICY,
        GOOD_METRICS,
        GOOD_STRESS,
        GOOD_CALIBRATION,
        {"rows": [{"regime": "trend", "trades": 40}], "unmatched_trades": 60},
    )
    assert decision.accepted is False
    assert "coverage 0.160" in decision.rejections[0]


@pytest.mark.parametrize(
    ("criteria", "code"),
    [
        (GateCriteria(min_trades=10), "missing_metric"),
        (GateCriteria(min_expectancy_r=0.01), "missing_metric"),
        (GateCriteria(min_profit_factor=1.1), "missing_metric"),
        (GateCriteria(max_drawdown=0.2), "missing_metric"),
    ],
)
def test_missing_metrics_reject(criteria: GateCriteria, code: str) -> None:
    decision = evaluate_candidate(criteria, {})
    assert decision.accepted is False
    assert code in decision.codes


def test_missing_calibration_rejects() -> None:
    decision = evaluate_candidate(GateCriteria(max_ece=0.1), GOOD_METRICS, None, None, GOOD_REGIME)
    assert decision.accepted is False
    assert decision.codes == ("missing_calibration",)


def test_missing_cost_stress_rejects() -> None:
    criteria = GateCriteria(require_positive_cost_stress=(1.5, 2.0))
    decision = evaluate_candidate(criteria, GOOD_METRICS, {"x1.5": {"net_profit": 10.0}})
    assert decision.accepted is False
    assert decision.codes == ("missing_cost_stress",)
    assert "x2" in decision.rejections[0]


def test_missing_regime_rejects() -> None:
    decision = evaluate_candidate(GateCriteria(min_regime_coverage=0.5), GOOD_METRICS)
    assert decision.accepted is False
    assert decision.codes == ("missing_regime",)


def test_missing_trades_rejects_regime_coverage_too() -> None:
    decision = evaluate_candidate(
        GateCriteria(min_regime_coverage=0.5), {}, None, None, {"rows": [], "unmatched_trades": 0}
    )
    assert decision.accepted is False
    assert decision.codes == ("missing_regime",)


def test_every_rejection_is_reported() -> None:
    decision = _rejected()
    assert decision.accepted is False
    assert decision.codes == (
        "min_trades",
        "min_expectancy_r",
        "min_profit_factor",
        "max_drawdown",
        "max_ece",
        "cost_stress",
        "min_regime_coverage",
    )
    assert len(decision.rejections) == len(decision.codes)
    assert decision.summary.startswith("gate rejected (")
    for code in decision.codes:
        assert code in decision.summary


def test_boundary_values_pass_and_a_hair_beyond_fails() -> None:
    at_limit = _metrics(trades=100, expectancy_r=0.05, profit_factor=1.3, max_drawdown=0.25)
    decision = evaluate_candidate(
        POLICY,
        at_limit,
        {"x1.5": {"net_profit": 1e-9}},
        {"ece": 0.1},
        {"rows": [{"regime": "trend", "trades": 90}], "unmatched_trades": 10},
    )
    assert decision.accepted is True, decision.rejections

    just_below = _metrics(trades=99, expectancy_r=0.0499, profit_factor=1.2999, max_drawdown=0.2501)
    decision = evaluate_candidate(
        POLICY,
        just_below,
        {"x1.5": {"net_profit": 0.0}},
        {"ece": 0.1001},
        {"rows": [{"regime": "trend", "trades": 89}], "unmatched_trades": 11},
    )
    assert decision.accepted is False
    assert decision.codes == (
        "min_trades",
        "min_expectancy_r",
        "min_profit_factor",
        "max_drawdown",
        "max_ece",
        "cost_stress",
        "min_regime_coverage",
    )


def test_stress_must_be_strictly_positive() -> None:
    criteria = GateCriteria(require_positive_cost_stress=(1.5,))
    assert evaluate_candidate(criteria, {}, {"x1.5": {"net_profit": 0.0}}).accepted is False
    assert evaluate_candidate(criteria, {}, {"x1.5": {"net_profit": 0.01}}).accepted is True


def test_cost_stress_multipliers_may_be_a_list() -> None:
    criteria = GateCriteria(require_positive_cost_stress=[1.5])  # type: ignore[arg-type]
    assert criteria.require_positive_cost_stress == (1.5,)
    assert evaluate_candidate(criteria, {}, {"x1.5": {"net_profit": 5.0}}).accepted is True


def test_decision_to_dict_is_json_serializable() -> None:
    payload = _rejected().to_dict()
    encoded = json.dumps(payload, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["accepted"] is False
    assert decoded["codes"][0] == "min_trades"
    assert decoded["criteria"]["min_trades"] == 100
    assert decoded["criteria"]["require_positive_cost_stress"] == [1.5]
    assert "gate rejected" in decoded["summary"]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"min_trades": -1}, ValueError),
        ({"min_trades": 1.5}, TypeError),
        ({"max_drawdown": 1.5}, ValueError),
        ({"max_ece": -0.1}, ValueError),
        ({"min_regime_coverage": 2.0}, ValueError),
        ({"min_expectancy_r": float("nan")}, ValueError),
        ({"require_positive_cost_stress": (0.0,)}, ValueError),
        ({"require_positive_cost_stress": (-1.0,)}, ValueError),
    ],
)
def test_criteria_validation(kwargs: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        GateCriteria(**kwargs)


# -------------------------------------------------------------------- promotion


def test_accepted_candidate_is_promoted_with_the_gate_summary(registry: ModelRegistry) -> None:
    _seed_models(registry, "candidate")
    decision = _accepted()
    record = promote_candidate(registry, "candidate", decision, reason="beats the baseline")

    assert record.model_id == "candidate"
    assert record.previous_model_id is None
    assert record.reason.startswith("beats the baseline; gate accepted")
    assert "trades=250" in record.reason
    champion = registry.champion()
    assert champion is not None and champion.model_id == "candidate"
    assert registry.champion_promotion().reason == record.reason  # type: ignore[union-attr]


def test_rejected_candidate_is_not_promoted(registry: ModelRegistry) -> None:
    _seed_models(registry, "candidate")
    with pytest.raises(PromotionRefused) as excinfo:
        promote_candidate(registry, "candidate", _rejected(), reason="please")

    assert isinstance(excinfo.value, RegistryError)
    assert "min_trades" in str(excinfo.value)
    assert registry.champion() is None
    assert registry.list_promotions() == []


def test_forced_promotion_requires_a_human_reason(registry: ModelRegistry) -> None:
    _seed_models(registry, "candidate")
    for reason in (None, "", "   "):
        with pytest.raises(ValueError, match="non-empty reason"):
            promote_candidate(registry, "candidate", _rejected(), reason=reason, force=True)
    assert registry.champion() is None
    assert registry.list_promotions() == []


def test_forced_rejected_promotion_is_recorded_as_forced(registry: ModelRegistry) -> None:
    _seed_models(registry, "candidate")
    record = promote_candidate(
        registry,
        "candidate",
        _rejected(),
        reason="manual override after review",
        force=True,
    )
    assert record.reason.startswith("forced: manual override after review; gate rejected")
    assert "min_trades" in record.reason
    champion = registry.champion()
    assert champion is not None and champion.model_id == "candidate"


def test_forced_accepted_promotion_is_also_marked_forced(registry: ModelRegistry) -> None:
    _seed_models(registry, "candidate")
    record = promote_candidate(
        registry, "candidate", _accepted(), reason="nightly roll", force=True
    )
    assert record.reason.startswith("forced: nightly roll; gate accepted")


def test_promotion_chain_keeps_the_previous_champion(registry: ModelRegistry) -> None:
    _seed_models(registry, "model-a", "model-b")
    first = promote_candidate(registry, "model-a", _accepted(), reason="a wins")
    second = promote_candidate(registry, "model-b", _accepted(), reason="b wins")

    assert first.previous_model_id is None
    assert second.previous_model_id == "model-a"
    history = registry.list_promotions()
    assert [item.model_id for item in history] == ["model-a", "model-b"]
    assert all("gate accepted" in item.reason for item in history)


def test_unknown_model_is_reported_through_the_registry(registry: ModelRegistry) -> None:
    _seed_models(registry, "candidate")
    with pytest.raises(UnknownModelError):
        promote_candidate(registry, "missing", _accepted())
    assert registry.champion() is None


def test_promote_candidate_type_checks(registry: ModelRegistry) -> None:
    _seed_models(registry, "candidate")
    with pytest.raises(TypeError):
        promote_candidate(registry, "candidate", "accepted")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        promote_candidate(registry, "candidate", _accepted(), reason=42)  # type: ignore[arg-type]
