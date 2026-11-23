"""DSL evaluator wiring: simulation, dataset guard, gate and cost stress.

The frame-level tests use a constant-price M1 frame so a hand-built strategy has
a known sign: an always-long candidate pays the spread on entry and exits at the
bid, so it loses deterministically; an always-flat candidate does nothing. That
makes the assertions about trades, gate codes and stress direction exact instead
of statistical.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from scalping.config import Settings
from scalping.data.synthetic import SyntheticProvider
from scalping.discovery import DiscoveryArchive, SearchConfig, TrialResult, evolve
from scalping.discovery.evaluate import DslEvaluator
from scalping.discovery.search import STOP_REASONS
from scalping.features.build import FeatureFrame, build_features
from scalping.registry.gate import GateCriteria
from scalping.strategy.dsl import (
    Compare,
    Const,
    EntryRule,
    ExitRule,
    HoldingLimit,
    StrategyFactory,
    StrategySpec,
)

DATASET_ID = "validation:XAUUSD:M1:abcd1234"
CODE_VERSION = "test-2026-11-23"
SYMBOL = "XAUUSD"
TIMEFRAME = "M1"
START = datetime(2026, 1, 5, tzinfo=UTC)

#: Conditions that are always true / always false as data, not as code.
ALWAYS = Compare(Const(0.0), "<", Const(1.0))
NEVER = Compare(Const(0.0), ">", Const(1.0))


def flat_frame(rows: int = 120, *, price: float = 1000.0) -> FeatureFrame:
    """Constant-price bars with unreachable stops; only the quote exit can close.

    ``atr`` is 5.0 and the bar range is one thousandth, so SL/TP never trigger
    and a holding limit is the only exit. The entry pays ask + slippage and the
    exit returns the bid, so a long loses spread plus slippage plus commission.
    """
    ts = np.datetime64("2026-01-05T00:00", "us") + np.arange(rows).astype("timedelta64[m]")
    ohlc = np.column_stack(
        [
            np.full(rows, price),
            np.full(rows, price + 0.001),
            np.full(rows, price - 0.001),
            np.full(rows, price),
        ]
    )
    return FeatureFrame(
        X=np.zeros((rows, 1), dtype=np.float32),
        ts=ts.astype("datetime64[us]"),
        ohlc=ohlc,
        spread=np.full(rows, 0.20),
        atr=np.full(rows, 5.0),
        valid_from=0,
        columns=["ret_1"],
    )


def long_spec(bars: int = 1) -> StrategySpec:
    """Always enter long when flat; the holding limit is the only exit."""
    return StrategySpec(
        entries=(EntryRule("long", ALWAYS),),
        exits=(ExitRule(NEVER),),
        holding=HoldingLimit(bars),
        max_lookback=5,
    )


def flat_spec() -> StrategySpec:
    """A valid strategy whose entry never fires, so the simulator stays flat."""
    return StrategySpec(
        entries=(EntryRule("long", NEVER),),
        exits=(ExitRule(NEVER),),
        holding=HoldingLimit(1),
        max_lookback=5,
    )


def evaluator(features: FeatureFrame, **overrides: object) -> DslEvaluator:
    values: dict[str, object] = {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "dataset_id": DATASET_ID,
    }
    values.update(overrides)
    return DslEvaluator(features, Settings(), **values)  # type: ignore[arg-type]


def codes(result: TrialResult) -> tuple[str, ...]:
    return tuple(reason.split(":", 1)[0].strip() for reason in result.rejections)


def run(
    features: FeatureFrame,
    spec: StrategySpec,
    **overrides: object,
) -> TrialResult:
    return evaluator(features, **overrides).evaluate(
        spec, dataset_id=DATASET_ID, code_version=CODE_VERSION, budget=1
    )


def test_always_long_trades_and_always_flat_does_not() -> None:
    frame = flat_frame()

    traded = run(frame, long_spec())
    assert traded.accepted is True
    assert traded.rejections == ()
    assert traded.metrics["trades"] > 0
    assert traded.metrics["net_profit"] < 0.0, "paying the spread on every round turn loses"
    assert traded.metrics["bars"] > 0

    idle = run(frame, flat_spec())
    assert idle.accepted is True
    assert idle.metrics["trades"] == 0
    assert idle.metrics["net_profit"] == 0.0
    assert idle.metrics["expectancy_r"] == 0.0


def test_metrics_are_deterministic_on_a_fixed_seed_frame() -> None:
    bars = SyntheticProvider(seed=11).fetch(SYMBOL, START, START + timedelta(days=1))
    frame = build_features(bars)
    spec = long_spec(bars=5)

    first = run(frame, spec, max_steps=300)
    second = run(frame, spec, max_steps=300)

    assert first.metrics == second.metrics
    assert first.accepted is second.accepted
    assert {"net_profit", "trades", "expectancy_r", "profit_factor", "bars"} <= set(first.metrics)
    assert first.metrics["bars"] == 300.0


def test_future_bars_do_not_change_the_run_up_to_that_point() -> None:
    """Mutate every bar after the truncation point; the run must not notice."""
    base = flat_frame(rows=300)
    future = 200
    matrix = base.X.copy()
    matrix[future:] = 5.0
    ohlc = base.ohlc.copy()
    ohlc[future:] = 2000.0
    spread = base.spread.copy()
    spread[future:] = 1.0
    atr = base.atr.copy()
    atr[future:] = 50.0
    mutated = replace(base, X=matrix, ohlc=ohlc, spread=spread, atr=atr)

    spec = long_spec()
    original = run(base, spec, max_steps=100)
    changed = run(mutated, spec, max_steps=100)

    assert original.metrics == changed.metrics
    assert original.metrics["bars"] == 100.0
    assert original.metrics["trades"] > 0, "the comparison must not be vacuous"


def test_dataset_id_mismatch_is_refused() -> None:
    instance = evaluator(flat_frame())
    with pytest.raises(ValueError, match="bound to dataset"):
        instance.evaluate(
            long_spec(),
            dataset_id="validation:XAUUSD:M5:other",
            code_version=CODE_VERSION,
            budget=1,
        )


def test_test_and_holdout_dataset_ids_are_refused_at_construction() -> None:
    for dataset_id in (
        "test:XAUUSD:M1:abcd1234",
        "test-set:XAUUSD:M1:abcd1234",
        "holdout:XAUUSD:M1:abcd1234",
        "final-holdout:XAUUSD:M1:abcd1234",
        "hold-out:XAUUSD:M1:abcd1234",
    ):
        with pytest.raises(ValueError, match="test/holdout"):
            evaluator(flat_frame(), dataset_id=dataset_id)

    # Token matching, not a substring search: "latest" is a legitimate label.
    allowed = evaluator(flat_frame(), dataset_id="validation:latest:XAUUSD:M1:abcd1234")
    assert allowed.dataset_id == "validation:latest:XAUUSD:M1:abcd1234"


def test_gate_rejects_min_trades_and_expectancy_with_its_codes() -> None:
    frame = flat_frame()

    too_few = run(frame, flat_spec(), gate=GateCriteria(min_trades=5))
    assert too_few.accepted is False
    assert codes(too_few) == ("min_trades",)

    weak = run(frame, long_spec(), gate=GateCriteria(min_expectancy_r=0.5))
    assert weak.accepted is False
    assert codes(weak) == ("min_expectancy_r",)
    assert weak.metrics["trades"] > 0


def test_gate_rejects_missing_calibration_and_regime_evidence() -> None:
    frame = flat_frame()

    uncalibrated = run(frame, flat_spec(), gate=GateCriteria(max_ece=0.1))
    assert uncalibrated.accepted is False
    assert codes(uncalibrated) == ("missing_calibration",)

    unregimed = run(frame, flat_spec(), gate=GateCriteria(min_regime_coverage=0.5))
    assert unregimed.accepted is False
    assert codes(unregimed) == ("missing_regime",)


def test_cost_stress_metrics_are_present_and_negative_net_is_rejected() -> None:
    frame = flat_frame()
    gate = GateCriteria(require_positive_cost_stress=(2.0,))
    result = run(frame, long_spec(), gate=gate, stress=(2.0,))

    assert result.accepted is False
    assert codes(result) == ("cost_stress",)
    assert result.metrics["net_profit"] < 0.0
    assert result.metrics["stress_x2_net_profit"] < result.metrics["net_profit"]
    assert 0 < result.metrics["stress_x2_trades"] <= result.metrics["trades"]
    assert "stress_x2_bars" in result.metrics


def test_gate_rejects_a_required_stress_run_that_was_not_requested() -> None:
    result = run(
        flat_frame(),
        long_spec(),
        gate=GateCriteria(require_positive_cost_stress=(3.0,)),
    )
    assert result.accepted is False
    assert codes(result) == ("missing_cost_stress",)
    assert "stress_x3_net_profit" not in result.metrics


def test_an_empty_gate_accepts_a_losing_candidate_but_keeps_its_metrics() -> None:
    result = run(flat_frame(), long_spec(), stress=(1.0, 2.0))
    assert result.accepted is True
    assert result.rejections == ()
    assert result.metrics["net_profit"] < 0.0
    assert result.metrics["stress_x1_net_profit"] == result.metrics["net_profit"]
    assert result.metrics["stress_x2_net_profit"] < result.metrics["stress_x1_net_profit"]


def test_misaligned_frame_is_rejected_once() -> None:
    frame = flat_frame()
    with pytest.raises(ValueError, match="columns has 1 names for 2 columns"):
        evaluator(replace(frame, X=np.zeros((len(frame.ts), 2), dtype=np.float32)))

    with pytest.raises(ValueError, match="non-finite"):
        matrix = frame.X.copy()
        matrix[10] = np.nan
        evaluator(replace(frame, X=matrix))


def test_evolve_and_dsl_evaluator_run_end_to_end(tmp_path) -> None:
    bars = SyntheticProvider(seed=3).fetch(SYMBOL, START, START + timedelta(days=1))
    frame = build_features(bars)
    config = SearchConfig(
        dataset_id=DATASET_ID,
        code_version=CODE_VERSION,
        population=3,
        generations=2,
        mutation_rate=0.4,
        crossover_rate=0.4,
        max_trials=4,
        seed=5,
        patience=1,
        complexity_cap=40,
        novelty_requirement=0,
        selection_metric="net_profit",
    )

    with DiscoveryArchive(tmp_path / "discovery.sqlite") as archive:
        result = evolve(
            StrategyFactory(max_lookback=5, max_depth=2),
            evaluator(frame, max_steps=400),
            archive,
            config,
        )
        stored = archive.connection.execute("SELECT COUNT(*) FROM trials").fetchone()

    assert result.stop_reason in STOP_REASONS
    assert result.stop_reason == "budget"
    assert result.trials > 0
    assert result.trials + result.skipped == 4
    assert stored is not None and int(stored[0]) == result.trials
