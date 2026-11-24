"""Fold-level candidate evidence: geometry, leakage, checksums and fail-closed.

The fixtures use a constant-price tape so every round turn loses exactly its
costs and the assertions are directional rather than statistical. A synthetic
search result is not involved: the tests build a hand-written always-long DSL
spec and the real dataset builder, then check the evidence bundle alone.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from scalping.datasets.builder import DatasetBundle, build_dataset
from scalping.discovery import evidence as evidence_module
from scalping.discovery.evaluate import SimulationResult
from scalping.discovery.evidence import (
    EvidenceConfig,
    EvidenceError,
    bootstrap_ci,
    build_evidence,
    verify_evidence,
)
from scalping.features.build import FeatureFrame, build_features
from scalping.strategy.dsl import (
    Compare,
    Const,
    EntryRule,
    ExitRule,
    HoldingLimit,
    StrategySpec,
)

SYMBOL = "XAUUSD"
TIMEFRAME = "M1"
START = datetime(2026, 1, 5, tzinfo=UTC)
ROWS = 1600

ALWAYS = Compare(Const(0.0), "<", Const(1.0))
NEVER = Compare(Const(0.0), ">", Const(1.0))


def constant_bars(rows: int = ROWS) -> pl.DataFrame:
    """Flat M1 tape with a 0.20 spread; every round turn loses its costs."""
    return pl.DataFrame(
        {
            "ts": [START + timedelta(minutes=index) for index in range(rows)],
            "open": [1000.0] * rows,
            "high": [1000.001] * rows,
            "low": [999.999] * rows,
            "close": [1000.0] * rows,
            "bid_close": [1000.0] * rows,
            "ask_close": [1000.2] * rows,
            "spread": [0.2] * rows,
            "volume": [10] * rows,
        }
    )


def long_spec() -> StrategySpec:
    """Always enter long when flat; the holding limit is the only exit."""
    return StrategySpec(
        entries=(EntryRule("long", ALWAYS),),
        exits=(ExitRule(NEVER),),
        holding=HoldingLimit(1),
        max_lookback=5,
    )


@pytest.fixture(scope="module")
def discovery() -> tuple[FeatureFrame, DatasetBundle]:
    """One built frame and chronological bundle shared by the read-only tests."""
    features = build_features(constant_bars())
    bundle = build_dataset(
        features,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        horizon=1,
        train_fraction=0.6,
        validation_fraction=0.2,
        purge_bars=3,
        embargo_bars=2,
    )
    return features, bundle


def config(**overrides: Any) -> EvidenceConfig:
    values: dict[str, Any] = {
        "code_version": "test-evidence-2026-11-24",
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "folds": 3,
        "purge_bars": 0,
        "embargo_bars": 0,
        "min_trades_per_fold": 2,
        "cost_stress": (1.5, 2.0),
        "regime_coverage_min": 0.0,
        "bootstrap_samples": 200,
        "bootstrap_confidence": 0.95,
        "seed": 7,
        "search_budget": 50,
        "trial_count": 4,
        "window": 60,
    }
    values.update(overrides)
    return EvidenceConfig(**values)


def build(
    discovery: tuple[FeatureFrame, DatasetBundle],
    **overrides: Any,
) -> Any:
    features, bundle = discovery
    return build_evidence(bundle, long_spec(), config(**overrides), features=features)


def code_of(exc: pytest.ExceptionInfo[EvidenceError]) -> str:
    return exc.value.code


# --- geometry -----------------------------------------------------------------


def test_folds_are_chronological_non_overlapping_and_purged(
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    features, bundle = discovery
    evidence = build(discovery)
    relative_end = int(bundle.manifest.validation.end) - int(features.valid_from)
    gap = int(bundle.manifest.purge_bars) + int(bundle.manifest.embargo_bars)

    folds = evidence.folds
    assert [fold.fold for fold in folds] == [1, 2, 3]
    assert folds[0].train_start_row == 0
    for index, fold in enumerate(folds):
        assert fold.train_end_row == fold.test_start_row - gap
        assert fold.test_start_row - fold.train_end_row == gap
        assert fold.test_end_row - fold.test_start_row == fold.test_rows
        assert fold.train_rows == fold.train_end_row - fold.train_start_row
        assert fold.test_rows > config().window
        assert fold.test_end_row <= relative_end
        assert fold.purge_bars == int(bundle.manifest.purge_bars)
        assert fold.embargo_bars == int(bundle.manifest.embargo_bars)
        assert datetime.fromisoformat(fold.train_end_ts) < datetime.fromisoformat(
            fold.test_start_ts
        )
        if index:
            previous = folds[index - 1]
            assert fold.test_start_row == previous.test_end_row
            assert fold.train_rows > previous.train_rows


def test_fold_timestamps_are_iso_utc_and_inside_the_frame(
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    evidence = build(discovery)
    for fold in evidence.folds:
        for value in (fold.train_start_ts, fold.train_end_ts, fold.test_start_ts, fold.test_end_ts):
            parsed = datetime.fromisoformat(value)
            assert parsed.tzinfo == UTC
            assert parsed.year == 2026


# --- leakage ------------------------------------------------------------------


def test_poisoned_future_rows_cannot_change_the_evidence(
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    features, bundle = discovery
    base = build(discovery)
    cut = int(bundle.manifest.validation.end)

    matrix = features.X.copy()
    matrix[cut:] = np.nan
    ohlc = features.ohlc.copy()
    ohlc[cut:] = 5000.0
    spread = features.spread.copy()
    spread[cut:] = 99.0
    atr = features.atr.copy()
    atr[cut:] = 99.0
    poisoned = replace(features, X=matrix, ohlc=ohlc, spread=spread, atr=atr)

    after = build_evidence(bundle, long_spec(), config(), features=poisoned)

    assert after.checksum == base.checksum
    assert after.folds == base.folds
    relative_end = int(bundle.manifest.validation.end) - int(features.valid_from)
    assert all(fold.test_end_row <= relative_end for fold in after.folds)


def test_a_holdout_dataset_id_is_refused(
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    features, bundle = discovery
    poisoned = replace(
        bundle,
        manifest=replace(bundle.manifest, dataset_id=f"{SYMBOL}:{TIMEFRAME}:final-test"),
    )

    with pytest.raises(EvidenceError) as exc:
        build_evidence(poisoned, long_spec(), config(), features=features)

    assert code_of(exc) == "holdout_dataset"


def test_a_mismatched_frame_is_refused(
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    features, bundle = discovery
    shifted = replace(features, ts=features.ts + np.timedelta64(1, "m"))

    with pytest.raises(EvidenceError) as exc:
        build_evidence(bundle, long_spec(), config(), features=shifted)

    assert code_of(exc) == "frame_mismatch"


# --- determinism and checksum -------------------------------------------------


def test_same_inputs_reproduce_the_same_checksum(
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    features, bundle = discovery
    first = build_evidence(
        bundle,
        long_spec(),
        config(),
        features=features,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    second = build_evidence(
        bundle,
        long_spec(),
        config(),
        features=features,
        created_at=datetime(2026, 2, 2, tzinfo=UTC),
    )

    assert first.checksum == second.checksum
    assert first.created_at != second.created_at
    assert first.folds == second.folds
    assert first.aggregate == second.aggregate
    verify_evidence(
        first,
        dataset_id=bundle.manifest.dataset_id,
        spec_hash=long_spec().canonical_hash(),
        code_version=config().code_version,
    )


def test_tampering_with_a_metric_breaks_the_checksum(
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    evidence = build(discovery)
    original = evidence.folds[0].net_profit
    object.__setattr__(evidence.folds[0], "net_profit", original + 1.0)

    with pytest.raises(EvidenceError) as exc:
        verify_evidence(
            evidence,
            dataset_id=evidence.dataset_id,
            spec_hash=evidence.spec_hash,
            code_version=evidence.code_version,
        )

    assert code_of(exc) == "checksum_mismatch"


# --- identity and verification ------------------------------------------------


def test_identity_exposes_the_trial_context(
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    features, bundle = discovery
    spec = long_spec()
    evidence = build_evidence(bundle, spec, config(), features=features)

    assert evidence.spec_hash == spec.canonical_hash()
    assert evidence.spec_version == spec.spec_version
    assert evidence.complexity == spec.complexity()
    assert evidence.dataset_id == bundle.manifest.dataset_id
    assert evidence.dataset_checksum == bundle.manifest.checksum
    assert evidence.symbol == SYMBOL
    assert evidence.timeframe == TIMEFRAME
    assert evidence.code_version == config().code_version
    assert evidence.seed == config().seed
    assert evidence.trial_count == config().trial_count
    assert evidence.search_budget == config().search_budget
    assert evidence.policy.folds == config().folds
    assert evidence.policy.cost_stress == config().cost_stress


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("dataset_id", "XAUUSD:M1:h1:different", "stale_dataset"),
        ("spec_hash", "0" * 64, "stale_spec"),
        ("code_version", "older-code", "stale_code_version"),
    ],
)
def test_verify_rejects_a_stale_identity(
    discovery: tuple[FeatureFrame, DatasetBundle],
    field: str,
    value: str,
    expected: str,
) -> None:
    evidence = build(discovery)
    kwargs = {
        "dataset_id": evidence.dataset_id,
        "spec_hash": evidence.spec_hash,
        "code_version": evidence.code_version,
    }
    kwargs[field] = value

    with pytest.raises(EvidenceError) as exc:
        verify_evidence(evidence, **kwargs)

    assert code_of(exc) == expected


def test_verify_rejects_a_missing_section(
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    evidence = build(discovery)
    object.__setattr__(evidence, "folds", ())

    with pytest.raises(EvidenceError) as exc:
        verify_evidence(
            evidence,
            dataset_id=evidence.dataset_id,
            spec_hash=evidence.spec_hash,
            code_version=evidence.code_version,
        )

    assert code_of(exc) == "missing_section"


# --- fail closed --------------------------------------------------------------


def test_too_few_folds_fail_closed(discovery: tuple[FeatureFrame, DatasetBundle]) -> None:
    with pytest.raises(EvidenceError) as exc:
        build(discovery, folds=2)

    assert code_of(exc) == "insufficient_folds"


def test_too_few_trades_per_fold_fail_closed(
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    with pytest.raises(EvidenceError) as exc:
        build(discovery, min_trades_per_fold=10_000)

    assert code_of(exc) == "fold_trades_below_minimum"


def test_missing_cost_stress_fails_closed(
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    with pytest.raises(EvidenceError) as exc:
        build(discovery, cost_stress=())

    assert code_of(exc) == "missing_cost_stress"


def test_non_finite_metrics_fail_closed(
    discovery: tuple[FeatureFrame, DatasetBundle],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = evidence_module.DslEvaluator.simulate

    def poisoned(self: Any, candidate: Any, *, multiplier: float = 1.0) -> SimulationResult:
        result = original(self, candidate, multiplier=multiplier)
        if multiplier != 1.0:
            return result
        return SimulationResult(
            metrics=replace(result.metrics, profit_factor=math.inf),
            trades=result.trades,
            equity=result.equity,
        )

    monkeypatch.setattr(evidence_module.DslEvaluator, "simulate", poisoned)

    with pytest.raises(EvidenceError) as exc:
        build(discovery)

    assert code_of(exc) == "non_finite_evidence"


def test_missing_regime_labels_fail_closed(
    discovery: tuple[FeatureFrame, DatasetBundle],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evidence_module,
        "_regime_labels",
        lambda frame: pl.DataFrame(schema={"ts": pl.Datetime("us", "UTC"), "regime": pl.String}),
    )

    with pytest.raises(EvidenceError) as exc:
        build(discovery)

    assert code_of(exc) == "missing_regime_evidence"


def test_insufficient_regime_coverage_fails_closed(
    discovery: tuple[FeatureFrame, DatasetBundle],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = evidence_module._regime_labels

    monkeypatch.setattr(evidence_module, "_regime_labels", lambda frame: original(frame).tail(1))

    with pytest.raises(EvidenceError) as exc:
        build(discovery, regime_coverage_min=1.0)

    assert code_of(exc) == "regime_coverage_below_minimum"


# --- stress and regimes -------------------------------------------------------


def test_every_fold_carries_each_cost_stress(discovery: tuple[FeatureFrame, DatasetBundle]) -> None:
    evidence = build(discovery)
    for fold in evidence.folds:
        assert [row.multiplier for row in fold.stress] == [1.5, 2.0]
        assert all(row.trades >= 0 for row in fold.stress)
        assert all(row.net_profit < fold.net_profit for row in fold.stress)


def test_regime_coverage_and_counts_are_pooled(
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    evidence = build(discovery)
    assert evidence.aggregate.regime_coverage == 1.0
    assert sum(row.trades for row in evidence.aggregate.regime_trades) == (
        evidence.aggregate.total_trades
    )
    assert all(row.regime != "unknown" for row in evidence.aggregate.regime_trades)
    for fold in evidence.folds:
        assert sum(row.trades for row in fold.regimes) == fold.trades
        assert all(row.regime != "unknown" for row in fold.regimes)


def test_aggregate_reports_consistency_and_bootstrap_cis(
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    evidence = build(discovery)
    aggregate = evidence.aggregate

    assert aggregate.total_trades == sum(fold.trades for fold in evidence.folds)
    assert aggregate.positive_folds == 0
    assert aggregate.positive_fold_fraction == 0.0
    assert aggregate.bootstrap_samples == config().bootstrap_samples
    assert aggregate.bootstrap_confidence == config().bootstrap_confidence
    assert aggregate.bootstrap_method == "iid-bootstrap"
    assert aggregate.expectancy_r_ci95_low <= aggregate.expectancy_r
    assert aggregate.expectancy_r <= aggregate.expectancy_r_ci95_high
    assert aggregate.total_r_ci95_low <= aggregate.total_r
    assert aggregate.total_r <= aggregate.total_r_ci95_high
    assert aggregate.max_drawdown == max(fold.max_drawdown for fold in evidence.folds)


# --- bootstrap ----------------------------------------------------------------


def test_bootstrap_ci_matches_a_hand_computable_sample() -> None:
    constant_mean = bootstrap_ci([0.5] * 5, samples=200, seed=3)
    constant_sum = bootstrap_ci([0.5] * 5, samples=200, seed=3, statistic="sum")
    two_values_mean = bootstrap_ci([-1.0, 1.0], samples=2000, seed=0)
    two_values_sum = bootstrap_ci([-1.0, 1.0], samples=2000, seed=0, statistic="sum")

    assert constant_mean == (0.5, 0.5)
    assert constant_sum == (2.5, 2.5)
    # Every resample mean is -1, 0 or 1 and every sum is -2, 0 or 2; the
    # 2.5%/97.5% percentiles therefore sit on the extreme values.
    assert two_values_mean == (-1.0, 1.0)
    assert two_values_sum == (-2.0, 2.0)


def test_bootstrap_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="statistic"):
        bootstrap_ci([1.0], samples=10, seed=0, statistic="median")
    with pytest.raises(ValueError, match="samples"):
        bootstrap_ci([1.0], samples=0, seed=0)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([1.0], samples=10, seed=0, confidence=1.0)
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([], samples=10, seed=0)
    with pytest.raises(ValueError, match="finite"):
        bootstrap_ci([1.0, math.nan], samples=10, seed=0)


# --- configuration guards -----------------------------------------------------


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="code_version"):
        config(code_version="")
    with pytest.raises(ValueError, match="bootstrap_confidence"):
        config(bootstrap_confidence=1.0)
    with pytest.raises(ValueError, match="cost_stress"):
        config(cost_stress=(1.0, 1.0))
    with pytest.raises(TypeError, match="folds"):
        config(folds=True)
