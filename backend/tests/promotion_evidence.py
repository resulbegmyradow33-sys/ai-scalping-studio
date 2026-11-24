"""Minimal hand-built evidence bundles for archive/provider tests.

These are not simulator results. They let a test seed an *evidence-mode*
champion without running a backtest, which is enough to check the governance,
storage and provider seams. ``test_promotion_policy.py`` owns the detailed
policy arithmetic with its own richer fixtures.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from scalping.discovery.evidence import (
    AggregateEvidence,
    CandidateEvidence,
    EvidencePolicy,
    FoldEvidence,
    RegimeEvidence,
    SplitBoundaries,
    StressEvidence,
    evidence_checksum,
)

START = datetime(2026, 1, 5, tzinfo=UTC)


def _ts(minutes: int) -> str:
    return (START + timedelta(minutes=minutes)).isoformat()


def _fold(index: int, *, trades: int, expectancy_r: float) -> FoldEvidence:
    test_start = 100 + (index - 1) * 50
    test_end = test_start + 50
    train_end = test_start - 5
    return FoldEvidence(
        fold=index,
        train_start_row=0,
        train_end_row=train_end,
        test_start_row=test_start,
        test_end_row=test_end,
        train_rows=train_end,
        test_rows=50,
        train_start_ts=_ts(0),
        train_end_ts=_ts(test_start - 1),
        test_start_ts=_ts(test_start),
        test_end_ts=_ts(test_end - 1),
        purge_bars=3,
        embargo_bars=2,
        trades=trades,
        net_profit=1.0,
        profit_factor=1.5,
        win_rate=0.5,
        expectancy_r=expectancy_r,
        max_drawdown=0.05,
        regimes=(
            RegimeEvidence(
                regime="trend_up",
                trades=trades,
                net_profit=1.0,
                win_rate=0.5,
                expectancy_r=expectancy_r,
                total_r=expectancy_r * trades,
            ),
        ),
        stress=tuple(
            StressEvidence(
                multiplier=multiplier,
                trades=trades,
                net_profit=1.0,
                profit_factor=1.5,
                max_drawdown=0.05,
                expectancy_r=expectancy_r,
            )
            for multiplier in (1.5, 2.0)
        ),
    )


def evidence_for(
    *,
    spec_hash: str,
    dataset_id: str,
    symbol: str,
    timeframe: str,
    code_version: str,
    dataset_checksum: str,
    settings_digest: str,
    expectancy_r: float = 0.3,
    trades_per_fold: int = 25,
) -> CandidateEvidence:
    folds = tuple(
        _fold(index, trades=trades_per_fold, expectancy_r=expectancy_r) for index in (1, 2, 3)
    )
    total_trades = trades_per_fold * len(folds)
    aggregate = AggregateEvidence(
        total_trades=total_trades,
        total_r=expectancy_r * total_trades,
        expectancy_r=expectancy_r,
        profit_factor=1.5,
        max_drawdown=0.10,
        positive_folds=len(folds),
        positive_fold_fraction=1.0,
        expectancy_r_ci95_low=max(0.0, expectancy_r - 0.05),
        expectancy_r_ci95_high=expectancy_r + 0.1,
        total_r_ci95_low=1.0,
        total_r_ci95_high=4.0,
        bootstrap_samples=1000,
        bootstrap_confidence=0.95,
        bootstrap_method="iid-bootstrap",
        regime_coverage=0.95,
        regime_trades=(),
    )
    bundle = CandidateEvidence(
        spec_hash=spec_hash,
        spec_version=1,
        complexity=10,
        dataset_id=dataset_id,
        dataset_checksum=dataset_checksum,
        symbol=symbol,
        timeframe=timeframe,
        code_version=code_version,
        settings_digest=settings_digest,
        seed=0,
        trial_count=10,
        search_budget=100,
        splits=SplitBoundaries(
            train_start_row=0,
            train_end_row=200,
            validation_start_row=200,
            validation_end_row=250,
            test_start_row=250,
            test_end_row=300,
        ),
        policy=EvidencePolicy(
            folds=3,
            purge_bars=3,
            embargo_bars=2,
            min_trades_per_fold=2,
            cost_stress=(1.5, 2.0),
            regime_coverage_min=0.85,
            bootstrap_samples=1000,
            bootstrap_confidence=0.95,
            window=60,
        ),
        folds=folds,
        aggregate=aggregate,
        checksum="",
        created_at="2026-11-24T00:00:00+00:00",
    )
    draft = replace(bundle, checksum="")
    return replace(draft, checksum=evidence_checksum(draft))


__all__: list[str] = ["evidence_for"]
