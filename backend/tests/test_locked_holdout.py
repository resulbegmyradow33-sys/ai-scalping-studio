"""One-shot locked final-holdout protocol: seals, refusals, ledger and result.

The fixture is a flat M1 tape so every round turn loses exactly its costs and
the assertions are about the protocol, not about an edge. The promotion decision
is supplied explicitly: this module is the gate that spends the holdout after a
policy has accepted the candidate, not a second policy implementation.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from promotion_evidence import evidence_for as synthetic_evidence_for
from scalping.config import Settings
from scalping.datasets.builder import DatasetBundle, build_dataset
from scalping.discovery.evidence import (
    CandidateEvidence,
    EvidenceConfig,
    build_evidence,
    evidence_checksum,
    evidence_from_dict,
    locked_oos_checksum,
    settings_digest,
    verify_evidence,
)
from scalping.discovery.holdout import (
    LEDGER_SCHEMA_VERSION,
    HoldoutConfig,
    HoldoutError,
    HoldoutLedger,
    HoldoutSpec,
    HoldoutSpent,
    LockedHoldoutResult,
    attach_holdout,
    evaluate_holdout,
    verify_holdout_result,
)
from scalping.discovery.promotion import (
    PRODUCTION_POLICIES,
    PROMOTION_V1,
    ExpectedIdentity,
    PromotionDecision,
    evaluate_promotion,
    is_production_policy,
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
CODE = "test-holdout-2026-11-24"
START = datetime(2026, 1, 5, tzinfo=UTC)
ROWS = 1600

#: A named non-production eligibility policy. It cannot demand the locked-OOS
#: section before that section exists; `unlock` records `PROMOTION_V1` as the
#: target policy that will judge the attached bundle.
PRE_HOLDOUT_POLICY = replace(
    PROMOTION_V1,
    policy_id="holdout-eligibility",
    require_locked_oos=False,
)

#: The same eligibility thresholds for a synthetic tape. This is an explicit,
#: named, non-production demo policy: ``PROMOTION_V1`` still refuses the bundle.
DEMO_HOLDOUT_POLICY = replace(
    PRE_HOLDOUT_POLICY,
    policy_id="holdout-demo",
    allow_synthetic=True,
)

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


def profitable_bars(rows: int = ROWS) -> pl.DataFrame:
    """A strongly trending tape with one small loss every seven bars.

    The always-long fixture captures the +1.0 moves and pays for the occasional
    -0.4 bar, so the holdout result has a positive expectancy, a finite profit
    factor well above 1 and a positive bootstrap lower bound. It is a protocol
    fixture, not a market claim.
    """
    prices: list[float] = []
    price = 1000.0
    for index in range(rows):
        # Large winners relative to costs keep the 1.5x/2.0x stress reruns well
        # inside the policy's degradation cap.
        price += -0.5 if index % 7 == 6 else 3.0
        prices.append(price)
    return pl.DataFrame(
        {
            "ts": [START + timedelta(minutes=index) for index in range(rows)],
            "open": prices,
            "high": [value + 0.05 for value in prices],
            "low": [value - 0.05 for value in prices],
            "close": prices,
            "bid_close": prices,
            "ask_close": [value + 0.2 for value in prices],
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


SPEC = long_spec()
SPEC_HASH = SPEC.canonical_hash()
SETTINGS = Settings()
SETTINGS_HASH = settings_digest(SETTINGS)


@pytest.fixture(scope="module")
def discovery() -> tuple[FeatureFrame, DatasetBundle]:
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
        source="test:fixture",
        synthetic=False,
    )
    return features, bundle


@pytest.fixture(scope="module")
def profitable_discovery() -> tuple[FeatureFrame, DatasetBundle]:
    features = build_features(profitable_bars())
    bundle = build_dataset(
        features,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        horizon=1,
        train_fraction=0.6,
        validation_fraction=0.2,
        purge_bars=3,
        embargo_bars=2,
        source="test:fixture",
        synthetic=False,
    )
    return features, bundle


def holdout_spec(
    bundle: DatasetBundle,
    *,
    spec_hash: str = SPEC_HASH,
    code_version: str = CODE,
    settings_hash: str = SETTINGS_HASH,
    dataset_id: str | None = None,
    dataset_checksum: str | None = None,
    source: str | None = "test:fixture",
    synthetic: bool | None = False,
) -> HoldoutSpec:
    return HoldoutSpec(
        dataset_id=bundle.manifest.dataset_id if dataset_id is None else dataset_id,
        dataset_checksum=bundle.manifest.checksum if dataset_checksum is None else dataset_checksum,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        spec_hash=spec_hash,
        code_version=code_version,
        settings_digest=settings_hash,
        source=source,
        synthetic=synthetic,
    )


def precheck(
    bundle: DatasetBundle,
    *,
    spec_hash: str = SPEC_HASH,
    dataset_id: str | None = None,
    dataset_checksum: str | None = None,
    code_version: str = CODE,
    settings_hash: str = SETTINGS_HASH,
    expectancy_r: float = 0.3,
    policy: Any = PRE_HOLDOUT_POLICY,
    source: str | None = "test:fixture",
    synthetic: bool | None = False,
) -> tuple[PromotionDecision, CandidateEvidence]:
    """A sealed accepted decision plus the synthetic bundle that produced it.

    The bundle is hand-built rather than simulated (the flat tape loses its costs
    by construction); this test is about the holdout gate, not about finding an
    edge. Its identity is the real dataset/spec/code/settings, so the holdout
    unlocks only when those match.
    """
    evidence = synthetic_evidence_for(
        spec_hash=spec_hash,
        dataset_id=bundle.manifest.dataset_id if dataset_id is None else dataset_id,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        code_version=code_version,
        dataset_checksum=(
            bundle.manifest.checksum if dataset_checksum is None else dataset_checksum
        ),
        settings_digest=settings_hash,
        expectancy_r=expectancy_r,
        source=source,
        synthetic=synthetic,
    )
    # The promotion policy requires the locked section's test range to equal the
    # bundle's declared test split. The synthetic bundle is built without a
    # simulator, so align its declaration with the real locked slice.
    offset = bundle.manifest.train.start
    splits = replace(
        evidence.splits,
        test_start_row=bundle.manifest.test.start - offset,
        test_end_row=bundle.manifest.test.end - offset,
    )
    aligned = replace(evidence, splits=splits, checksum="")
    evidence = replace(aligned, checksum=evidence_checksum(aligned))
    decision = evaluate_promotion(evidence, policy)
    assert decision.accepted, decision.codes
    return decision, evidence


def accepted_decision(bundle: DatasetBundle, **overrides: Any) -> PromotionDecision:
    return precheck(bundle, **overrides)[0]


def rejected_decision(bundle: DatasetBundle) -> PromotionDecision:
    """A sealed rejection: PROMOTION_V1 rejects until the holdout is attached."""
    decision = evaluate_promotion(precheck(bundle)[1], PROMOTION_V1)
    assert not decision.accepted
    return decision


def holdout_config(
    features: FeatureFrame | None,
    *,
    dataset_id: str | None = None,
    dataset_checksum: str | None = None,
    **overrides: Any,
) -> HoldoutConfig:
    values: dict[str, Any] = {
        "code_version": CODE,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "window": 60,
        "cost_stress": (1.5, 2.0),
        "bootstrap_samples": 1000,
        "bootstrap_confidence": 0.95,
        "seed": 7,
        "settings": SETTINGS,
        "features": features,
        "dataset_id": dataset_id,
        "dataset_checksum": dataset_checksum,
        "source": "test:fixture",
        "synthetic": False,
    }
    values.update(overrides)
    return HoldoutConfig(**values)


def evidence_config() -> EvidenceConfig:
    return EvidenceConfig(
        code_version=CODE,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        folds=3,
        purge_bars=0,
        embargo_bars=0,
        min_trades_per_fold=2,
        cost_stress=(1.5, 2.0),
        regime_coverage_min=0.0,
        bootstrap_samples=200,
        bootstrap_confidence=0.95,
        seed=7,
        search_budget=50,
        trial_count=4,
        window=60,
    )


def evidence_for(
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> CandidateEvidence:
    features, bundle = discovery
    return build_evidence(bundle, SPEC, evidence_config(), features=features)


# --- seals, one-shot spend and reuse -----------------------------------------


def test_seal_spend_once_and_reuse_refused(
    tmp_path: Path,
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    features, bundle = discovery
    spec = holdout_spec(bundle)

    with HoldoutLedger(tmp_path / "holdout.sqlite") as ledger:
        assert ledger.schema_version == LEDGER_SCHEMA_VERSION
        sealed = ledger.seal(spec, actor="tester", reason="final split registered")
        assert sealed.status == "sealed"
        assert sealed.spec_hash == SPEC_HASH

        decision = accepted_decision(bundle)
        token = ledger.unlock(
            spec,
            decision=decision,
            policy=PROMOTION_V1,
            actor="tester",
            reason="policy accepted the candidate",
        )
        spent = ledger.get(bundle.manifest.dataset_id, bundle.manifest.checksum)
        assert spent is not None and spent.status == "spent"
        assert spent.spent_at is not None
        assert spent.policy_id == PROMOTION_V1.policy_id
        assert spent.policy_version == PROMOTION_V1.version
        assert spent.policy_digest == PROMOTION_V1.digest
        assert spent.decision_json is not None
        assert decision.evidence_checksum is not None
        assert decision.evidence_checksum in spent.decision_json

        result = evaluate_holdout(
            bundle,
            SPEC,
            holdout_config(features),
            ledger=ledger,
            unlock_token=token,
        )
        assert result.trades > 0
        assert result.bootstrap_method == "iid-bootstrap"
        assert result.test_rows > 60

        with pytest.raises(HoldoutSpent):
            ledger.unlock(
                spec,
                decision=accepted_decision(bundle),
                policy=PROMOTION_V1,
                actor="tester",
                reason="a second attempt",
            )

        other_spec = replace(spec, spec_hash="f" * 64)
        with pytest.raises(HoldoutSpent):
            ledger.unlock(
                other_spec,
                decision=accepted_decision(bundle, spec_hash="f" * 64),
                policy=PROMOTION_V1,
                actor="tester",
                reason="a different spec on the same data",
            )


def test_a_different_dataset_checksum_is_a_new_seal(
    tmp_path: Path,
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    _features, bundle = discovery
    other_checksum = "a" * 64
    first = holdout_spec(bundle)
    second = holdout_spec(bundle, dataset_checksum=other_checksum)

    with HoldoutLedger(tmp_path / "holdout.sqlite") as ledger:
        ledger.seal(first)
        ledger.unlock(first, decision=accepted_decision(bundle), policy=PROMOTION_V1)
        assert ledger.get(bundle.manifest.dataset_id, bundle.manifest.checksum).status == "spent"

        sealed_second = ledger.seal(second, actor="tester", reason="new dataset build")
        assert sealed_second.status == "sealed"
        seeded = ledger.get(bundle.manifest.dataset_id, other_checksum)
        assert seeded is not None and seeded.status == "sealed"

        token = ledger.unlock(
            second,
            decision=accepted_decision(bundle, dataset_checksum=other_checksum),
            policy=PROMOTION_V1,
        )
        assert token.dataset_checksum == other_checksum


# --- decision and identity gates ----------------------------------------------


def test_unlock_requires_an_accepted_decision(
    tmp_path: Path,
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    _features, bundle = discovery
    spec = holdout_spec(bundle)

    with HoldoutLedger(tmp_path / "holdout.sqlite") as ledger:
        ledger.seal(spec)
        with pytest.raises(HoldoutError) as exc:
            ledger.unlock(spec, decision=rejected_decision(bundle), policy=PROMOTION_V1)
        assert exc.value.code == "decision_rejected"
        sealed = ledger.get(bundle.manifest.dataset_id, bundle.manifest.checksum)
        assert sealed is not None and sealed.status == "sealed"

        decision = accepted_decision(bundle)
        token = ledger.unlock(spec, decision=decision, policy=PROMOTION_V1)
        assert token.evidence_checksum == decision.evidence_checksum


def test_identity_mismatch_is_refused(
    tmp_path: Path,
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    _features, bundle = discovery
    spec = holdout_spec(bundle)

    with HoldoutLedger(tmp_path / "holdout.sqlite") as ledger:
        ledger.seal(spec)

        mismatches = (
            accepted_decision(bundle, dataset_checksum="b" * 64),
            accepted_decision(bundle, code_version="other-code"),
            accepted_decision(bundle, settings_hash="c" * 64),
            accepted_decision(bundle, spec_hash="d" * 64),
        )
        for decision in mismatches:
            with pytest.raises(HoldoutError) as exc:
                ledger.unlock(spec, decision=decision, policy=PROMOTION_V1)
            assert exc.value.code == "decision_identity_mismatch"
            assert (
                ledger.get(bundle.manifest.dataset_id, bundle.manifest.checksum).status == "sealed"
            )

        with pytest.raises(HoldoutError) as exc:
            ledger.unlock(
                replace(spec, spec_hash="d" * 64),
                decision=accepted_decision(bundle, spec_hash="d" * 64),
                policy=PROMOTION_V1,
            )
        assert exc.value.code == "holdout_identity_mismatch"

        with pytest.raises(HoldoutError) as exc:
            ledger.unlock(
                spec,
                decision=accepted_decision(bundle),
                policy=PRE_HOLDOUT_POLICY,
            )
        assert exc.value.code == "non_production_policy"

        unsealed = object.__new__(PromotionDecision)
        with pytest.raises(HoldoutError) as exc:
            ledger.unlock(spec, decision=unsealed, policy=PROMOTION_V1)
        assert exc.value.code == "invalid_decision"


def test_access_log_records_grants_and_refusals(
    tmp_path: Path,
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    _features, bundle = discovery
    spec = holdout_spec(bundle)

    with HoldoutLedger(tmp_path / "holdout.sqlite") as ledger:
        ledger.seal(spec, actor="tester", reason="register the final split")
        with pytest.raises(HoldoutError):
            ledger.unlock(
                spec,
                decision=rejected_decision(bundle),
                policy=PROMOTION_V1,
                actor="tester",
                reason="rejected decision",
            )
        ledger.unlock(
            spec,
            decision=accepted_decision(bundle),
            policy=PROMOTION_V1,
            actor="tester",
            reason="policy accepted the candidate",
        )
        with pytest.raises(HoldoutSpent):
            ledger.unlock(
                spec,
                decision=accepted_decision(bundle),
                policy=PROMOTION_V1,
                actor="tester",
                reason="a second attempt",
            )

        entries = ledger.access_entries(
            dataset_id=bundle.manifest.dataset_id,
            dataset_checksum=bundle.manifest.checksum,
        )
        assert {entry.outcome for entry in entries} == {"granted", "refused"}
        unlocks = [entry for entry in entries if entry.action == "unlock"]
        assert [entry.outcome for entry in unlocks].count("granted") == 1
        assert [entry.outcome for entry in unlocks].count("refused") >= 1
        assert all(entry.actor == "tester" for entry in unlocks)
        assert all(entry.reason for entry in unlocks)
        assert any(entry.detail for entry in unlocks)


# --- evaluation, immutability and verification --------------------------------


def test_result_checksum_is_stable_and_tamper_is_detected(
    tmp_path: Path,
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    features, bundle = discovery
    spec = holdout_spec(bundle)
    evidence = evidence_for(discovery)

    with HoldoutLedger(tmp_path / "holdout.sqlite") as ledger:
        ledger.seal(spec)
        token = ledger.unlock(spec, decision=accepted_decision(bundle), policy=PROMOTION_V1)
        result = evaluate_holdout(
            bundle, SPEC, holdout_config(features), ledger=ledger, unlock_token=token
        )

    clone = LockedHoldoutResult.from_dict(result.to_dict())
    assert clone.checksum == result.checksum
    assert clone.trades == result.trades
    assert clone.regimes == result.regimes
    assert clone.stress == result.stress
    verify_holdout_result(clone, evidence)

    object.__setattr__(clone, "net_profit", clone.net_profit + 1.0)
    with pytest.raises(HoldoutError) as exc:
        verify_holdout_result(clone, evidence)
    assert exc.value.code == "checksum_mismatch"


def test_result_cannot_be_recomputed(
    tmp_path: Path,
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    features, bundle = discovery
    spec = holdout_spec(bundle)

    with HoldoutLedger(tmp_path / "holdout.sqlite") as ledger:
        ledger.seal(spec)
        token = ledger.unlock(spec, decision=accepted_decision(bundle), policy=PROMOTION_V1)
        first = evaluate_holdout(
            bundle, SPEC, holdout_config(features), ledger=ledger, unlock_token=token
        )
        stored = ledger.get(bundle.manifest.dataset_id, bundle.manifest.checksum)
        assert stored is not None and stored.result_json is not None
        stored_json = stored.result_json

        with pytest.raises(HoldoutError) as exc:
            evaluate_holdout(
                bundle, SPEC, holdout_config(features), ledger=ledger, unlock_token=token
            )
        assert exc.value.code == "holdout_result_exists"

        after = ledger.get(bundle.manifest.dataset_id, bundle.manifest.checksum)
        assert after is not None and after.result_json == stored_json
        assert after.result is not None and after.result.checksum == first.checksum


def test_an_invalid_token_does_not_read_or_spend_the_result(
    tmp_path: Path,
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    features, bundle = discovery
    spec = holdout_spec(bundle)

    with HoldoutLedger(tmp_path / "holdout.sqlite") as ledger:
        ledger.seal(spec)
        token = ledger.unlock(spec, decision=accepted_decision(bundle), policy=PROMOTION_V1)

        with pytest.raises(HoldoutError) as exc:
            evaluate_holdout(
                bundle, SPEC, holdout_config(features), ledger=ledger, unlock_token="not-a-token"
            )
        assert exc.value.code == "holdout_invalid_token"
        untouched = ledger.get(bundle.manifest.dataset_id, bundle.manifest.checksum)
        assert untouched is not None and untouched.result_json is None
        assert any(
            entry.action == "evaluate"
            and entry.outcome == "refused"
            and entry.reason == "holdout_invalid_token"
            for entry in ledger.access_entries()
        )

        result = evaluate_holdout(
            bundle, SPEC, holdout_config(features), ledger=ledger, unlock_token=token
        )
        assert result.checksum


def test_a_failed_evaluation_still_spends_the_holdout(
    tmp_path: Path,
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    features, bundle = discovery
    spec = holdout_spec(bundle)

    with HoldoutLedger(tmp_path / "holdout.sqlite") as ledger:
        ledger.seal(spec)
        token = ledger.unlock(spec, decision=accepted_decision(bundle), policy=PROMOTION_V1)

        with pytest.raises(HoldoutError) as exc:
            evaluate_holdout(
                bundle,
                SPEC,
                holdout_config(None),
                ledger=ledger,
                unlock_token=token,
            )
        assert exc.value.code == "missing_features"

        record = ledger.get(bundle.manifest.dataset_id, bundle.manifest.checksum)
        assert record is not None
        assert record.status == "spent"
        assert record.result_json is None
        assert record.unlock_token is None
        assert any(
            entry.action == "evaluate"
            and entry.outcome == "refused"
            and entry.reason == "missing_features"
            for entry in ledger.access_entries()
        )

        with pytest.raises(HoldoutError) as retry:
            evaluate_holdout(
                bundle,
                SPEC,
                holdout_config(features),
                ledger=ledger,
                unlock_token=token,
            )
        assert retry.value.code == "holdout_invalid_token"


def test_ledger_survives_reopen(
    tmp_path: Path,
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    features, bundle = discovery
    spec = holdout_spec(bundle)
    evidence = evidence_for(discovery)
    path = tmp_path / "holdout.sqlite"

    with HoldoutLedger(path) as ledger:
        ledger.seal(spec, actor="tester", reason="durable ledger")
        token = ledger.unlock(
            spec,
            decision=accepted_decision(bundle),
            policy=PROMOTION_V1,
            actor="tester",
            reason="accepted",
        )
        result = evaluate_holdout(
            bundle, SPEC, holdout_config(features), ledger=ledger, unlock_token=token
        )

    with HoldoutLedger(path) as reopened:
        record = reopened.get(bundle.manifest.dataset_id, bundle.manifest.checksum)
        assert record is not None
        assert record.status == "spent"
        assert record.result is not None
        assert record.result.checksum == result.checksum
        verify_holdout_result(record.result, evidence)
        assert reopened.access_entries()


# --- no read before unlock ----------------------------------------------------


class _TestRowTripwire(np.ndarray):
    """An array that raises when a test-split row is read while armed."""

    armed = False
    test_start = 0

    def __getitem__(self, key: Any) -> Any:
        if type(self).armed and _touches_test_rows(key, type(self).test_start, len(self)):
            raise AssertionError("test rows were read before the holdout was unlocked")
        return super().__getitem__(key)


def _touches_test_rows(key: Any, test_start: int, length: int) -> bool:
    if isinstance(key, slice):
        stop = length if key.stop is None else key.stop
        return stop > test_start
    if isinstance(key, tuple) and key:
        return _touches_test_rows(key[0], test_start, length)
    if isinstance(key, (int, np.integer)):
        return int(key) >= test_start
    if isinstance(key, np.ndarray):
        return bool((key >= test_start).any())
    if isinstance(key, list):
        return any(int(item) >= test_start for item in key)
    return False


def test_test_rows_are_not_read_before_unlock(
    tmp_path: Path,
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    features, bundle = discovery
    spec = holdout_spec(bundle)
    test_start = int(bundle.manifest.test.start)

    def wire(array: np.ndarray) -> _TestRowTripwire:
        return np.asarray(array).view(_TestRowTripwire)

    poisoned = replace(
        features,
        X=wire(features.X),
        ts=wire(features.ts),
        ohlc=wire(features.ohlc),
        spread=wire(features.spread),
        atr=wire(features.atr),
    )
    _TestRowTripwire.test_start = test_start
    _TestRowTripwire.armed = True
    try:
        with HoldoutLedger(tmp_path / "holdout.sqlite") as ledger:
            ledger.seal(spec)
            with pytest.raises(HoldoutError):
                ledger.unlock(spec, decision=rejected_decision(bundle), policy=PROMOTION_V1)
            token = ledger.unlock(spec, decision=accepted_decision(bundle), policy=PROMOTION_V1)
            assert (
                ledger.get(bundle.manifest.dataset_id, bundle.manifest.checksum).status == "spent"
            )

            _TestRowTripwire.armed = False
            result = evaluate_holdout(
                bundle,
                SPEC,
                holdout_config(poisoned),
                ledger=ledger,
                unlock_token=token,
            )
            assert result.test_rows > 0
    finally:
        _TestRowTripwire.armed = False


def test_a_bare_feature_frame_is_accepted_as_the_locked_slice(
    tmp_path: Path,
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    features, bundle = discovery
    start = int(bundle.manifest.test.start)
    end = int(bundle.manifest.test.end)
    locked = FeatureFrame(
        X=np.ascontiguousarray(features.X[start:end]),
        ts=np.asarray(features.ts[start:end]),
        ohlc=np.asarray(features.ohlc[start:end]),
        spread=np.asarray(features.spread[start:end]),
        atr=np.asarray(features.atr[start:end]),
        valid_from=0,
        columns=list(features.columns),
    )
    spec = holdout_spec(bundle)

    with HoldoutLedger(tmp_path / "holdout.sqlite") as ledger:
        ledger.seal(spec)
        token = ledger.unlock(spec, decision=accepted_decision(bundle), policy=PROMOTION_V1)
        result = evaluate_holdout(
            locked,
            SPEC,
            holdout_config(
                None,
                dataset_id=bundle.manifest.dataset_id,
                dataset_checksum=bundle.manifest.checksum,
            ),
            ledger=ledger,
            unlock_token=token,
        )
    assert result.dataset_id == bundle.manifest.dataset_id
    assert result.test_rows == end - start


# --- evidence attachment ------------------------------------------------------


def test_attach_holdout_returns_a_new_bundle_and_never_mutates(
    tmp_path: Path,
    profitable_discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    features, bundle = profitable_discovery
    decision, evidence = precheck(bundle)
    assert evidence.locked_oos is None
    spec = holdout_spec(bundle)

    with HoldoutLedger(tmp_path / "holdout.sqlite") as ledger:
        ledger.seal(spec)
        token = ledger.unlock(spec, decision=decision, policy=PROMOTION_V1)
        result = evaluate_holdout(
            bundle, SPEC, holdout_config(features), ledger=ledger, unlock_token=token
        )

    # These are the numbers the production policy will judge once the locked
    # section is part of the bundle; assert them explicitly rather than relying
    # on the policy pass alone.
    assert result.trades >= PROMOTION_V1.min_trades
    assert result.expectancy_r >= PROMOTION_V1.min_expectancy_r
    assert result.profit_factor >= PROMOTION_V1.min_profit_factor
    assert result.max_drawdown <= PROMOTION_V1.max_drawdown
    assert result.expectancy_r_ci95_low > PROMOTION_V1.min_ci_lower_r
    assert result.regime_coverage >= PROMOTION_V1.min_regime_coverage
    for multiplier in PROMOTION_V1.require_cost_stress:
        stress = next(row for row in result.stress if row.multiplier == multiplier)
        assert stress.net_profit > 0.0
        assert stress.profit_factor > 1.0

    attached = attach_holdout(evidence, result)
    assert type(attached) is CandidateEvidence
    assert attached is not evidence
    assert evidence.locked_oos is None
    assert attached.locked_oos is not None
    assert attached.locked_oos.checksum == locked_oos_checksum(attached.locked_oos)
    assert attached.locked_oos.evidence_checksum == evidence.checksum
    assert attached.locked_oos.schema_version == 2
    assert attached.locked_oos.test_start_row == result.test_start_row
    assert attached.locked_oos.test_end_row == result.test_end_row
    assert attached.checksum != evidence.checksum
    verify_evidence(
        attached,
        dataset_id=attached.dataset_id,
        dataset_checksum=attached.dataset_checksum,
        spec_hash=attached.spec_hash,
        code_version=attached.code_version,
        settings_digest=attached.settings_digest,
    )
    verify_holdout_result(result, attached)
    assert attach_holdout(attached, result) is attached

    # The archive stores canonical JSON and rebuilds it later for champion
    # comparison; the locked section must round-trip without losing its checksum.
    rebuilt = evidence_from_dict(attached.to_dict())
    assert rebuilt.locked_oos == attached.locked_oos
    assert evidence_checksum(rebuilt) == attached.checksum

    # End-to-end production policy: PROMOTION_V1 requires the locked-OOS section,
    # so this is the step that makes the bundle promotable.
    promotion = evaluate_promotion(
        attached,
        PROMOTION_V1,
        expected_identity=ExpectedIdentity.current(
            dataset_checksum=bundle.manifest.checksum,
            settings=SETTINGS,
            code_version=CODE,
        ),
    )
    assert promotion.accepted, promotion.codes
    assert "missing_locked_oos" not in promotion.codes
    assert is_production_policy(PROMOTION_V1)
    assert PROMOTION_V1.digest in PRODUCTION_POLICIES
    assert not is_production_policy(PRE_HOLDOUT_POLICY)


def test_a_holdout_result_for_other_provenance_is_refused(
    tmp_path: Path,
    profitable_discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    """A synthetic holdout cannot be attached to real-data evidence.

    The sealed identity and the spent result both say ``synthetic=True``; the
    evidence bundle says ``False``. The attach gate must compare the provenance,
    not just the dataset id and checksum, because identical bars can be labelled
    differently.
    """
    features, _ = profitable_discovery
    synthetic_bundle = build_dataset(
        features,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        horizon=1,
        train_fraction=0.6,
        validation_fraction=0.2,
        purge_bars=3,
        embargo_bars=2,
        source="test:fixture",
        synthetic=True,
    )
    demo_decision, demo_evidence = precheck(
        synthetic_bundle,
        policy=DEMO_HOLDOUT_POLICY,
        synthetic=True,
    )
    real_decision, real_evidence = precheck(profitable_discovery[1])
    assert demo_decision.accepted and real_decision.accepted
    spec = holdout_spec(synthetic_bundle, synthetic=True)

    with HoldoutLedger(tmp_path / "holdout.sqlite") as ledger:
        ledger.seal(spec)
        token = ledger.unlock(spec, decision=demo_decision, policy=PROMOTION_V1)
        result = evaluate_holdout(
            synthetic_bundle,
            SPEC,
            holdout_config(features, source="test:fixture", synthetic=True),
            ledger=ledger,
            unlock_token=token,
        )

    assert (result.source, result.synthetic) == ("test:fixture", True)
    assert demo_evidence.synthetic is True
    assert real_evidence.synthetic is False
    with pytest.raises(HoldoutError) as exc:
        attach_holdout(real_evidence, result)

    assert exc.value.code == "stale_synthetic"


def test_attach_refuses_a_zero_trade_holdout(
    tmp_path: Path,
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    features, bundle = discovery
    never = StrategySpec(
        entries=(EntryRule("long", NEVER),),
        exits=(ExitRule(NEVER),),
        holding=HoldingLimit(1),
        max_lookback=5,
    )
    spec_hash = never.canonical_hash()
    decision, evidence = precheck(bundle, spec_hash=spec_hash)
    spec = replace(holdout_spec(bundle), spec_hash=spec_hash)

    with HoldoutLedger(tmp_path / "holdout.sqlite") as ledger:
        ledger.seal(spec)
        token = ledger.unlock(spec, decision=decision, policy=PROMOTION_V1)
        result = evaluate_holdout(
            bundle, never, holdout_config(features), ledger=ledger, unlock_token=token
        )
    assert result.trades == 0

    with pytest.raises(HoldoutError) as exc:
        attach_holdout(evidence, result)
    assert exc.value.code == "locked_oos_requires_trades"


def test_attach_refuses_a_result_from_another_evidence_bundle(
    tmp_path: Path,
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    features, bundle = discovery
    _decision, evidence = precheck(bundle)
    other_decision, other_evidence = precheck(bundle, expectancy_r=0.4)
    assert other_evidence.checksum != evidence.checksum
    spec = holdout_spec(bundle)

    with HoldoutLedger(tmp_path / "holdout.sqlite") as ledger:
        ledger.seal(spec)
        token = ledger.unlock(spec, decision=other_decision, policy=PROMOTION_V1)
        result = evaluate_holdout(
            bundle, SPEC, holdout_config(features), ledger=ledger, unlock_token=token
        )

    with pytest.raises(HoldoutError) as exc:
        attach_holdout(evidence, result)
    assert exc.value.code == "evidence_checksum_mismatch"


# --- concurrency --------------------------------------------------------------


def test_concurrent_unlock_has_one_winner(
    tmp_path: Path,
    discovery: tuple[FeatureFrame, DatasetBundle],
) -> None:
    _features, bundle = discovery
    spec = holdout_spec(bundle)
    path = tmp_path / "holdout.sqlite"
    decision = accepted_decision(bundle)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    with HoldoutLedger(path) as ledger:
        ledger.seal(spec)

    def worker() -> None:
        with HoldoutLedger(path) as ledger:
            barrier.wait()
            try:
                ledger.unlock(spec, decision=decision, policy=PROMOTION_V1, actor="thread")
            except HoldoutSpent:
                outcome = "spent"
            except Exception as exc:  # noqa: BLE001 - the outcome is reported, not hidden
                outcome = f"error:{type(exc).__name__}:{exc}"
            else:
                outcome = "won"
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert sorted(outcomes) == ["spent", "won"]

    with HoldoutLedger(path) as ledger:
        record = ledger.get(bundle.manifest.dataset_id, bundle.manifest.checksum)
        assert record is not None and record.status == "spent"
        unlocks = [entry for entry in ledger.access_entries() if entry.action == "unlock"]
        assert [entry.outcome for entry in unlocks].count("granted") == 1
        assert [entry.outcome for entry in unlocks].count("refused") == 1
