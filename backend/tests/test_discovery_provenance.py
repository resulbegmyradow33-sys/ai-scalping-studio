"""Provenance gates: synthetic or unknown data must not reach promotion.

These are policy tests, not simulator runs. The evidence bundle is hand-built
so each assertion is about the provenance rule and nothing else: the same
metrics are reused with ``synthetic`` set to ``None``, ``False`` and ``True``
and the verdict must differ only because of that flag.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from promotion_evidence import evidence_for
from promotion_support import registered_test_policies
from scalping.datasets.builder import DatasetManifest, SplitRange
from scalping.discovery import reproduce as reproduce_module
from scalping.discovery.evidence import (
    EVIDENCE_SCHEMA_VERSION,
    CandidateEvidence,
    EvidenceError,
    LockedOOSEvidence,
    evidence_checksum,
    evidence_from_dict,
    locked_oos_checksum,
)
from scalping.discovery.promotion import (
    PROMOTION_V1,
    PromotionPolicy,
    evaluate_promotion,
    is_production_policy,
)
from scalping.discovery.reproduce import ReproduceError

SPEC_HASH = "a" * 64
DATASET_ID = "XAUUSD:M1:h1:provenance"
DATASET_CHECKSUM = "b" * 32
CODE_VERSION = "test-provenance-2026-11-25"
SETTINGS_DIGEST = "c" * 64

#: PROMOTION_V1 without the locked-OOS requirement, so the provenance gate can
#: be observed on its own. Every other threshold and the allow-list stay real.
PROVENANCE_POLICY: PromotionPolicy = replace(
    PROMOTION_V1,
    policy_id="provenance-test-policy",
    require_locked_oos=False,
)


def bundle(**overrides: Any) -> CandidateEvidence:
    values: dict[str, Any] = {
        "spec_hash": SPEC_HASH,
        "dataset_id": DATASET_ID,
        "symbol": "XAUUSD",
        "timeframe": "M1",
        "code_version": CODE_VERSION,
        "dataset_checksum": DATASET_CHECKSUM,
        "settings_digest": SETTINGS_DIGEST,
        "source": "test:real",
        "synthetic": False,
    }
    values.update(overrides)
    return evidence_for(**values)


# --- production gate ----------------------------------------------------------


def test_promotion_v1_rejects_a_synthetic_bundle() -> None:
    decision = evaluate_promotion(bundle(synthetic=True), PROMOTION_V1)

    assert decision.accepted is False
    assert "synthetic_data" in decision.codes


def test_promotion_v1_rejects_unknown_provenance() -> None:
    decision = evaluate_promotion(bundle(synthetic=None), PROMOTION_V1)

    assert decision.accepted is False
    assert "provenance_unknown" in decision.codes


def test_an_explicit_real_bundle_clears_the_provenance_gate() -> None:
    evidence = bundle(synthetic=False)

    decision = evaluate_promotion(evidence, PROVENANCE_POLICY)

    assert decision.accepted is True, decision.codes
    assert decision.synthetic is False
    assert decision.source == "test:real"


def test_provenance_policy_is_not_the_production_policy() -> None:
    assert is_production_policy(PROVENANCE_POLICY) is False
    assert is_production_policy(PROMOTION_V1) is True


def test_the_real_data_rule_does_not_depend_on_the_source_spelling() -> None:
    """``synthetic is False`` passes whatever label the producer chose.

    Source equality is a traceability check, not a realness check: the audit's
    G17 notes that producer spellings already differ, so the gate keys on the
    explicit flag and a production policy never infers it.
    """
    decision = evaluate_promotion(bundle(source="csv:2026"), PROVENANCE_POLICY)

    assert decision.accepted is True, decision.codes


# --- synthetic opt-in is demo-only -------------------------------------------


def test_allow_synthetic_passes_on_a_non_production_policy() -> None:
    demo = replace(PROVENANCE_POLICY, policy_id="provenance-demo", allow_synthetic=True)

    decision = evaluate_promotion(bundle(synthetic=True), demo)

    assert decision.accepted is True, decision.codes
    assert decision.synthetic is True


def test_allow_synthetic_is_never_a_production_opt_in() -> None:
    """Registering the demo policy makes it production; ``True`` is then refused."""
    demo = replace(PROVENANCE_POLICY, policy_id="provenance-demo", allow_synthetic=True)

    with registered_test_policies(demo):
        assert is_production_policy(demo) is True
        decision = evaluate_promotion(bundle(synthetic=True), demo)

    assert decision.accepted is False
    assert "synthetic_data" in decision.codes


def test_the_production_policy_cannot_enable_allow_synthetic() -> None:
    with pytest.raises(ValueError, match="synthetic"):
        replace(PROMOTION_V1, allow_synthetic=True)


# --- identity and checksum ---------------------------------------------------


def test_checksum_changes_when_only_provenance_changes() -> None:
    real = bundle(synthetic=False)
    synthetic = bundle(synthetic=True)
    unknown = bundle(synthetic=None)

    assert real.checksum != synthetic.checksum
    assert real.checksum != unknown.checksum
    assert synthetic.checksum != unknown.checksum


def test_provenance_is_re_derived_from_the_payload() -> None:
    evidence = bundle(synthetic=True)

    restored = evidence_from_dict(evidence.to_dict())

    assert (restored.source, restored.synthetic) == (evidence.source, True)
    assert evidence_checksum(restored) == evidence.checksum


def test_a_stored_bundle_without_provenance_decodes_as_unknown() -> None:
    payload = dict(bundle(synthetic=False).to_dict())
    payload.pop("source")
    payload.pop("synthetic")

    restored = evidence_from_dict(payload)

    assert restored.source is None
    assert restored.synthetic is None


def test_a_mistyped_provenance_flag_is_refused() -> None:
    payload = dict(bundle(synthetic=False).to_dict())
    payload["synthetic"] = 0

    with pytest.raises(EvidenceError) as exc:
        evidence_from_dict(payload)

    assert "synthetic" in str(exc.value)


def test_locked_section_provenance_is_part_of_its_checksum() -> None:
    evidence = bundle(synthetic=False)
    section_values = {
        "checksum": "0" * 64,
        "evidence_checksum": evidence_checksum(evidence),
        "dataset_id": evidence.dataset_id,
        "dataset_checksum": evidence.dataset_checksum,
        "symbol": evidence.symbol,
        "timeframe": evidence.timeframe,
        "spec_hash": evidence.spec_hash,
        "spec_version": evidence.spec_version,
        "complexity": evidence.complexity,
        "code_version": evidence.code_version,
        "settings_digest": evidence.settings_digest,
        "policy_id": PROVENANCE_POLICY.policy_id,
        "policy_version": PROVENANCE_POLICY.version,
        "test_start_row": evidence.splits.test_start_row,
        "test_end_row": evidence.splits.test_end_row,
        "test_rows": evidence.splits.test_end_row - evidence.splits.test_start_row,
        "test_start_ts": "2026-02-01T00:00:00+00:00",
        "test_end_ts": "2026-02-02T00:00:00+00:00",
        "trades": 75,
        "net_profit": 5.0,
        "profit_factor": 1.6,
        "win_rate": 0.5,
        "expectancy_r": 0.3,
        "total_r": 22.5,
        "max_drawdown": 0.1,
        "expectancy_r_ci95_low": 0.1,
        "expectancy_r_ci95_high": 0.5,
        "total_r_ci95_low": 2.0,
        "total_r_ci95_high": 20.0,
        "bootstrap_samples": 1000,
        "bootstrap_confidence": 0.95,
        "bootstrap_method": "iid-bootstrap",
        "regime_coverage": 0.95,
        "created_at": "2026-11-25T00:00:00+00:00",
        "source": "test:real",
        "synthetic": False,
    }
    real = LockedOOSEvidence(**section_values)
    synthetic = LockedOOSEvidence(**{**section_values, "synthetic": True})

    assert locked_oos_checksum(real) != locked_oos_checksum(synthetic)


# --- manifest JSON compatibility ---------------------------------------------


def _manifest_payload(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "symbol": "XAUUSD",
        "timeframe": "M1",
        "horizon": 1,
        "groups": ["base"],
        "columns": ["ret_1"],
        "rows": 300,
        "splits": {
            "train": {"start": 0, "end": 200},
            "validation": {"start": 205, "end": 250},
            "test": {"start": 255, "end": 300},
        },
        "purge_bars": 3,
        "embargo_bars": 2,
        "checksum": DATASET_CHECKSUM,
        "created_at": "2026-11-25T00:00:00+00:00",
        "notes": "",
        "source": "test:real",
        "synthetic": False,
    }
    values.update(overrides)
    return values


def test_a_manifest_without_provenance_decodes_as_unknown_not_real() -> None:
    payload = _manifest_payload()
    payload.pop("source")
    payload.pop("synthetic")

    manifest = reproduce_module._manifest(payload)

    assert manifest.source is None
    assert manifest.synthetic is None


def test_manifest_json_keeps_provenance_when_present() -> None:
    manifest = reproduce_module._manifest(_manifest_payload(source="csv:2026", synthetic=False))

    assert (manifest.source, manifest.synthetic) == ("csv:2026", False)
    restored = DatasetManifest(
        dataset_id=manifest.dataset_id,
        symbol=manifest.symbol,
        timeframe=manifest.timeframe,
        horizon=manifest.horizon,
        groups=manifest.groups,
        columns=manifest.columns,
        rows=manifest.rows,
        train=SplitRange(0, 200),
        validation=SplitRange(205, 250),
        test=SplitRange(255, 300),
        purge_bars=manifest.purge_bars,
        embargo_bars=manifest.embargo_bars,
        checksum=manifest.checksum,
        created_at=manifest.created_at,
        source=manifest.source,
        synthetic=manifest.synthetic,
    )
    assert restored.to_dict()["source"] == "csv:2026"
    assert restored.to_dict()["synthetic"] is False


def test_a_coerced_manifest_flag_is_refused() -> None:
    payload = _manifest_payload(synthetic=0)

    with pytest.raises(ReproduceError) as exc:
        reproduce_module._manifest(payload)

    assert "synthetic" in str(exc.value)


def test_old_manifest_json_defaults_to_unknown() -> None:
    """Construction from the pre-provenance field set stays valid and unknown."""
    manifest = DatasetManifest(
        dataset_id=DATASET_ID,
        symbol="XAUUSD",
        timeframe="M1",
        horizon=1,
        groups=("base",),
        columns=("ret_1",),
        rows=300,
        train=SplitRange(0, 200),
        validation=SplitRange(205, 250),
        test=SplitRange(255, 300),
        purge_bars=3,
        embargo_bars=2,
        checksum=DATASET_CHECKSUM,
        created_at="2026-11-25T00:00:00+00:00",
    )

    assert manifest.source is None
    assert manifest.synthetic is None
    assert manifest.to_dict()["source"] is None
    assert manifest.to_dict()["synthetic"] is None


def test_evidence_schema_is_bumped_for_the_new_identity() -> None:
    # Version 4 added source/synthetic; version 5 adds provenance_verified.
    assert EVIDENCE_SCHEMA_VERSION == 5


def test_the_verification_flag_round_trips_through_the_bundle_json() -> None:
    draft = replace(bundle(), checksum="", provenance_verified=True)
    evidence = replace(draft, checksum=evidence_checksum(draft))

    payload = evidence.to_dict()

    assert payload["provenance_verified"] is True
    rebuilt = evidence_from_dict(payload)
    assert rebuilt.provenance_verified is True
    assert rebuilt.source == "test:real"
    assert rebuilt.synthetic is False
    assert evidence_checksum(rebuilt) == evidence_checksum(evidence)
