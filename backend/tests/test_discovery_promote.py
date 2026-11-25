"""Runtime promotion: evidence bundle -> governed champion, holdout one-shot.

The fixture is the profitable trend tape from the locked-holdout tests plus a
hand-built evidence bundle whose identity is the real dataset. The policy
arithmetic is owned by ``test_promotion_policy``; this module checks the runtime
wiring and the governance rules around it:

* a dry run evaluates only and writes nothing;
* a real promotion refuses a non-production policy, a missing owner approval and
  an already-spent holdout;
* the locked holdout is spent once, attached, re-evaluated and only then stored;
* a second promotion is compared with the current champion evidence;
* a synthetic or unknown-provenance bundle never reaches the holdout;
* the approval token is redacted from every record and database;
* the promotion writes the archive audit rows and does not use the network.
"""

from __future__ import annotations

import json
import socket
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from promotion_evidence import evidence_for
from promotion_support import promote_for_tests, registered_test_policies
from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.datasets.builder import DatasetBundle, build_dataset
from scalping.discovery.archive import DiscoveryArchive, TrialResult
from scalping.discovery.evidence import (
    CandidateEvidence,
    evidence_checksum,
    settings_digest,
)
from scalping.discovery.holdout import (
    HoldoutConfig,
    HoldoutLedger,
    HoldoutSpec,
    attach_holdout,
    evaluate_holdout,
)
from scalping.discovery.persist import save_dataset_artifact, save_evidence_bundle
from scalping.discovery.promote import PromoteRequest, promote_evidence
from scalping.discovery.promotion import (
    PROMOTION_V1,
    ExpectedIdentity,
    evaluate_promotion,
)
from scalping.discovery.run import code_version as discovery_code_version
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
SCOPE = f"{SYMBOL}:{TIMEFRAME}"
CODE = "test-promote-2026-11-25"
START = datetime(2026, 1, 5, tzinfo=UTC)
ROWS = 1600

SETTINGS = Settings()
SETTINGS_HASH = settings_digest(SETTINGS)

ALWAYS = Compare(Const(0.0), "<", Const(1.0))
NEVER = Compare(Const(0.0), ">", Const(1.0))

#: The eligibility policy the holdout protocol requires: production thresholds
#: without the section that does not exist yet. It never promotes.
ELIGIBILITY = replace(
    PROMOTION_V1,
    policy_id="promotion-v1-eligibility-test",
    require_locked_oos=False,
)


def profitable_bars(rows: int = ROWS) -> pl.DataFrame:
    """A strongly trending tape with one small loss every seven bars.

    Copied from the locked-holdout fixture: the always-long spec captures the
    moves and pays the occasional loss, so the held-out result can clear the
    metric gate. It is a protocol fixture, not a market claim.
    """
    prices: list[float] = []
    price = 1000.0
    for index in range(rows):
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


def long_spec(*, holding: int = 1) -> StrategySpec:
    """Always enter long when flat; the holding limit is the only exit."""
    return StrategySpec(
        entries=(EntryRule("long", ALWAYS),),
        exits=(ExitRule(NEVER),),
        holding=HoldingLimit(holding),
        max_lookback=5,
    )


@dataclass
class Context:
    """One dataset build, its evidence bundle and the archive that judges it."""

    tmp_path: Path
    store_root: Path
    features: FeatureFrame
    bundle: DatasetBundle
    spec: StrategySpec
    evidence: CandidateEvidence
    evidence_path: Path
    archive_path: Path
    identity: ExpectedIdentity

    @property
    def candidate_id(self) -> str:
        return self.spec.canonical_hash()


def aligned_evidence(
    bundle: DatasetBundle,
    spec: StrategySpec,
    *,
    code_version: str = CODE,
    expectancy_r: float = 0.3,
    source: str | None = "test:fixture",
    synthetic: bool | None = False,
) -> CandidateEvidence:
    """A hand-built bundle whose identity and test range are the real dataset's.

    The metric arithmetic comes from ``promotion_evidence``; this function only
    aligns the declared test rows with the artifact so the one-shot section the
    holdout attaches can match the bundle.
    """
    evidence = evidence_for(
        spec_hash=spec.canonical_hash(),
        dataset_id=bundle.manifest.dataset_id,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        code_version=code_version,
        dataset_checksum=bundle.manifest.checksum,
        settings_digest=SETTINGS_HASH,
        expectancy_r=expectancy_r,
        source=source,
        synthetic=synthetic,
    )
    # Evidence split rows are relative to the bundle arrays; the manifest
    # ranges are absolute feature-frame rows. The holdout evaluator maps the
    # absolute test range through the same offset.
    offset = bundle.manifest.train.start
    splits = replace(
        evidence.splits,
        test_start_row=bundle.manifest.test.start - offset,
        test_end_row=bundle.manifest.test.end - offset,
    )
    adjusted = replace(
        evidence,
        complexity=int(spec.complexity()),
        spec_version=int(spec.spec_version),
        splits=splits,
        checksum="",
    )
    return replace(adjusted, checksum=evidence_checksum(adjusted))


def build_context(
    tmp_path: Path,
    *,
    spec: StrategySpec | None = None,
    expectancy_r: float = 0.3,
    source: str | None = "test:fixture",
    synthetic: bool | None = False,
    evidence_name: str = "candidate.evidence.json",
    write_artifact: bool = True,
    code_version: str = CODE,
) -> Context:
    """Write the bar store, dataset artifact, evidence bundle and sidecar."""
    candidate = long_spec() if spec is None else spec
    store_root = tmp_path / "data"
    frame = profitable_bars()
    ParquetStore(store_root).write(SYMBOL, TIMEFRAME, frame)
    features = build_features(frame)
    bundle = build_dataset(
        features,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        horizon=1,
        train_fraction=0.6,
        validation_fraction=0.2,
        purge_bars=3,
        embargo_bars=2,
        source=source,
        synthetic=synthetic,
    )
    if write_artifact:
        save_dataset_artifact(store_root, bundle, features)
    evidence = aligned_evidence(
        bundle,
        candidate,
        code_version=code_version,
        expectancy_r=expectancy_r,
        source=source,
        synthetic=synthetic,
    )
    evidence_path = tmp_path / evidence_name
    save_evidence_bundle(evidence_path, evidence, candidate)
    return Context(
        tmp_path=tmp_path,
        store_root=store_root,
        features=features,
        bundle=bundle,
        spec=candidate,
        evidence=evidence,
        evidence_path=evidence_path,
        archive_path=tmp_path / "discovery.sqlite",
        identity=ExpectedIdentity(
            code_version=code_version,
            dataset_checksum=bundle.manifest.checksum,
            settings_digest=SETTINGS_HASH,
        ),
    )


def seed_archive(
    context: Context,
    *,
    candidate: StrategySpec | None = None,
    code_version: str = CODE,
) -> None:
    """Register the candidate and one accepted trial on the exact dataset."""
    spec = context.spec if candidate is None else candidate
    candidate_id = spec.canonical_hash()
    archive = DiscoveryArchive(context.archive_path)
    try:
        archive.add_candidate(spec, code_version=code_version)
        archive.record_trial(
            candidate_id,
            TrialResult(
                metrics={"expectancy_r": 0.3, "trades": 75.0},
                accepted=True,
                rejections=(),
            ),
            dataset_id=context.bundle.manifest.dataset_id,
            code_version=code_version,
            seed=0,
            dataset_checksum=context.bundle.manifest.checksum,
            settings_digest=SETTINGS_HASH,
        )
    finally:
        archive.close()


def request(context: Context, **overrides: Any) -> PromoteRequest:
    values: dict[str, Any] = {
        "archive_path": context.archive_path,
        "candidate_id": context.candidate_id,
        "evidence_path": context.evidence_path,
        "scope": SCOPE,
        "expected_identity": context.identity,
        "data_root": context.store_root,
        "reason": "test promotion",
    }
    values.update(overrides)
    return PromoteRequest(**values)


def fixture_attach_holdout(context: Context) -> CandidateEvidence:
    """Spend a *fixture* ledger and attach the section for the dry-run path.

    This is the real one-shot protocol on a separate ledger; it does not touch
    the ledger the promotion would use.
    """
    decision = evaluate_promotion(
        context.evidence,
        ELIGIBILITY,
        expected_identity=context.identity,
    )
    assert decision.accepted, decision.codes
    spec = HoldoutSpec(
        dataset_id=context.bundle.manifest.dataset_id,
        dataset_checksum=context.bundle.manifest.checksum,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        spec_hash=context.spec.canonical_hash(),
        code_version=CODE,
        settings_digest=SETTINGS_HASH,
        source="test:fixture",
        synthetic=False,
    )
    with HoldoutLedger(context.tmp_path / "fixture-holdout.sqlite") as ledger:
        ledger.seal(spec, actor="fixture", reason="dry-run fixture")
        token = ledger.unlock(
            spec,
            decision=decision,
            policy=PROMOTION_V1,
            actor="fixture",
        )
        config = HoldoutConfig(
            code_version=CODE,
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            window=context.evidence.policy.window,
            cost_stress=context.evidence.policy.cost_stress,
            bootstrap_samples=context.evidence.policy.bootstrap_samples,
            bootstrap_confidence=context.evidence.policy.bootstrap_confidence,
            seed=context.evidence.seed,
            settings=SETTINGS,
            features=context.features,
            dataset_id=context.bundle.manifest.dataset_id,
            dataset_checksum=context.bundle.manifest.checksum,
            source="test:fixture",
            synthetic=False,
        )
        result = evaluate_holdout(
            context.bundle,
            context.spec,
            config,
            ledger=ledger,
            unlock_token=token,
        )
    return attach_holdout(context.evidence, result)


# --- dry run ------------------------------------------------------------------


def test_dry_run_rejects_without_writing_the_archive(tmp_path: Path) -> None:
    context = build_context(tmp_path)

    result = promote_evidence(request(context, dry_run=True))

    assert result.status == "rejected"
    assert result.codes == ("missing_locked_oos",)
    assert result.decision is not None and result.decision["accepted"] is False
    assert not context.archive_path.exists()
    assert not (context.archive_path.parent / "holdout.sqlite").exists()


def test_dry_run_accepts_an_attached_holdout_without_promoting(tmp_path: Path) -> None:
    context = build_context(tmp_path)
    attached = fixture_attach_holdout(context)
    path = tmp_path / "attached.evidence.json"
    save_evidence_bundle(path, attached, context.spec)

    result = promote_evidence(request(context, evidence_path=path, dry_run=True))

    assert result.status == "accepted", result.to_dict()
    assert result.codes == ()
    assert result.champion is None
    assert not context.archive_path.exists()
    assert not (context.archive_path.parent / "holdout.sqlite").exists()


# --- refusals -----------------------------------------------------------------


def test_non_production_policy_is_refused_for_a_real_promotion(tmp_path: Path) -> None:
    context = build_context(tmp_path)
    seed_archive(context)

    result = promote_evidence(request(context, policy_id="holdout-eligibility"))

    assert result.status == "refused"
    assert result.codes == ("non_production_policy",)
    with DiscoveryArchive(context.archive_path) as archive:
        assert archive.champion(SCOPE) is None


def test_missing_owner_approval_refuses_without_spending(tmp_path: Path) -> None:
    context = build_context(tmp_path)
    seed_archive(context)

    result = promote_evidence(request(context, spend_holdout=True, approval_token=None))

    assert result.status == "refused"
    assert result.codes == ("approval_missing",)
    ledger_path = context.archive_path.parent / "holdout.sqlite"
    assert not ledger_path.exists()
    with DiscoveryArchive(context.archive_path) as archive:
        assert archive.champion(SCOPE) is None


def test_candidate_identity_mismatch_refuses(tmp_path: Path) -> None:
    context = build_context(tmp_path)

    result = promote_evidence(request(context, candidate_id="0" * 64, dry_run=True))

    assert result.status == "refused"
    assert result.codes == ("candidate_mismatch",)


# --- the one-shot holdout -----------------------------------------------------


def _evaluate_entries(ledger_path: Path) -> int:
    with sqlite3.connect(ledger_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM access_log WHERE action = 'evaluate'").fetchone()
    return int(row[0])


def test_holdout_is_spent_once_and_reuse_is_refused(tmp_path: Path) -> None:
    context = build_context(tmp_path)
    seed_archive(context)
    token = "owner-approval-token-7f3c9d"
    req = request(context, spend_holdout=True, approval_token=token)

    first = promote_evidence(req)

    assert first.status == "accepted", first.to_dict()
    assert first.champion is not None
    assert first.champion["mode"] == "evidence"
    assert first.holdout is not None and first.holdout["spent"] is True
    ledger_path = context.archive_path.parent / "holdout.sqlite"
    assert ledger_path.is_file()
    assert _evaluate_entries(ledger_path) == 1

    second = promote_evidence(req)

    assert second.status == "refused", second.to_dict()
    assert second.codes == ("holdout_spent",)
    assert _evaluate_entries(ledger_path) == 1

    # The same answer even when the caller forgets --spend-holdout: the ledger,
    # not the flag, decides that the build is exhausted.
    no_flag = promote_evidence(request(context))
    assert no_flag.status == "refused"
    assert no_flag.codes == ("holdout_spent",)
    assert _evaluate_entries(ledger_path) == 1


def test_promotion_writes_the_archive_audit_rows(tmp_path: Path) -> None:
    context = build_context(tmp_path)
    seed_archive(context)

    result = promote_evidence(
        request(context, spend_holdout=True, approval_token="owner-token-audit")
    )
    assert result.status == "accepted", result.to_dict()

    with DiscoveryArchive(context.archive_path) as archive:
        champion = archive.champion(SCOPE)
        assert champion is not None
        assert champion.mode == "evidence"
        assert champion.policy_id == PROMOTION_V1.policy_id
        assert champion.policy_version == PROMOTION_V1.version
        assert champion.policy_digest == PROMOTION_V1.digest
        assert champion.evidence_checksum is not None
        record = archive.evidence_record(champion.evidence_checksum)
        assert record is not None
        assert record.candidate_id == context.candidate_id

        promotions = archive.connection.execute(
            "SELECT * FROM champion_promotions WHERE scope = ?", (SCOPE,)
        ).fetchall()
        assert len(promotions) == 1
        assert promotions[0]["mode"] == "evidence"

        decisions = archive.connection.execute(
            "SELECT * FROM decisions WHERE decision = 'promoted'"
        ).fetchall()
        assert len(decisions) == 1
        detail = json.loads(decisions[0]["detail_json"])
        assert detail["mode"] == "evidence"
        assert detail["policy_digest"] == PROMOTION_V1.digest


def test_second_promotion_is_compared_with_the_champion_evidence(tmp_path: Path) -> None:
    first = build_context(tmp_path, evidence_name="first.evidence.json")
    second_spec = long_spec(holding=2)
    second = build_context(
        tmp_path,
        spec=second_spec,
        expectancy_r=0.2,
        evidence_name="second.evidence.json",
    )
    seed_archive(first)
    seed_archive(second)

    with registered_test_policies(ELIGIBILITY), DiscoveryArchive(first.archive_path) as archive:
        promote_for_tests(
            archive,
            first.candidate_id,
            first.evidence,
            ELIGIBILITY,
            scope=SCOPE,
            reason="seed champion",
            expected_identity=first.identity,
        )

    rejected = promote_evidence(request(second, dry_run=True))
    assert rejected.status == "rejected"
    assert "champion_improvement" in rejected.codes

    # Remove the stored champion bundle pointer: the second promotion must
    # refuse rather than silently evaluate as a first promotion.
    with DiscoveryArchive(first.archive_path) as archive:
        archive.connection.execute(
            "UPDATE champions SET evidence_checksum = NULL WHERE scope = ?", (SCOPE,)
        )
    missing = promote_evidence(request(second, dry_run=True))
    assert missing.status == "refused"
    assert missing.codes == ("champion_evidence_missing",)


def test_synthetic_bundle_is_rejected_and_the_holdout_is_not_spent(tmp_path: Path) -> None:
    context = build_context(
        tmp_path,
        source="synthetic:fixture",
        synthetic=True,
        evidence_name="synthetic.evidence.json",
    )
    seed_archive(context)

    result = promote_evidence(
        request(context, spend_holdout=True, approval_token="owner-token-synthetic")
    )

    assert result.status == "rejected"
    assert "synthetic_data" in result.codes
    assert not (context.archive_path.parent / "holdout.sqlite").exists()
    with DiscoveryArchive(context.archive_path) as archive:
        assert archive.champion(SCOPE) is None


def test_unknown_provenance_is_rejected_and_the_holdout_is_not_spent(tmp_path: Path) -> None:
    context = build_context(
        tmp_path,
        source=None,
        synthetic=None,
        evidence_name="unknown.evidence.json",
    )
    seed_archive(context)

    result = promote_evidence(
        request(context, spend_holdout=True, approval_token="owner-token-unknown")
    )

    assert result.status == "rejected"
    assert "provenance_unknown" in result.codes
    assert not (context.archive_path.parent / "holdout.sqlite").exists()


# --- redaction and network ----------------------------------------------------


def test_approval_token_never_appears_in_records_or_databases(tmp_path: Path) -> None:
    context = build_context(tmp_path)
    seed_archive(context)
    token = "OWNER-TOKEN-CANARY-9c1e77"

    req = request(context, spend_holdout=True, approval_token=token)
    result = promote_evidence(req)
    assert result.status == "accepted"

    serialized = json.dumps(result.to_dict())
    assert token not in serialized
    assert token not in repr(req)
    assert result.approval_source == "inline"
    assert token not in context.archive_path.read_bytes().decode("latin-1")
    ledger_path = context.archive_path.parent / "holdout.sqlite"
    assert token not in ledger_path.read_bytes().decode("latin-1")


def test_promotion_path_does_not_use_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a promotion must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)

    context = build_context(tmp_path)
    seed_archive(context)

    result = promote_evidence(request(context, dry_run=True))

    assert result.status == "rejected"
    assert result.codes == ("missing_locked_oos",)


# --- CLI ----------------------------------------------------------------------


BACKEND = Path(__file__).resolve().parents[1]


def cli_promote(context: Context, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scalping.cli",
            "discovery",
            "promote",
            "--archive",
            str(context.archive_path),
            "--candidate",
            context.candidate_id,
            "--evidence",
            str(context.evidence_path),
            "--data-root",
            str(context.store_root),
            "--reason",
            "cli promotion test",
            *extra,
        ],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_cli_dry_run_reject_exits_1_and_writes_the_record(tmp_path: Path) -> None:
    context = build_context(tmp_path, code_version=discovery_code_version())
    out = tmp_path / "decision.json"

    result = cli_promote(context, "--dry-run", "--out", str(out))

    assert result.returncode == 1, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "rejected"
    assert payload["codes"] == ["missing_locked_oos"]
    assert not context.archive_path.exists()
    assert json.loads(out.read_text()) == payload


def test_cli_unreadable_evidence_still_writes_a_decision_record(tmp_path: Path) -> None:
    context = build_context(tmp_path, code_version=discovery_code_version())
    context.evidence_path.unlink()
    out = tmp_path / "missing.json"

    result = cli_promote(context, "--dry-run", "--out", str(out))

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "refused"
    assert payload["codes"] == ["bundle_missing"]
    assert payload["policy_digest"] == PROMOTION_V1.digest
    assert json.loads(out.read_text()) == payload


def test_cli_non_production_policy_exits_2(tmp_path: Path) -> None:
    context = build_context(tmp_path, code_version=discovery_code_version())

    result = cli_promote(context, "--policy", "not-a-production-policy", "--dry-run")

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "refused"
    assert payload["codes"] == ["non_production_policy"]


def test_cli_owner_token_file_is_never_printed(tmp_path: Path) -> None:
    context = build_context(tmp_path, code_version=discovery_code_version())
    seed_archive(context, code_version=discovery_code_version())
    token = "CLI-OWNER-TOKEN-CANARY-4b21"
    token_file = tmp_path / "approval.token"
    token_file.write_text(token + "\n", encoding="utf-8")
    out = tmp_path / "decision.json"

    result = cli_promote(
        context,
        "--spend-holdout",
        "--approval-token-file",
        str(token_file),
        "--out",
        str(out),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "accepted"
    assert payload["approval_source"] == "file"
    assert token not in result.stdout
    assert token not in result.stderr
    assert token not in out.read_text(encoding="utf-8")
    assert token not in context.archive_path.read_bytes().decode("latin-1")
