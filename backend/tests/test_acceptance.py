"""End-to-end acceptance harness: real chain, synthetic data, honest claims.

The tests pin what the harness is allowed to say. A passing run proves the
plumbing executes; it never proves market validation, paper-trading evidence or
profitability. The failure test forces a stage error and checks that the run
stops there, writes a failed report and raises the typed result.

Paper governance is pinned here too. The harness must never paper-trade the
discovery candidate that promotion rejected: without a governed evidence
champion the paper and outcome stages are auditable skips. The champion cases
seed a fixture archive through the real promotion/provider APIs; they exercise
the mechanism, not a market claim.
"""

from __future__ import annotations

import json
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

import scalping.acceptance.run as acceptance_run
from promotion_evidence import evidence_for
from promotion_support import promote_for_tests, registered_test_policies
from scalping.acceptance import (
    STAGE_NAMES,
    AcceptanceConfig,
    AcceptanceError,
    AcceptanceStage,
    run_acceptance,
)
from scalping.discovery.archive import DiscoveryArchive
from scalping.discovery.promotion import (
    PROMOTION_V1,
    EvidenceIdentity,
    PromotionRefused,
    evaluate_promotion,
)
from scalping.discovery.protocols import TrialResult
from scalping.paper.ledger import PaperLedger

START = "2026-01-05"

# Fixture champion: an evidence-mode promotion recorded through the real archive
# API on a hand-built, non-synthetic bundle. It proves provider wiring only.
SCOPE = "XAUUSD:M1"
CHAMPION_CODE = "test-acceptance-champion"
CHAMPION_DATASET_ID = "validation:XAUUSD:M1:h1:fixture"
CHAMPION_DATASET_CHECKSUM = "c1" * 32
CHAMPION_SETTINGS_DIGEST = "c2" * 32
FIXTURE_POLICY = replace(
    PROMOTION_V1,
    policy_id="acceptance-harness-fixture",
    version=1,
    require_locked_oos=False,
)


def config(tmp_path: Path, **overrides: Any) -> AcceptanceConfig:
    values: dict[str, Any] = {
        "start": date.fromisoformat(START),
        "days": 1,
        "seed": 7,
        "window": 20,
        "max_trials": 4,
        "population": 3,
        "generations": 2,
        "paper_bars": 60,
        "out": tmp_path / "acceptance",
        "keep": False,
    }
    values.update(overrides)
    return AcceptanceConfig(**values)


def stage(result: Any, name: str) -> Any:
    return next(item for item in result.stages if item.name == name)


def fixture_evidence(spec_hash: str) -> Any:
    """A hand-built evidence bundle for the champion fixture; no simulation."""
    return evidence_for(
        spec_hash=spec_hash,
        dataset_id=CHAMPION_DATASET_ID,
        symbol="XAUUSD",
        timeframe="M1",
        code_version=CHAMPION_CODE,
        dataset_checksum=CHAMPION_DATASET_CHECKSUM,
        settings_digest=CHAMPION_SETTINGS_DIGEST,
        expectancy_r=0.3,
    )


def seed_champion(path: Path, *, mode: str = "evidence") -> tuple[str, int]:
    """Promote a fixture spec into the harness archive through the real APIs.

    ``exploratory`` records a metric-only champion that the provider must
    refuse; ``invalid`` tampers with the stored canonical payload after an
    evidence promotion so the provider's hash check fails.
    """
    spec = acceptance_run.acceptance_spec()
    with DiscoveryArchive(path) as archive:
        record = archive.add_candidate(spec, code_version=CHAMPION_CODE)
        archive.record_trial(
            record.candidate_id,
            TrialResult(
                metrics={"expectancy_r": 0.5, "net_profit": 3.0, "trades": 25.0},
                accepted=True,
            ),
            dataset_id=CHAMPION_DATASET_ID,
            code_version=CHAMPION_CODE,
            seed=0,
            dataset_checksum=CHAMPION_DATASET_CHECKSUM,
            settings_digest=CHAMPION_SETTINGS_DIGEST,
        )
        if mode == "exploratory":
            champion = archive.promote_exploratory(
                record.candidate_id,
                scope=SCOPE,
                reason="acceptance exploratory provider fixture",
                force=True,
            )
        else:
            champion = promote_for_tests(
                archive,
                record.candidate_id,
                fixture_evidence(record.candidate_id),
                FIXTURE_POLICY,
                scope=SCOPE,
                reason="acceptance paper-provider mechanism fixture",
                expected_identity=EvidenceIdentity(
                    code_version=CHAMPION_CODE,
                    dataset_checksum=CHAMPION_DATASET_CHECKSUM,
                    settings_digest=CHAMPION_SETTINGS_DIGEST,
                ),
            )
    if mode == "invalid":
        with sqlite3.connect(path) as connection:
            row = connection.execute(
                "SELECT canonical_json FROM candidates WHERE candidate_id = ?",
                (record.candidate_id,),
            ).fetchone()
            assert row is not None
            payload = json.loads(row[0])
            payload["max_lookback"] = 3  # changes the canonical hash after promotion
            connection.execute(
                "UPDATE candidates SET canonical_json = ? WHERE candidate_id = ?",
                (json.dumps(payload, sort_keys=True), record.candidate_id),
            )
    return record.candidate_id, champion.promotion_id


def run_governed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    decision_spec_hash: str | None = None,
    seed_mode: str = "evidence",
    drop_archive: bool = False,
    production_rejected: bool = False,
) -> tuple[Any, str, int]:
    """Run the harness with a fixture decision and a controlled archive state.

    The synthetic holdout ledger is a separate mechanism fixture; unless
    ``production_rejected`` is set it is deliberately skipped. These fixtures
    exercise the provider/seam wiring only, never a market claim.
    """
    real_discovery = acceptance_run._AcceptanceRun._stage_discovery
    real_promotion = acceptance_run._AcceptanceRun._stage_promotion
    real_paper = acceptance_run._AcceptanceRun._stage_paper
    champion_id: str = ""
    promotion_id: int = 0
    fixture_hash = decision_spec_hash or acceptance_run.acceptance_spec().canonical_hash()

    def seeding_discovery(run: Any) -> Any:
        nonlocal champion_id, promotion_id
        result = real_discovery(run)
        if seed_mode != "none":
            path = run.run_dir / "discovery.sqlite"
            champion_id, promotion_id = seed_champion(path, mode=seed_mode)
        return result

    def skipped_holdout(run: Any) -> AcceptanceStage:
        if production_rejected:
            decision = evaluate_promotion(fixture_evidence(fixture_hash), PROMOTION_V1)
            run.production_decision = decision
            run._write_json(
                run.run_dir / "holdout.json",
                {"production_decision": decision.to_dict(), "fixture": True},
            )
        return AcceptanceStage(
            name="holdout",
            status="skipped",
            detail="synthetic holdout ledger is a separate mechanism fixture",
            artifacts={},
            checksums={},
            counts={"attached": False, "spent": False},
        )

    def fixture_promotion(run: Any) -> Any:
        result = real_promotion(run)
        run.pre_decision = evaluate_promotion(fixture_evidence(fixture_hash), FIXTURE_POLICY)
        run._write_json(
            run.run_dir / "promotion.json",
            {"decision": run.pre_decision.to_dict(), "fixture": True},
        )
        return replace(
            result,
            detail="fixture decision accepted (mechanism test, not market validation)",
            counts={
                **result.counts,
                "accepted": True,
                "rejections": list(run.pre_decision.codes),
            },
            checksums={
                **result.checksums,
                "decision": acceptance_run._decision_digest(run.pre_decision),
            },
        )

    def paper_with_state(run: Any) -> Any:
        if drop_archive:
            (run.run_dir / "discovery.sqlite").unlink(missing_ok=True)
        return real_paper(run)

    monkeypatch.setattr(acceptance_run._AcceptanceRun, "_stage_discovery", seeding_discovery)
    monkeypatch.setattr(acceptance_run._AcceptanceRun, "_stage_holdout", skipped_holdout)
    monkeypatch.setattr(acceptance_run._AcceptanceRun, "_stage_promotion", fixture_promotion)
    if drop_archive:
        monkeypatch.setattr(acceptance_run._AcceptanceRun, "_stage_paper", paper_with_state)
    with registered_test_policies(FIXTURE_POLICY):
        result = run_acceptance(config(tmp_path))
    return result, champion_id, promotion_id


def run_with_champion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, str, int]:
    """Run the harness with a matching fixture champion and accepted decision."""
    return run_governed(tmp_path, monkeypatch)


def test_happy_path_runs_every_stage_with_honest_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One run: every stage executes, no network, no real store, no market claim."""
    calls: list[str] = []

    def no_network(*_args: object, **_kwargs: object) -> None:
        calls.append("network")
        raise AssertionError("the acceptance harness must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)

    result = run_acceptance(config(tmp_path))

    assert result.status == "passed"
    assert calls == []
    assert [item.name for item in result.stages] == list(STAGE_NAMES)
    assert all(item.status != "failed" for item in result.stages)
    assert stage(result, "holdout").status in {"skipped", "passed"}

    report_path = Path(result.report_path)
    assert report_path.is_file()
    report = json.loads(report_path.read_text())
    assert report["status"] == "passed"
    assert report["synthetic"] is True
    assert report["claims"] == {
        "implemented": True,
        "tested": True,
        "validated": False,
        "paper_traded": False,
        "profitable": False,
    }
    assert report["scope"] == {
        "synthetic_data": True,
        "out_of_sample": False,
        "market_evidence": False,
        "real_money": False,
    }
    assert report["claims"]["validated"] is False
    assert report["claims"]["paper_traded"] is False
    assert report["claims"]["profitable"] is False

    evidence = stage(result, "evidence")
    assert evidence.counts["candidate_source"] in {"discovery_best", "acceptance_fixture"}
    assert evidence.counts["fold_count"] == 3
    assert evidence.counts["trades"] >= 1

    promotion = stage(result, "promotion")
    assert promotion.counts["policy_id"] == "acceptance-test-policy"
    assert promotion.counts["policy_id"] != "promotion-v1"

    # Promotion rejected on this synthetic sample and no governed champion
    # exists, so neither the paper engine nor the outcome reader may run.
    paper = stage(result, "paper_advance")
    assert paper.status == "skipped"
    assert paper.error_code is not None
    assert paper.counts["paper_traded"] is False
    assert paper.counts["real_money"] is False
    assert (Path(result.report_path).parent / "paper.sqlite").exists() is False

    outcomes = stage(result, "matured_outcomes")
    assert outcomes.status == "skipped"
    assert outcomes.counts["paper_traded"] is False

    # The synthetic store is a temp directory and is removed unless --keep.
    assert Path(result.store_path).parent == Path(tempfile.gettempdir())
    assert Path(result.store_path).exists() is False

    # No real-money client is even imported by this module.
    source = Path(acceptance_run.__file__).read_text()
    for forbidden in ("MetaTrader5", "scalping.paper.mt5", "requests", "httpx"):
        assert forbidden not in source


def test_same_config_is_deterministic(tmp_path: Path) -> None:
    """The same inputs produce the same identity, dataset, evidence and outcomes."""
    first = run_acceptance(config(tmp_path / "one"))
    second = run_acceptance(config(tmp_path / "two"))

    assert first.acceptance_id == second.acceptance_id
    assert first.config_digest == second.config_digest
    assert [item.name for item in first.stages] == [item.name for item in second.stages]

    def checksums(result: Any) -> dict[str, str]:
        return {
            "dataset": stage(result, "dataset").checksums["dataset"],
            "evidence": stage(result, "evidence").checksums["evidence"],
            "promotion": stage(result, "promotion").checksums["decision"],
        }

    assert checksums(first) == checksums(second)
    assert stage(first, "evidence").counts == stage(second, "evidence").counts
    assert stage(first, "paper_advance").counts == stage(second, "paper_advance").counts
    assert stage(first, "matured_outcomes").counts == stage(second, "matured_outcomes").counts


def test_rejected_promotion_never_reaches_the_paper_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected discovery candidate must never be paper traded by the harness."""

    def forbidden_runner(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the paper engine ran after promotion rejected")

    monkeypatch.setattr(acceptance_run, "AutonomousPaperRunner", forbidden_runner)
    result = run_acceptance(config(tmp_path))

    promotion = stage(result, "promotion")
    assert promotion.counts["accepted"] is False
    paper = stage(result, "paper_advance")
    assert paper.status == "skipped"
    assert paper.error_code == "promotion_rejected"
    assert paper.counts["paper_traded"] is False
    assert paper.counts["real_money"] is False
    assert paper.counts["rejections"] == promotion.counts["rejections"]
    assert stage(result, "matured_outcomes").status == "skipped"
    report = json.loads(Path(result.report_path).read_text())
    assert report["claims"]["paper_traded"] is False
    assert report["claims"]["validated"] is False


def test_promotion_refusal_prevents_any_paper_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A structural promotion refusal stops the run before the paper stage."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise PromotionRefused("structural promotion check failed")

    def forbidden_runner(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the paper engine ran after a promotion refusal")

    monkeypatch.setattr(acceptance_run, "evaluate_promotion", refuse)
    monkeypatch.setattr(acceptance_run, "AutonomousPaperRunner", forbidden_runner)

    with pytest.raises(AcceptanceError) as caught:
        run_acceptance(config(tmp_path))

    assert caught.value.stage == "promotion"
    assert [item.name for item in caught.value.result.stages] == [
        "import_validate",
        "dataset",
        "discovery",
        "evidence",
        "promotion",
    ]
    assert (Path(caught.value.result.report_path).parent / "paper.sqlite").exists() is False


def test_governed_champion_is_the_only_paper_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paper uses the provider's exact champion, never the discovery best spec."""
    result, candidate_id, promotion_id = run_with_champion(tmp_path, monkeypatch)

    best_candidate = stage(result, "discovery").counts["best_candidate"]
    assert candidate_id != best_candidate
    assert stage(result, "promotion").counts["accepted"] is True

    paper = stage(result, "paper_advance")
    assert paper.status == "passed"
    assert paper.counts["spec_hash"] == candidate_id
    assert paper.counts["candidate_id"] == candidate_id
    assert paper.counts["promotion_id"] == promotion_id
    assert paper.counts["champion_mode"] == "evidence"
    assert paper.counts["paper_traded"] is False  # synthetic bars are not a market trial
    assert paper.counts["decision_source"] == "promotion"
    assert paper.counts["decision_artifact"] == "promotion.json"
    assert paper.counts["decision_spec_hash"] == candidate_id
    assert paper.counts["champion_spec_hash"] == candidate_id
    assert Path(paper.artifacts["decision"]) == Path(result.report_path).parent / "promotion.json"
    assert paper.counts["decision_digest"] == stage(result, "promotion").checksums["decision"]

    ledger_path = Path(result.report_path).parent / "paper.sqlite"
    with PaperLedger.open_reader(ledger_path) as ledger:
        session = ledger.session(paper.counts["session_id"])
    assert session is not None
    assert session.model_version == candidate_id
    assert session.config["extra"]["dsl"]["spec_hash"] == candidate_id

    outcomes = stage(result, "matured_outcomes")
    assert outcomes.status == "passed"
    assert outcomes.counts["strategy_hash"] == candidate_id


def test_decision_and_champion_must_name_the_same_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An accepted decision for another spec never authorizes the archive champion."""
    decision_hash = "b" * 64
    runner_calls: list[Any] = []

    def forbidden_runner(*_args: object, **_kwargs: object) -> None:
        runner_calls.append("paper")
        raise AssertionError("the paper engine ran on a decision/champion mismatch")

    monkeypatch.setattr(acceptance_run, "AutonomousPaperRunner", forbidden_runner)
    result, champion_id, _ = run_governed(tmp_path, monkeypatch, decision_spec_hash=decision_hash)

    paper = stage(result, "paper_advance")
    assert paper.status == "skipped"
    assert paper.error_code == "champion_decision_mismatch"
    assert paper.counts["reason_code"] == "champion_decision_mismatch"
    assert paper.counts["paper_traded"] is False
    assert paper.counts["decision_spec_hash"] == decision_hash
    assert paper.counts["champion_spec_hash"] == champion_id
    assert runner_calls == []


@pytest.mark.parametrize(
    ("fixture", "expected_code"),
    (
        ({"drop_archive": True}, "missing_archive"),
        ({"seed_mode": "none"}, "no_champion"),
        ({"seed_mode": "exploratory"}, "no_evidence_champion"),
        ({"seed_mode": "invalid"}, "invalid_champion"),
    ),
)
def test_unavailable_champion_skips_paper_without_running_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture: dict[str, Any],
    expected_code: str,
) -> None:
    """Each archive/provider refusal is an auditable skip with no paper run."""
    runner_calls: list[Any] = []

    def forbidden_runner(*_args: object, **_kwargs: object) -> None:
        runner_calls.append("paper")
        raise AssertionError(f"the paper engine ran for {expected_code}")

    monkeypatch.setattr(acceptance_run, "AutonomousPaperRunner", forbidden_runner)
    result, _, _ = run_governed(tmp_path, monkeypatch, **fixture)

    paper = stage(result, "paper_advance")
    assert paper.status == "skipped"
    assert paper.error_code == expected_code
    assert paper.counts["reason_code"] == expected_code
    assert paper.counts["paper_traded"] is False
    assert runner_calls == []
    assert (Path(result.report_path).parent / "paper.sqlite").exists() is False


def test_production_rejection_cites_the_holdout_decision_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A holdout-stage rejection must not be attributed to promotion.json."""
    runner_calls: list[Any] = []

    def forbidden_runner(*_args: object, **_kwargs: object) -> None:
        runner_calls.append("paper")
        raise AssertionError("the paper engine ran after a production rejection")

    monkeypatch.setattr(acceptance_run, "AutonomousPaperRunner", forbidden_runner)
    result, _, _ = run_governed(tmp_path, monkeypatch, production_rejected=True)

    paper = stage(result, "paper_advance")
    assert paper.status == "skipped"
    assert paper.error_code == "production_promotion_rejected"
    assert paper.counts["paper_traded"] is False
    assert paper.counts["decision_source"] == "production"
    assert paper.counts["decision_artifact"] == "holdout.json"
    assert Path(paper.artifacts["decision"]).name == "holdout.json"

    run_dir = Path(result.report_path).parent
    holdout = json.loads((run_dir / "holdout.json").read_text())
    promotion = json.loads((run_dir / "promotion.json").read_text())

    def digest(payload: dict[str, Any]) -> str:
        return acceptance_run._digest_json(
            {key: value for key, value in payload.items() if key != "created_at"}
        )

    assert paper.counts["decision_digest"] == digest(holdout["production_decision"])
    assert paper.counts["decision_digest"] != digest(promotion["decision"])
    assert paper.counts["promotion_accepted"] is False
    assert paper.counts["rejections"] == holdout["production_decision"]["codes"]
    assert runner_calls == []


def test_synthetic_fixture_never_claims_real_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even with a passed governed paper stage, the run is still synthetic."""
    result, _, _ = run_with_champion(tmp_path, monkeypatch)
    assert stage(result, "paper_advance").status == "passed"

    report = json.loads(Path(result.report_path).read_text())
    assert report["synthetic"] is True
    assert report["claims"] == {
        "implemented": True,
        "tested": True,
        "validated": False,
        "paper_traded": False,
        "profitable": False,
    }
    assert report["scope"] == {
        "synthetic_data": True,
        "out_of_sample": False,
        "market_evidence": False,
        "real_money": False,
    }
    assert any(
        "synthetic" in note.lower() and "holdout" in note.lower() for note in report["notes"]
    )


def test_stage_failure_is_typed_and_writes_a_failed_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage exception aborts, records the failure and leaves later stages out."""

    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic discovery failure")

    monkeypatch.setattr(acceptance_run, "run_discovery", explode)

    with pytest.raises(AcceptanceError) as caught:
        run_acceptance(config(tmp_path))

    error = caught.value
    assert error.stage == "discovery"
    assert error.code == "RuntimeError"
    assert error.result.status == "failed"
    assert error.result.claims["validated"] is False
    assert error.result.claims["profitable"] is False
    assert [item.name for item in error.result.stages] == [
        "import_validate",
        "dataset",
        "discovery",
    ]
    failed = error.result.stages[-1]
    assert failed.status == "failed"
    assert failed.error_code == "RuntimeError"
    assert "synthetic discovery failure" in failed.detail

    report_path = Path(error.result.report_path)
    assert report_path.is_file()
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["stages"][-1]["status"] == "failed"
    assert report["claims"]["implemented"] is True
    assert report["claims"]["validated"] is False
    assert Path(error.result.store_path).exists() is False


def test_cli_prints_one_json_line_and_keep_retains_the_store(tmp_path: Path) -> None:
    """The CLI contract stays one JSON line; --keep leaves the temp store behind."""
    out = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scalping.cli",
            "acceptance",
            "run",
            "--out",
            str(out),
            "--keep",
            "--max-trials",
            "3",
            "--paper-bars",
            "20",
            "--window",
            "10",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "passed"
    assert payload["synthetic"] is True
    assert payload["claims"]["validated"] is False
    assert payload["store_retained"] is True
    report_path = Path(payload["report"])
    assert report_path.is_file()
    store_path = Path(payload["store_path"])
    assert store_path.is_dir()
    shutil.rmtree(store_path, ignore_errors=True)


def test_cli_failure_prints_the_typed_report_and_exits_non_zero(tmp_path: Path) -> None:
    """A weekend-only range fails at import; the CLI gates on it with exit 1."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scalping.cli",
            "acceptance",
            "run",
            "--out",
            str(tmp_path / "failed"),
            "--start",
            "2026-01-03",
            "--days",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 1, completed.stderr
    assert "Traceback" not in completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "failed"
    assert payload["stages"][0]["name"] == "import_validate"
    assert payload["stages"][0]["status"] == "failed"
    assert payload["stages"][0]["error_code"] == "empty_synthetic_store"
    assert payload["claims"]["implemented"] is True
    assert payload["claims"]["validated"] is False
    report_path = Path(payload["report"])
    assert report_path.is_file()
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["stages"][0]["status"] == "failed"
