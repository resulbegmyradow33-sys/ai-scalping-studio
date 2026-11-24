"""End-to-end acceptance harness: real chain, synthetic data, honest claims.

The tests pin what the harness is allowed to say. A passing run proves the
plumbing executes; it never proves market validation, paper-trading evidence or
profitability. The failure test forces a stage error and checks that the run
stops there, writes a failed report and raises the typed result.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import pytest

import scalping.acceptance.run as acceptance_run
from scalping.acceptance import (
    STAGE_NAMES,
    AcceptanceConfig,
    AcceptanceError,
    run_acceptance,
)

START = "2026-01-05"


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

    paper = stage(result, "paper_advance")
    assert paper.counts["real_money"] is False
    assert paper.counts["bars_processed"] > 0

    outcomes = stage(result, "matured_outcomes")
    assert outcomes.counts["closed_trades"] >= 1
    assert outcomes.counts["open_trades"] >= 0

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
            "outcomes": stage(result, "matured_outcomes").checksums["fingerprint"],
        }

    assert checksums(first) == checksums(second)
    assert stage(first, "evidence").counts == stage(second, "evidence").counts
    assert stage(first, "matured_outcomes").counts == stage(second, "matured_outcomes").counts


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
