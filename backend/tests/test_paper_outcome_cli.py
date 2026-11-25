"""CLI wiring for matured paper outcomes as the next research trigger.

The source and the proposal-only handler already exist; these tests pin the
operator-facing seam:

* ``--watch-paper-ledger`` is repeatable and binds one source per path;
* the emitted job reaches the ``paper.outcomes`` proposal handler and executes
  no discovery;
* synthetic batches are refused unless ``--allow-synthetic`` is given, and the
  handler shares that opt-in;
* the cursor dedupes across a CLI restart, and the store key dedupes after the
  cursor is lost;
* no flag means no source: the archive-root paper ledger is never bound
  implicitly, and the read-only HTTP API cannot bind one either.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scalping import cli
from scalping.api.app import create_app
from scalping.orchestrator import JobStore
from scalping.orchestrator.paper_triggers import (
    EVENT_NOT_QUEUED,
    KIND_PAPER_OUTCOMES,
    REASON_SYNTHETIC,
    PaperOutcomeTriggerSource,
)
from scalping.paper.ledger import PaperLedger

SYMBOL = "XAUUSD"
TIMEFRAME = "M1"
BASE = datetime(2026, 1, 5, 10, tzinfo=UTC)
DISCOVERY_KIND = "discovery.run"


def _seed_ledger(
    path: Path,
    session_id: str = "session-1",
    *,
    synthetic: bool = False,
) -> None:
    """One session with one closed, attributable trade."""
    config: dict[str, object] = {
        "origin": "test",
        "data": {"dataset_id": "dataset-cli"},
    }
    if synthetic:
        config["provenance"] = {"source": "synthetic", "synthetic": True}
    with PaperLedger(path) as ledger:
        ledger.open_session(
            session_id,
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            strategy="dsl",
            model_version="hash-cli",
            started_at=BASE,
            config=config,
        )
        ledger.record_trade(
            session_id,
            side="long",
            entry_ts=BASE + timedelta(minutes=1),
            entry_price=2000.0,
            exit_ts=BASE + timedelta(minutes=5),
            exit_price=2001.0,
            lot=0.10,
            pnl=10.0,
            r_multiple=1.0,
            exit_reason="take_profit",
        )


def _cli_args(
    tmp_path: Path,
    *extra: str,
    max_jobs_per_tick: str = "0",
) -> list[str]:
    return [
        "orchestrator",
        "run",
        "--once",
        "--profile",
        "none",
        "--db",
        str(tmp_path / "orchestrator.sqlite3"),
        "--root",
        str(tmp_path / "data"),
        "--runs-root",
        str(tmp_path / "runs"),
        "--archive-root",
        str(tmp_path / "archive"),
        "--max-jobs-per-tick",
        max_jobs_per_tick,
        *extra,
    ]


def _tick(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    payload = json.loads(capsys.readouterr().out.strip())
    assert "tick" in payload
    return payload["tick"]


def _jobs(db: Path, *, kind: str = KIND_PAPER_OUTCOMES) -> list:
    store = JobStore(db)
    try:
        return store.list_jobs(kind=kind)
    finally:
        store.close()


def test_cli_binds_a_ledger_and_reaches_the_proposal_handler(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = tmp_path / "paper.sqlite"
    _seed_ledger(ledger)
    args = _cli_args(
        tmp_path,
        "--watch-paper-ledger",
        str(ledger),
        max_jobs_per_tick="10",
    )

    assert cli.main(args) == 0
    tick = _tick(capsys)
    assert tick["error"] is None
    scan = tick["scan"]
    assert scan is not None
    assert [job["kind"] for job in scan["queued"]] == [KIND_PAPER_OUTCOMES]
    assert scan["queued"][0]["payload"]["session_id"] == "session-1"
    assert [outcome["status"] for outcome in tick["outcomes"]] == ["succeeded"]
    assert [outcome["job"]["kind"] for outcome in tick["outcomes"]] == [KIND_PAPER_OUTCOMES]

    db = tmp_path / "orchestrator.sqlite3"
    store = JobStore(db)
    try:
        jobs = store.list_jobs(kind=KIND_PAPER_OUTCOMES)
        assert len(jobs) == 1
        job = jobs[0]
        assert job.status == "succeeded"
        assert job.result is not None and job.result["ok"] is True
        proposal = job.result["data"]
        assert proposal["action"] == "queue_discovery"
        assert proposal["queued"] is False and proposal["executed"] is False
        assert proposal["session_id"] == "session-1"
        assert proposal["strategy_hash"] == "hash-cli"
        assert proposal["dataset_id"] == "dataset-cli"
        # A proposal is not execution: no discovery and no paper advance job.
        assert store.list_jobs(kind=DISCOVERY_KIND) == []
        assert store.list_jobs(kind="paper.advance") == []
    finally:
        store.close()
    assert not (tmp_path / "runs" / "discovery").exists()


def test_cli_repeatable_flag_binds_one_source_per_ledger(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = tmp_path / "paper-a.sqlite"
    second = tmp_path / "paper-b.sqlite"
    _seed_ledger(first, "session-a")
    _seed_ledger(second, "session-b")
    args = _cli_args(
        tmp_path,
        "--watch-paper-ledger",
        str(first),
        "--watch-paper-ledger",
        str(second),
    )

    assert cli.main(args) == 0
    scan = _tick(capsys)["scan"]
    assert scan is not None
    assert len(scan["sources"]) == 2
    queued = {job["payload"]["session_id"] for job in scan["queued"]}
    assert queued == {"session-a", "session-b"}
    assert {job["kind"] for job in scan["queued"]} == {KIND_PAPER_OUTCOMES}

    jobs = _jobs(tmp_path / "orchestrator.sqlite3")
    assert {job.payload["session_id"] for job in jobs} == {"session-a", "session-b"}


def test_cli_refuses_synthetic_batches_until_the_opt_in(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = tmp_path / "paper.sqlite"
    _seed_ledger(ledger, "demo", synthetic=True)
    refused_args = _cli_args(tmp_path, "--watch-paper-ledger", str(ledger))

    assert cli.main(refused_args) == 0
    tick = _tick(capsys)
    scan = tick["scan"]
    assert scan is not None and scan["queued"] == []
    assert tick["outcomes"] == []
    db = tmp_path / "orchestrator.sqlite3"
    assert _jobs(db) == []

    store = JobStore(db)
    try:
        reasons = [
            json.loads(event.detail)["reason"]
            for event in store.events()
            if event.event == EVENT_NOT_QUEUED and event.detail is not None
        ]
    finally:
        store.close()
    assert reasons == [REASON_SYNTHETIC]

    # The opt-in is explicit and shared by the source and the handler. The
    # cursor recorded the refusal, not a lost batch, so the next scan emits it.
    allowed_args = _cli_args(
        tmp_path,
        "--watch-paper-ledger",
        str(ledger),
        "--allow-synthetic",
        max_jobs_per_tick="10",
    )
    assert cli.main(allowed_args) == 0
    allowed = _tick(capsys)
    assert allowed["scan"]["queued"][0]["payload"]["synthetic"] is True
    proposal = allowed["outcomes"][0]["job"]["result"]["data"]
    assert proposal["action"] == "queue_discovery"
    assert proposal["synthetic"] is True


def test_cli_dedupes_across_a_restart_and_a_lost_cursor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = tmp_path / "paper.sqlite"
    _seed_ledger(ledger)
    args = _cli_args(tmp_path, "--watch-paper-ledger", str(ledger))
    db = tmp_path / "orchestrator.sqlite3"

    assert cli.main(args) == 0
    first = _tick(capsys)["scan"]
    assert first is not None and len(first["queued"]) == 1

    # A second process run reuses the cursor stored in the queue database.
    assert cli.main(args) == 0
    second = _tick(capsys)["scan"]
    assert second is not None
    assert second["queued"] == [] and second["duplicates"] == []
    assert len(_jobs(db)) == 1

    # A crash between enqueue and acknowledgement loses the cursor, not the
    # dedupe key: the same batch is re-emitted and the store returns the job.
    store = JobStore(db)
    try:
        cursor_key = PaperOutcomeTriggerSource(ledger, store=store).cursor_key
        store.clear_state(cursor_key)
    finally:
        store.close()
    assert cli.main(args) == 0
    third = _tick(capsys)["scan"]
    assert third is not None
    assert third["queued"] == []
    assert [job["kind"] for job in third["duplicates"]] == [KIND_PAPER_OUTCOMES]
    assert len(_jobs(db)) == 1


def test_cli_without_the_flag_binds_no_paper_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    # The ledger sits exactly where the archive-root default would look, and is
    # still not read: the path must be explicit on the command line.
    _seed_ledger(archive_root / "paper.sqlite")
    args = _cli_args(tmp_path)

    assert cli.main(args) == 0
    tick = _tick(capsys)
    assert tick["scan"] is None
    assert tick["outcomes"] == []
    db = tmp_path / "orchestrator.sqlite3"
    assert _jobs(db) == []
    store = JobStore(db)
    try:
        assert [event.event for event in store.events()] == []
    finally:
        store.close()


def test_http_api_cannot_bind_a_paper_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "paper.sqlite"
    _seed_ledger(ledger)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    db = runs_root / "orchestrator.sqlite3"
    JobStore(db).close()

    app = create_app(
        data_root=tmp_path / "data",
        runs_root=runs_root,
        paper_ledger=ledger,
        orchestrator_db=db,
        discovery_archive=runs_root / "discovery.sqlite",
    )
    with TestClient(app) as client:
        response = client.get(
            "/api/autonomy/status",
            params={"watch_paper_ledger": str(ledger), "allow_synthetic": "true"},
        )
        assert response.status_code == 200
        assert client.post("/api/autonomy/status").status_code == 405

    store = JobStore(db)
    try:
        assert store.list_jobs() == []
        assert [event.event for event in store.events()] == []
    finally:
        store.close()
