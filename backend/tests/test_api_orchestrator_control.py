"""Owner control API: persisted flags, an audit line, no queue or daemon effect.

The route is the only HTTP writer in the autonomous surface. These tests pin the
contract the owner relies on: each action changes exactly one flag and appends
one control event, a missing database is a 409, a malformed body is a 422, and
nothing ever claims, enqueues or runs a job.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from scalping.api.app import create_app
from scalping.orchestrator.core import STATE_PAUSED, STATE_RESEARCH_DISABLED, STATE_SAFE_MODE
from scalping.orchestrator.store import JobStore

STATE_KEYS = {
    "paused": STATE_PAUSED,
    "safe_mode": STATE_SAFE_MODE,
    "research_disabled": STATE_RESEARCH_DISABLED,
}


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "runs" / "orchestrator.sqlite3"


@pytest.fixture
def seeded(db_path: Path) -> Path:
    """A real job store with one queued job so side effects are visible."""
    db_path.parent.mkdir(parents=True)
    with JobStore(db_path) as store:
        store.enqueue("knowledge.ingest", dedupe_key="seed:doc")
    return db_path


@pytest.fixture
def client(tmp_path: Path, seeded: Path) -> TestClient:
    app = create_app(
        data_root=tmp_path / "data",
        runs_root=tmp_path / "runs",
        paper_ledger=tmp_path / "data" / "paper.sqlite",
        orchestrator_db=seeded,
        discovery_archive=tmp_path / "runs" / "discovery" / "discovery.sqlite",
    )
    return TestClient(app)


def _state(path: Path, key: str) -> dict[str, Any] | None:
    with JobStore(path) as store:
        raw = store.get_state(key)
    return None if raw is None else json.loads(raw)


def _jobs(path: Path) -> list[Any]:
    with JobStore(path) as store:
        return store.connection.execute("SELECT * FROM jobs ORDER BY id").fetchall()


def _events(path: Path) -> list[Any]:
    with JobStore(path) as store:
        return store.events()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ persistence


@pytest.mark.parametrize(
    ("action", "reason", "flag", "on", "event"),
    [
        ("pause", "owner review", "paused", True, "paused"),
        ("safe_mode_on", "incident drill", "safe_mode", True, "safe_mode_on"),
        ("safe_mode_off", "drill finished", "safe_mode", False, "safe_mode_off"),
        ("disable_research", "free the machine", "research_disabled", True, "research_disabled"),
        ("enable_research", None, "research_disabled", False, "research_enabled"),
    ],
)
def test_each_action_persists_its_flag_and_writes_one_audit_line(
    client: TestClient,
    seeded: Path,
    action: str,
    reason: str | None,
    flag: str,
    on: bool,
    event: str,
) -> None:
    body = {"action": action} if reason is None else {"action": action, "reason": reason}

    response = client.post("/api/orchestrator/control", json=body)

    assert response.status_code == 200, response.text
    assert response.json()[flag] is on
    persisted = _state(seeded, STATE_KEYS[flag])
    assert persisted is not None
    assert persisted["on"] is on
    assert persisted["reason"] == reason

    control_events = [row for row in _events(seeded) if row.event == event]
    assert len(control_events) == 1
    detail = json.loads(control_events[0].detail or "{}")
    assert detail["on"] is on
    assert detail["reason"] == reason
    assert detail["actor"] == "local-owner"
    assert control_events[0].job_id is None

    body_state = client.get("/api/orchestrator/control").json()
    expected_change: dict[str, Any] = {
        "action": action,
        "actor": "local-owner",
        "at": control_events[0].ts,
    }
    if reason is not None:
        expected_change["reason"] = reason
    assert body_state["last_change"] == expected_change


def test_resume_after_pause_clears_the_flag_and_is_audited(
    client: TestClient, seeded: Path
) -> None:
    assert (
        client.post(
            "/api/orchestrator/control", json={"action": "pause", "reason": "stop spending"}
        ).status_code
        == 200
    )
    assert client.post("/api/orchestrator/control", json={"action": "resume"}).status_code == 200

    assert _state(seeded, STATE_PAUSED)["on"] is False
    controls = [row for row in _events(seeded) if row.event in {"paused", "resumed"}]
    assert [row.event for row in controls] == ["paused", "resumed"]
    assert client.get("/api/orchestrator/control").json()["paused"] is False
    assert client.get("/api/orchestrator/control").json()["last_change"]["action"] == "resume"


def test_pause_is_idempotent(client: TestClient, seeded: Path) -> None:
    first = client.post("/api/orchestrator/control", json={"action": "pause", "reason": "first"})
    second = client.post("/api/orchestrator/control", json={"action": "pause", "reason": "again"})

    assert first.status_code == second.status_code == 200
    assert second.json()["paused"] is True
    assert _state(seeded, STATE_PAUSED)["on"] is True
    # Two calls, two audit lines: the log is append-only, not de-duplicated.
    assert [row.event for row in _events(seeded)].count("paused") == 2
    assert client.get("/api/orchestrator/control").json()["last_change"]["reason"] == "again"


def test_get_reports_absent_controls_as_off_with_no_last_change(
    client: TestClient,
) -> None:
    body = client.get("/api/orchestrator/control")

    assert body.status_code == 200
    assert body.json() == {
        "configured": True,
        "readable": True,
        "paused": False,
        "safe_mode": False,
        "research_disabled": False,
    }


# ---------------------------------------------------------------------- errors


def test_post_without_a_database_is_409_and_creates_nothing(tmp_path: Path) -> None:
    target = tmp_path / "runs" / "orchestrator.sqlite3"
    app = create_app(data_root=tmp_path / "data", runs_root=tmp_path / "runs")
    response = TestClient(app).post(
        "/api/orchestrator/control", json={"action": "pause", "reason": "why not"}
    )

    assert response.status_code == 409, response.text
    assert "orchestrator database is not available" in response.json()["detail"]
    assert not target.exists()
    assert str(tmp_path) not in response.text


def test_get_without_a_database_is_unconfigured_not_an_error(tmp_path: Path) -> None:
    app = create_app(data_root=tmp_path / "data", runs_root=tmp_path / "runs")

    response = TestClient(app).get("/api/orchestrator/control")

    assert response.status_code == 200
    assert response.json() == {"configured": False, "readable": False}
    assert not (tmp_path / "runs" / "orchestrator.sqlite3").exists()


@pytest.mark.parametrize(
    "body",
    [
        {"action": "explode"},
        {"action": "PAUSE", "reason": "case sensitive"},
        {"action": "pause"},
        {"action": "pause", "reason": ""},
        {"action": "pause", "reason": "   "},
        {"action": "safe_mode_on"},
        {"action": "safe_mode_off"},
        {"action": "safe_mode_on", "reason": None},
        {"action": "pause", "reason": "ok", "path": "../elsewhere"},
        {"action": "resume", "actor": "root"},
        {"action": "pause", "reason": "ok", "extra": 1},
        {"reason": "no action"},
        {},
    ],
)
def test_invalid_bodies_are_422_and_do_not_touch_the_store(
    client: TestClient, seeded: Path, body: dict[str, Any]
) -> None:
    before = _sha256(seeded)

    response = client.post("/api/orchestrator/control", json=body)

    assert response.status_code == 422, (body, response.text)
    assert _sha256(seeded) == before
    assert not any(
        row.event in {"paused", "safe_mode_on", "safe_mode_off"} for row in _events(seeded)
    )


def test_reason_is_optional_for_resume_and_enable(client: TestClient) -> None:
    for action in ("resume", "enable_research", "safe_mode_off"):
        reason = None if action != "safe_mode_off" else "closing the drill"
        body = {"action": action} if reason is None else {"action": action, "reason": reason}
        assert client.post("/api/orchestrator/control", json=body).status_code == 200, action


# ---------------------------------------------------------------- no side effects


def test_controls_never_enqueue_claim_or_run_a_job(client: TestClient, seeded: Path) -> None:
    before = _jobs(seeded)
    events_before = len(_events(seeded))

    for body in (
        {"action": "pause", "reason": "stop"},
        {"action": "resume"},
        {"action": "safe_mode_on", "reason": "drill"},
        {"action": "safe_mode_off", "reason": "drill over"},
        {"action": "disable_research", "reason": "free the machine"},
        {"action": "enable_research"},
    ):
        assert client.post("/api/orchestrator/control", json=body).status_code == 200

    after = _jobs(seeded)
    assert len(after) == len(before) == 1
    assert after[0]["status"] == "queued"
    assert after[0]["attempts"] == 0
    assert after[0]["dedupe_key"] == "seed:doc"

    events = _events(seeded)
    assert len(events) == events_before + 6
    assert not any(
        row.event in {"started", "succeeded", "enqueued"} for row in events[events_before:]
    )


def test_get_is_read_only_on_a_seeded_database(client: TestClient, seeded: Path) -> None:
    before = _sha256(seeded)
    control_events_before = len(_events(seeded))

    for _ in range(3):
        assert client.get("/api/orchestrator/control").status_code == 200

    assert _sha256(seeded) == before
    assert len(_events(seeded)) == control_events_before
