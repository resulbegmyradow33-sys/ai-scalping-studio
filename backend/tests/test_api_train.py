"""Training API: background PPO runs, registry listing, control and backtest.

Training is started on a worker thread, so every test that starts a run either
waits for a terminal status or lets the app lifespan stop and join it. The
fixture keeps the ``TestClient`` inside a context manager: closing it runs the
lifespan shutdown, which is the path that must not leave a thread behind.
"""

from __future__ import annotations

import json
import time
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scalping.api.app import create_app
from scalping.backtest.metrics import Metrics
from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider
from scalping.rl.policy import load_policy
from scalping.rl.registry import save_manifest

BODY = {"total_steps": 256, "n_envs": 1, "seed": 0, "symbol": "XAUUSD"}
#: A run long enough that a second request arrives while it is still alive.
LONG_BODY = {**BODY, "total_steps": 200_000}


def _wait_status(
    client: TestClient, run_id: str, statuses: set[str], timeout: float = 60.0
) -> dict:
    """Poll one run's detail until its status is in ``statuses``."""
    deadline = time.monotonic() + timeout
    last: dict | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        if response.status_code == 200:
            last = response.json()
            if last["status"] in statuses:
                return last
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not reach {statuses}; last detail {last}")


@pytest.fixture
def client(tmp_path):
    app = create_app(data_root=tmp_path, runs_root=tmp_path / "runs")
    with TestClient(app) as test_client:
        yield test_client


def test_train_rejects_without_data(client):
    r = client.post("/api/train", json={"total_steps": 256, "n_envs": 1, "seed": 0, "symbol": "XAUUSD"})
    assert r.status_code == 409
    assert "not enough bars" in r.json()["detail"]


def test_runs_empty(client, tmp_path):
    assert client.get("/api/runs").json() == []
    # The registry directory is created lazily: listing an absent root is not a
    # reason to create it.
    assert not (tmp_path / "runs").exists()


def test_unknown_run_404(client):
    assert client.get("/api/runs/nope").status_code == 404


def test_train_starts_and_is_listed(client, seed_data):
    r = client.post("/api/train", json={"total_steps": 256, "n_envs": 1, "seed": 0, "symbol": "XAUUSD"})
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    listed = [x["run_id"] for x in client.get("/api/runs").json()]
    assert run_id in listed


def test_train_returns_while_the_thread_is_still_working(client, seed_data):
    """The request only prepares the run; 200 000 steps happen off the loop."""
    started = time.monotonic()
    r = client.post("/api/train", json=LONG_BODY)
    elapsed = time.monotonic() - started
    assert r.status_code == 202
    assert elapsed < 5.0, f"/api/train blocked for {elapsed:.2f}s"

    run_id = r.json()["run_id"]
    assert client.post(f"/api/runs/{run_id}/control", json={"action": "pause"}).status_code == 200
    assert client.post(f"/api/runs/{run_id}/control", json={"action": "stop"}).status_code == 200
    assert _wait_status(client, run_id, {"interrupted"})["status"] == "interrupted"


def test_double_start_on_the_same_run_id_is_409(client, seed_data):
    first = client.post("/api/train", json=LONG_BODY)
    assert first.status_code == 202
    run_id = first.json()["run_id"]
    assert client.post(f"/api/runs/{run_id}/control", json={"action": "pause"}).status_code == 200

    second = client.post("/api/train", json=LONG_BODY)
    assert second.status_code == 409
    assert run_id in second.json()["detail"]

    client.post(f"/api/runs/{run_id}/control", json={"action": "stop"})
    _wait_status(client, run_id, {"interrupted"})


def test_pause_resume_and_stop_leave_a_loadable_model(client, seed_data, tmp_path):
    r = client.post("/api/train", json=LONG_BODY)
    run_id = r.json()["run_id"]

    assert client.post(f"/api/runs/{run_id}/control", json={"action": "pause"}).json() == {"ok": True}
    assert _wait_status(client, run_id, {"paused"})["status"] == "paused"

    assert client.post(f"/api/runs/{run_id}/control", json={"action": "resume"}).json() == {"ok": True}
    assert _wait_status(client, run_id, {"running"})["status"] == "running"

    assert client.post(f"/api/runs/{run_id}/control", json={"action": "stop"}).json() == {"ok": True}
    detail = _wait_status(client, run_id, {"interrupted"})
    assert detail["status"] == "interrupted"
    manifest = json.loads((tmp_path / "runs" / run_id / "manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    policy = load_policy(tmp_path / "runs" / run_id)
    assert policy.obs_dim == detail["config"]["obs_dim"]


def test_shutdown_stops_and_joins_live_threads(client, seed_data):
    r = client.post("/api/train", json=LONG_BODY)
    run_id = r.json()["run_id"]
    assert client.post(f"/api/runs/{run_id}/control", json={"action": "pause"}).status_code == 200

    runner = client.app.state.trains
    assert [thread.name for thread in runner.active_threads()] == [f"train-{run_id}"]
    assert runner.active_threads()  # the worker outlives the request

    runner.shutdown(timeout=30.0)
    assert runner.active_threads() == []
    _wait_status(client, run_id, {"interrupted"})


def test_control_unknown_run_404(client):
    r = client.post("/api/runs/nope/control", json={"action": "pause"})
    assert r.status_code == 404


def test_control_on_a_completed_run_is_409(client, seed_data):
    r = client.post("/api/train", json=BODY)
    run_id = r.json()["run_id"]
    assert _wait_status(client, run_id, {"completed"})

    for action in ("pause", "resume", "stop"):
        response = client.post(f"/api/runs/{run_id}/control", json={"action": action})
        assert response.status_code == 409, action
    assert client.get(f"/api/runs/{run_id}").json()["status"] == "completed"


def test_detail_returns_config_metrics_and_status(client, seed_data):
    r = client.post("/api/train", json=BODY)
    run_id = r.json()["run_id"]
    detail = _wait_status(client, run_id, {"completed"})

    assert detail["config"]["run_id"] == run_id
    assert detail["config"]["params"]["total_steps"] == 256
    assert detail["metrics"]["step"] >= 256
    assert isinstance(detail["metrics"]["action_dist"], list)


def test_runs_are_newest_first(client, tmp_path):
    root = tmp_path / "runs"
    for run_id, created in (("old", "2026-01-01T00:00:00+00:00"), ("new", "2026-02-01T00:00:00+00:00")):
        save_manifest(
            root / run_id,
            {
                "run_id": run_id,
                "created_at": created,
                "status": "completed",
                "steps": 10,
                "reward_mean": 0.5,
            },
        )
    listed = client.get("/api/runs").json()
    assert [item["run_id"] for item in listed] == ["new", "old"]
    assert set(listed[0]) == {"run_id", "status", "created_at", "steps", "reward_mean"}


def test_backtest_uses_the_stored_policy(client, seed_data, tmp_path):
    r = client.post("/api/train", json=BODY)
    run_id = r.json()["run_id"]
    _wait_status(client, run_id, {"completed"})

    response = client.post("/api/backtest", json={"run_id": run_id, "symbol": "XAUUSD"})
    assert response.status_code == 200
    body = response.json()
    report = Path(body["report_path"])
    assert report.is_file()
    assert report.parent == tmp_path / "runs" / run_id
    stored = json.loads(report.read_text())
    assert stored["run_id"] == run_id
    assert body["metrics"] == stored["metrics"]
    assert set(body["metrics"]) == {field.name for field in fields(Metrics)}
    assert body["metrics"]["trades"] >= 0


def test_backtest_unknown_run_404(client, seed_data):
    r = client.post("/api/backtest", json={"run_id": "nope", "symbol": "XAUUSD"})
    assert r.status_code == 404


def test_train_rejects_a_store_below_the_bar_floor(client, tmp_path):
    frame = SyntheticProvider(seed=7).fetch(
        "XAUUSD", datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 6, tzinfo=UTC)
    )
    ParquetStore(tmp_path).write("XAUUSD", "M1", frame)

    r = client.post("/api/train", json=BODY)
    assert r.status_code == 409
    assert "not enough bars: need 4000" in r.json()["detail"]
    assert not (tmp_path / "runs").exists()
