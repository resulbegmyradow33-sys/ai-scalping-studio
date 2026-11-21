"""API and WebSocket runner tests.

The WebSocket tests are deterministic by construction: connecting sends a
``status`` message with ``seq == 0`` before any simulator work happens, and the
runner is started with ``speed=0`` (as fast as possible) only after the socket
is open, so the first two frames are always ``seq 0`` and ``seq 1``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scalping.api.app import create_app

BODY = {
    "provider": "synthetic",
    "symbol": "XAUUSD",
    "seed": 7,
    "start": "2026-01-05T00:00:00Z",
    "end": "2026-01-07T00:00:00Z",
}


@pytest.fixture
def client(tmp_path):
    app = create_app(data_root=tmp_path)
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_data_status_empty_store(client):
    r = client.get("/api/data/status", params={"symbol": "XAUUSD"})
    assert r.status_code == 200 and r.json()["rows"] == 0


def test_sim_start_requires_data(client):
    r = client.post("/api/sim/start", json={"speed": 10, "mode": "replay"})
    assert r.status_code == 409
    assert "no data" in r.json()["detail"].lower()


def test_ws_sends_sequenced_messages(client, seed_data):
    with client.websocket_connect("/ws/sim") as ws:
        client.post("/api/sim/start", json={"speed": 0, "mode": "replay"})
        first = ws.receive_json()
        second = ws.receive_json()
        assert first["seq"] == 0 and second["seq"] == 1
        assert first["type"] in {"bar", "decision", "equity", "status"}


def test_control_stop(client, seed_data):
    client.post("/api/sim/start", json={"speed": 0, "mode": "replay"})
    assert client.post("/api/sim/control", json={"action": "stop"}).json()["ok"] is True


def test_synthetic_download_then_status(client):
    r = client.post("/api/data/download", json=BODY)
    assert r.status_code == 200 and r.json()["rows"] > 0
    status = client.get("/api/data/status", params={"symbol": "XAUUSD"}).json()
    assert status["rows"] == r.json()["rows"]
    assert status["providers"] == ["synthetic"]


# --- runner contract beyond the plan's smoke tests -------------------------


def test_sim_start_returns_session_id(client, seed_data):
    r = client.post("/api/sim/start", json={"speed": 0, "mode": "replay"})
    assert r.status_code == 200
    assert isinstance(r.json()["session_id"], str) and r.json()["session_id"]


def test_stop_marks_runner_not_playing(client, seed_data):
    client.post("/api/sim/start", json={"speed": 0, "mode": "replay"})
    runner = client.app.state.sim
    assert runner.playing is True
    client.post("/api/sim/control", json={"action": "stop"})
    assert runner.playing is False


def test_stop_halts_stepping(client, seed_data):
    with client.websocket_connect("/ws/sim") as ws:
        ws.receive_json()  # status
        client.post("/api/sim/start", json={"speed": 10, "mode": "replay"})
        ws.receive_json()
        ws.receive_json()
        assert client.post("/api/sim/control", json={"action": "stop"}).json()["ok"] is True
        runner = client.app.state.sim
        stopped_at = runner.seq
        time.sleep(0.3)  # three scheduling intervals: no step may happen now
        assert runner.seq == stopped_at


def test_ws_resume_replays_only_newer_messages(client, seed_data):
    # One bar per second keeps the ring buffer small enough that ``seq 2`` is
    # still buffered when the second connection resumes from ``seq 1``.
    with client.websocket_connect("/ws/sim") as ws:
        status = ws.receive_json()
        assert status["seq"] == 0 and status["type"] == "status"
        client.post("/api/sim/start", json={"speed": 1, "mode": "replay"})
        for _ in range(3):
            ws.receive_json()
        assert client.post("/api/sim/control", json={"action": "stop"}).json()["ok"] is True
    with client.websocket_connect("/ws/sim") as ws:
        status = ws.receive_json()
        assert status["seq"] == 0 and status["type"] == "status"
        ws.send_json({"type": "resume", "seq": 1})
        replay = ws.receive_json()
        assert replay["seq"] == 2
        assert replay["type"] == "decision"


def test_two_apps_do_not_share_state(tmp_path):
    app_a = create_app(data_root=tmp_path / "a")
    app_b = create_app(data_root=tmp_path / "b")
    with TestClient(app_a) as a:
        assert a.post("/api/data/download", json=BODY).json()["rows"] > 0
        assert a.get("/api/data/status", params={"symbol": "XAUUSD"}).json()["rows"] > 0
    with TestClient(app_b) as b:
        assert b.get("/api/data/status", params={"symbol": "XAUUSD"}).json()["rows"] == 0
    assert app_a.state.store is not app_b.state.store
    assert app_a.state.sim is not app_b.state.sim


def test_runs_root_is_not_created_eagerly(tmp_path):
    runs_root = tmp_path / "runs"
    app = create_app(data_root=tmp_path / "data", runs_root=runs_root)
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 200
    assert not runs_root.exists()


def test_download_rejects_unknown_provider(client):
    body = dict(BODY, provider="nope")
    assert client.post("/api/data/download", json=body).status_code == 422


def test_csv_download_uses_the_given_offset(client):
    path = Path(__file__).parent / "fixtures" / "xauusd_gmt3.csv"
    body = {
        "provider": "csv",
        "symbol": "XAUUSD",
        "start": "2026-01-05T00:00:00Z",
        "end": "2026-01-06T00:00:00Z",
        "path": str(path),
        "tz_offset": 3,
    }
    r = client.post("/api/data/download", json=body)
    assert r.status_code == 200 and r.json()["rows"] == 5
    status = client.get("/api/data/status", params={"symbol": "XAUUSD"}).json()
    assert status["providers"] == ["csv_no_spread"]


def test_dukascopy_unavailable_maps_to_503(client, monkeypatch):
    from datetime import UTC, datetime

    from scalping.api import routes
    from scalping.data.dukascopy import DataUnavailable

    def unavailable(self, symbol, start=None, end=None):
        raise DataUnavailable(symbol, datetime(2026, 1, 5, tzinfo=UTC), 429)

    monkeypatch.setattr(routes.DukascopyProvider, "fetch", unavailable)
    body = {
        "provider": "dukascopy",
        "symbol": "XAUUSD",
        "start": "2026-01-05T00:00:00Z",
        "end": "2026-01-05T02:00:00Z",
    }
    r = client.post("/api/data/download", json=body)
    assert r.status_code == 503
    assert "unavailable" in r.json()["detail"]


def test_settings_path_is_loaded(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("contract:\n  initial_balance: 1234.0\n", encoding="utf-8")
    app = create_app(data_root=tmp_path / "data", settings_path=path)
    assert app.state.settings.contract.initial_balance == 1234.0
