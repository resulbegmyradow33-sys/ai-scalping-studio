"""API and WebSocket runner tests.

The WebSocket tests are deterministic by construction: connecting sends a
``status`` message with ``seq == 0`` before any simulator work happens, and the
runner is started with ``speed=0`` (as fast as possible) only after the socket
is open, so the first two frames are always ``seq 0`` and ``seq 1``.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scalping.api.app import create_app
from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider

BODY = {
    "provider": "synthetic",
    "symbol": "XAUUSD",
    "seed": 7,
    "start": "2026-01-05T00:00:00Z",
    "end": "2026-01-07T00:00:00Z",
}


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    """Poll a predicate until it holds or the deadline passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


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


# --- history guard, sequence lifecycle and transport errors -----------------


def test_start_rejects_short_history(tmp_path):
    short = SyntheticProvider(seed=7).fetch(
        "XAUUSD", datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 5, 1, tzinfo=UTC)
    )
    ParquetStore(tmp_path).write("XAUUSD", "M1", short)
    with TestClient(create_app(data_root=tmp_path)) as client:
        r = client.post("/api/sim/start", json={"speed": 1, "mode": "replay"})
    assert r.status_code == 409
    assert r.json()["detail"] == "not enough bars: need 120"


def test_csv_download_missing_file_is_400(client, tmp_path):
    body = {
        "provider": "csv",
        "symbol": "XAUUSD",
        "start": "2026-01-05T00:00:00Z",
        "end": "2026-01-06T00:00:00Z",
        "path": str(tmp_path / "missing.csv"),
    }
    r = client.post("/api/data/download", json=body)
    assert r.status_code == 400
    assert "cannot read data source" in r.json()["detail"]


def test_dukascopy_unknown_symbol_is_400(client):
    body = {
        "provider": "dukascopy",
        "symbol": "BOGUS",
        "start": "2026-01-05T00:00:00Z",
        "end": "2026-01-05T02:00:00Z",
    }
    r = client.post("/api/data/download", json=body)
    assert r.status_code == 400
    assert "BOGUS" in r.json()["detail"]


def test_ws_sequence_increases_by_one(client, seed_data):
    with client.websocket_connect("/ws/sim") as ws:
        ws.receive_json()  # status
        client.post("/api/sim/start", json={"speed": 0, "mode": "replay"})
        seqs = [ws.receive_json()["seq"] for _ in range(50)]
    assert seqs == list(range(1, 51))


def test_ring_keeps_only_the_last_500_messages(client, seed_data):
    runner = client.app.state.sim
    with client.websocket_connect("/ws/sim") as ws:
        ws.receive_json()  # status
        client.post("/api/sim/start", json={"speed": 0, "mode": "replay"})
        assert _wait_for(lambda: runner.state == "finished")
        head = runner.seq
        assert head > 500
        assert runner.buffered_size == 500
    with client.websocket_connect("/ws/sim") as ws:
        ws.receive_json()  # status
        ws.send_json({"type": "resume", "seq": 0})
        seqs = [ws.receive_json()["seq"] for _ in range(500)]
    assert seqs == list(range(head - 499, head + 1))


def test_speed_ten_emits_eight_bars_within_a_second(client, seed_data):
    with client.websocket_connect("/ws/sim") as ws:
        ws.receive_json()  # status
        client.post("/api/sim/start", json={"speed": 10, "mode": "replay"})
        started = time.monotonic()
        events = []
        while time.monotonic() - started < 1.0:
            events.append(ws.receive_json())
    # Collect for a full second and require a tolerant floor. No upper bound:
    # speed 10 is the schedule, not a ceiling on what may arrive.
    assert len(events) >= 8


def test_two_sockets_receive_the_same_sequence(client, seed_data):
    with (
        client.websocket_connect("/ws/sim") as first,
        client.websocket_connect("/ws/sim") as second,
    ):
        assert first.receive_json()["seq"] == 0
        assert second.receive_json()["seq"] == 0
        client.post("/api/sim/start", json={"speed": 1, "mode": "replay"})
        left = [first.receive_json() for _ in range(3)]
        right = [second.receive_json() for _ in range(3)]
    assert [message["seq"] for message in left] == [1, 2, 3]
    assert [message["seq"] for message in right] == [1, 2, 3]
    assert [message["type"] for message in left] == [message["type"] for message in right]
    assert [message["type"] for message in left[:2]] == ["bar", "decision"]


def test_status_carries_session_and_head(client, seed_data):
    runner = client.app.state.sim
    with client.websocket_connect("/ws/sim") as ws:
        status = ws.receive_json()
        assert status["seq"] == 0 and status["type"] == "status"
        assert status["payload"]["session_id"] == runner.session_id
        assert status["payload"]["head_seq"] == runner.seq
        assert status["payload"]["state"] == "idle"
        client.post("/api/sim/start", json={"speed": 1, "mode": "replay"})
        ws.receive_json()


def test_resume_above_head_resends_the_status_handshake(client, seed_data):
    runner = client.app.state.sim
    with client.websocket_connect("/ws/sim") as ws:
        ws.receive_json()  # status
        client.post("/api/sim/start", json={"speed": 1, "mode": "replay"})
        for _ in range(3):
            ws.receive_json()
        assert client.post("/api/sim/control", json={"action": "stop"}).json()["ok"] is True
        head = runner.seq
    with client.websocket_connect("/ws/sim") as ws:
        assert ws.receive_json()["seq"] == 0
        ws.send_json({"type": "resume", "seq": head + 25})
        handshake = ws.receive_json()
        assert handshake["seq"] == 0 and handshake["type"] == "status"
        assert handshake["payload"]["head_seq"] == head
        assert handshake["payload"]["session_id"] == runner.session_id


def test_restart_resets_the_sequence_and_bumps_the_session(client, seed_data):
    runner = client.app.state.sim
    with client.websocket_connect("/ws/sim") as ws:
        ws.receive_json()  # status
        client.post("/api/sim/start", json={"speed": 1, "mode": "replay"})
        for _ in range(3):
            ws.receive_json()
        first_session = runner.session_id
        client.post("/api/sim/control", json={"action": "stop"})
        client.post("/api/sim/start", json={"speed": 1, "mode": "replay"})
        # Frames from the old session that were already queued arrive first;
        # the handshake is the first status frame after the restart.
        for _ in range(10):
            handshake = ws.receive_json()
            if handshake["type"] == "status":
                break
        else:
            pytest.fail("the restarted run never sent a status handshake")
        assert handshake["seq"] == 0
        assert handshake["payload"]["session_id"] != first_session
        assert handshake["payload"]["head_seq"] == 0
        assert ws.receive_json()["seq"] == 1


def test_restart_handshake_matches_the_rewound_cursor(client, seed_data):
    """A live second client must not be told a head above its rewind point."""
    with (
        client.websocket_connect("/ws/sim") as first,
        client.websocket_connect("/ws/sim") as second,
    ):
        first.receive_json()  # status
        second.receive_json()  # status
        client.post("/api/sim/start", json={"speed": 1, "mode": "replay"})
        for _ in range(3):
            first.receive_json()
            second.receive_json()
        client.post("/api/sim/control", json={"action": "stop"})
        client.post("/api/sim/start", json={"speed": 0, "mode": "replay"})
        handshake = None
        for _ in range(20):
            message = second.receive_json()
            if message["type"] == "status":
                handshake = message
                break
        assert handshake is not None, "the second client never received a restart handshake"
        following = second.receive_json()
    assert handshake["payload"]["head_seq"] <= following["seq"]


def test_step_failure_ends_the_episode_with_an_error(client, seed_data, monkeypatch, caplog):
    runner = client.app.state.sim

    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(runner, "_step_locked", boom)
    with client.websocket_connect("/ws/sim") as ws:
        ws.receive_json()  # status
        with caplog.at_level(logging.ERROR, logger="scalping.api.runner"):
            client.post("/api/sim/start", json={"speed": 0, "mode": "replay"})
            message = ws.receive_json()
    assert message["type"] == "episode_end"
    assert message["payload"] == {"reason": "error"}
    assert runner.state == "error"
    assert "simulator step failed" in caplog.text
