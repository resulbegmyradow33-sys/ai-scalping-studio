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
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scalping.api.app import create_app
from scalping.backtest.metrics import Metrics
from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider
from scalping.rl.policy import load_policy
from scalping.rl.registry import save_manifest
from scalping.strategy.bandit import LinUCB
from scalping.strategy.labeling import CONTEXT_DIM
from scalping.strategy.signals import ALL_STRATEGIES

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
    r = client.post(
        "/api/train", json={"total_steps": 256, "n_envs": 1, "seed": 0, "symbol": "XAUUSD"}
    )
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
    r = client.post(
        "/api/train", json={"total_steps": 256, "n_envs": 1, "seed": 0, "symbol": "XAUUSD"}
    )
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

    assert client.post(f"/api/runs/{run_id}/control", json={"action": "pause"}).json() == {
        "ok": True
    }
    assert _wait_status(client, run_id, {"paused"})["status"] == "paused"

    assert client.post(f"/api/runs/{run_id}/control", json={"action": "resume"}).json() == {
        "ok": True
    }
    assert _wait_status(client, run_id, {"running"})["status"] == "running"

    assert client.post(f"/api/runs/{run_id}/control", json={"action": "stop"}).json() == {
        "ok": True
    }
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


def test_stop_moves_the_visible_status_off_running(client, seed_data):
    r = client.post("/api/train", json=LONG_BODY)
    run_id = r.json()["run_id"]
    assert client.post(f"/api/runs/{run_id}/control", json={"action": "pause"}).status_code == 200
    _wait_status(client, run_id, {"paused"})

    assert client.post(f"/api/runs/{run_id}/control", json={"action": "stop"}).json() == {
        "ok": True
    }
    stopped = client.get(f"/api/runs/{run_id}").json()["status"]
    # The worker may already have written ``interrupted``; it must never still
    # report the run as controllable.
    assert stopped in {"stopping", "interrupted"}, stopped
    assert stopped != "paused"
    assert client.get("/api/runs").json()[0]["status"] in {"stopping", "interrupted"}

    detail = _wait_status(client, run_id, {"interrupted"})
    assert detail["status"] == "interrupted"


def test_pause_and_resume_after_stop_are_409(client, seed_data):
    r = client.post("/api/train", json=LONG_BODY)
    run_id = r.json()["run_id"]
    assert client.post(f"/api/runs/{run_id}/control", json={"action": "pause"}).status_code == 200
    _wait_status(client, run_id, {"paused"})
    assert client.post(f"/api/runs/{run_id}/control", json={"action": "stop"}).status_code == 200

    for action in ("pause", "resume"):
        response = client.post(f"/api/runs/{run_id}/control", json={"action": action})
        assert response.status_code == 409, action
    _wait_status(client, run_id, {"interrupted"})


def test_train_rejects_too_many_envs_before_reserving(client, seed_data, tmp_path):
    r = client.post("/api/train", json={**BODY, "n_envs": 100_000})
    assert r.status_code == 409
    assert "n_envs 100000 exceeds" in r.json()["detail"]
    assert not (tmp_path / "runs").exists()


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
    for run_id, created in (
        ("old", "2026-01-01T00:00:00+00:00"),
        ("new", "2026-02-01T00:00:00+00:00"),
    ):
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
    assert set(listed[0]) == {
        "run_id",
        "status",
        "created_at",
        "steps",
        "reward_mean",
        "reward_mean_normalized",
        "degenerate",
        "baseline_hold_net_profit",
    }


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


def test_backtest_rejects_a_run_that_is_still_training(client, seed_data):
    r = client.post("/api/train", json=LONG_BODY)
    run_id = r.json()["run_id"]

    response = client.post("/api/backtest", json={"run_id": run_id, "symbol": "XAUUSD"})
    assert response.status_code == 409
    assert "not finished" in response.json()["detail"]

    client.post(f"/api/runs/{run_id}/control", json={"action": "stop"})
    _wait_status(client, run_id, {"interrupted"})


def test_backtest_rejects_a_failed_run(client, tmp_path):
    save_manifest(
        tmp_path / "runs" / "dead",
        {
            "run_id": "dead",
            "created_at": "2026-01-01T00:00:00+00:00",
            "status": "failed",
            "steps": 0,
            "reward_mean": None,
        },
    )
    response = client.post("/api/backtest", json={"run_id": "dead", "symbol": "XAUUSD"})
    assert response.status_code == 409
    assert "failed" in response.json()["detail"]


def test_backtest_rejects_mismatched_data(client, seed_data, tmp_path):
    r = client.post("/api/train", json=BODY)
    run_id = r.json()["run_id"]
    _wait_status(client, run_id, {"completed"})

    other = SyntheticProvider(seed=99).fetch(
        "XAUUSD", datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 8, tzinfo=UTC)
    )
    ParquetStore(tmp_path).write("XAUUSD", "M1", other)

    response = client.post("/api/backtest", json={"run_id": run_id, "symbol": "XAUUSD"})
    assert response.status_code == 409
    assert "different data" in response.json()["detail"]


def test_backtest_report_records_both_data_hashes(client, seed_data, tmp_path):
    r = client.post("/api/train", json=BODY)
    run_id = r.json()["run_id"]
    _wait_status(client, run_id, {"completed"})

    body = client.post("/api/backtest", json={"run_id": run_id, "symbol": "XAUUSD"}).json()
    stored = json.loads(Path(body["report_path"]).read_text())
    assert stored["data_hash"] == stored["run_data_hash"]
    assert len(stored["data_hash"]) == 64


def test_listing_survives_a_non_numeric_step_count(client, tmp_path):
    save_manifest(
        tmp_path / "runs" / "odd",
        {
            "run_id": "odd",
            "created_at": "2026-01-01T00:00:00+00:00",
            "status": "completed",
            "steps": "not-a-number",
            "reward_mean": 0.1,
        },
    )
    response = client.get("/api/runs")
    assert response.status_code == 200
    assert response.json()[0]["steps"] == 0


def _fake_run(
    tmp_path: Path,
    run_id: str,
    *,
    config: dict | None = None,
    metrics: dict | None = None,
    history: list[dict] | None = None,
) -> Path:
    """Create a registry directory the detail endpoint can serve."""
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    save_manifest(
        run_dir,
        {
            "run_id": run_id,
            "created_at": "2026-01-01T00:00:00+00:00",
            "status": "completed",
            "steps": 0,
            "reward_mean": None,
        },
    )
    if config is not None:
        (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if metrics is not None:
        (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if history is not None:
        lines = "\n".join(json.dumps(row) for row in history)
        (run_dir / "metrics.jsonl").write_text(lines + "\n", encoding="utf-8")
    return run_dir


def test_detail_builds_ordered_series_from_the_metrics_history(client, seed_data, tmp_path):
    r = client.post("/api/train", json=BODY)
    run_id = r.json()["run_id"]
    _wait_status(client, run_id, {"completed"})

    history = [
        {
            "step": 100,
            "reward_mean_interval": 0.10,
            "ep_len_mean": 10.0,
            "entropy_loss": 0.20,
            "policy_loss": 0.30,
        },
        {
            "step": 200,
            "reward_mean": 0.05,
            "ep_len_mean": 20.0,
            "entropy_loss": None,
            "policy_loss": 0.40,
        },
        {
            "step": 300,
            "reward_mean_interval": 0.30,
            "ep_len_mean": 30.0,
            "entropy_loss": 0.60,
            "policy_loss": 0.70,
        },
    ]
    run_dir = tmp_path / "runs" / run_id
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in history) + "\n", encoding="utf-8"
    )

    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["total_steps"] == 256
    latest = json.loads((run_dir / "metrics.json").read_text())
    assert detail["step"] == latest["step"]

    assert detail["series"]["reward"] == [
        {"step": 100, "value": 0.10},
        {"step": 200, "value": 0.05},  # falls back to reward_mean
        {"step": 300, "value": 0.30},
    ]
    assert detail["series"]["episode_length"] == [
        {"step": 100, "value": 10.0},
        {"step": 200, "value": 20.0},
        {"step": 300, "value": 30.0},
    ]
    # The null in row 2 is skipped, not turned into a zero.
    assert detail["series"]["entropy"] == [
        {"step": 100, "value": 0.20},
        {"step": 300, "value": 0.60},
    ]
    assert [point["step"] for point in detail["series"]["loss"]] == [100, 200, 300]


def test_detail_without_history_omits_series(client, tmp_path):
    _fake_run(
        tmp_path,
        "nohist",
        config={"params": {"total_steps": 1000}},
        metrics={"step": 42, "reward_mean": 0.5},
    )

    detail = client.get("/api/runs/nohist").json()
    assert "series" not in detail
    assert detail["step"] == 42
    assert detail["total_steps"] == 1000


def test_detail_returns_bandit_weights_in_strategy_order(client, tmp_path):
    bandit = LinUCB(n_arms=len(ALL_STRATEGIES), dim=CONTEXT_DIM)
    bandit.fit_offline(
        np.eye(CONTEXT_DIM)[: len(ALL_STRATEGIES)],
        np.arange(len(ALL_STRATEGIES)),
        np.arange(1, len(ALL_STRATEGIES) + 1, dtype=float),
    )
    expected = [float(np.linalg.norm(theta)) for theta in bandit.weights()]
    run_dir = _fake_run(tmp_path, "bandit", metrics={"step": 10})
    bandit.save(run_dir / "bandit.npz")

    rows = client.get("/api/runs/bandit").json()["bandit"]
    assert [row["strategy"] for row in rows] == [strategy.value for strategy in ALL_STRATEGIES]
    assert [row["weight"] for row in rows] == pytest.approx(expected)


def test_train_rejects_a_store_below_the_bar_floor(client, tmp_path):
    frame = SyntheticProvider(seed=7).fetch(
        "XAUUSD", datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 6, tzinfo=UTC)
    )
    ParquetStore(tmp_path).write("XAUUSD", "M1", frame)

    r = client.post("/api/train", json=BODY)
    assert r.status_code == 409
    assert "not enough bars: need 4000" in r.json()["detail"]
    assert not (tmp_path / "runs").exists()


def test_train_uses_the_fitted_bandit_by_default(client, seed_data, tmp_path):
    r = client.post("/api/train", json=BODY)
    run_id = r.json()["run_id"]
    detail = _wait_status(client, run_id, {"completed"})
    config = detail["config"]
    assert config["use_strategy"] is True
    assert config["obs_dim"] == 3551
    assert (tmp_path / "runs" / run_id / "bandit.npz").is_file()
    assert detail["status"] == "completed"


def test_train_without_strategy_keeps_the_plain_observation(client, seed_data, tmp_path):
    r = client.post("/api/train", json={**BODY, "use_strategy": False})
    run_id = r.json()["run_id"]
    detail = _wait_status(client, run_id, {"completed"})
    assert detail["config"]["use_strategy"] is False
    assert detail["config"]["obs_dim"] == 3545
    assert not (tmp_path / "runs" / run_id / "bandit.npz").exists()


def test_train_end_records_the_split(client, seed_data):
    r = client.post(
        "/api/train",
        json={**BODY, "use_strategy": False, "train_end": "2026-01-07T23:00:00+00:00"},
    )
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    detail = _wait_status(client, run_id, {"completed"})
    config = detail["config"]
    assert config["train_end"] == "2026-01-07T23:00:00+00:00"
    assert config["eval_start"] == "2026-01-07T23:01:00+00:00"


def test_backtest_marks_a_split_run_out_of_sample(client, seed_data, tmp_path):
    r = client.post(
        "/api/train",
        json={**BODY, "use_strategy": False, "train_end": "2026-01-07T23:00:00+00:00"},
    )
    run_id = r.json()["run_id"]
    _wait_status(client, run_id, {"completed"})

    body = client.post("/api/backtest", json={"run_id": run_id, "symbol": "XAUUSD"}).json()
    stored = json.loads(Path(body["report_path"]).read_text())
    assert stored["in_sample"] is False
    assert stored["train_end"] == "2026-01-07T23:00:00+00:00"


def test_backtest_marks_a_full_history_run_in_sample(client, seed_data):
    r = client.post("/api/train", json=BODY)
    run_id = r.json()["run_id"]
    _wait_status(client, run_id, {"completed"})
    body = client.post("/api/backtest", json={"run_id": run_id, "symbol": "XAUUSD"}).json()
    stored = json.loads(Path(body["report_path"]).read_text())
    assert stored["in_sample"] is True
    assert stored["train_end"] is None


def test_runs_surface_the_degeneracy_gate(client, seed_data):
    r = client.post("/api/train", json=BODY)
    run_id = r.json()["run_id"]
    _wait_status(client, run_id, {"completed"})
    row = next(item for item in client.get("/api/runs").json() if item["run_id"] == run_id)
    assert "degenerate" in row
    assert "baseline_hold_net_profit" in row
    assert "reward_mean_normalized" in row


def _seed_long_history(tmp_path: Path, days: int = 160, seed: int = 33) -> None:
    frame = SyntheticProvider(seed=seed).fetch(
        "XAUUSD",
        datetime(2026, 1, 5, tzinfo=UTC),
        datetime(2026, 1, 5, tzinfo=UTC) + timedelta(days=days),
    )
    ParquetStore(tmp_path).write("XAUUSD", "M1", frame)


def test_evaluate_returns_three_walkforward_folds(client, tmp_path):
    # Train and evaluate on the same stored history: the identity guard must pass.
    _seed_long_history(tmp_path, days=160, seed=7)
    r = client.post("/api/train", json={**BODY, "use_strategy": False})
    run_id = r.json()["run_id"]
    _wait_status(client, run_id, {"completed"})

    response = client.post("/api/evaluate", json={"run_id": run_id, "symbol": "XAUUSD"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["folds"]) == 3
    assert len(body["stress"]) == 3
    assert isinstance(body["overfit_warning"], bool)
    assert {"net_profit", "trades", "profit_factor"} <= set(body["overall"])
    assert body["in_sample"] is False
    assert body["train_end"] is None
    assert body["eval_start"] is None
    assert body["data_hash"] == body["eval_data_hash"]
    report = Path(body["report_path"])
    assert report.is_file()
    assert report.parent == tmp_path / "runs" / run_id
    stored = json.loads(report.read_text())
    assert stored["run_id"] == run_id
    assert len(stored["folds"]) == 3
    assert stored["data_hash"] == stored["eval_data_hash"]
    assert stored["in_sample"] is False


def test_evaluate_rejects_a_different_data_store(client, seed_data, tmp_path):
    r = client.post("/api/train", json={**BODY, "use_strategy": False})
    run_id = r.json()["run_id"]
    _wait_status(client, run_id, {"completed"})

    _seed_long_history(tmp_path, days=160, seed=33)
    response = client.post("/api/evaluate", json={"run_id": run_id, "symbol": "XAUUSD"})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "different data" in detail
    assert "data_hash" in detail
    assert len(detail.split("data_hash ")) == 3  # both hashes are named


def test_train_with_strategy_rejects_an_empty_label_batch(client, tmp_path):
    import polars as pl

    flat = SyntheticProvider(seed=7).fetch(
        "XAUUSD",
        datetime(2026, 1, 5, tzinfo=UTC),
        datetime(2026, 1, 8, tzinfo=UTC),
    ).with_columns(
        [
            pl.lit(2350.0).alias("open"),
            pl.lit(2350.01).alias("high"),
            pl.lit(2349.99).alias("low"),
            pl.lit(2350.0).alias("close"),
        ]
    )
    ParquetStore(tmp_path).write("XAUUSD", "M1", flat)

    response = client.post("/api/train", json=BODY)
    assert response.status_code == 409
    assert "no strategy labels" in response.json()["detail"]
    assert not (tmp_path / "runs").exists()


def test_listing_sanitises_a_non_bool_degenerate(client, tmp_path):
    run_dir = tmp_path / "runs" / "odd"
    save_manifest(
        run_dir,
        {
            "run_id": "odd",
            "created_at": "2026-01-01T00:00:00+00:00",
            "status": "completed",
            "steps": 10,
            "reward_mean": 0.5,
            "degenerate": "yes",
        },
    )
    row = client.get("/api/runs").json()[0]
    assert row["degenerate"] is None


def test_evaluate_rejects_a_live_run(client, seed_data):
    r = client.post("/api/train", json=LONG_BODY)
    run_id = r.json()["run_id"]
    response = client.post("/api/evaluate", json={"run_id": run_id, "symbol": "XAUUSD"})
    assert response.status_code == 409
    client.post(f"/api/runs/{run_id}/control", json={"action": "stop"})
    _wait_status(client, run_id, {"interrupted"})
