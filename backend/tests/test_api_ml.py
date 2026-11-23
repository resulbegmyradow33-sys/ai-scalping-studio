"""Classical ML API: one-job slot, run lifecycle, reports and trading evaluation.

Every test uses a roughly 15-hour synthetic M1 store and a logistic model on one
feature group, so a fit is tens of milliseconds. Nothing here reads the real
17-year store. Runs are started on a worker thread: a test either waits for a
terminal status or lets the app lifespan cancel and join the run.
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from scalping.api import ml_runner as ml_runner_module
from scalping.api.app import create_app
from scalping.api.ml_runner import UnknownMLRunError
from scalping.api.schemas import is_ml_run_id
from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider

SYMBOL = "XAUUSD"
START = datetime(2026, 1, 5, tzinfo=UTC)
#: 15 hours of M1: enough for a 70/15/15 split after warm-up, small enough to fit
#: in a couple of hundred milliseconds.
BARS = 900

BODY = {
    "symbol": SYMBOL,
    "timeframe": "M1",
    "model": "logistic",
    "horizon": 5,
    "groups": ["price"],
    "purge_bars": 5,
    "embargo_bars": 5,
}

RUN_ID_RE = re.compile(r"^[A-Za-z0-9]+$")


def _wait_status(
    client: TestClient, run_id: str, statuses: set[str], timeout: float = 30.0
) -> dict:
    """Poll one run's detail until its status is in ``statuses``."""
    deadline = time.monotonic() + timeout
    last: dict | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/ml/runs/{run_id}")
        if response.status_code == 200:
            last = response.json()
            if last["status"] in statuses:
                return last
        time.sleep(0.02)
    raise AssertionError(f"ML run {run_id} did not reach {statuses}; last detail {last}")


@pytest.fixture
def client(tmp_path):
    frame = SyntheticProvider(seed=5).fetch(SYMBOL, START, START + timedelta(minutes=BARS))
    ParquetStore(tmp_path).write(SYMBOL, "M1", frame)
    app = create_app(data_root=tmp_path, runs_root=tmp_path / "runs")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def real_train(monkeypatch):
    """The genuine trainer, for fakes that delegate after releasing a latch."""
    return ml_runner_module.train_classifier


def _blocking_trainer(real, started: threading.Event, release: threading.Event):
    def slow(*args, **kwargs):
        started.set()
        if not release.wait(10.0):
            raise RuntimeError("test never released the trainer")
        return real(*args, **kwargs)

    return slow


def test_train_returns_202_and_writes_a_full_report(client, tmp_path):
    response = client.post("/api/ml/train", json=BODY)
    assert response.status_code == 202, response.text
    run_id = response.json()["run_id"]
    assert RUN_ID_RE.fullmatch(run_id)
    assert is_ml_run_id(run_id)
    assert len(run_id) <= 64

    detail = _wait_status(client, run_id, {"completed"})
    assert detail["status"] == "completed"
    assert detail["run_id"] == run_id
    assert detail["model"] == "logistic"
    assert detail["symbol"] == SYMBOL
    assert detail["timeframe"] == "M1"
    assert detail["dataset_id"].startswith(f"{SYMBOL}:M1:h5:")
    assert detail["spec"]["model"] == "logistic"
    assert detail["spec"]["groups"] == ["price"]
    assert detail["config"]["split"] == "validation"
    assert detail["config"]["spec"]["horizon"] == 5

    validation = detail["validation"]
    assert validation["rows"] > 0
    assert {"accuracy", "log_loss", "brier", "calibration"} <= set(validation)
    assert validation["calibration"]["bins"]
    assert detail["test"] is None  # evaluate_test defaults to false
    assert detail["baseline"]["majority_class"] in (-1, 1)

    calibration = detail["calibration"]
    assert set(calibration) == {"brier", "log_loss", "ece", "mce"}
    assert all(value is not None for value in calibration.values())

    assert set(detail["split_counts"]) == {
        "train",
        "validation",
        "test",
        "purged",
        "flat_dropped",
    }
    assert detail["columns"]
    assert len(detail["columns"]) == 14  # the "price" group

    run_dir = tmp_path / "runs" / "ml" / run_id
    assert (run_dir / "meta.json").is_file()
    assert (run_dir / "model.joblib").is_file()
    report = json.loads((run_dir / "report.json").read_text())
    assert report["status"] == "completed"
    assert report["dataset_id"] == detail["dataset_id"]
    assert report["validation"] == validation
    assert report["evaluation"]["split"] == "validation"
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["finished_at"] is not None


def test_evaluate_shares_the_one_job_slot(client, monkeypatch, real_train):
    """The slot is shared by every ML entry point, not only by /train."""
    finished = client.post("/api/ml/train", json=BODY).json()["run_id"]
    _wait_status(client, finished, {"completed"})

    started, release = threading.Event(), threading.Event()
    monkeypatch.setattr(
        ml_runner_module,
        "train_classifier",
        _blocking_trainer(real_train, started, release),
    )
    blocked = client.post("/api/ml/train", json=BODY).json()["run_id"]
    assert started.wait(10.0)

    response = client.post("/api/ml/evaluate", json={"run_id": finished})
    assert response.status_code == 409
    assert "another ML job is running" in response.json()["detail"]

    release.set()
    _wait_status(client, blocked, {"completed"})


def test_a_second_start_while_a_job_runs_is_409_and_the_slot_recovers(
    client, tmp_path, monkeypatch, real_train
):
    started, release = threading.Event(), threading.Event()
    monkeypatch.setattr(
        ml_runner_module,
        "train_classifier",
        _blocking_trainer(real_train, started, release),
    )

    first = client.post("/api/ml/train", json=BODY)
    assert first.status_code == 202
    run_id = first.json()["run_id"]
    assert started.wait(10.0), "the worker never entered the fit"

    second = client.post("/api/ml/train", json=BODY)
    assert second.status_code == 409
    assert "another ML job is running" in second.json()["detail"]

    release.set()
    assert _wait_status(client, run_id, {"completed"})["status"] == "completed"

    # The worker released the slot: the same request now starts a new run.
    third = client.post("/api/ml/train", json=BODY)
    assert third.status_code == 202
    _wait_status(client, third.json()["run_id"], {"completed"})


def test_slot_is_released_after_a_failed_fit(client, monkeypatch, real_train):
    calls = {"count": 0}

    def flaky(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("boom")
        return real_train(*args, **kwargs)

    monkeypatch.setattr(ml_runner_module, "train_classifier", flaky)
    first = client.post("/api/ml/train", json=BODY)
    assert first.status_code == 202
    run_id = first.json()["run_id"]
    detail = _wait_status(client, run_id, {"failed"})
    assert "boom" in detail["error"]
    assert detail["validation"] is None

    # A failed run must not keep the one-job slot.
    second = client.post("/api/ml/train", json=BODY)
    assert second.status_code == 202
    _wait_status(client, second.json()["run_id"], {"completed"})


def test_cancel_goes_running_then_cancelling_then_cancelled(
    client, tmp_path, monkeypatch, real_train
):
    started, release = threading.Event(), threading.Event()
    monkeypatch.setattr(
        ml_runner_module,
        "train_classifier",
        _blocking_trainer(real_train, started, release),
    )
    response = client.post("/api/ml/train", json=BODY)
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert started.wait(10.0)

    control = client.post(f"/api/ml/runs/{run_id}/control", json={"action": "cancel"})
    assert control.status_code == 200
    assert control.json() == {"ok": True}
    # The worker is inside the fit and cannot be interrupted; the API must say
    # "cancelling", not pretend the run already stopped.
    assert client.get(f"/api/ml/runs/{run_id}").json()["status"] == "cancelling"
    meta = json.loads((tmp_path / "runs" / "ml" / run_id / "meta.json").read_text())
    assert meta["status"] == "cancelling"

    release.set()
    detail = _wait_status(client, run_id, {"cancelled"})
    assert detail["error"] == "cancelled by request"
    meta = json.loads((tmp_path / "runs" / "ml" / run_id / "meta.json").read_text())
    assert meta["status"] == "cancelled"

    # A terminal run cannot be cancelled again.
    again = client.post(f"/api/ml/runs/{run_id}/control", json={"action": "cancel"})
    assert again.status_code == 409

    # And the slot is free.
    monkeypatch.setattr(ml_runner_module, "train_classifier", real_train)
    replacement = client.post("/api/ml/train", json=BODY)
    assert replacement.status_code == 202
    _wait_status(client, replacement.json()["run_id"], {"completed"})


def test_evaluate_returns_headline_hold_stress_and_regime(client):
    started = client.post("/api/ml/train", json=BODY)
    assert started.status_code == 202
    run_id = started.json()["run_id"]
    detail = _wait_status(client, run_id, {"completed"})

    response = client.post(
        "/api/ml/evaluate",
        json={
            "run_id": run_id,
            "split": "validation",
            "stress": [2, 3],
            "regime": True,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["run_id"] == run_id
    assert body["symbol"] == SYMBOL
    assert body["timeframe"] == "M1"
    assert body["dataset_id"] == detail["dataset_id"]
    assert body["split"] == "validation"
    assert body["threshold"] == 0.55
    assert {
        "trades",
        "net_profit",
        "profit_factor",
        "win_rate",
        "expectancy_r",
        "max_drawdown",
        "total_costs",
    } == set(body["metrics"])
    assert set(body["hold"]) == set(body["metrics"])
    assert list(body["stress"]) == ["x2", "x3"]
    assert all(set(row) == set(body["metrics"]) for row in body["stress"].values())
    assert set(body["regime"]) == {"rows", "unmatched_trades"}
    assert isinstance(body["regime"]["rows"], list)
    assert isinstance(body["regime"]["unmatched_trades"], int)


def test_evaluate_without_regime_returns_an_empty_table(client):
    run_id = client.post("/api/ml/train", json=BODY).json()["run_id"]
    _wait_status(client, run_id, {"completed"})
    body = client.post(
        "/api/ml/evaluate",
        json={"run_id": run_id, "split": "test", "stress": [], "regime": False},
    ).json()
    assert body["split"] == "test"
    assert body["regime"] == {}


def test_unknown_run_and_missing_artefact_are_404(client, tmp_path):
    assert client.get("/api/ml/runs/nope123").status_code == 404
    assert client.post("/api/ml/runs/nope123/control", json={"action": "cancel"}).status_code == 404
    assert client.post("/api/ml/evaluate", json={"run_id": "nope123"}).status_code == 404

    # A completed-shaped run without model.joblib cannot be evaluated.
    fake = tmp_path / "runs" / "ml" / "fakeabc123"
    fake.mkdir(parents=True)
    (fake / "meta.json").write_text(
        json.dumps(
            {
                "run_id": "fakeabc123",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00+00:00",
                "symbol": SYMBOL,
                "timeframe": "M1",
                "model": "logistic",
                "dataset_id": None,
            }
        ),
        encoding="utf-8",
    )
    assert client.get("/api/ml/runs/fakeabc123").status_code == 200
    response = client.post("/api/ml/evaluate", json={"run_id": "fakeabc123"})
    assert response.status_code == 404
    assert "model artefact" in response.json()["detail"]


def test_bad_requests_are_422(client):
    cases = [
        {**BODY, "model": "nonexistent"},
        {**BODY, "timeframe": "W1"},
        {**BODY, "horizon": 0},
        {**BODY, "horizon": 10_000},
        {**BODY, "groups": ["nonexistent"]},
        {**BODY, "groups": []},
        {**BODY, "symbol": "../XAUUSD"},
        {**BODY, "train_fraction": 0.9, "validation_fraction": 0.2},
        {**BODY, "stress": [0]},
        {**BODY, "stress": [1, 2, 3, 4, 5, 6]},
        {**BODY, "split": "holdout"},
    ]
    for body in cases:
        response = client.post("/api/ml/train", json=body)
        assert response.status_code == 422, (body, response.text)
    assert client.get("/api/ml/runs").json() == []


def test_run_ids_cannot_traverse(client):
    for path in (
        "/api/ml/runs/..",
        "/api/ml/runs/%2e%2e",
        "/api/ml/runs/..%2F..%2Fetc",
        "/api/ml/runs/a%2Fb",
    ):
        response = client.get(path)
        assert response.status_code in {404, 422}, (path, response.status_code)
        assert response.status_code != 200

    response = client.post("/api/ml/evaluate", json={"run_id": "../secret"})
    assert response.status_code == 422
    response = client.post("/api/ml/runs/..%2Fetc/control", json={"action": "cancel"})
    assert response.status_code in {404, 422}


def test_list_reports_the_registry_fields(client):
    run_id = client.post("/api/ml/train", json=BODY).json()["run_id"]
    _wait_status(client, run_id, {"completed"})
    rows = client.get("/api/ml/runs").json()
    assert len(rows) == 1
    assert set(rows[0]) == {
        "run_id",
        "status",
        "created_at",
        "model",
        "symbol",
        "timeframe",
        "dataset_id",
    }
    assert rows[0]["run_id"] == run_id
    assert rows[0]["status"] == "completed"
    assert rows[0]["dataset_id"].startswith(f"{SYMBOL}:M1:h5:")


def test_runner_rejects_an_invalid_id_without_touching_disk(tmp_path):
    """The runner's own guard, independent of the HTTP schema."""
    from scalping.api.ml_runner import MLRunner
    from scalping.config import Settings

    runner = MLRunner(ParquetStore(tmp_path), Settings(), tmp_path / "runs")
    with pytest.raises(UnknownMLRunError):
        runner.detail("../etc")
    assert runner.list_runs() == []


def test_report_and_meta_are_the_only_run_files(client, tmp_path):
    run_id = client.post("/api/ml/train", json=BODY).json()["run_id"]
    _wait_status(client, run_id, {"completed"})
    files = sorted(path.name for path in (tmp_path / "runs" / "ml" / run_id).iterdir())
    assert files == ["meta.json", "metrics.json", "model.joblib", "report.json"]


def test_a_store_too_large_for_the_api_is_rejected(client, tmp_path, monkeypatch):
    """The row guard is enforced before a matrix is materialised."""
    monkeypatch.setattr(ml_runner_module, "MAX_BARS", 100)
    response = client.post("/api/ml/train", json=BODY)
    assert response.status_code == 409
    assert "API limit is 100" in response.json()["detail"]
    assert not (tmp_path / "runs" / "ml").exists()


def test_a_run_started_by_another_process_is_not_controllable(client, tmp_path):
    run_dir = tmp_path / "runs" / "ml" / "diskonly1"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps({"run_id": "diskonly1", "status": "running", "created_at": None}),
        encoding="utf-8",
    )
    assert client.get("/api/ml/runs/diskonly1").json()["status"] == "running"
    assert (
        client.post("/api/ml/runs/diskonly1/control", json={"action": "cancel"}).status_code == 404
    )


def test_shutdown_cancels_a_worker_that_is_still_fitting(tmp_path, monkeypatch, real_train):
    frame = SyntheticProvider(seed=5).fetch(SYMBOL, START, START + timedelta(minutes=BARS))
    ParquetStore(tmp_path).write(SYMBOL, "M1", frame)
    started, release = threading.Event(), threading.Event()
    monkeypatch.setattr(
        ml_runner_module,
        "train_classifier",
        _blocking_trainer(real_train, started, release),
    )

    app = create_app(data_root=tmp_path, runs_root=tmp_path / "runs")
    with TestClient(app) as test_client:
        response = test_client.post("/api/ml/train", json=BODY)
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        assert started.wait(10.0)
        runner = test_client.app.state.ml
        release.set()
        runner.shutdown(timeout=10.0)
        assert runner.active_threads() == []
        detail = _wait_status(test_client, run_id, {"cancelled"})
        assert detail["status"] == "cancelled"
