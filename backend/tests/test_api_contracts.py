"""Shape contracts for the merged HTTP surface.

These tests are guards against **silent schema drift**, not behaviour proofs.
They pin the response envelope of ``/api/paper``, ``/api/ml``, ``/api/autonomy``,
``/api/data/status``, ``/api/regime`` and ``/api/strategy/card``: exact key sets,
JSON types, status codes and the absence of fields a client must never receive
(filesystem paths, handler errors, the orchestrator's free-form event detail).

A passing test here means "the wire shape did not change", never "the feature
is correct". Behaviour, persistence and concurrency are covered by the per-route
suites (``test_api_paper.py``, ``test_api_ml.py``, ``test_api_autonomy.py``,
``test_api_data.py``, ``test_api_regime.py``, ``test_api_strategy.py``).

The two invariants worth stating explicitly:

* Missing **and** unreadable stores are different answers. A missing store is
  ``configured: false``; a file that exists but cannot be read is
  ``configured: true, readable: false`` and carries **no** items and no counts.
* No success body exposes a local path. The checks below scan every key and the
  serialised text for the request's temporary root.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from scalping.api import ml_runner as ml_runner_module
from scalping.api.app import create_app
from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider
from scalping.orchestrator.core import Orchestrator
from scalping.orchestrator.store import JobStore
from scalping.paper.ledger import PaperLedger
from scalping.strategy.card import PolicyCard, RuleMetrics, RuleParams, save_card
from scalping.types import Side

T0 = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)

# ---------------------------------------------------------------- leak guard

#: Keys that must never appear in a success body: they name the server's disk
#: layout, an internal stack, or the FastAPI error envelope. ``error`` itself is
#: part of the ML run contract (a terminal run reports why it failed) and is
#: deliberately not listed.
_LEAKY_KEYS = frozenset(
    {
        "absolute_path",
        "cwd",
        "detail",
        "env",
        "file",
        "filename",
        "path",
        "stack",
        "stacktrace",
        "traceback",
    }
)


def _walk_keys(node: Any) -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.append(str(key))
            found.extend(_walk_keys(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_walk_keys(item))
    return found


def _assert_no_leak(response, root: Path) -> Any:
    """Assert a success body carries no path, stack or error-envelope field."""
    assert 200 <= response.status_code < 300, response.text
    body = response.json()
    leaked = sorted(set(_walk_keys(body)) & _LEAKY_KEYS)
    assert not leaked, f"leaky keys in {response.request.url}: {leaked}"
    assert str(root) not in response.text, f"absolute path in {response.request.url}"
    return body


def _assert_keys(payload: dict[str, Any], expected: set[str], where: str) -> None:
    assert set(payload) == expected, f"{where}: {sorted(set(payload) ^ expected)}"


def _assert_int(value: Any, where: str) -> None:
    assert isinstance(value, int) and not isinstance(value, bool), f"{where}: {value!r}"


def _assert_float(value: Any, where: str) -> None:
    assert isinstance(value, float) and not isinstance(value, bool), f"{where}: {value!r}"


def _assert_str(value: Any, where: str) -> None:
    assert isinstance(value, str) and value, f"{where}: {value!r}"


# =============================================================== /api/paper

PAPER_BODY = {"symbol": "XAUUSD", "timeframe": "M1", "strategy": "flow_momentum"}

SESSION_CREATED_KEYS = {"session_id", "status", "started_at"}
SESSION_ITEM_KEYS = {
    "session_id",
    "symbol",
    "timeframe",
    "strategy",
    "model_version",
    "status",
    "started_at",
    "ended_at",
}
SESSION_SUMMARY_KEYS = {
    "session_id",
    "symbol",
    "timeframe",
    "strategy",
    "model_version",
    "status",
    "started_at",
    "ended_at",
    "balance",
    "equity",
    "open_position",
    "trade_count",
    "last_bar_ts",
    "config",
}
PAGE_KEYS = {"items", "total", "limit", "offset"}
POSITION_KEYS = {
    "position_id",
    "side",
    "lot",
    "entry_ts",
    "entry_price",
    "sl",
    "tp",
    "status",
    "closed_ts",
    "closed_price",
}
ORDER_KEYS = {
    "order_id",
    "signal_id",
    "ts",
    "side",
    "lot",
    "requested_price",
    "status",
    "rejection_reason",
}
TRADE_KEYS = {
    "trade_id",
    "side",
    "entry_ts",
    "entry_price",
    "exit_ts",
    "exit_price",
    "lot",
    "pnl",
    "r_multiple",
    "exit_reason",
}
EQUITY_KEYS = {"ts", "balance", "equity"}


@pytest.fixture
def paper_client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(data_root=tmp_path))


def _seed_ledger(path: Path, session_id: str = "seeded", *, close: bool = True) -> None:
    """One session with an order, an open position and three marks.

    ``close=False`` keeps the position open so the nested ``open_position``
    object on the summary is exercised; ``close=True`` settles it into a trade.
    """
    with PaperLedger(path) as book:
        book.open_session(session_id, symbol="XAUUSD", timeframe="M1", strategy="seed")
        signal_id = book.record_signal(session_id, T0, "buy")
        order_id = book.record_order(
            session_id,
            signal_id=signal_id,
            ts=T0,
            side=Side.LONG,
            lot=0.1,
            requested_price=2350.0,
        )
        book.record_fill(order_id, ts=T0, price=2350.1, lot=0.1, sl=2349.0, tp=2352.0)
        for index in range(3):
            book.record_equity(session_id, T0 + timedelta(minutes=index), 10_000.0, 10_000.0)
        if close:
            book.record_trade(
                session_id,
                side=Side.LONG,
                entry_ts=T0,
                entry_price=2350.1,
                exit_ts=T0 + timedelta(minutes=5),
                exit_price=2352.1,
                lot=0.1,
                pnl=18.0,
                r_multiple=1.4,
                exit_reason="tp",
            )


def test_paper_open_list_get_close_shape(paper_client: TestClient, tmp_path: Path) -> None:
    created = paper_client.post("/api/paper/sessions", json=PAPER_BODY)
    body = _assert_no_leak(created, tmp_path)
    assert created.status_code == 201
    _assert_keys(body, SESSION_CREATED_KEYS, "POST /api/paper/sessions")
    _assert_str(body["session_id"], "session_id")
    assert body["status"] == "running"
    _assert_str(body["started_at"], "started_at")
    session_id = body["session_id"]

    page = _assert_no_leak(paper_client.get("/api/paper/sessions"), tmp_path)
    _assert_keys(page, PAGE_KEYS, "GET /api/paper/sessions")
    _assert_int(page["total"], "total")
    _assert_int(page["limit"], "limit")
    _assert_int(page["offset"], "offset")
    assert isinstance(page["items"], list) and page["items"]
    _assert_keys(page["items"][0], SESSION_ITEM_KEYS, "session item")
    _assert_str(page["items"][0]["started_at"], "session item started_at")

    summary = _assert_no_leak(paper_client.get(f"/api/paper/sessions/{session_id}"), tmp_path)
    _assert_keys(summary, SESSION_SUMMARY_KEYS, "session summary")
    assert summary["balance"] is None and summary["equity"] is None
    assert summary["open_position"] is None
    _assert_int(summary["trade_count"], "trade_count")
    assert summary["config"] == {"origin": "api", "extra": {}}

    closed = _assert_no_leak(paper_client.post(f"/api/paper/sessions/{session_id}/close"), tmp_path)
    _assert_keys(closed, SESSION_SUMMARY_KEYS, "closed summary")
    assert closed["status"] == "closed"
    _assert_str(closed["ended_at"], "ended_at")


def test_paper_item_shapes_from_a_settled_ledger(paper_client: TestClient, tmp_path: Path) -> None:
    _seed_ledger(tmp_path / "paper.sqlite")
    _seed_ledger(tmp_path / "paper.sqlite", "open", close=False)

    positions = _assert_no_leak(paper_client.get("/api/paper/sessions/seeded/positions"), tmp_path)
    _assert_keys(positions, PAGE_KEYS, "positions page")
    _assert_keys(positions["items"][0], POSITION_KEYS, "position item")
    _assert_int(positions["items"][0]["position_id"], "position_id")
    _assert_float(positions["items"][0]["entry_price"], "entry_price")

    orders = _assert_no_leak(paper_client.get("/api/paper/sessions/seeded/orders"), tmp_path)
    _assert_keys(orders["items"][0], ORDER_KEYS, "order item")
    _assert_int(orders["items"][0]["order_id"], "order_id")

    history = _assert_no_leak(paper_client.get("/api/paper/sessions/seeded/history"), tmp_path)
    _assert_keys(history["items"][0], TRADE_KEYS, "trade item")
    _assert_float(history["items"][0]["pnl"], "pnl")

    equity = _assert_no_leak(paper_client.get("/api/paper/sessions/seeded/equity"), tmp_path)
    _assert_keys(equity["items"][0], EQUITY_KEYS, "equity item")
    _assert_float(equity["items"][0]["balance"], "balance")

    summary = _assert_no_leak(paper_client.get("/api/paper/sessions/seeded"), tmp_path)
    _assert_keys(summary, SESSION_SUMMARY_KEYS, "summary")
    assert summary["open_position"] is None
    assert summary["trade_count"] == 1

    still_open = _assert_no_leak(paper_client.get("/api/paper/sessions/open"), tmp_path)
    _assert_keys(still_open, SESSION_SUMMARY_KEYS, "open summary")
    _assert_keys(still_open["open_position"], POSITION_KEYS, "summary open_position")
    assert still_open["trade_count"] == 0


@pytest.mark.parametrize("suffix", ["", "/positions", "/orders", "/history", "/equity"])
@pytest.mark.parametrize("limit,offset", [(0, 0), (1001, 0), (100, -1), (-1, 0)])
def test_paper_pagination_bounds(
    paper_client: TestClient, suffix: str, limit: int, offset: int
) -> None:
    session_id = paper_client.post("/api/paper/sessions", json=PAPER_BODY).json()["session_id"]
    base = "/api/paper/sessions" if not suffix else f"/api/paper/sessions/{session_id}{suffix}"
    response = paper_client.get(base, params={"limit": limit, "offset": offset})
    assert response.status_code == 422, (base, limit, offset, response.text)


@pytest.mark.parametrize("limit", [1, 1000])
def test_paper_pagination_accepts_the_documented_range(
    paper_client: TestClient, limit: int
) -> None:
    body = paper_client.get("/api/paper/sessions", params={"limit": limit}).json()
    assert body["limit"] == limit


def test_paper_error_codes(paper_client: TestClient) -> None:
    unknown = paper_client.get("/api/paper/sessions/unknown")
    assert unknown.status_code == 404
    _assert_keys(unknown.json(), {"detail"}, "paper 404")
    assert paper_client.post("/api/paper/sessions/unknown/close").status_code == 404

    session_id = paper_client.post("/api/paper/sessions", json=PAPER_BODY).json()["session_id"]
    assert paper_client.post(f"/api/paper/sessions/{session_id}/close").status_code == 200
    retry = paper_client.post(f"/api/paper/sessions/{session_id}/close")
    assert retry.status_code == 409
    _assert_keys(retry.json(), {"detail"}, "paper 409")

    for body in (
        {**PAPER_BODY, "symbol": "../x"},
        {**PAPER_BODY, "timeframe": "Y1"},
        {**PAPER_BODY, "strategy": ""},
    ):
        assert paper_client.post("/api/paper/sessions", json=body).status_code == 422
    assert paper_client.get("/api/paper/sessions/.hidden").status_code == 422


# ================================================================== /api/ml

ML_SYMBOL = "XAUUSD"
ML_START = datetime(2026, 1, 5, tzinfo=UTC)
#: 15 hours of M1: enough for a 70/15/15 split after warm-up, fast to fit.
ML_BARS = 900
ML_BODY = {
    "symbol": ML_SYMBOL,
    "timeframe": "M1",
    "model": "logistic",
    "horizon": 5,
    "groups": ["price"],
    "purge_bars": 5,
    "embargo_bars": 5,
}
ML_DETAIL_KEYS = {
    "run_id",
    "status",
    "created_at",
    "finished_at",
    "model",
    "symbol",
    "timeframe",
    "dataset_id",
    "config",
    "spec",
    "columns",
    "validation",
    "test",
    "baseline",
    "split_counts",
    "calibration",
    "error",
}


def _wait_status(
    client: TestClient, run_id: str, statuses: set[str], timeout: float = 30.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/ml/runs/{run_id}")
        if response.status_code == 200:
            last = response.json()
            if last["status"] in statuses:
                return last
        time.sleep(0.02)
    raise AssertionError(f"ML run {run_id} never reached {statuses}; last {last}")


@pytest.fixture
def ml_client(tmp_path: Path):
    frame = SyntheticProvider(seed=5).fetch(
        ML_SYMBOL, ML_START, ML_START + timedelta(minutes=ML_BARS)
    )
    ParquetStore(tmp_path).write(ML_SYMBOL, "M1", frame)
    app = create_app(data_root=tmp_path, runs_root=tmp_path / "runs")
    with TestClient(app) as client:
        yield client


@pytest.fixture
def real_train():
    return ml_runner_module.train_classifier


def test_ml_models_shape(ml_client: TestClient) -> None:
    response = ml_client.get("/api/ml/models")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list) and body
    assert all(isinstance(name, str) and name for name in body)
    assert len(body) == len(set(body))
    assert "logistic" in body


def test_ml_train_202_and_run_detail_shape(ml_client: TestClient, tmp_path: Path) -> None:
    started = ml_client.post("/api/ml/train", json=ML_BODY)
    assert started.status_code == 202, started.text
    _assert_keys(started.json(), {"run_id"}, "POST /api/ml/train")
    run_id = started.json()["run_id"]

    _wait_status(ml_client, run_id, {"completed"})
    detail = _assert_no_leak(ml_client.get(f"/api/ml/runs/{run_id}"), tmp_path)
    _assert_keys(detail, ML_DETAIL_KEYS, "GET /api/ml/runs/{id}")
    assert detail["status"] == "completed"
    _assert_str(detail["dataset_id"], "dataset_id")
    assert isinstance(detail["config"], dict) and detail["config"]
    assert isinstance(detail["spec"], dict) and detail["spec"]["model"] == "logistic"
    assert isinstance(detail["columns"], list) and detail["columns"]
    assert isinstance(detail["validation"], dict)
    assert isinstance(detail["baseline"], dict)
    assert isinstance(detail["split_counts"], dict)
    _assert_keys(detail["calibration"], {"brier", "log_loss", "ece", "mce"}, "calibration")
    assert detail["test"] is None
    assert detail["error"] is None

    rows = _assert_no_leak(ml_client.get("/api/ml/runs"), tmp_path)
    assert isinstance(rows, list) and len(rows) == 1
    _assert_keys(
        rows[0],
        {"run_id", "status", "created_at", "model", "symbol", "timeframe", "dataset_id"},
        "run summary",
    )


def test_ml_train_rejects_bad_fields_with_422(ml_client: TestClient) -> None:
    for payload in (
        {**ML_BODY, "model": "nonexistent"},
        {**ML_BODY, "timeframe": "W1"},
        {**ML_BODY, "horizon": 0},
        {**ML_BODY, "groups": ["nonexistent"]},
        {**ML_BODY, "symbol": "../XAUUSD"},
    ):
        response = ml_client.post("/api/ml/train", json=payload)
        assert response.status_code == 422, (payload, response.text)
        _assert_keys(response.json(), {"detail"}, "ML 422")


def test_ml_train_conflict_is_409(ml_client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(ml_runner_module, "MAX_BARS", 100)
    response = ml_client.post("/api/ml/train", json=ML_BODY)
    assert response.status_code == 409, response.text
    _assert_keys(response.json(), {"detail"}, "ML 409")


def test_ml_evaluate_404_and_409(ml_client: TestClient, tmp_path: Path) -> None:
    unknown = ml_client.post("/api/ml/evaluate", json={"run_id": "nope123"})
    assert unknown.status_code == 404

    # A run whose meta says "running" cannot be evaluated on this process.
    live = tmp_path / "runs" / "ml" / "livefake1"
    live.mkdir(parents=True)
    (live / "meta.json").write_text(
        json.dumps({"run_id": "livefake1", "status": "running", "created_at": None}),
        encoding="utf-8",
    )
    busy = ml_client.post("/api/ml/evaluate", json={"run_id": "livefake1"})
    assert busy.status_code == 409, busy.text

    # A finished-looking run without the model artefact is a 404, not a 409.
    completed = tmp_path / "runs" / "ml" / "nofile1"
    completed.mkdir(parents=True)
    (completed / "meta.json").write_text(
        json.dumps({"run_id": "nofile1", "status": "completed", "created_at": None}),
        encoding="utf-8",
    )
    assert ml_client.post("/api/ml/evaluate", json={"run_id": "nofile1"}).status_code == 404


def test_ml_cancel_unknown_and_bad_action(ml_client: TestClient) -> None:
    assert (
        ml_client.post("/api/ml/runs/nope123/control", json={"action": "cancel"}).status_code == 404
    )
    assert (
        ml_client.post("/api/ml/runs/nope123/control", json={"action": "stop"}).status_code == 422
    )


def test_ml_cancel_live_run_shape(ml_client: TestClient, monkeypatch, real_train) -> None:
    started, release = threading.Event(), threading.Event()

    def blocking(*args, **kwargs):
        started.set()
        if not release.wait(10.0):
            raise RuntimeError("test never released the trainer")
        return real_train(*args, **kwargs)

    monkeypatch.setattr(ml_runner_module, "train_classifier", blocking)
    run_id = ml_client.post("/api/ml/train", json=ML_BODY).json()["run_id"]
    assert started.wait(10.0), "the worker never entered the fit"

    try:
        cancelled = ml_client.post(f"/api/ml/runs/{run_id}/control", json={"action": "cancel"})
        assert cancelled.status_code == 200, cancelled.text
        _assert_keys(cancelled.json(), {"ok"}, "cancel")
        assert cancelled.json()["ok"] is True

        retry = ml_client.post(f"/api/ml/runs/{run_id}/control", json={"action": "cancel"})
        assert retry.status_code == 409, retry.text
        _assert_keys(retry.json(), {"detail"}, "cancel 409")
    finally:
        release.set()
    _wait_status(ml_client, run_id, {"cancelled"})


# ============================================================ /api/autonomy

AUTONOMY_PAPER_KEYS = {"configured", "readable", "sessions", "open_positions", "closed_trades"}
AUTONOMY_ORCHESTRATOR_KEYS = {
    "configured",
    "readable",
    "queued",
    "running",
    "failed",
    "paused",
    "safe_mode",
    "research_disabled",
    "last_event",
}
AUTONOMY_CHAMPION_KEYS = {"scope", "candidate_id", "mode", "promoted_at", "policy_digest"}
AUTONOMY_JOB_KEYS = {
    "id",
    "kind",
    "status",
    "attempts",
    "defer_count",
    "priority",
    "created_at",
    "scheduled_at",
}
AUTONOMY_EVENT_KEYS = {"id", "ts", "event"}


@pytest.fixture
def autonomy_paths(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "data_root": tmp_path / "data",
        "runs_root": tmp_path / "runs",
        "paper": tmp_path / "paper.sqlite",
        "orchestrator": tmp_path / "runs" / "orchestrator.sqlite3",
        "archive": tmp_path / "runs" / "discovery" / "discovery.sqlite",
    }
    paths["data_root"].mkdir()
    paths["runs_root"].mkdir()
    return paths


@pytest.fixture
def autonomy_client(autonomy_paths: dict[str, Path]) -> TestClient:
    app = create_app(
        data_root=autonomy_paths["data_root"],
        runs_root=autonomy_paths["runs_root"],
        paper_ledger=autonomy_paths["paper"],
        orchestrator_db=autonomy_paths["orchestrator"],
        discovery_archive=autonomy_paths["archive"],
    )
    return TestClient(app)


def _seed_autonomy_paper(path: Path) -> None:
    with PaperLedger(path) as book:
        book.open_session("seed", symbol="XAUUSD", timeframe="M1", strategy="seed")
        book.record_equity("seed", T0, 10_000.0, 10_000.0)


def _seed_autonomy_orchestrator(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with JobStore(path) as store:
        store.enqueue("discovery.run", dedupe_key="contract")
        Orchestrator(store).pause("contract")


def _seed_autonomy_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE champions (scope TEXT, candidate_id TEXT, promoted_at TEXT)")
        conn.execute(
            "INSERT INTO champions VALUES ('XAUUSD:M5', 'abc123', '2026-01-05T00:00:00+00:00')"
        )
        conn.execute("CREATE TABLE trials (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE decisions (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()


def test_autonomy_missing_stores_shape(
    autonomy_client: TestClient, autonomy_paths: dict[str, Path]
) -> None:
    status = _assert_no_leak(
        autonomy_client.get("/api/autonomy/status"), autonomy_paths["data_root"].parent
    )
    _assert_keys(
        status,
        {
            "paper",
            "orchestrator",
            "champions",
            "champions_readable",
            "discoveries",
            "data",
        },
        "autonomy status",
    )
    assert status["paper"] == {"configured": False, "readable": False}
    assert status["orchestrator"] == {"configured": False, "readable": False}
    assert status["champions"] == []
    assert status["champions_readable"] is False
    assert status["discoveries"] == {"configured": False, "readable": False}
    assert status["data"] == {"configured": True, "readable": True, "symbols": []}

    champions = autonomy_client.get("/api/autonomy/champions")
    assert champions.json() == {"configured": False, "readable": False}
    jobs = autonomy_client.get("/api/autonomy/jobs")
    assert jobs.json() == {"configured": False, "readable": False, "limit": 50}
    events = autonomy_client.get("/api/autonomy/events")
    assert events.json() == {"configured": False, "readable": False, "limit": 100}


def test_autonomy_seeded_shapes(
    autonomy_client: TestClient, autonomy_paths: dict[str, Path]
) -> None:
    _seed_autonomy_paper(autonomy_paths["paper"])
    _seed_autonomy_orchestrator(autonomy_paths["orchestrator"])
    _seed_autonomy_archive(autonomy_paths["archive"])

    status = _assert_no_leak(
        autonomy_client.get("/api/autonomy/status"), autonomy_paths["data_root"].parent
    )
    _assert_keys(status["paper"], AUTONOMY_PAPER_KEYS, "status.paper")
    assert status["paper"]["sessions"] == 1
    _assert_keys(status["orchestrator"], AUTONOMY_ORCHESTRATOR_KEYS, "status.orchestrator")
    assert status["orchestrator"]["queued"] == 1
    assert status["orchestrator"]["paused"] is True
    _assert_keys(status["orchestrator"]["last_event"], AUTONOMY_EVENT_KEYS, "status last_event")
    assert status["champions_readable"] is True
    assert isinstance(status["champions"], list) and len(status["champions"]) == 1
    _assert_keys(status["champions"][0], AUTONOMY_CHAMPION_KEYS, "status champion")
    assert status["champions"][0]["policy_digest"] == "abc123"
    _assert_keys(
        status["discoveries"],
        {"configured", "readable", "runs", "decisions"},
        "status.discoveries",
    )
    assert status["discoveries"] == {
        "configured": True,
        "readable": True,
        "runs": 0,
        "decisions": 0,
    }
    _assert_keys(status["data"], {"configured", "readable", "symbols"}, "status.data")

    champions = _assert_no_leak(
        autonomy_client.get("/api/autonomy/champions"), autonomy_paths["data_root"].parent
    )
    _assert_keys(champions, {"configured", "readable", "items"}, "champions")
    _assert_keys(champions["items"][0], AUTONOMY_CHAMPION_KEYS, "champion item")

    jobs = _assert_no_leak(
        autonomy_client.get("/api/autonomy/jobs"), autonomy_paths["data_root"].parent
    )
    _assert_keys(jobs, {"configured", "readable", "items", "limit"}, "jobs")
    _assert_keys(jobs["items"][0], AUTONOMY_JOB_KEYS, "job item")
    _assert_int(jobs["items"][0]["defer_count"], "defer_count")

    events = _assert_no_leak(
        autonomy_client.get("/api/autonomy/events"), autonomy_paths["data_root"].parent
    )
    _assert_keys(events, {"configured", "readable", "items", "limit"}, "events")
    assert events["items"], "the pause control must record an event"
    for item in events["items"]:
        assert {"id", "ts", "event"} <= set(item) <= {"id", "job_id", "ts", "event"}
        assert "detail" not in item


def test_autonomy_unreadable_store_has_no_items(
    autonomy_client: TestClient, autonomy_paths: dict[str, Path]
) -> None:
    for key in ("paper", "orchestrator", "archive"):
        autonomy_paths[key].parent.mkdir(parents=True, exist_ok=True)
        autonomy_paths[key].write_bytes(b"this is not a sqlite database")

    status = autonomy_client.get("/api/autonomy/status").json()
    assert status["paper"] == {"configured": True, "readable": False}
    assert status["orchestrator"] == {"configured": True, "readable": False}
    assert status["champions"] == []
    assert status["champions_readable"] is False
    assert status["discoveries"] == {"configured": True, "readable": False}

    assert autonomy_client.get("/api/autonomy/champions").json() == {
        "configured": True,
        "readable": False,
    }
    assert autonomy_client.get("/api/autonomy/jobs").json() == {
        "configured": True,
        "readable": False,
        "limit": 50,
    }
    assert autonomy_client.get("/api/autonomy/events").json() == {
        "configured": True,
        "readable": False,
        "limit": 100,
    }


@pytest.mark.parametrize("route", ["jobs", "events"])
@pytest.mark.parametrize("limit", [1, 500])
def test_autonomy_limits_accept_the_documented_range(
    autonomy_client: TestClient, route: str, limit: int
) -> None:
    body = autonomy_client.get(f"/api/autonomy/{route}", params={"limit": limit}).json()
    assert body["limit"] == limit


@pytest.mark.parametrize("route", ["jobs", "events"])
@pytest.mark.parametrize("limit", [0, 501, -1])
def test_autonomy_limits_are_bounded(autonomy_client: TestClient, route: str, limit: int) -> None:
    response = autonomy_client.get(f"/api/autonomy/{route}", params={"limit": limit})
    assert response.status_code == 422, (route, limit, response.text)


# =============================================== data, regime and strategy card

DATA_STATUS_KEYS = {
    "symbol",
    "timeframe",
    "rows",
    "first_ts",
    "last_ts",
    "providers",
    "provenance",
}
PROVENANCE_KEYS = {
    "symbol",
    "timeframe",
    "source",
    "synthetic",
    "rows",
    "first_ts",
    "last_ts",
    "checksum",
    "created_at",
    "notes",
    "dataset_id",
}
DOWNLOAD_BODY = {
    "provider": "synthetic",
    "symbol": "XAUUSD",
    "seed": 7,
    "start": "2026-01-05T00:00:00Z",
    "end": "2026-01-07T00:00:00Z",
}


def test_data_status_shape(tmp_path: Path) -> None:
    with TestClient(create_app(data_root=tmp_path)) as client:
        empty = _assert_no_leak(
            client.get("/api/data/status", params={"symbol": "XAUUSD"}), tmp_path
        )
        _assert_keys(empty, DATA_STATUS_KEYS, "empty data status")
        assert empty["rows"] == 0
        assert empty["first_ts"] is None and empty["last_ts"] is None
        assert empty["providers"] == [] and empty["provenance"] is None

        assert client.post("/api/data/download", json=DOWNLOAD_BODY).status_code == 200
        filled = _assert_no_leak(
            client.get("/api/data/status", params={"symbol": "XAUUSD"}), tmp_path
        )
        _assert_keys(filled, DATA_STATUS_KEYS, "filled data status")
        _assert_int(filled["rows"], "rows")
        _assert_str(filled["first_ts"], "first_ts")
        _assert_str(filled["last_ts"], "last_ts")
        assert filled["providers"] == ["synthetic"]
        _assert_keys(filled["provenance"], PROVENANCE_KEYS, "provenance")
        assert filled["provenance"]["synthetic"] is True


def test_regime_shape(tmp_path: Path, seed_data) -> None:
    with TestClient(create_app(data_root=tmp_path)) as client:
        body = _assert_no_leak(
            client.get("/api/regime", params={"symbol": "XAUUSD", "timeframe": "M1"}), tmp_path
        )
        empty = _assert_no_leak(
            client.get("/api/regime", params={"symbol": "EURUSD", "timeframe": "M1"}), tmp_path
        )

    _assert_keys(body, {"symbol", "timeframe", "rows", "current", "counts", "series"}, "regime")
    _assert_int(body["rows"], "regime rows")
    _assert_keys(
        body["current"],
        {
            "ts",
            "trend",
            "trend_strength",
            "volatility",
            "expansion",
            "breakout",
            "unusual",
            "regime",
        },
        "regime current",
    )
    assert isinstance(body["current"]["unusual"], bool)
    _assert_keys(body["counts"][0], {"regime", "bars", "share"}, "regime count")
    _assert_float(body["counts"][0]["share"], "regime share")
    _assert_keys(body["series"][0], {"ts", "regime", "trend", "volatility"}, "regime point")

    _assert_keys(empty, {"symbol", "timeframe", "rows", "current", "counts", "series"}, "empty")
    assert empty["rows"] == 0
    assert empty["current"] is None and empty["counts"] == [] and empty["series"] == []


def test_strategy_card_inactive_shape(tmp_path: Path) -> None:
    with TestClient(create_app(data_root=tmp_path, runs_root=tmp_path / "runs")) as client:
        body = _assert_no_leak(client.get("/api/strategy/card"), tmp_path)

    _assert_keys(body, {"active", "strategy", "timeframe", "forward"}, "inactive card")
    assert body["active"] is False
    _assert_str(body["strategy"], "strategy")
    _assert_str(body["timeframe"], "timeframe")
    assert body["forward"] is None


def test_strategy_card_active_shape(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    card = PolicyCard.build(
        RuleParams(
            timeframe="M1",
            flow_quantile=0.8,
            flow_window=60,
            sl_atr_mult=3.0,
            tp_atr_mult=6.0,
            max_bars_in_trade=48,
        ),
        RuleMetrics(trades=10, expectancy_r=0.1),
        RuleMetrics(trades=5, expectancy_r=0.05),
        "digest",
    )
    save_card(runs / "strategy_card.json", card)

    with TestClient(create_app(data_root=tmp_path, runs_root=runs)) as client:
        body = _assert_no_leak(client.get("/api/strategy/card"), tmp_path)

    _assert_keys(
        body,
        {
            "active",
            "version",
            "strategy",
            "symbol",
            "params",
            "train",
            "holdout",
            "digest",
            "created_at",
            "note",
            "forward",
        },
        "active card",
    )
    assert body["active"] is True
    _assert_int(body["version"], "card version")
    _assert_keys(
        body["params"],
        {
            "timeframe",
            "flow_quantile",
            "flow_window",
            "sl_atr_mult",
            "tp_atr_mult",
            "max_bars_in_trade",
        },
        "card params",
    )
    metric_keys = {
        "trades",
        "expectancy_r",
        "profit_factor",
        "win_rate",
        "max_drawdown",
        "total_r",
    }
    _assert_keys(body["train"], metric_keys, "card train")
    _assert_keys(body["holdout"], metric_keys, "card holdout")
    assert body["forward"] is None
