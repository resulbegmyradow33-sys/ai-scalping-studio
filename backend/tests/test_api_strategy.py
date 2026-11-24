"""Strategy card endpoints and the runner's card path."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from scalping.api.app import create_app
from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.strategy.card import PolicyCard, RuleMetrics, RuleParams, save_card

SYMBOL = "XAUUSD"


@pytest.fixture
def flow_root(tmp_path: Path, seed_data) -> Path:
    """One flow row per stored M1 bar, so a card can be built on M1."""
    store = ParquetStore(tmp_path)
    frame = store.read(SYMBOL, "M1")
    root = tmp_path / "flow" / SYMBOL / "M1"
    root.mkdir(parents=True)
    pl.DataFrame(
        {
            "ts": frame["ts"],
            "buy_vol": np.full(frame.height, 2.0),
            "sell_vol": np.full(frame.height, 1.0),
            "flow": np.full(frame.height, 0.5),
        }
    ).write_parquet(root / "part-0001.parquet")
    return tmp_path / "flow"


def _card(timeframe: str = "M1") -> PolicyCard:
    return PolicyCard.build(
        RuleParams(
            timeframe=timeframe,
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


def test_card_endpoint_is_inactive_without_a_card(tmp_path) -> None:
    with TestClient(create_app(data_root=tmp_path, runs_root=tmp_path / "runs")) as client:
        body = client.get("/api/strategy/card").json()
    assert body["active"] is False
    assert body["classification"] == "fallback"
    assert body["validation"] == "legacy-heuristic"
    assert body["fallback_reason"] == "no_strategy_card"


def test_card_endpoint_reads_the_promoted_card(tmp_path, flow_root) -> None:
    runs = tmp_path / "runs"
    save_card(runs / "strategy_card.json", _card())
    with TestClient(create_app(data_root=tmp_path, runs_root=runs, flow_root=flow_root)) as client:
        body = client.get("/api/strategy/card").json()
    assert body["active"] is True
    assert body["params"]["timeframe"] == "M1"
    assert body["holdout"]["expectancy_r"] == pytest.approx(0.05)
    assert body["classification"] == "legacy"
    assert body["validation"] == "legacy-heuristic"
    assert body["active_rule"] == "flow_momentum"


def test_calibrate_requires_a_flow_store(tmp_path) -> None:
    with TestClient(create_app(data_root=tmp_path, runs_root=tmp_path / "runs")) as client:
        response = client.post("/api/strategy/calibrate", json={"timeframe": "M1"})
    assert response.status_code == 409


def test_calibrate_dry_run_does_not_write_a_card(tmp_path, flow_root, seed_data) -> None:
    runs = tmp_path / "runs"
    with TestClient(create_app(data_root=tmp_path, runs_root=runs, flow_root=flow_root)) as client:
        response = client.post(
            "/api/strategy/calibrate",
            json={
                "symbol": SYMBOL,
                "timeframe": "M1",
                "min_train_trades": 10,
                "min_holdout_trades": 5,
                "dry_run": True,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert "candidate" in body
    assert "reason" in body
    assert not (runs / "strategy_card.json").exists()


def test_runner_starts_on_the_card_timeframe(tmp_path, flow_root, seed_data) -> None:
    runs = tmp_path / "runs"
    save_card(runs / "strategy_card.json", _card())
    app = create_app(
        data_root=tmp_path,
        runs_root=runs,
        flow_root=flow_root,
        settings_path=None,
    )
    with TestClient(app) as client:
        response = client.post("/api/sim/start", json={"speed": 0, "mode": "replay"})
        assert response.status_code == 200
        assert response.json()["session_id"]
    assert isinstance(app.state.settings, Settings)
    assert app.state.sim.strategy_card()["active"] is True


def test_runner_falls_back_when_the_card_frame_is_missing(tmp_path, flow_root) -> None:
    runs = tmp_path / "runs"
    save_card(runs / "strategy_card.json", _card(timeframe="H1"))
    app = create_app(data_root=tmp_path, runs_root=runs, flow_root=flow_root)
    with TestClient(app) as client:
        # No H1 bars in the store: the runner must fall back to the M1 replay
        # instead of failing on a card it cannot build.
        response = client.post("/api/sim/start", json={"speed": 0, "mode": "replay"})
        body = client.get("/api/strategy/card").json()
    assert response.status_code == 200
    # The card is on disk but did not run: the endpoint must say so rather than
    # repeating the old active: true misreport.
    assert body["active"] is False
    assert body["classification"] == "fallback"
    assert body["validation"] == "legacy-heuristic"
    assert body["fallback_reason"] == "strategy_card_not_buildable"
    assert body["legacy_fallback"] is True


def test_runner_flag_off_fails_closed_without_a_card(tmp_path, seed_data) -> None:
    app = create_app(
        data_root=tmp_path,
        runs_root=tmp_path / "runs",
        legacy_fallback=False,
    )
    with TestClient(app) as client:
        with client.websocket_connect("/ws/sim") as ws:
            assert ws.receive_json()["type"] == "status"
            assert (
                client.post("/api/sim/start", json={"speed": 0, "mode": "replay"}).status_code
                == 200
            )
            decision = None
            for _ in range(10):
                message = ws.receive_json()
                if message["type"] == "decision":
                    decision = message["payload"]
                    break
        body = client.get("/api/strategy/card").json()
    assert decision is not None
    assert decision["strategy"] == "no_trade"
    assert decision["action"] == 0
    assert decision["no_trade"] is True
    assert decision["reason"] == "legacy_fallback_disabled"
    assert decision["blocked"] == "legacy_fallback_disabled"
    assert body["active"] is False
    assert body["classification"] == "fallback"
    assert body["fallback_reason"] == "legacy_fallback_disabled"
    assert body["legacy_fallback"] is False


def test_runner_flag_on_keeps_the_legacy_replay(tmp_path, seed_data) -> None:
    app = create_app(data_root=tmp_path, runs_root=tmp_path / "runs")
    with TestClient(app) as client, client.websocket_connect("/ws/sim") as ws:
        assert ws.receive_json()["type"] == "status"
        assert (
            client.post("/api/sim/start", json={"speed": 0, "mode": "replay"}).status_code
            == 200
        )
        decision = None
        for _ in range(10):
            message = ws.receive_json()
            if message["type"] == "decision":
                decision = message["payload"]
                break
    assert decision is not None
    assert decision["strategy"] == "momentum"
    assert decision["classification"] == "fallback"
    assert "no_trade" not in decision


def test_calibrate_writes_an_audit_record(tmp_path, flow_root, seed_data) -> None:
    runs = tmp_path / "runs"
    with TestClient(create_app(data_root=tmp_path, runs_root=runs, flow_root=flow_root)) as client:
        response = client.post(
            "/api/strategy/calibrate",
            json={
                "symbol": SYMBOL,
                "timeframe": "M1",
                "min_train_trades": 10,
                "min_holdout_trades": 5,
                "dry_run": True,
            },
        )
    assert response.status_code == 200
    audit_path = runs / "strategy_calibration_audit.jsonl"
    assert audit_path.exists()
    rows = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["event"] == "calibration"
    assert row["classification"] == "legacy"
    assert row["dry_run"] is True
    assert row["card_written"] is False
    assert row["actor"].startswith("api:")
    assert row["result"]["promoted"] is False
    assert row["result"]["reason"] == response.json()["reason"]
    assert datetime.fromisoformat(row["at"]).tzinfo is not None
    assert response.json()["audit"]["actor"] == row["actor"]
    assert not (runs / "strategy_card.json").exists()


def test_saved_card_is_labelled_legacy(tmp_path) -> None:
    from scalping.strategy.card import load_card

    path = tmp_path / "strategy_card.json"
    save_card(path, _card())
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["classification"] == "legacy"
    assert load_card(path) is not None


def test_card_endpoint_includes_the_latest_forward_result(tmp_path, flow_root) -> None:
    runs = tmp_path / "runs"
    save_card(runs / "strategy_card.json", _card())
    (runs / "forward_history.jsonl").write_text(
        json.dumps(
            {
                "at": "2026-11-22T06:41:59+00:00",
                "start": "2026-06-01",
                "end": "2026-11-21",
                "forward": {
                    "trades": 24,
                    "expectancy_r": 0.502,
                    "profit_factor": 2.15,
                    "net_profit": 480.6,
                    "max_drawdown": 0.024,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with TestClient(create_app(data_root=tmp_path, runs_root=runs, flow_root=flow_root)) as client:
        body = client.get("/api/strategy/card").json()
    assert body["forward"]["trades"] == 24
    assert body["forward"]["expectancy_r"] == pytest.approx(0.502)
    assert body["forward"]["start"] == "2026-06-01"
