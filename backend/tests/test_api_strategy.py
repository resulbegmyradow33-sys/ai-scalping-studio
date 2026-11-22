"""Strategy card endpoints and the runner's card path."""

from __future__ import annotations

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


def test_card_endpoint_reads_the_promoted_card(tmp_path, flow_root) -> None:
    runs = tmp_path / "runs"
    save_card(runs / "strategy_card.json", _card())
    with TestClient(
        create_app(data_root=tmp_path, runs_root=runs, flow_root=flow_root)
    ) as client:
        body = client.get("/api/strategy/card").json()
    assert body["active"] is True
    assert body["params"]["timeframe"] == "M1"
    assert body["holdout"]["expectancy_r"] == pytest.approx(0.05)


def test_calibrate_requires_a_flow_store(tmp_path) -> None:
    with TestClient(create_app(data_root=tmp_path, runs_root=tmp_path / "runs")) as client:
        response = client.post("/api/strategy/calibrate", json={"timeframe": "M1"})
    assert response.status_code == 409


def test_calibrate_dry_run_does_not_write_a_card(tmp_path, flow_root, seed_data) -> None:
    runs = tmp_path / "runs"
    with TestClient(
        create_app(data_root=tmp_path, runs_root=runs, flow_root=flow_root)
    ) as client:
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
    assert response.status_code == 200
    assert app.state.sim.strategy_card()["active"] is True
